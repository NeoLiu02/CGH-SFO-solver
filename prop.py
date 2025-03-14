import torch
from torch import nn
import torch.fft as fft
from PIL import Image
import cv2
import numpy as np


class ASM_split(nn.Module):
    def __init__(self, wavelength, res, roi, pitch, apply_constraint=False):
        super().__init__()
        self.pitch = pitch #mm
        self.wavelength = wavelength*1e-6 #mm
        self.pixelx = res[0]
        self.pixely = res[1]

        self.pad = 1
        self.fx = torch.linspace(-1/2/self.pitch, 1/2/self.pitch, (1+self.pad)*self.pixelx).cuda()
        self.dfx = 1 / ((1+self.pad)*self.pixelx * self.pitch)
        self.fy = torch.linspace(-1/2/self.pitch, 1/2/self.pitch, (1+self.pad)*self.pixely).cuda()
        self.dfy = 1 / ((1+self.pad)*self.pixely * self.pitch)
        self.fx, self.fy = torch.meshgrid(self.fx, self.fy)
        self.padding = nn.ZeroPad2d((int(self.pad/2*self.pixely), int(self.pad/2*self.pixely),
                                     int(self.pad/2*self.pixelx), int(self.pad/2*self.pixelx))).cuda()

        self.cropx = (res[0]-roi[0])//2
        self.cropy = (res[1]-roi[1])//2
        self.apply_constraint = apply_constraint

    def forward(self, amplitude, phase, distance): #holo[4,1,2160,3840] #distance[4,1]

        batchsize = distance.shape[0]
        fX = self.fx.unsqueeze(0).unsqueeze(0).repeat(batchsize, 1, 1, 1)
        fY = self.fy.unsqueeze(0).unsqueeze(0).repeat(batchsize, 1, 1, 1)

        distance = distance.unsqueeze(2).unsqueeze(3) ##[4,1,1,1]
        holor = amplitude * torch.cos(phase)
        holoi = amplitude * torch.sin(phase)

        holor = self.padding(holor)
        holoi = self.padding(holoi)

        bandlimitx = 1 / (torch.sqrt((2 * self.dfx * distance) ** 2 + 1) * self.wavelength)  # [4,1,1,1]
        bandlimity = 1 / (torch.sqrt((2 * self.dfy * distance) ** 2 + 1) * self.wavelength)
        bandlimit = (torch.abs(fX) < bandlimitx) & (torch.abs(fY) < bandlimity)

        if self.apply_constraint: # apply constraint on freq domain
            fxlimit = 1/3/self.pitch
            fylimit = 1/3/self.pitch
            bandlimit_constraint = (torch.abs(fX) < fxlimit) & (torch.abs(fY) < fylimit)
            bandlimit = bandlimit*bandlimit_constraint

        gamma = (torch.sqrt(1 - (self.wavelength*fX) ** 2 - (self.wavelength*fY) ** 2))

        Hr = torch.cos(2 * torch.pi / self.wavelength * distance * gamma) * bandlimit
        Hi = torch.sin(2 * torch.pi / self.wavelength * distance * gamma) * bandlimit

        #propagation
        Fre1 = fft.fftshift(fft.fft2(fft.fftshift(holor, dim=(-2, -1)), dim=(-2, -1), norm='ortho'), dim=(-2, -1))
        Fre2 = fft.fftshift(fft.fft2(fft.fftshift(holoi, dim=(-2, -1)), dim=(-2, -1), norm='ortho'), dim=(-2, -1))
        Frer = torch.real(Fre1) - torch.imag(Fre2)
        Frei = torch.imag(Fre1) + torch.real(Fre2)
        prop_fr = Frer*Hr - Frei*Hi
        prop_fi = Frei*Hr + Frer*Hi
        reconstruct1 = fft.ifftshift(fft.ifft2(fft.ifftshift(prop_fr, dim=(-2, -1)), dim=(-2, -1), norm='ortho'), dim=(-2, -1))
        reconstruct2 = fft.ifftshift(fft.ifft2(fft.ifftshift(prop_fi, dim=(-2, -1)), dim=(-2, -1), norm='ortho'), dim=(-2, -1))
        reconstructr = torch.real(reconstruct1) - torch.imag(reconstruct2)
        reconstructi = torch.imag(reconstruct1) + torch.real(reconstruct2)
        reconstructr = reconstructr[:,:,int(self.pad/2*self.pixelx):int((2+self.pad)/2*self.pixelx),
                       int(self.pad/2*self.pixely):int((2+self.pad)/2*self.pixely)]
        reconstructi = reconstructi[:,:,int(self.pad/2*self.pixelx):int((2+self.pad)/2*self.pixelx),
                       int(self.pad/2*self.pixely):int((2+self.pad)/2*self.pixely)]

        # crop to roi region
        reconstructr = reconstructr[:,:,self.cropx:self.pixelx-self.cropx,
                       self.cropy:self.pixely-self.cropy]
        reconstructi = reconstructi[:,:,self.cropx:self.pixelx-self.cropx,
                       self.cropy:self.pixely-self.cropy]
        reconstruct_inten = torch.abs(reconstructr)**2 + torch.abs(reconstructi)**2

        return reconstructr, reconstructi, reconstruct_inten #/ torch.max(reconstruct)


class Fresnel(nn.Module):
    # change batchsize to 1
    def __init__(self):
        super().__init__()

        self.pitch = 0.008 #mm
        self.wavelength = 0.000532 #mm
        self.pixelx = 1080
        self.pixely = 1920


    def forward(self, holo, distance):  # holo[4,1,2160,3840] #distance[4,1]
        batchsize = distance.shape[0]
        distance = distance.unsqueeze(2).unsqueeze(3)  ##[4,1,1,1]
        scale = self.wavelength * distance / (self.pitch) ** 2
        # print(xscale)
        padx = int((scale-self.pixelx+1)/2)
        pady = int((scale-self.pixely+1)/2)
        # padx = 0
        # pady = 0
        pad = nn.ZeroPad2d((pady, pady, padx, padx)).cuda()

        x = torch.linspace(-1/2 * (self.pixelx+2*padx) * self.pitch,
                           1/2 * (self.pixelx+2*padx) * self.pitch,
                            self.pixelx+2*padx).cuda()
        y = torch.linspace(-1 / 2 * (self.pixely + 2 * pady) * self.pitch,
                           1 / 2 * (self.pixely + 2 * pady) * self.pitch,
                           self.pixely + 2 * pady).cuda()

        X, Y = torch.meshgrid(x, y)

        X = X.unsqueeze(0).unsqueeze(0).repeat(batchsize, 1, 1, 1)
        Y = Y.unsqueeze(0).unsqueeze(0).repeat(batchsize, 1, 1, 1)

        circle = torch.exp(1j*torch.pi*(X**2+Y**2)/(self.wavelength*distance))
        holo = torch.complex(torch.cos(holo), torch.sin(holo))
        holo = pad(holo)

        reconstruct = fft.fftshift(fft.fft2(fft.fftshift(holo*circle, dim=(-2, -1)), dim=(-2, -1)), dim=(-2, -1))
        reconstruct = reconstruct[:,:,padx:padx+self.pixelx,
                        pady:pady+self.pixely]

        reconstruct = torch.abs(reconstruct)**2

        return reconstruct /torch.max(reconstruct), padx, pady


if __name__ == '__main__':

    prop = ASM_split(wavelength=520, res=(2160, 3840), roi=(2160, 3840), pitch=3.74e-3, apply_constraint=True).cuda()
    # prop = Fresnel().cuda()
    # input = np.array(cv2.imread('cgh.bmp',-1))/255
    input = np.array(cv2.imread('Iterative_algorithms/data/animation/DPAC_ASM/green/animation.png', -1))/255
    # input = np.array(cv2.imread('hologram.png', -1)) / 255
    holo = torch.tensor(input, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
    holo = 2*holo*torch.pi - torch.pi
    distance = torch.tensor([[100]]).cuda()
    # m,_,_ = prop(holo,distance)
    _, _, m = prop(torch.ones_like(holo), holo, distance)
    m = m.cpu().squeeze().numpy()*255
    im = Image.fromarray(m)
    im.show()
    cv2.imwrite('Iterative_algorithms/data/animation/DPAC_ASM/green/display.png', m)


    #'''
    # a = torch.randn(2,2,2,2,dtype=float)
    # print(a.shape)
