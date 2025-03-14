import torch
from torch import nn
import torch.fft as fft
import math
from prop import ASM_split
import torch.nn.functional as F
from PIL import Image
import numpy
import scipy.io as sio

class Flex_FourierNet(nn.Module):
    def __init__(self, wl, center_distance):
        super().__init__()
        self.prop = ASM_split(wavelength=wl, res=(2160, 3840),
                              roi=(2160, 3840), pitch=3.74e-3)  #required_grad = False
        self.phase_generator = flex_phase_generator(input_dim=2, center=center_distance,
                                                    wl=wl, pitch=3.74e-3)
        self.center = center_distance
    def forward(self, x, distance):
        amp0 = torch.sqrt(x)   #amplitude
        center_distance = torch.ones_like(distance) * self.center
        ampr, ampi, _ = self.prop(amp0, torch.zeros_like(amp0), center_distance)
        input = torch.cat([ampr, ampi], dim=1)
        holo = self.phase_generator(input, distance)
        return holo


class FourierNet(nn.Module):
    def __init__(self, wl):
        super().__init__()
        self.prop = ASM_split(wavelength=wl, res=(2160, 3840),
                              roi=(2160, 3840), pitch=3.74e-3)  #required_grad = False
        self.phase_generator = phase_generator(input_dim=2)
    def forward(self, x, distance):
        amp0 = torch.sqrt(x)   #amplitude
        ampr, ampi, _ = self.prop(amp0, torch.zeros_like(amp0), distance)
        input = torch.cat([ampr, ampi], dim=1)
        holo = self.phase_generator(input)
        return holo


class phase_generator(nn.Module):
    def __init__(self, input_dim, dim=4, init_h=2160, init_w=3840):
        super().__init__()
        self.inc = nn.Conv2d(input_dim, dim, kernel_size=3, stride=1, padding=1)
        self.SF1d = SF(dim, dim * 2, init_h, init_w)
        self.SF2d = SF(dim * 2, dim * 4, init_h // 2, init_w//2)
        self.SF3d = SF(dim * 4, dim * 8, init_h // 4, init_w//4)

        self.down1 = nn.Conv2d(dim * 2, dim * 2, kernel_size=4, stride=2, padding=1)
        self.down2 = nn.Conv2d(dim * 4, dim * 4, kernel_size=4, stride=2, padding=1)
        self.down3 = nn.Conv2d(dim * 8, dim * 8, kernel_size=4, stride=2, padding=1)

        self.bottleneck = nn.Conv2d(dim*8, dim*8, 3, 1, 1)
        self.up3 = nn.ConvTranspose2d(dim * 8, dim * 4, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(dim * 4, dim * 2, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(dim * 2, dim * 1, kernel_size=2, stride=2)

        self.SF3u = SF(dim * 8, dim * 4, init_h // 4, init_w//4)
        self.SF2u = SF(dim * 4, dim * 2, init_h // 2, init_w//2)
        self.SF1u = SF(dim * 2, dim, init_h, init_w)

        self.Out = nn.Sequential(
            nn.Conv2d(dim, dim//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim//2, 1, kernel_size=1, stride=1),
            nn.Hardtanh(0, 2*math.pi)
        )


    def forward(self, x):
        x0 = self.inc(x)

        x1 = self.SF1d(x0)
        x1 = self.down1(x1)

        x2 = self.SF2d(x1)
        x2 = self.down2(x2)

        x3 = self.SF3d(x2)
        x3 = self.down3(x3)

        x = self.bottleneck(x3)

        x = self.up3(x)
        x = self.SF3u(torch.cat([x, x2], dim=1))

        x = self.up2(x)
        x = self.SF2u(torch.cat([x, x1], dim=1))

        x = self.up1(x)
        x = self.SF1u(torch.cat([x, x0], dim=1))

        phase = self.Out(x)-math.pi

        return phase


class flex_phase_generator(nn.Module):
    def __init__(self, input_dim, center, wl, pitch, dim=4, init_h=2160, init_w=3840):
        super().__init__()
        self.inc = nn.Conv2d(input_dim, dim, kernel_size=3, stride=1, padding=1)
        self.SF1d = Flex_SF(dim, dim * 2, init_h, init_w)
        self.SF2d = Flex_SF(dim * 2, dim * 4, init_h // 2, init_w // 2)
        self.SF3d = Flex_SF(dim * 4, dim * 8, init_h // 4, init_w // 4)

        self.down1 = nn.Conv2d(dim * 2, dim * 2, kernel_size=4, stride=2, padding=1)
        self.down2 = nn.Conv2d(dim * 4, dim * 4, kernel_size=4, stride=2, padding=1)
        self.down3 = nn.Conv2d(dim * 8, dim * 8, kernel_size=4, stride=2, padding=1)

        self.bottleneck = nn.Conv2d(dim*8, dim*8, 3, 1, 1)
        self.up3 = nn.ConvTranspose2d(dim * 8, dim * 4, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(dim * 4, dim * 2, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(dim * 2, dim * 1, kernel_size=2, stride=2)

        self.SF3u = Flex_SF(dim * 8, dim * 4, init_h // 4, init_w // 4)
        self.SF2u = Flex_SF(dim * 4, dim * 2, init_h // 2, init_w // 2)
        self.SF1u = Flex_SF(dim * 2, dim, init_h, init_w)

        self.Out = nn.Sequential(
            nn.Conv2d(dim, dim//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim//2, 1, kernel_size=1, stride=1),
            nn.Hardtanh(0, 2*math.pi)
        )

        self.center = center
        self.distance_encoder = Distance_kernel(channel=dim, wl=wl, pitch=pitch, n_embed=528, kernel_dim=528)


    def forward(self, x0, distance):

        dist_norm = distance-self.center
        c1, c2, c3, c11, c22, c33 = self.distance_encoder(dist_norm)
        x0 = self.inc(x0)

        x = self.SF1d(x0, c1)
        x1s = self.down1(x)

        x = self.SF2d(x1s, c2)
        x2s = self.down2(x)

        x = self.SF3d(x2s, c3)
        x = self.down3(x)

        x = self.bottleneck(x)

        x = self.up3(x)
        x = self.SF3u(torch.cat([x, x2s], dim=1), c33)

        x = self.up2(x)
        x = self.SF2u(torch.cat([x, x1s], dim=1), c22)

        x = self.up1(x)
        x = self.SF1u(torch.cat([x, x0], dim=1), c11)

        phase = self.Out(x)-math.pi

        return phase

class SF(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 filter_h, filter_w):
        super().__init__()

        self.iniF = nn.Conv2d(in_channels, out_channels//4, kernel_size=3,
                      stride=1, padding=1)

        self.F = Fourier_filter(out_channels//4, filter_h, filter_w)

        self.S = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//2, kernel_size=3, dilation=1,
                      stride=1, padding=1),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//2, out_channels//2, kernel_size=3, dilation=1,
                      stride=1, padding=1),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU(inplace=True),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=1,
                      stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        xf0 = self.iniF(x)
        xf = self.F(xf0)
        xs = self.S(x)
        x = torch.cat([xs, xf], dim=1)
        return self.conv(x)

class Flex_SF(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 filter_h, filter_w):
        super().__init__()

        self.iniF = nn.Conv2d(in_channels, out_channels//4, kernel_size=3,
                      stride=1, padding=1)

        self.F = Flex_Fourier_filter(out_channels//4, filter_h, filter_w)
        # out_channels//4 --> out_channels//2

        self.S = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//2, kernel_size=3, dilation=1,
                      stride=1, padding=1),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//2, out_channels//2, kernel_size=3, dilation=1,
                      stride=1, padding=1),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU(inplace=True),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=1,
                      stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, c):
        xf0 = self.iniF(x)
        xf = self.F(xf0, c)
        xs = self.S(x)
        x = torch.cat([xs, xf], dim=1)
        return self.conv(x)

class Fourier_filter(nn.Module):
    def __init__(self, F_dim, filter_h, filter_w):
        super().__init__()
        self.scale = 1/(F_dim)
        self.complex_weight = nn.Parameter(torch.randn(F_dim, filter_h, filter_w, 2, dtype=torch.float32)*self.scale)

    def forward(self, x):
        batch, channel, a, b = x.shape
        x = fft.fft2(x, dim=(2, 3), norm='ortho')
        # print(x.shape)
        weight = self.complex_weight
        # print(weight.shape)
        if not weight.shape[1:3] == x.shape[2:4]:
            weight = F.interpolate(weight.permute(3, 0, 1, 2), size=x.shape[2:4],
                                   mode='bilinear', align_corners=True).permute(1, 2, 3, 0)
        weight = torch.view_as_complex(weight.contiguous())
        x = x * weight
        x = fft.ifft2(x, dim=(2, 3), norm='ortho')
        y = torch.cat([x.real, x.imag], dim=1)
        return y

class Flex_Fourier_filter(nn.Module):
    def __init__(self, F_dim, filter_h, filter_w):
        super().__init__()
        self.scale = 1/(F_dim)
        self.weight = nn.Parameter(torch.randn(F_dim, filter_h, filter_w, 2, dtype=torch.float32)*self.scale)

    def forward(self, x, c):
        batch, channel, a, b = x.shape
        assert c.shape[1] == channel*2
        x = fft.fft2(x, dim=(2, 3), norm='ortho')
        # print(x.shape)
        weight = self.weight
        # print(weight.shape)
        if not weight.shape[1:3] == x.shape[2:4]:
            weight = F.interpolate(weight.permute(3, 0, 1, 2), size=x.shape[2:4],
                                   mode='bilinear', align_corners=True).permute(1, 2, 3, 0)
        if not c.shape[2:4] == x.shape[2:4]:
            c = F.interpolate(c, size=x.shape[2:4],
                            mode='bilinear', align_corners=True)
        weight = torch.view_as_complex(weight.contiguous()) # complex weight
        c1r, c1i = torch.chunk(c, chunks=2, dim=1)
        c_complex = torch.view_as_complex(torch.stack([c1r, c1i], dim=4))
        x = x * weight * fft.ifftshift(c_complex, dim=(-2, -1))
        x = fft.ifft2(x, dim=(2, 3), norm='ortho')
        y = torch.cat([x.real, x.imag], dim=1)
        return y

class Distance_kernel(nn.Module): # Generate Spatial Representation of distance kernel
    def __init__(self, channel, wl, pitch, n_embed, kernel_dim):
        super().__init__()
        self.wl = wl
        self.pitch = pitch
        self.F_embed = n_embed//2
        self.channel = channel
        self.kernel_dim = kernel_dim
        self.net1 = nn.Sequential(
            nn.Linear(n_embed, 512),
            nn.LayerNorm(512),
            nn.Tanh(),
            nn.Linear(512, channel*kernel_dim//2*3),
            nn.LayerNorm(channel*kernel_dim//2*3),
            nn.Tanh())
        self.net2 = nn.Sequential(
            nn.Linear(channel*kernel_dim//2*3, channel*kernel_dim*3),
            nn.LayerNorm(channel*kernel_dim*3),
            nn.Tanh()
        )


    def forward(self, x):
        distance = embed(x, self.F_embed, self.wl, self.pitch) #(batchsize, F_embed)
        kv = self.net1(distance)
        kv1, kv2, kv3 = torch.chunk(kv, 3, dim=1)

        c11 = circular_padding(kv1, channel=self.channel//4)
        c22 = circular_padding(kv2, channel=self.channel//4*2)
        c33 = circular_padding(kv3, channel=self.channel//4*4)

        kv = self.net2(kv)
        kv1, kv2, kv3 = torch.chunk(kv, 3, dim=1)

        c1 = circular_padding(kv1, channel=self.channel//2)
        c2 = circular_padding(kv2, channel=self.channel//2 * 2)
        c3 = circular_padding(kv3, channel=self.channel//2 * 4)

        return c1, c2, c3, c11, c22, c33

def embed(distance, N_freqs, wv, pitch):
    # distance: [batchsize,1]
    # N_freqs: int
    assert distance.shape[-1] == 1
    wavelength = wv*1e-6
    batchsize = distance.shape[0]
    min_fre = 2*torch.pi/wavelength * (1-2*(wavelength/pitch/2)**2)**(1/2)
    max_fre = 2*torch.pi/wavelength
    # freq_bands = 2 ** torch.linspace(0, N_freqs-1, steps=N_freqs).to(distance.device)
    freq_bands = (max_fre-min_fre)/N_freqs*torch.linspace(1, N_freqs, steps=N_freqs).to(distance.device)+min_fre
    distance_scaled = distance*freq_bands.repeat(batchsize, 1)
    distance_scaled = torch.cat((torch.cos(distance_scaled), torch.sin(distance_scaled)), dim=-1)

    return distance_scaled

def circular_padding(x, channel):
    batch, kernel_dim = x.shape
    length = kernel_dim//(channel*2)
    components = torch.reshape(x, (batch, channel*2, length)).unsqueeze(-1).unsqueeze(-2)
    x_axis = y_axis = torch.linspace(-length, length, 2*length).cuda()
    x_axis, y_axis = torch.meshgrid(x_axis, y_axis)
    dis_axis = torch.sqrt(x_axis**2 + y_axis**2)
    interval = torch.max(dis_axis)/length
    dis_index = dis_axis//(interval+1e-4)

    index_tensor = torch.arange(length).unsqueeze(1).unsqueeze(2).cuda()
    area = dis_index.unsqueeze(0)==index_tensor
    y = components*area.unsqueeze(0).unsqueeze(1)
    circle = torch.sum(y, dim=2)
    # print(circle.shape)

    return circle #[batch, channel*2, size, size]

if __name__ == '__main__':
    model = FourierNet(wl=520).cuda()
    # FNET = torch.load('8/FourierNet_fix_150.pth')
    # model.load_state_dict(FNET)
    # x = torch.ones((1, 1, 108*10, 192*10), dtype=torch.float).cuda()
    # distance = 150*torch.ones((1, 1)).cuda()
    # y = model(x, distance)
    # y = y / (2 * torch.pi) + 0.5
    # out = Image.fromarray(y.squeeze().cpu().data.numpy() * 255)
    # a = y.squeeze().cpu().data.numpy()
    # sio.savemat(f'tensor.mat', {'tensor':a})
    # out.show()


