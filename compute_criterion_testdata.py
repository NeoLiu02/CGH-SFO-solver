import numpy as np
import os
import configargparse
import cv2
import scipy.io as sio
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# python compute_criterion_testdata.py
def srgb_gamma2lin(x):
    # convert from srgb to linear color space
    thresh = 0.04045
    y = np.where(x <= thresh, x/12.92, ((x+0.055)/1.055)**2.4)
    return y

def srgb_lin2gamma(x):
    # convert from  linear color to srgb space
    thresh = 0.0031308
    y = np.where(x <= thresh, 12.92*x, 1.055*(x**(1/2.4))-0.055)
    return y

def get_pnsr_ssim(recon_amp, target, multichannel=False):
    psnrs, ssims = {}, {}

    # linear
    target_lin = target
    recon_lin = recon_amp
    psnrs['lin'] = psnr(target_lin, recon_lin)
    ssims['lin'] = ssim(target_lin, recon_lin, channel_axis=multichannel)

    # srgb
    target_srgb = srgb_lin2gamma(np.clip(target_lin, 0.0, 1.0))
    recon_srgb = srgb_lin2gamma(np.clip(recon_lin, 0.0, 1.0))
    psnrs['srgb'] = psnr(target_srgb, recon_srgb)
    ssims['srgb'] = ssim(target_srgb, recon_srgb, channel_axis=multichannel)

    return psnrs, ssims

def crop(image, target_res, roi_res):
    image_c = image[(target_res[0] - roi_res[0]) // 2:(target_res[0] + roi_res[0]) // 2,
                    (target_res[1] - roi_res[1]) // 2:(target_res[1] + roi_res[1]) // 2]
    return image_c


def main():
    p = configargparse.ArgumentParser()
    p.add_argument('--number', type=int, default=31, help='image number')
    parse = p.parse_args()
    channels = [0, 1, 2]
    # for i in range(10):
    for i in [9]:
        psnr_fullcolor = {'r': [], 'g': [], 'b': []}
        ssim_fullcolor = {'r': [], 'g': [], 'b': []}
        for channel in channels:
            color = ('r', 'g', 'b')[channel]
            path = "testdata/" + str(i + 1) + '/' + color + '/flex'
            truth_path = "testdata/" + str(i + 1) + ".png"
            psnr_list = {'lin': [], 'srgb': []}
            ssim_list = {'lin': [], 'srgb': []}

            target_res = (2160, 3840)
            roi_res = (2160, 3840)

            for index in range(parse.number):
                print('Start calculating criterion for image %d, color: %s, distance: %d' %(i+1, color, index+85))
                recon_path = path + '/display_' + str(index + 85) + ".png"
                target = cv2.imread(truth_path)
                target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
                target = target.astype(np.float32)/255

                # recon_iten = cv2.imread(recon_path, cv2.IMREAD_GRAYSCALE)
                recon_iten = cv2.imread(recon_path)
                recon_iten = cv2.cvtColor(recon_iten, cv2.COLOR_BGR2RGB)
                recon_iten = recon_iten.astype(np.float32)/255

                target_iten = cv2.split(target)[channel]
                recon_iten = cv2.split(recon_iten)[channel]

                recon_iten = crop(recon_iten, target_res, roi_res)
                target_iten = crop(target_iten, target_res, roi_res)

                # Normalize (which can be achieved by adjusting laser input intensity)
                recon_iten *= np.sum(target_iten, (0, 1), keepdims=True)/np.sum(recon_iten, (0, 1), keepdims=True)

                psnrs, ssims = get_pnsr_ssim(recon_iten, target_iten, multichannel=(channel == 3))

                for domain in ['lin', 'srgb']:
                    psnr_list[domain].append(psnrs[domain])
                    ssim_list[domain].append(ssims[domain])

                psnr_fullcolor[color].append(psnrs['lin'])
                ssim_fullcolor[color].append(ssims['lin'])

            sio.savemat(os.path.join(path, f'psnr.mat'), psnr_list)
            sio.savemat(os.path.join(path, f'ssim.mat'), ssim_list)

        sio.savemat(os.path.join('testdata', str(i + 1), f'psnr_fullcolor.mat'), psnr_fullcolor)
        sio.savemat(os.path.join('testdata', str(i + 1), f'ssim_fullcolor.mat'), ssim_fullcolor)

if __name__ == "__main__":
    main()