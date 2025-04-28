"""
Modified from Neural holography:

This is the main executive script used for the phase generation using iterative optimization mehtods (GS/DPAC/SGD).

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from Stanford).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.

@article{Peng:2020:NeuralHolography,
author = {Y. Peng, S. Choi, N. Padmanaban, G. Wetzstein},
title = {{Neural Holography with Camera-in-the-loop Training}},
journal = {ACM Trans. Graph. (SIGGRAPH Asia)},
year = {2020},
}

-----
cd to this path first
$ python main.py --channel=1 --method=SGD --root_path=data/usaf --data_path=data/usaf --distance=100 --use_writer=1
$ python main.py --channel=1 --method=GS --root_path=data/usaf --data_path=data/usaf --distance=100 --use_writer=1
$ python main.py --channel=1 --method=DPAC --root_path=data/animation --data_path=data/animation --distance=100
"""

import os
import time
import cv2
import torch
import torch.nn as nn
import configargparse
import utils
from augmented_image_loader import ImageLoader
from modules import SGD, GS, DPAC
from propagation_ASM import propagation_ASM
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

# Command line argument processing
p = configargparse.ArgumentParser()
p.add_argument('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')
p.add_argument('--distance', type=int, default=100, help='reconstruction distance mm')
p.add_argument('--channel', type=int, default=1, help='Red:0, green:1, blue:2')
p.add_argument('--method', type=str, default='SGD', help='Type of algorithm, GS/SGD/DPAC/HOLONET/UNET')
p.add_argument('--prop_model', type=str, default='ASM', help='Type of propagation model, ASM or model')
p.add_argument('--root_path', type=str, default='data', help='Directory where optimized phases will be saved.')
p.add_argument('--data_path', type=str, default='data', help='Directory for the dataset')
p.add_argument('--lr', type=float, default=8e-3, help='Learning rate for phase variables (for SGD)')
p.add_argument('--lr_s', type=float, default=2e-3, help='Learning rate for learnable scale (for SGD)')
p.add_argument('--num_iters', type=int, default=2000, help='Number of iterations (GS, SGD)')
p.add_argument('--use_writer', type=int, default=1, help='Whether or not use summary writer (extra time cost)')

# parse arguments
opt = p.parse_args()
run_id = f'{opt.method}_{opt.prop_model}'  # {algorithm}_{prop_model} format

channel = opt.channel  # Red:0 / Green:1 / Blue:2
chan_str = ('red', 'green', 'blue')[channel]
print(f'   - optimizing phase with {opt.method}/{opt.prop_model} ... ')

# Hyperparameters setting
cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
prop_dist = (opt.distance * mm, opt.distance * mm, opt.distance * mm)[channel]  # propagation distance from SLM plane to target plane
wavelength = (638 * nm, 520 * nm, 450 * nm)[channel]  # wavelength of each color
feature_size = (3.74 * um, 3.74 * um)  # SLM pitch
slm_res = (2160, 3840)  # resolution of SLM
image_res = (2160, 3840)
roi_res = (2160, 3840)  # regions of interest (to penalize for SGD)
dtype = torch.float32  # default datatype (Note: the result may be slightly different if you use float64, etc.)
device = torch.device('cuda')  # The gpu you are using

# Options for the algorithm
loss = nn.MSELoss().to(device)  # loss functions to use (try other loss functions!)
s0 = 1.0  # initial scale

root_path = os.path.join(opt.root_path, run_id, chan_str)  # path for saving out optimized phases

# Tensorboard writer
if opt.use_writer:
    summaries_dir = os.path.join(root_path, 'summaries')
    utils.cond_mkdir(summaries_dir)
    writer = SummaryWriter(log_dir=summaries_dir)
else:
    writer = None

# Simulation model
if opt.prop_model == 'ASM':
    propagator = propagation_ASM  # Ideal model

# Select Phase generation method, algorithm
if opt.method == 'SGD':
    phase_only_algorithm = SGD(prop_dist, wavelength, feature_size, opt.num_iters, roi_res, root_path,
                               opt.prop_model, propagator, loss, opt.lr, opt.lr_s, s0, citl=False, camera_prop=None,
                               writer=writer, device=device)
elif opt.method == 'GS':
    phase_only_algorithm = GS(prop_dist, wavelength, feature_size, opt.num_iters, root_path,
                              opt.prop_model, propagator, writer, device)
elif opt.method == 'DPAC':
    phase_only_algorithm = DPAC(prop_dist, wavelength, feature_size, opt.prop_model, propagator, device)

# Augmented image loader (if you want to shuffle, augment dataset, put options accordingly.)
image_loader = ImageLoader(opt.data_path, channel=channel,
                           image_res=image_res, homography_res=roi_res,
                           crop_to_homography=True,
                           shuffle=False, vertical_flips=False, horizontal_flips=False)

# Loop over the dataset
for k, target in enumerate(image_loader):
    # print('111')
    # get target image
    target_amp, target_res, target_filename = target
    target_path, target_filename = os.path.split(target_filename[0])
    target_idx = target_filename.split('_')[-1]
    target_amp = target_amp.to(device)
    m = target_amp.cpu().squeeze().numpy() * 255
    # display[:, :, i] = m
    im = Image.fromarray(m)
    print(target_idx)

    # if you want to separate folders by target_idx or whatever, you can do so here.
    phase_only_algorithm.init_scale = s0 * utils.crop_image(target_amp, roi_res, stacked_complex=False).mean()
    phase_only_algorithm.phase_path = os.path.join(root_path)

    time_start = time.time()
    # run algorithm (See algorithm_modules.py and algorithms.py)
    if opt.method in ['DPAC', 'HOLONET', 'UNET']:
        # direct methods
        # for i in range(10):
        _, final_phase = phase_only_algorithm(target_amp)
    else:
        # iterative methods, initial phase: random guess
        init_phase = (-0.5 + 1.0 * torch.rand(1, 1, *slm_res)).to(device)
        final_phase, _ = phase_only_algorithm(target_amp, init_phase)
    time_end = time.time()

    print(final_phase.shape)
    print('totally cost', time_end - time_start)

    # save the final result somewhere.
    phase_out_8bit = utils.phasemap_8bit(final_phase.cpu().detach(), inverted=False)

    utils.cond_mkdir(root_path)
    cv2.imwrite(os.path.join(root_path, f'{target_idx}.png'), phase_out_8bit)
    if not opt.use_writer:
        file = open(os.path.join(root_path, f'time.txt'), 'w')
        file.write('Iterations: ' + str(opt.num_iters) + '\n')
        file.write('totally cost: '+str(time_end-time_start)+'s')
        file.close()

print(f'    - Done, result: --root_path={root_path}')
