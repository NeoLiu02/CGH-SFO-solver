import cv2
import numpy
import torch.fft
import time
from model.FourierNet import FourierNet, Flex_FourierNet
import argparse
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image
import os
from prop import ASM_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# python eval_dual.py --channel=0 --name=hololab

class Eval_dual_Dataset(Dataset):
    def __init__(self, image_dir, color, distance,
                 target_res, roi_res):
        assert color in ['r', 'g', 'b']
        self.color = color
        self.images = [image_dir+"/1.PNG", image_dir+"/2.PNG"]
        self.data = [distance[0], distance[1]]
        pad_h = (target_res[1] - roi_res[1])//2
        pad_w = (target_res[0] - roi_res[0])//2
        self.transform = transforms.Compose([
            # transforms.Grayscale(),
            transforms.Resize(roi_res),
            transforms.Pad((pad_h, pad_w, pad_h, pad_w)),
            transforms.ToTensor()
            # transforms.Normalize(mean=[0.448,0.437,0.404],std=[0.282,0.267,0.290])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])

        input = self.transform(image)
        if input.shape[0] == 1:
            input = input[0, :, :]
        elif self.color == 'r':
            input = input[0, :, :]  # red
        elif self.color == 'g':
            input = input[1, :, :]  # green
        elif self.color == 'b':
            input = input[2, :, :]  # blue

        data = self.data[idx]
        return {'image': input, 'distance': data}

p = argparse.ArgumentParser()
p.add_argument('--channel', type=int, default=1, help='red:0, green:1, blue:2')
p.add_argument('--model_path', type=str, default='model/', help='torch save trained model')
p.add_argument('--distance', type=list, default=[85, 115], help='near -> far distance')
p.add_argument('--name', type=str, default="hololab or symbol")

arg = p.parse_args()
channel = arg.channel
color = ('r', 'g', 'b')[channel]
wl = (638, 520, 450)[channel]
distance_near = arg.distance[0]
distance_far = arg.distance[1]
target_res = (2160, 3840)
roi_res = (2160, 3840)

prop = ASM_split(wavelength=wl, res=(2160, 3840), roi=(2160, 3840), pitch=3.74e-3, apply_constraint=True).cuda()
prop.eval()

model_dir = arg.model_path + color + '/' + 'FourierNet_flex_100.pth'
print("Model:", model_dir)
model = Flex_FourierNet(wl=wl, center_distance=100).cuda()

pretrained = torch.load(model_dir)
model.load_state_dict(pretrained)
# model.eval()
# Training batchsize=1 does not require model.eval()

imagedir = "Dual_plane/" + arg.name
dataset = Eval_dual_Dataset(image_dir=imagedir, color=color, distance=arg.distance, target_res=target_res, roi_res=roi_res)
Loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

output_list = []
with torch.no_grad():
    for batch in Loader:
        image = batch['image'].unsqueeze(1).cuda()
        distance = batch['distance'].unsqueeze(1).to(torch.float32).cuda()
        print('Distance:', distance.item())
        time_start = time.time()
        output = model(image, distance)
        time_end = time.time()
        output_list.append(output)

# output = torch.cat([output_list[0][:, :, :, 0:1920], output_list[1][:, :, :, 1920:3840]], dim=-1)
output = torch.angle(torch.exp(1j*output_list[0]) + torch.exp(1j*output_list[1])) # Phase dominant property
# print(output.shape)

_, _, final1 = prop(torch.ones_like(output), output, distance_near*torch.ones(1, 1).to(torch.float32).cuda())
final1 = final1/torch.max(final1)
m1 = final1.cpu().squeeze().numpy() * 255

_, _, final2 = prop(torch.ones_like(output), output, distance_far*torch.ones(1, 1).to(torch.float32).cuda())
final2 = final2/torch.max(final2)
m2 = final2.cpu().squeeze().numpy() * 255

# save images
output = torch.squeeze(output)
holo = output / (2 * torch.pi) + 0.5
# holo = 1-holo  # reverse phase
print('Each plane costs', time_end - time_start)
holo = numpy.uint8(holo.cpu().data.numpy() * 255)
holo_flip = cv2.flip(holo, 0)

save_dir = imagedir + '/distance_' + str(distance_near) + '_' + str(distance_far) + '/' + color
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
cv2.imwrite(save_dir + '/hologram.png', holo)
cv2.imwrite(save_dir + '/hologram_flip.png', holo_flip)
cv2.imwrite(save_dir + '/reconstruction_near.png', m1)
cv2.imwrite(save_dir + '/reconstruction_far.png', m2)
