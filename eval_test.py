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

# python eval_test.py
class EvalDataset(Dataset):
    def __init__(self, image_dir, color, distance,
                 target_res, roi_res):
        assert color in ['r', 'g', 'b']
        self.color = color
        self.images = image_dir
        # repeat each image to match the number of int values
        self.images = [image_dir for _ in range(distance[1]-distance[0]+1)]
        self.data = list(range(distance[0], distance[1]+1))
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
        # image.show()
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
p.add_argument('--model_path', type=str, default='model/', help='torch save trained model')
p.add_argument('--distance', type=list, default=[85, 115], help='near -> far distance')
p.add_argument('--test_name', type=str, default='animation', help='wolf, Rome, animation, THU1......')

arg = p.parse_args()
channels = [0, 1, 2]
# for i in range(10):
for i in [9]:
    imagedir = 'testdata/' + str(i + 1) + '.png'
    for channel in channels:
        color = ('r', 'g', 'b')[channel]
        wl = (638, 520, 450)[channel]
        distance_near = arg.distance[0]
        distance_far = arg.distance[1]
        center_distance = (distance_near + distance_far)/2
        target_res = (2160, 3840)
        roi_res = (2160, 3840)

        # ASM propagation (Constraint can be experimentally applied by 4f filter)
        prop = ASM_split(wavelength=wl, res=(2160, 3840), roi=(2160, 3840), pitch=3.74e-3, apply_constraint=True).cuda()
        prop.eval()

        model_dir = arg.model_path + color + '/' + 'FourierNet_flex_' + str(int(center_distance)) + '.pth'
        print("Model:", model_dir)
        model = Flex_FourierNet(wl=wl, center_distance=100).cuda()
        pretrained = torch.load(model_dir)
        model.load_state_dict(pretrained)
        # model.eval()
        # Training batchsize=1 does not require model.eval()

        dataset = EvalDataset(image_dir=imagedir, color=color, distance=arg.distance, target_res=target_res, roi_res=roi_res)
        Loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

        k = 0
        with torch.no_grad():
            for batch in Loader:
                image = batch['image'].unsqueeze(1).cuda()
                distance = batch['distance'].unsqueeze(1).to(torch.float32).cuda()
                print('Distance:', distance.item())
                time_start = time.time()
                # Run SFO-solver to generate hologram
                output = model(image, distance)
                time_end = time.time()
                # Simulate reconstructions
                _, _, final = prop(torch.ones_like(output), output, distance)
                # Match scale (This can be experimentally done by controlling the laser power)
                scale = torch.sum(image, dim=(-2, -1), keepdim=True) / torch.sum(final, dim=(-2, -1), keepdim=True)
                final *= scale
                m = final.cpu().squeeze().numpy() * 255 # Simulated reconstruction

                # save images
                output = torch.squeeze(output)
                holo = output / (2 * torch.pi) + 0.5
                print('totally cost', time_end - time_start)
                holo = numpy.uint8(holo.cpu().data.numpy() * 255)

                save_dir = 'testdata/' + str(i+1) + '/' + color + '/flex/'
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                cv2.imwrite(save_dir + 'display_'+str(k+distance_near)+'.png', m)
                k += 1