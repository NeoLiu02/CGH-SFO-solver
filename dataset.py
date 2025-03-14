import random
import torch
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import numpy
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


trainpath='/home/user/lnh_2023/Ada-Holo/dataset/DIV2K_train_HR'
validpath='/home/user/lnh_2023/Ada-Holo/dataset/DIV2K_valid_HR'
distance_path = '/home/user/lnh_2023/Ada-Holo/dataset/distance.txt'
distance_path2 = '/home/user/lnh_2023/Ada-Holo/dataset/distance-valid.txt'

def high_freq_noise(image, level):
    if isinstance(image, torch.Tensor):
        pepper_values = torch.randn(image.shape)
        salt_values = torch.randn(image.shape)
        image[pepper_values < level] = 1
        image[salt_values < level] = 0
        return image
    else:
        raise ValueError('Unsupported input type')


# def pad_crop_to_res(image, target_res):
#     """Pads with 0 and crops as needed to force image to be target_res
#
#     image: an array with dims [..., channel, height, width]
#     target_res: [height, width]
#     """
#     return utils.crop_image(utils.pad_image(image,
#                                             target_res, pytorch=False),
#                             target_res, pytorch=False)


class MyDataset(Dataset):
    def __init__(self, image_dir, data_file, color,
                 target_res, roi_res, noise=False):
        assert color in ['r', 'g', 'b']
        self.color = color
        # self.target_res = target_res
        # self.roi_res = roi_res
        self.images = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
        with open(data_file, 'r') as f:
            self.data = [int(x.strip()) for x in f.readlines()]

        # repeat each image to match the number of int values
        # image augmentation
        self.images = [x for x in self.images for _ in range(40)]
        pad_h = (target_res[1] - roi_res[1])//2
        pad_w = (target_res[0] - roi_res[0])//2
        self.transform = transforms.Compose([
            # transforms.Grayscale(),
            transforms.Resize(roi_res),
            transforms.Pad((pad_h, pad_w, pad_h, pad_w)),
            transforms.ToTensor()
            # transforms.Normalize(mean=[0.448,0.437,0.404],std=[0.282,0.267,0.290])
        ])
        self.transform1 = transforms.RandomHorizontalFlip(p=1)
        self.transform2 = transforms.RandomVerticalFlip(p=1)
        self.transform3 = transforms.RandomRotation(degrees=(180, 180))

        if noise:
            self.noise = True
            self.transform4 = transforms.RandomApply([transforms.Lambda(lambda x:high_freq_noise(x, level=0.1))], p=0.3)
        else:
            self.noise = False

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])

        if idx % 4 == 3:
            image = self.transform3(image)
        elif idx % 4 == 2:
            image = self.transform2(image)
        elif idx % 4 == 1:
            image = self.transform1(image)

        # image = np.array(image)
        # if self.roi_res:
        #     image = pad_crop_to_res(image, self.roi_res)
        # image = pad_crop_to_res(image, self.target_res)

        input = self.transform(image)
        # print(input.shape)
        if self.color == 'r':
            input = input[0, :, :]  # red
        elif self.color == 'g':
            input = input[1, :, :]  # green
        else:
            input = input[2, :, :]  # blue

        if self.noise:
            input = self.transform4(input)

        data = self.data[idx]
        return {'image': input, 'distance': data}



if __name__ == '__main__':
    # test
    trainset = MyDataset(trainpath, distance_path, color='g', target_res=(1080, 1920), roi_res=(880, 1600))
    validset = MyDataset(validpath, distance_path2, color='g', target_res=(1080, 1920), roi_res=(880, 1600))
    dataloader = DataLoader(trainset, batch_size=1, shuffle=False)


    for batch in dataloader:
        images = batch['image'].unsqueeze(1)
        data = batch['distance'].unsqueeze(1).to(torch.float32)
        f = torch.fft.rfft2(images, dim=(2,3))
        # print(f.shape)
        im = Image.fromarray(images.cpu().squeeze(1).numpy()[0]*255)
        im.show()

