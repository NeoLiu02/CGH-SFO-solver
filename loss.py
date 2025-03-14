import torch
import torch.nn as nn
import torchvision.models
import lpips

class NPCC(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image, target):
        batch, channel, _, _ = image.shape
        x = image.contiguous().view(batch, channel,-1)
        target = target.contiguous().view(batch, channel, -1)
        mean1 = torch.mean(x, dim=2).unsqueeze(2)
        mean2 = torch.mean(target, dim=2).unsqueeze(2)
        cov1 = torch.matmul(x-mean1, (target-mean2).transpose(1,2))
        diag1 = torch.matmul(x-mean1, (x-mean1).transpose(1,2))
        diag2 = torch.matmul(target-mean2, (target-mean2).transpose(1,2))
        pearson = cov1 / torch.sqrt(diag2 * diag1)
        return 1-pearson.squeeze().mean()


class TV(nn.Module):
    def __init__(self):
        super(TV, self).__init__()

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = x[:, :, 1:, :] - x[:, :, :h_x - 1, :]
        w_tv = x[:, :, :, 1:] - x[:, :, :, :w_x - 1]
        return h_tv, w_tv
        # h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        # w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        # return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class Merge1(nn.Module): # MSE+Perceptual
    def __init__(self, percep_weight):
        super().__init__()
        self.mse = nn.MSELoss()
        self.percep = lpips.LPIPS(net='vgg').cuda()
        self.weight = percep_weight
    def forward(self, x, y):
        return self.mse(x, y)+self.weight * torch.mean(self.percep(x, y))

class Merge2(nn.Module): # MSE+npcc
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.npcc = NPCC()
    def forward(self, x, y):
        return 2*self.mse(x, y)+self.npcc(x, y)

