import os.path

import torch
from torch import nn
import argparse
import numpy
import lpips
import torch.fft
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
from prop import ASM_split
from model.FourierNet import Flex_FourierNet
from dataset import MyDataset
from torch.utils.data import DataLoader
from loss import NPCC, TV, Merge1, Merge2
from distance_generation import gen_flex_distance

# python flex_train.py --channel=0 --lr=1e-3 --lr_decay=1 --loss=Merge2 --batch_size=1 --resume=1

def plot_model(writer, model, step):
    for name, param in model.named_parameters():
        writer.add_histogram(f"{name}", param, step)

p = argparse.ArgumentParser()
p.add_argument('--channel', type=int, default=1, help='red:0, green:1, blue:2, rgb:3')
p.add_argument('--model_path', type=str, default='model', help='torch save trained model')
p.add_argument('--ckpt_path', type=str, default='checkpoint', help='torch save checkpoint model')
p.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
p.add_argument('--epoch', type=int, default=1, help='total epoch')
p.add_argument('--batch_size', type=int, default=4, help='total epoch')
p.add_argument('--distance', type=list, default=[85, 115], help='near -> far distance')
p.add_argument('--loss', type=str, default='merge2', help='Options: npcc, mse, perceptual, merge1, merge2')
p.add_argument('--TV_w', type=float, default=1e-2, help='TV loss')
p.add_argument('--resume', type=int, default=0, help='Resume or not')
p.add_argument('--noise', type=int, default=0, help='if add noise')
p.add_argument('--lr_decay', type=int, default=1, help='if weight decay')
p.add_argument('--force_scale', type=int, default=1, help='if force_scale(default), if not force_scale then scale_loss')

arg = p.parse_args()
epoch = arg.epoch
batchsize = arg.batch_size
channel = arg.channel
color = ('r', 'g', 'b')[channel]
wl = (638, 520, 450)[channel]
lr = arg.lr

target_res = (2160, 3840)
roi_res = (2160, 3840)
cropx = (target_res[0] - roi_res[0]) // 2
cropy = (target_res[1] - roi_res[1]) // 2
pitch = 3.74e-3 #mm

distance_near = arg.distance[0]
distance_far = arg.distance[1]
center_distance = (distance_near + distance_far)/2
print("Near:", distance_near)
print("Far:", distance_far)
gen_flex_distance(distance_near, distance_far)

loss_fun = arg.loss
loss_fun = ['npcc', 'mse', 'perceptual', 'merge1', 'merge2'].index(loss_fun.lower())
L1 = nn.L1Loss().cuda()
if loss_fun == 0:
    criterion = NPCC().cuda()
    print('Using NPCC loss function')
elif loss_fun == 1:
    criterion = nn.MSELoss().cuda()
    print('Using MSE loss function')
elif loss_fun == 2:
    criterion = lpips.LPIPS(net='vgg').cuda()
    print('Using Perceptual loss function')
elif loss_fun == 3: # MSE+Perceptual
    criterion = Merge1(percep_weight=0.2).cuda()
    print('Using MSE + Perceptual loss function')
elif loss_fun == 4: # MSE+npcc
    criterion = Merge2().cuda()
    print('Using MSE + NPCC loss function')
TV_w = arg.TV_w
compute_TV = TV().cuda()

ckpt_dir = arg.ckpt_path + '/' + color + '/' + 'FourierNet_flex_checkpoint.pth'
# fix_dir = arg.model_path + '/' + color + '/' + 'FourierNet_fix_'+str(int(center_distance))+'.pth'
save_dir = arg.model_path + '/' + color + '/' + 'FourierNet_flex_'+str(int(center_distance))+'.pth'

model = Flex_FourierNet(wl=wl, center_distance=center_distance).cuda()
if not os.path.exists('writer'):
    os.makedirs('writer')
writer = SummaryWriter(f'writer')
ASM = ASM_split(wavelength=wl, res=target_res, roi=roi_res, pitch=pitch, apply_constraint=True).cuda()
ASM.eval()

if not arg.force_scale:
# if not force_scale then use scaleLoss
    scaleLoss = torch.ones(1)*1.1
    scaleLoss = scaleLoss.cuda()
    scaleLoss.requires_grad = True
    optvars = [scaleLoss, *model.parameters()]
    print("Using scale loss...")
else:
    scaleLoss = torch.ones(1)
    scaleLoss = scaleLoss.cuda()
    scaleLoss.requires_grad = False
    optvars = model.parameters()
    print("Force equal scale...")

optimizer = torch.optim.Adam(optvars, lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

trainpath = '/home/user/lnh_2023/Ada-Holo/dataset/DIV2K_train_HR'
validpath = '/home/user/lnh_2023/Ada-Holo/dataset/DIV2K_valid_HR'
distance_path = '/home/user/lnh_2023/Ada-Holo/dataset/distance.txt'
distance_path2 = '/home/user/lnh_2023/Ada-Holo/dataset/distance-valid.txt'

if arg.noise:
    trainset = MyDataset(trainpath, distance_path, color, target_res, roi_res, noise=True)
else:
    trainset = MyDataset(trainpath, distance_path, color, target_res, roi_res, noise=False)
validset = MyDataset(validpath, distance_path2, color, target_res, roi_res, noise=True)
trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=True, drop_last=True)
validloader = DataLoader(trainset, batch_size=batchsize, shuffle=False, drop_last=True)

if arg.resume:
    pretrained = torch.load(ckpt_dir)
    # pretrained = torch.load(save_dir)
    print("load from checkpoint Flex_FourierNet")
    model.load_state_dict(pretrained)
    model.requires_grad_(True)
    model.prop.requires_grad_(False)

    # pretrained = torch.load(g_dir)
    # print("load from pretrained Flex_FourierNet")
    # model.load_state_dict(pretrained)
    # model.requires_grad_(True)
    # model.prop.requires_grad_(False)


l = []
tl = []
global_step = 0
# pad = nn.ZeroPad2d((512,512,512,512)).cuda()

for k in trange(epoch):
    currenttloss = 0
    for batch in trainloader:
        global_step += 1
        image = batch['image'].unsqueeze(1).cuda()
        distance = batch['distance'].unsqueeze(1).to(torch.float32).cuda()
        output = model(image, distance)
        image_c = image[:, :, (target_res[0] - roi_res[0]) // 2:(target_res[0] + roi_res[0]) // 2,
                  (target_res[1] - roi_res[1]) // 2:(target_res[1] + roi_res[1]) // 2]
        _, _, final = ASM(torch.ones_like(output), output, distance)

        if arg.force_scale:
            scale = torch.sum(image_c, dim=(-2, -1), keepdim=True) / torch.sum(final, dim=(-2, -1), keepdim=True)
            final *= scale
        else:# scaleLoss
            final *= scaleLoss
            writer.add_scalar("scale", scaleLoss.item(), global_step=global_step)

        final_TVx, final_TVy = compute_TV(final)
        image_TVx, image_TVy = compute_TV(image_c)

        loss = criterion(final, image_c) +\
               TV_w * L1(final_TVx, image_TVx) + TV_w * L1(final_TVy, image_TVy) \
                + 0.1*(1-scaleLoss)*(1-scaleLoss)

        writer.add_scalar("loss", loss.item(), global_step=global_step)

        # plot_model(writer, model, global_step)

        if global_step % (500*1//batchsize) == 0:
            writer.add_images("train_image", final, global_step=global_step)
            writer.add_images("image", image_c, global_step=global_step)
            torch.save(model.state_dict(), ckpt_dir)
            torch.save(model.state_dict(), save_dir)
            if arg.lr_decay:
                writer.add_scalar("lr", optimizer.param_groups[0]['lr'], global_step=global_step)
                scheduler.step()
            #plot_model(writer, model, global_step)

        # print('loss:', loss.cpu().data.numpy())
        currenttloss = currenttloss + loss.cpu().data.numpy()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    tl.append(currenttloss / len(trainloader))
    print('trainloss:', currenttloss / len(trainloader))

''' Validation
with torch.no_grad():
    currenttloss = 0
    global_step = 0
    for batch in validloader:
        global_step += 1
        image = batch['image'].unsqueeze(1).cuda()
        distance = batch['distance'].unsqueeze(1).to(torch.float32).cuda()
        output = model(image, distance)
        image_c = image[:, :, (target_res[0] - roi_res[0]) // 2:(target_res[0] + roi_res[0]) // 2,
                  (target_res[1] - roi_res[1]) // 2:(target_res[1] + roi_res[1]) // 2]
        _, _, final = ASM(torch.ones_like(output), output, distance)
        scale = torch.sum(image_c, dim=(-2, -1), keepdim=True) / torch.sum(final, dim=(-2, -1), keepdim=True)
        final *= scale

        loss = criterion(final, image_c)
               
        print('validate_loss:', loss.cpu().data.numpy())
        currenttloss = currenttloss + loss.cpu().data.numpy()

    l.append(currenttloss / len(validloader))
    print('validloss:',currenttloss / len(validloader))
#'''

torch.save(model.state_dict(), save_dir)
