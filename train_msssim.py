import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import Model.model as model
from Model.context_model import Weighted_Gaussian
import time
import Util.torch_msssim as torch_msssim


def adjust_learning_rate(optimizer, epoch, init_lr):

    if epoch < 3:
        lr = init_lr
    else:
        lr = init_lr * (0.5 ** ((epoch-3) // 2))
    if lr < 1e-6:
        lr = 1e-6
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr



def train(args):
    train_data = ImageFolder(root=args.data, transform=transforms.Compose(
        [transforms.RandomCrop(256), transforms.ToTensor()]))
    train_loader = DataLoader(train_data, batch_size=args.b_size,
                              shuffle=True, num_workers=8)

    image_comp = model.Image_coding(3, args.M, args.N2, args.M, args.M // 2).cuda()
    context = Weighted_Gaussian(args.M).cuda()
    net = model.NIC_Modnet(3, args.M, args.N2, args.M, args.M // 2).cuda()


    model_existed = os.path.exists(os.path.join(args.model, 'fr_mse.pkl')) and \
                    os.path.exists(os.path.join(args.model, 'fr_msep.pkl'))

    if model_existed:
        image_comp.load_state_dict(torch.load(os.path.join(args.model, 'fr_mse.pkl')))
        context.load_state_dict(torch.load(os.path.join(args.model, 'fr_msep.pkl')))
        net.encoder = image_comp.encoder.cuda()
        net.decoder = image_comp.decoder.cuda()
        net.hyper_dec = image_comp.hyper_dec.cuda()
        net.factorized_entropy_func = image_comp.factorized_entropy_func.cuda()
        net.p = image_comp.p.cuda()
        net.context = context.cuda()
        print('resumed from the fixed-rate model')

    else:
        print("fixed-rate model not found")

    if args.gpu > 1:
        gpu_id = [id for id in range(args.gpu)]
        net = nn.DataParallel(net, device_ids=gpu_id)

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    msssim_func = torch_msssim.MS_SSIM(max_val=1.).cuda()
    mse_func = nn.MSELoss()

    for epoch in range(20):
        rec_loss_tmp = 0
        last_time = time.time()
        train_bpp_tmp = 0
        mse_tmp=0
        msssim_tmp = 0
        cur_lr = adjust_learning_rate(opt, epoch, args.lr)

        for step, batch_x in enumerate(train_loader):
            batch_x = batch_x[0]
            num_pixels = batch_x.size()[0] * \
                         batch_x.size()[2] * batch_x.size()[3]
            batch_x = Variable(batch_x).cuda()
            b = batch_x.size()[0]

            lmd_info,fake,xp2,xp3=net(batch_x)

            msssim = msssim_func(fake, batch_x)

            if epoch<1:
                delta = (fake - batch_x) ** 2
                delta = delta.view(b, -1)
                dloss = torch.mean(delta, dim=1, keepdim=False).cuda()
            else:
                dloss = 1.0 - msssim
                dloss = dloss.cuda()

            train_bpp_total = torch.sum(torch.log(xp2)) / (-np.log(2) * num_pixels) + torch.sum(torch.log(xp3)) / (
                    -np.log(2) * num_pixels)

            lambda_dloss = torch.dot(lmd_info,dloss)/float(args.b_size)

            if epoch<1:
                l_rec = lambda_dloss + 0.01 * train_bpp_total
            else :
                l_rec = 0.01 * lambda_dloss + 0.01 * train_bpp_total

            opt.zero_grad()

            l_rec.backward()

            torch.nn.utils.clip_grad_norm_(net.parameters(), 5)

            opt.step()

            rec_loss_tmp += l_rec.item()
            train_bpp_tmp += train_bpp_total.item()
            mse_tmp += torch.mean((fake-batch_x)**2).item()
            msssim_tmp += torch.mean(msssim).item()


            if step % 100 == 0:

                with open(os.path.join(args.out_dir, 'train_msssim.txt'), 'a') as fd:
                    time_used = time.time() - last_time
                    last_time = time.time()
                    msssim_dB = -10.0*np.log10(1-(msssim_tmp/(step+1)))
                    psnr = -10.0*np.log10(mse_tmp/(step+1))
                    bpp_total = train_bpp_tmp / (1+step)
                    fd.write(
                        'ep:%d step:%d time:%.1f lr:%.8f loss:%.6f bpp:%.4f psnr:%.2f msssim:%.2f\n'
                        % (epoch, step, time_used,cur_lr,rec_loss_tmp/(step+1), bpp_total ,psnr, msssim_dB))
                fd.close()

            if (step + 1) % 2000 == 0:
                torch.save(net.module.state_dict(),
                           os.path.join(args.out_dir, 'Modnetmsssim.pkl'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=256, help="the value of M")
    parser.add_argument("--b_size", type=int, default=12)
    parser.add_argument("--N2", type=int, default=192, help="the value of N2")
    parser.add_argument("--lr", type=float, default=5e-5, help="initial learning rate.")
    parser.add_argument('--out_dir', type=str, default='proposed_model/')
    parser.add_argument('--model', type=str, default='baseline_model/')
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument("--gpu", type=int, default=2)

    args = parser.parse_args()
    print(args)
    train(args)
