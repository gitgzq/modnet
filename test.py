import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import Model.model as model
import Util.torch_msssim as torch_msssim
import time
import glob
import struct
from Model.context_model import Weighted_Gaussian
from Util.metrics import evaluate
import AE


def test(args):

    test_images = []

    if os.path.isdir(args.input):
        dirs = os.listdir(args.input)
        for dir in dirs:
            path = os.path.join(args.input, dir)
            if os.path.isdir(path):
                test_images += glob.glob(path + '/*.png')
            if os.path.isfile(path):
                test_images.append(path)

    else:
        test_images.append(args.input)

    im_dirs = test_images

    net = model.NIC_Modnet(3, args.M, args.N2, args.M, args.M // 2).cuda()

    model_existed = os.path.exists(args.model)

    if model_existed:
        net.load_state_dict(torch.load(args.model))
        print('resumed the trained model')
    else:
        print('model not exists')

    for im_dir in im_dirs:

        bin_dir = os.path.join(args.out_dir, 'enc.bin')
        rec_dir = os.path.join(args.out_dir, 'dec.png')
        file_object = open(bin_dir, 'wb')
        img = Image.open(im_dir)
        ori_img = np.array(img)
        img = ori_img
        H, W, _ = img.shape
        num_pixels = H * W
        C = 3
        H_offset = 0
        W_offset = 0
        out_img = np.zeros([H, W, C])

        Block_Num_in_Width = int(np.ceil(W / 2048))
        Block_Num_in_Height = int(np.ceil(H / 1024))
        img_block_list = []
        for i in range(Block_Num_in_Height):
            for j in range(Block_Num_in_Width):
                img_block_list.append(img[i * 1024:np.minimum((i + 1) * 1024, H),
                                      j * 2048:np.minimum((j + 1) * 2048, W), ...])

        Block_Idx = 0
        for img in img_block_list:
            block_H = img.shape[0]
            block_W = img.shape[1]

            tile = 64.
            block_H_PAD = int(tile * np.ceil(block_H / tile))
            block_W_PAD = int(tile * np.ceil(block_W / tile))
            im = np.zeros([block_H_PAD, block_W_PAD, 3], dtype='float32')
            im[:block_H, :block_W, :] = img[:, :, :3] / 255.0
            im = torch.FloatTensor(im)
            im = im.permute(2, 0, 1).contiguous()
            im = im.view(1, C, block_H_PAD, block_W_PAD).cuda()

            Block_Idx += 1

            with torch.no_grad():

                y_main, y_hyper = net.encoder(im)

                y_main_q = net.SQL1(y_main,args.lmd)
                y_main_q = net.SQL2(y_main_q,args.lmd)

                y_main_q = torch.round(y_main_q)
                y_main_q = torch.Tensor(y_main_q.cpu().numpy().astype(np.int)).cuda()

                rec = net.decoder(y_main_q)
                output_ = torch.clamp(rec, min=0., max=1.0)
                out = output_.data[0].cpu().numpy()
                out = out.transpose(1, 2, 0)
                out_img[H_offset: H_offset + block_H, W_offset: W_offset + block_W, :] = out[:block_H, :block_W, :]
                W_offset += block_W
                if W_offset >= W:
                    W_offset = 0
                    H_offset += block_H

                y_hyper_q, xp2 = net.factorized_entropy_func(y_hyper, 2)
                y_hyper_q = torch.Tensor(y_hyper_q.cpu().numpy().astype(np.int)).cuda()

                hyper_dec = net.p(net.hyper_dec(y_hyper_q))

                xp3, params_prob = net.context(y_main_q, hyper_dec)
                bpp_hyper = (torch.sum(torch.log(xp2)) / (-np.log(2) * num_pixels)).item()
                bpp_main = (torch.sum(torch.log(xp3)) / (-np.log(2) * num_pixels)).item()
                print('bpp_hyper_info:', bpp_hyper, 'bpp_main_info:', bpp_main, 'bpp_total_info:',
                      bpp_hyper + bpp_main)

            Datas = torch.reshape(y_main_q, [-1]).cpu().numpy().astype(np.int).tolist()
            Max_Main = max(Datas)
            Min_Main = min(Datas)
            sample = np.arange(Min_Main, Max_Main + 1 + 1)
            _, c, h, w = y_main_q.shape
            print("Main Channel:", c)
            sample = torch.FloatTensor(np.tile(sample, [1, c, h, w, 1])).cpu()

            params_prob = params_prob.cpu()
            prob0, mean0, scale0, prob1, mean1, scale1, prob2, mean2, scale2 = [
                torch.chunk(params_prob, 9, dim=1)[i].squeeze(1) for i in range(9)]
            del params_prob

            probs = torch.stack([prob0, prob1, prob2], dim=-1)
            del prob0, prob1, prob2

            probs = F.softmax(probs, dim=-1)

            scale0 = torch.abs(scale0)
            scale1 = torch.abs(scale1)
            scale2 = torch.abs(scale2)
            scale0[scale0 < 1e-6] = 1e-6
            scale1[scale1 < 1e-6] = 1e-6
            scale2[scale2 < 1e-6] = 1e-6

            m0 = torch.distributions.normal.Normal(mean0, scale0)
            m1 = torch.distributions.normal.Normal(mean1, scale1)
            m2 = torch.distributions.normal.Normal(mean2, scale2)
            lower = torch.zeros(1, c, h, w, Max_Main - Min_Main + 2)
            for i in range(sample.shape[4]):

                lower0 = m0.cdf(sample[:, :, :, :, i] - 0.5)
                lower1 = m1.cdf(sample[:, :, :, :, i] - 0.5)
                lower2 = m2.cdf(sample[:, :, :, :, i] - 0.5)
                lower[:, :, :, :, i] = probs[:, :, :, :, 0] * lower0 + \
                                       probs[:, :, :, :, 1] * lower1 + probs[:, :, :, :, 2] * lower2
            del probs, lower0, lower1, lower2

            precise = 16
            cdf_m = lower.data.cpu().numpy() * ((1 << precise) - (Max_Main -
                                                                  Min_Main + 1))  # [1, c, h, w ,Max-Min+1]
            cdf_m = cdf_m.astype(np.int32) + sample.numpy().astype(np.int32) - Min_Main
            cdf_main = np.reshape(cdf_m, [len(Datas), -1])


            Cdf_lower = list(map(lambda x, y: int(y[x - Min_Main]), Datas, cdf_main))
            Cdf_upper = list(map(lambda x, y: int(
                y[x - Min_Main]), Datas, cdf_main[:, 1:]))
            AE.encode_cdf(Cdf_lower, Cdf_upper, "/output/main.bin")
            FileSizeMain = os.path.getsize("/output/main.bin")
            print("/output/main.bin: %d bytes" % (FileSizeMain))


            Min_V_HYPER = torch.min(y_hyper_q).cpu().numpy().astype(np.int).tolist()
            Max_V_HYPER = torch.max(y_hyper_q).cpu().numpy().astype(np.int).tolist()

            Datas_hyper = torch.reshape(
                y_hyper_q, [c, -1]).cpu().numpy().astype(np.int).tolist()

            sample = np.arange(Min_V_HYPER, Max_V_HYPER + 1 + 1)
            sample = np.tile(sample, [c, 1, 1])
            sample = torch.FloatTensor(sample).cuda()
            lower = torch.sigmoid(net.factorized_entropy_func._logits_cumulative(
                sample - 0.5, stop_gradient=False))
            cdf_h = lower.data.cpu().numpy() * ((1 << precise) - (Max_V_HYPER -
                                                                  Min_V_HYPER + 1))

            cdf_h = cdf_h.astype(np.int) + sample.detach().cpu().numpy().astype(np.int) - Min_V_HYPER
            cdf_hyper = np.reshape(np.tile(cdf_h, [len(Datas_hyper[0]), 1, 1, 1]), [
                len(Datas_hyper[0]), c, -1])


            Cdf_0, Cdf_1 = [], []
            for i in range(c):
                Cdf_0.extend(list(map(lambda x, y: int(
                    y[x - Min_V_HYPER]), Datas_hyper[i], cdf_hyper[:, i, :])))
                Cdf_1.extend(list(map(lambda x, y: int(
                    y[x - Min_V_HYPER]), Datas_hyper[i], cdf_hyper[:, i, 1:])))
            AE.encode_cdf(Cdf_0, Cdf_1, "/output/hyper.bin")
            FileSizeHyper = os.path.getsize("/output/hyper.bin")
            print("/output/hyper.bin: %d bytes" % (FileSizeHyper))

            Head_block = struct.pack('2H4h2I', block_H, block_W, Min_Main, Max_Main, Min_V_HYPER, Max_V_HYPER,
                                     FileSizeMain, FileSizeHyper)
            file_object.write(Head_block)

            with open("/output/main.bin", 'rb') as f:
                bits = f.read()
                file_object.write(bits)
            f.close()
            with open("/output/hyper.bin", 'rb') as f:
                bits = f.read()
                file_object.write(bits)
            f.close()
            del im

        file_object.close()
        with open(bin_dir, "rb") as f:
            bpp = len(f.read()) * 8. / num_pixels
            print('bpp_total_true:', bpp)
        f.close()

        out_img = np.round(out_img * 255.0)
        out_img = out_img.astype('uint8')
        img = Image.fromarray(out_img[:H, :W, :])
        img.save(rec_dir)
        [rgb_psnr, rgb_msssim, yuv_psnr, y_msssim] = evaluate(ori_img, out_img)

        class_name = im_dir.split('/')[-2]
        image_name = im_dir.split('/')[-1].replace('.png', '')

        with open(os.path.join(args.out_dir, 'mse_RD.txt'),
                  "a") as f:
            f.write(class_name + '/' + image_name + '\t' + str(bpp) + '\t' + str(rgb_psnr) + '\t' + str(
                rgb_msssim) + '\t' + str(-10 * np.log10(1 - rgb_msssim)) +
                    '\t' + str(yuv_psnr) + '\t' + str(y_msssim) + '\t' + str(-10 * np.log10(1 - y_msssim)) + '\n')
        f.close()
        del out_img

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=256, help="the value of M")
    parser.add_argument("--b_size", type=int, default=1)
    parser.add_argument("--N2", type=int, default=192, help="the value of N2")
    parser.add_argument("--lmd", type=float, required=True , help="Lambda for rate-distortion tradeoff.")
    parser.add_argument('--out_dir', type=str, default='/output/')

    parser.add_argument("-i", "--input", type=str, help="Input Image")
    parser.add_argument('--model', type=str, default='/proposed_model/modnet.pkl')
    parser.add_argument("--gpu", type=int, default=1)

    args = parser.parse_args()
    print(args)
    test(args)
