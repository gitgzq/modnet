import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as f
from torch.distributions.uniform import Uniform
from Model.context_model import Weighted_Gaussian
from Model.basic_module import Non_local_Block, ResBlock
from Model.context_model import P_Model
from Model.factorized_entropy_model import Entropy_bottleneck
from Model.gaussian_entropy_model import Distribution_for_entropy
from Model.sign_conv2d import SignConv2d
from Model.gdn import GDN2d



class bm(nn.Module):
    def __init__(self):
        super(bm, self).__init__()

        self.fcn1 = nn.Linear(1, 100)
        self.fcn2 = nn.Linear(100, 100)
        self.fcn3 = nn.Linear(100, 100)
        self.Relu = nn.ReLU()

    def forward(self, y):
        y1 = self.fcn1(y)
        y1 = self.Relu(y1)
        y2 = self.fcn2(y1)
        y2 = self.Relu(y2)
        mask = f.sigmoid(self.fcn3(y2))

        return mask


class Modnet(nn.Module):
    def __init__(self, in_channels, latent_channels):
        super(Modnet, self).__init__()

        self.m = int(in_channels)
        self.c = int(latent_channels)
        self.convs = nn.ModuleList([nn.Conv2d(self.m, self.c, 1, 1, 0)])
        for i in range(6):
            self.convs.append(nn.Conv2d(self.c, self.c, 1, 1, 0))
        self.convs.append(nn.Conv2d(self.c, self.m, 1, 1, 0))

        self.lmd_map = nn.ModuleList([])
        for i in range(7):
            self.lmd_map.append(bm())

    def forward(self, x, lmd):

        b=x.size()[0]
        y=lmd.cuda()
        masks = []

        for i in range(7):
            mask_i = self.lmd_map[i](y)
            masks.append(torch.reshape(mask_i,(b,self.c,1,1)).cuda())

        x0 = self.convs[0](x)
        for i in range(7):
            x0 = masks[i] * x0
            x0 = self.convs[i+1](x0)

        output = x * f.sigmoid(x0)

        return output


class Enc(nn.Module):
    def __init__(self, num_features, N1, N2, M, M1):

        super(Enc, self).__init__()
        self.N1 = int(N1)
        self.N2 = int(N2)
        self.M = int(M)
        self.M1 = int(M1)
        self.n_features = int(num_features)

        self.conv1 = nn.Conv2d(self.n_features, self.M1, 5, 1, 2)
        self.trunk1 = nn.Sequential(ResBlock(self.M1, self.M1, 3, 1, 1), ResBlock(
            self.M1, self.M1, 3, 1, 1), nn.Conv2d(self.M1, 2*self.M1, 5, 2, 2))

        self.down1 = nn.Conv2d(2*self.M1, self.M, 5, 2, 2)
        self.trunk2 = nn.Sequential(ResBlock(2*self.M1, 2*self.M1, 3, 1, 1), ResBlock(2*self.M1, 2*self.M1, 3, 1, 1),
                                    ResBlock(2*self.M1, 2*self.M1, 3, 1, 1))
        self.mask1 = nn.Sequential(Non_local_Block(2*self.M1, self.M1), ResBlock(2*self.M1, 2*self.M1, 3, 1, 1),
                                   ResBlock(
                                       2*self.M1, 2*self.M1, 3, 1, 1), ResBlock(2*self.M1, 2*self.M1, 3, 1, 1),
                                   nn.Conv2d(2*self.M1, 2*self.M1, 1, 1, 0))
        self.trunk3 = nn.Sequential(ResBlock(self.M, self.M, 3, 1, 1), ResBlock(self.M, self.M, 3, 1, 1),
                                    ResBlock(self.M, self.M, 3, 1, 1), nn.Conv2d(self.M, self.M, 5, 2, 2))

        self.trunk4 = nn.Sequential(ResBlock(self.M, self.M, 3, 1, 1), ResBlock(self.M, self.M, 3, 1, 1),
                                    ResBlock(self.M, self.M, 3, 1, 1), nn.Conv2d(self.M, self.M, 5, 2, 2))

        self.trunk5 = nn.Sequential(ResBlock(self.M, self.M, 3, 1, 1), ResBlock(self.M, self.M, 3, 1, 1),
                                    ResBlock(self.M, self.M, 3, 1, 1))
        self.mask2 = nn.Sequential(Non_local_Block(self.M, self.M // 2), ResBlock(self.M, self.M, 3, 1, 1),
                                   ResBlock(self.M, self.M, 3, 1, 1),
                                   ResBlock(self.M, self.M, 3, 1, 1), nn.Conv2d(self.M, self.M, 1, 1, 0))


        self.trunk6 = nn.Sequential(ResBlock(self.M, self.M, 3, 1, 1), ResBlock(self.M, self.M, 3, 1, 1),
                                    nn.Conv2d(self.M, self.M, 5, 2, 2))
        self.trunk7 = nn.Sequential(ResBlock(self.M, self.M, 3, 1, 1), ResBlock(self.M, self.M, 3, 1, 1),
                                    nn.Conv2d(self.M, self.M, 5, 2, 2))

        self.trunk8 = nn.Sequential(ResBlock(self.M, self.M, 3, 1, 1), ResBlock(self.M, self.M, 3, 1, 1),
                                    ResBlock(self.M, self.M, 3, 1, 1))
        self.mask3 = nn.Sequential(Non_local_Block(self.M, self.M // 2), ResBlock(self.M, self.M, 3, 1, 1),
                                   ResBlock(self.M, self.M, 3, 1, 1),
                                   ResBlock(self.M, self.M, 3, 1, 1), nn.Conv2d(self.M, self.M, 1, 1, 0))
        self.conv2 = nn.Conv2d(self.M, self.N2, 3, 1, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.trunk1(x1)
        x3 = self.trunk2(x2)+x2
        x3 = self.down1(x3)
        x4 = self.trunk3(x3)
        x5 = self.trunk4(x4)
        x6 = self.trunk5(x5)*f.sigmoid(self.mask2(x5)) + x5


        x7 = self.trunk6(x6)
        x8 = self.trunk7(x7)
        x9 = self.trunk8(x8)*f.sigmoid(self.mask3(x8)) + x8
        x10 = self.conv2(x9)

        return x6, x10



class Hyper_Dec(nn.Module):
    def __init__(self, N2, M):
        super(Hyper_Dec, self).__init__()

        self.N2 = N2
        self.M = M
        self.conv1 = nn.Conv2d(self.N2, M, 3, 1, 1)
        self.trunk1 = nn.Sequential(ResBlock(self.M, self.M, 3, 1, 1), ResBlock(self.M, self.M, 3, 1, 1),
                                    ResBlock(self.M, self.M, 3, 1, 1))
        self.mask1 = nn.Sequential(Non_local_Block(self.M, self.M // 2), ResBlock(self.M, self.M, 3, 1, 1),
                                   ResBlock(self.M, self.M, 3, 1, 1),
                                   ResBlock(self.M, self.M, 3, 1, 1), nn.Conv2d(self.M, self.M, 1, 1, 0))

        self.trunk2 = nn.Sequential(ResBlock(self.M, self.M, 3, 1, 1), ResBlock(self.M, self.M, 3, 1, 1),
                                    nn.ConvTranspose2d(M, M, 5, 2, 2, 1))
        self.trunk3 = nn.Sequential(ResBlock(self.M, self.M, 3, 1, 1), ResBlock(self.M, self.M, 3, 1, 1),
                                    nn.ConvTranspose2d(M, M, 5, 2, 2, 1))

    def forward(self, xq2):
        x1 = self.conv1(xq2)
        x2 = self.trunk1(x1) * f.sigmoid(self.mask1(x1)) + x1
        x3 = self.trunk2(x2)
        x4 = self.trunk3(x3)

        return x4


class Dec(nn.Module):
    def __init__(self, input_features, N1, M, M1):
        super(Dec, self).__init__()

        self.N1 = N1
        self.M = M
        self.M1 = M1
        self.input = input_features

        self.trunk1 = nn.Sequential(ResBlock(self.M, self.M, 3, 1, 1), ResBlock(self.M, self.M, 3, 1, 1),
                                    ResBlock(self.M, self.M, 3, 1, 1))
        self.mask1 = nn.Sequential(Non_local_Block(self.M, self.M // 2), ResBlock(self.M, self.M, 3, 1, 1),
                                   ResBlock(self.M, self.M, 3, 1, 1),
                                   ResBlock(self.M, self.M, 3, 1, 1), nn.Conv2d(self.M, self.M, 1, 1, 0))

        self.up1 = nn.ConvTranspose2d(M, M, 5, 2, 2, 1)
        self.trunk2 = nn.Sequential(ResBlock(self.M, self.M, 3, 1, 1), ResBlock(self.M, self.M, 3, 1, 1),
                                    ResBlock(self.M, self.M, 3, 1, 1), nn.ConvTranspose2d(M, M, 5, 2, 2, 1))
        self.trunk3 = nn.Sequential(ResBlock(self.M, self.M, 3, 1, 1), ResBlock(self.M, self.M, 3, 1, 1),
                                    ResBlock(self.M, self.M, 3, 1, 1), nn.ConvTranspose2d(M, 2 * self.M1, 5, 2, 2, 1))

        self.trunk4 = nn.Sequential(ResBlock(2 * self.M1, 2 * self.M1, 3, 1, 1),
                                    ResBlock(2 * self.M1, 2 * self.M1, 3, 1, 1),
                                    ResBlock(2 * self.M1, 2 * self.M1, 3, 1, 1))
        self.mask2 = nn.Sequential(Non_local_Block(2 * self.M1, self.M1), ResBlock(2 * self.M1, 2 * self.M1, 3, 1, 1),
                                   ResBlock(2 * self.M1, 2 * self.M1, 3, 1, 1),
                                   ResBlock(2 * self.M1, 2 * self.M1, 3, 1, 1),
                                   nn.Conv2d(2 * self.M1, 2 * self.M1, 1, 1, 0))

        self.trunk5 = nn.Sequential(nn.ConvTranspose2d(2 * M1, M1, 5, 2, 2, 1), ResBlock(self.M1, self.M1, 3, 1, 1),
                                    ResBlock(self.M1, self.M1, 3, 1, 1),
                                    ResBlock(self.M1, self.M1, 3, 1, 1))

        self.conv1 = nn.Conv2d(self.M1, self.input, 5, 1, 2)

    def forward(self, x):
        x1 = self.trunk1(x) * f.sigmoid(self.mask1(x)) + x
        x1 = self.up1(x1)
        x2 = self.trunk2(x1)
        x3 = self.trunk3(x2)
        x4 = self.trunk4(x3) + x3
        x5 = self.trunk5(x4)
        output = self.conv1(x5)
        return output


class Image_coding(nn.Module):
    def __init__(self, input_features, N1, N2, M, M1):

        super(Image_coding, self).__init__()
        self.N1 = N1
        self.encoder = Enc(input_features, N1, N2, M, M1)
        self.factorized_entropy_func = Entropy_bottleneck(N2)
        self.hyper_dec = Hyper_Dec(N2, M)
        self.p = P_Model(M)
        self.gaussin_entropy_func = Distribution_for_entropy()
        self.decoder = Dec(input_features, N1, M, M1)

    def add_noise(self, x):
        noise = np.random.uniform(-0.5, 0.5, x.size())
        noise = torch.Tensor(noise).cuda()
        return x + noise

    def forward(self, x, if_training):
        x1, x2 = self.encoder(x)
        xq2, xp2 = self.factorized_entropy_func(x2, if_training)
        x3 = self.hyper_dec(xq2)
        hyper_dec = self.p(x3)
        if if_training == 0:
            xq1 = self.add_noise(x1)
        elif if_training == 1:
            xq1 = UniverseQuant.apply(x1)
        else:
            xq1 = torch.round(x1)
        xp1 = self.gaussin_entropy_func(xq1, hyper_dec)

        output = self.decoder(xq1)

        return [output, xp1, xp2, xq1, hyper_dec]


class NIC_Modnet(nn.Module):
    def __init__(self, input_features, N1, N2, M, M1):
        super(NIC_Modnet, self).__init__()

        self.Modnet1 = Modnet(256,100)
        self.Modnet2 = Modnet(256,100)
        self.encoder = Enc(input_features, N1, N2, M, M1)
        self.decoder = Dec(input_features, N1, M, M1)
        self.factorized_entropy_func = Entropy_bottleneck(N2)
        self.hyper_dec = Hyper_Dec(N2, M)
        self.p = P_Model(M)
        self.context = Weighted_Gaussian(M)

    def forward(self, x):

        b = x.size()[0]
        rand_lambda = np.random.rand(b)
        rand_lambda = 256*rand_lambda + 1
        rand_lambda = np.array(rand_lambda, dtype=np.int)

        lmd_info = np.array(rand_lambda, dtype=np.float32)
        lmd_info = torch.from_numpy(lmd_info).cuda()
        lmd_info = torch.reshape(lmd_info,(b,1)).cuda()

        x1,x2 = self.encoder(x)

        x1 = self.Modnet1(x1, lmd_info)
        x1 = self.Modnet2(x1, lmd_info)

        xq2, xp2 = self.factorized_entropy_func(x2, 1)
        x3 = self.hyper_dec(xq2)

        hyper_dec_p = self.p(x3)
        xq1 = UniverseQuant.apply(x1)
        xp3, _ = self.context(xq1, hyper_dec_p)

        fake = self.decoder(xq1)

        lmd_info = torch.squeeze(lmd_info).cuda()

        return lmd_info, fake, xp2, xp3


class UniverseQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        b = np.random.uniform(-1, 1)
        uniform_distribution = Uniform(-0.5 * torch.ones(x.size())
                                       * (2 ** b), 0.5 * torch.ones(x.size()) * (2 ** b)).sample().cuda()
        return torch.round(x + uniform_distribution) - uniform_distribution

    @staticmethod
    def backward(ctx, g):
        return g
