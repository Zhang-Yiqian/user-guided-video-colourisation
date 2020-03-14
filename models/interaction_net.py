from __future__ import division
import torch
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# general libs
import numpy as np
import math
from torch import optim

print('Interaction Network: initialized')


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # [ab values, binary mask]
        self.conv1_strokes = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True)
        # previous round frame, ab space
        self.conv1_prev = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=True)
        # grayscale frame
        self.conv1_gray = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=True)

        # initialisation methods in Oh's paper
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        resnet = models.resnet50(pretrained=True)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1  # 1/4, 256
        self.res3 = resnet.layer2  # 1/8, 512
        self.res4 = resnet.layer3  # 1/16, 1024
        self.res5 = resnet.layer4  # 1/32, 2048

        # freeze BNs
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                for p in m.parameters():
                    p.requires_grad = False

    def forward(self, in_gray, in_strokes, in_prev):
        f = (in_gray - Variable(self.mean)) / Variable(self.std)
        s = torch.unsqueeze(in_strokes, dim=1).float()  # add channel dim
        p = torch.unsqueeze(in_prev, dim=1).float()  # add channel dim

        x = self.conv1_gray(f) + self.conv1_strokes(s) + self.conv1_prev(p)
        x = self.bn1(x)
        c1 = self.relu(x)     # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)     # 1/4, 64
        r3 = self.res3(r2)    # 1/8, 128
        r4 = self.res4(r3)    # 1/16, 256
        r5 = self.res5(r4)    # 1/32, 512

        return r5, r4, r3, r2


class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)

        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)

    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))

        if self.downsample is not None:
            x = self.downsample(x)
        
        return x + r 


class Refine(nn.Module):
    def __init__(self, inplanes, planes, scale_factor=2):
        super(Refine, self).__init__()
        self.ResFS = ResBlock(inplanes, planes)
        self.ResMM = ResBlock(planes, planes)
        self.scale_factor = scale_factor

    def forward(self, f, pm):
        s = self.ResFS(f)
        m = s + F.upsample(pm, scale_factor=self.scale_factor, mode='bilinear')
        m = self.ResMM(m)
        return m


class Decoder(nn.Module):
    def __init__(self, mdim):
        super(Decoder, self).__init__()
        self.ResFM = ResBlock(2048, mdim)
        self.RF4 = Refine(1024, mdim)  # 1/16 -> 1/8
        self.RF3 = Refine(512, mdim)   # 1/8 -> 1/4
        self.RF2 = Refine(256, mdim)   # 1/4 -> 1

        self.pred5 = nn.Conv2d(mdim, 2, kernel_size=(3,3), padding=(1,1), stride=1)
        self.pred4 = nn.Conv2d(mdim, 2, kernel_size=(3,3), padding=(1,1), stride=1)
        self.pred3 = nn.Conv2d(mdim, 2, kernel_size=(3,3), padding=(1,1), stride=1)
        self.pred2 = nn.Conv2d(mdim, 2, kernel_size=(3,3), padding=(1,1), stride=1)

    def forward(self, r5, r4, r3, r2):
        m5 = self.ResFM(r5)
        m4 = self.RF4(r4, m5)  # out: 1/16, 256
        m3 = self.RF3(r3, m4)  # out: 1/8, 256
        m2 = self.RF2(r2, m3)  # out: 1/4, 256

        p2 = self.pred2(F.relu(m2))
        p3 = self.pred3(F.relu(m3))
        p4 = self.pred4(F.relu(m4))
        p5 = self.pred5(F.relu(m5))

        p = F.upsample(p2, scale_factor=4, mode='bilinear')
        
        return p, p2, p3, p4, p5


class HuberLoss(nn.Module):
    def __init__(self, delta=.01):
        super(HuberLoss, self).__init__()
        self.delta=delta

    def __call__(self, in0, in1):
        mask = torch.zeros_like(in0)
        mann = torch.abs(in0-in1)
        eucl = .5 * (mann**2)
        mask[...] = mann < self.delta

        # loss = eucl*mask + self.delta*(mann-.5*self.delta)*(1-mask)
        loss = eucl*mask/self.delta + (mann-.5*self.delta)*(1-mask)
        return torch.sum(loss,dim=1,keepdim=True)


class Inet(nn.Module):
    def __init__(self, opt):
        super(Inet, self).__init__()
        mdim = 256
        self.Encoder = Encoder()      # inputs: ref: rf, rm / tar: tf, tm
        self.Decoder = Decoder(mdim)  # input: m5, r4, r3, r2 >> p
        self.opt = opt
        self.isTrain = opt.isTrain
        self.criterion = HuberLoss()
        self.model_names = 'Interaction net'

    def forward(self, in_frame, in_strokes, in_prev):
        tr5, tr4, tr3, tr2 = self.Encoder(in_frame, in_strokes, in_prev)
        em_ab = self.Decoder(tr5, tr4, tr3, tr2)

        return em_ab

    # load and print networks; create schedulers
    def setup(self, opt, parser=None):
        if self.isTrain:
            self.optimizer = optim.Adam(self.parameters(), lr = opt.lr, weight_decay = opt.weight_decay) 

        if not self.isTrain or opt.load_model:
            self.load_networks(opt.which_epoch)

    def setup_input(self, input):
        self.input = input

    # not changed but might be used
    def load_networks(self, which_epoch):
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (which_epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    def optimize_parameters(self):
        em_ab = self.forward()
        # update G
        self.optimizer.zero_grad()
        self.loss = self.criterion(em_ab, self.input['B'])
        self.loss.backward()
        self.optimizer.step()








