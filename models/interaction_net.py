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
import warnings
import os

print('Interaction Network: initialized')


class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        # clicks(2) & binary mask
        self.conv1_clicks = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True)
        # previous round frame, ab space
        self.conv1_prev = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=True)
        # grayscale
        self.conv1_gray = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=True)

        for m in self.modules():
          if isinstance(m, nn.Conv2d):
              nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
              if m.bias is not None:
                  nn.init.normal_(m.bias.data)
          elif isinstance(m, nn.BatchNorm2d):
              nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
              m.bias.data.zero_()

        resnet = models.resnet50(pretrained=True)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1  # 1/4, 256
        self.res3 = resnet.layer2  # 1/8, 512
        self.res4 = resnet.layer3  # 1/16, 1024
        self.res5 = resnet.layer4  # 1/32, 2048

        if opt.phase == 'I_auto':
            self.auto_colourisation = True  
        else:
            self.auto_colourisation = False  

    def forward(self, gray, clicks, prev):
        m = self.conv1_clicks(clicks) + self.conv1_prev(prev)
        if self.auto_colourisation:
            x = m.detach() + self.conv1_gray(gray)
        else:
            x = m + self.conv1_gray(gray)
        x = self.bn1(x)
        x = self.relu(x)     # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        r2 = self.res2(x)    # 1/4, 64  
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
        m = s + F.interpolate(pm, scale_factor=self.scale_factor, mode='bilinear')
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
        
        for m in self.modules():
          if isinstance(m, nn.Conv2d):
              nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
              if m.bias is not None:
                  nn.init.normal_(m.bias.data)
          elif isinstance(m, nn.BatchNorm2d):
              nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
              m.bias.data.zero_()

    def forward(self, r5, r4, r3, r2):
        x = self.ResFM(r5)
        x = self.RF4(r4, x)  # out: 1/16, 256
        x = self.RF3(r3, x)  # out: 1/8, 256
        m2 = self.RF2(r2, x)  # out: 1/4, 256
       
        # m5 = self.ResFM(r5)
        # m4 = self.RF4(r4, m5)  # out: 1/16, 256
        # m3 = self.RF3(r3, m4)  # out: 1/8, 256
        # m2 = self.RF2(r2, m3)  # out: 1/4, 256
        p2 = self.pred2(F.relu(m2))

        # p3 = self.pred3(F.relu(m3))
        # p4 = self.pred4(F.relu(m4))
        # p5 = self.pred5(F.relu(m5))

        p = F.interpolate(p2, scale_factor=4, mode='bilinear')
        
        return p#, p2, p3, p4, p5


class HuberLoss(nn.Module):
    def __init__(self, delta=.01):
        super(HuberLoss, self).__init__()
        self.delta=delta

    def __call__(self, in0, in1):
        mask = torch.zeros_like(in0)
        mann = torch.abs(in0-in1)
        eucl = .5 * (mann**2)
        mask[...] = mann < self.delta

        loss = eucl*mask/self.delta + (mann-.5*self.delta)*(1-mask)
        return torch.sum(loss,dim=1,keepdim=True)


class Inet(nn.Module):
    def __init__(self, opt):
        super(Inet, self).__init__()
        mdim = 228
        self.Encoder = Encoder(opt)      # inputs: ref: rf, rm / tar: tf, tm
        self.Decoder = Decoder(mdim)  # input: m5, r4, r3, r2 >> p
        self.opt = opt
        self.isTrain = opt.isTrain
        self.criterion = nn.SmoothL1Loss() 

    def forward(self, gray, clicks, prev):
        tr5, tr4, tr3, tr2 = self.Encoder(gray, clicks, prev)
        em_ab = self.Decoder(tr5, tr4, tr3, tr2)

        return em_ab

    # load and print networks; create schedulers
    def setup(self, opt):
        if self.isTrain:
            self.optimizer = optim.Adam(self.parameters(), lr = opt.lr, weight_decay = opt.weight_decay) 

        if not self.isTrain or opt.load_model:
            self.load_networks(opt.which_epoch)

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






