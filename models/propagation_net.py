from __future__ import division
import torch

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch import optim
# general libs
import math
print('Propagation Network: initialising')

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # current grayscale
        self.conv1_gray = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=True)
        # previous round ab
        self.conv1_prev_r = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=True)
        # previous time ab
        self.conv1_prev_t = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=True)

        for m in self.modules():
          if isinstance(m, nn.Conv2d):
              nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
              if m.bias is not None:
                  nn.init.normal_(m.bias.data)
          elif isinstance(m, nn.BatchNorm2d):
              nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
              m.bias.data.zero_()

        # extract ResNet layers
        resnet = models.resnet50(pretrained=True)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool
        self.res2 = resnet.layer1  # 1/4, 256
        self.res3 = resnet.layer2  # 1/8, 512
        self.res4 = resnet.layer3  # 1/16, 1024
        self.res5 = resnet.layer4  # 1/32, 2048

    def forward(self, gray, prev_r, prev_t):
        
        x = self.conv1_gray(gray) + self.conv1_prev_r(prev_r) + self.conv1_prev_t(prev_t)
        x = self.bn1(x)
        c1 = self.relu(x)   # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)   # 1/4, 64
        r3 = self.res3(r2) # 1/8, 128
        r4 = self.res4(r3) # 1/16, 256
        r5 = self.res5(r4) # 1/32, 512

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
    def __init__(self, mdim, opt):
        super(Decoder, self).__init__()
        self.ResFM1 = ResBlock(4096, 1024)
        self.ResFM2= ResBlock(1024, mdim)
        self.RF4 = Refine(1024, mdim) # 1/16 -> 1/8
        self.RF3 = Refine(512, mdim) # 1/8 -> 1/4
        self.RF2 = Refine(256, mdim) # 1/4 -> 1
        self.pred2 = nn.Conv2d(mdim, 2, kernel_size=(3,3), padding=(1,1), stride=1)
        self.opt = opt

    def forward(self, rr5, r5, r4, r3, r2):
        
        x = torch.cat([rr5, r5], dim=1)
        x = self.ResFM1(x)   
        x = self.ResFM2(x)
        x = self.RF4(r4, x)  # out: 1/16, 256
        x = self.RF3(r3, x)  # out: 1/8, 256
        x = self.RF2(r2, x)  # out: 1/4, 256
        x = self.pred2(F.relu(x))
        x = F.interpolate(x, scale_factor=4, mode='bilinear')
        out_reg = self.tanh(x)
        return out_reg

        
class SEFA(nn.Module):
    # Sequeeze-Expectation Feature Aggregation 
    def __init__(self, inplanes, r=4):
        super(SEFA, self).__init__()
        self.inplanes = inplanes
        self.fc1 = nn.Linear(2*inplanes, int(2*inplanes/r))
        self.fc2 = nn.Linear(int(2*inplanes/r), 2*inplanes)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = x.mean(-1).mean(-1) # global pool # 2048*2
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # b, 4096 
        x = F.softmax(x.view(-1, self.inplanes, 2), dim=2)
        w1 = x[:,:,0].contiguous().view(-1, self.inplanes, 1, 1)
        w2 = x[:,:,1].contiguous().view(-1, self.inplanes, 1, 1)
        out = x1*w1 + x2*w2 
        return out

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
        # return torch.sum(loss,dim=1,keepdim=True)
        return torch.mean(loss)

class Pnet(nn.Module):
    def __init__(self, opt):
        super(Pnet, self).__init__()
        mdim = 228
        self.Encoder = Encoder() # inputs:: ref: rf, rm / tar: tf, tm 
        self.SEFA = SEFA(2048, r=2)
        self.Decoder = Decoder(mdim, opt) # input: m5, r4, r3, r2 >> p
        self.isTrain = opt.isTrain
        self.load_P = opt.load_P
        self.P_path = opt.P_path
        self.opt = opt

    def forward(self, gray, prev_r, prev_t, crt_fam, prev_fam=None):
        tr5, tr4, tr3, tr2 = self.Encoder(gray, prev_r, prev_t)

        if prev_fam is None:
            crt_fam = crt_fam.detach()
        else:
            crt_fam = self.SEFA(crt_fam.detach(), prev_fam.detach())
        fake_ab = self.Decoder(crt_fam, tr5, tr4, tr3, tr2)

        return fake_ab, crt_fam
    
    # load and print networks; create schedulers
    def setup(self, opt):
        if self.isTrain:
            self.optimizer = optim.Adam(self.parameters(), lr = opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
            
        self.criterion = HuberLoss(delta=1. / opt.ab_norm)
        
        if self.load_P:
            self.load_state_dict(torch.load(opt.P_path, map_location='cuda:'+str(opt.gpu_ids)).state_dict())
            print('loading Pnet sccess')
        
    def calc_loss(self, real, fake):
        self.fake = fake
        self.real = real
        self.real = encode_ab_ind(self.real[:, :, ::4, ::4], self.opt)[:, 0, :, :].long()
        self.fake = self.fake.float()
        loss = self.criterion(self.fake, self.real)

        return loss