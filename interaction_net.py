from __future__ import division
import torch
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# general libs
import numpy as np
import math
from utils import ToCudaVariable

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
        self.conv1 = resnet.conv1
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

        ###############################
        ############ rewrite ##########
        ###############################
        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        ###############################
        ############ rewrite ##########
        ###############################

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


class Inet(nn.Module):
    def __init__(self):
        super(Inet, self).__init__()
        mdim = 256
        self.Encoder = Encoder()      # inputs: ref: rf, rm / tar: tf, tm
        self.Decoder = Decoder(mdim)  # input: m5, r4, r3, r2 >> p
        self.cnt = 0

    def is_there_scribble(self, p, n):
        num_pixel_p = np.sum(p.data.cpu().numpy(), axis=(1,2))
        num_pixel_n = np.sum(n.data.cpu().numpy(), axis=(1,2))
        num_pixel = num_pixel_p + num_pixel_n
        yes = (num_pixel > 0).astype(np.float32)
        mulplier = 1 / (np.mean(yes) + 0.001)
        yes = yes * mulplier
        return ToCudaVariable([torch.from_numpy(yes.copy()).float()])[0]

    def forward(self, in_frame, in_strokes, in_prev):
        tr5, tr4, tr3, tr2 = self.Encoder(in_frame, in_strokes, in_prev)
        em_ab = self.Decoder(tr5, tr4, tr3, tr2)

        return em_ab

    def forward1(self, tf, tm, tp, tn, gm, loss_weight):  # b,c,h,w // b,4 (y,x,h,w)
        if tm is None:
            # tm = ToCudaVariable([0.5*torch.ones(gm.size())], requires_grad=False)[0]
            pass

        # run Siamese Encoder
        tr5, tr4, tr3, tr2 = self.Encoder(tf_roi, tm_roi, tp_roi, tn_roi)
        em_roi = self.Decoder(tr5, tr4, tr3, tr2)

        # Losses are computed within ROI
        # CE loss
        gm_roi = F.grid_sample(torch.unsqueeze(gm, dim=1).float(), fw_grid)[:,0]
        gm_roi = gm_roi.detach()
        # CE loss
        CE = nn.CrossEntropyLoss(reduce=False)
        batch_CE = ToCudaVariable([torch.zeros(gm_roi.size()[0])])[0] # batch sized loss container 
        sizes=[(256,256), (64,64), (32,32), (16,16), (8,8)]
        for s in range(5):
            if s == 0:
                CE_s = CE(em_roi[s], torch.round(gm_roi).long()).mean(-1).mean(-1) # mean over h,w
                batch_CE += loss_weight[s] * CE_s
            else:
                if loss_weight[s]:
                    gm_roi_s = torch.round(F.upsample(torch.unsqueeze(gm_roi, dim=1), size=sizes[s], mode='bilinear')[:,0]).long()
                    CE_s = CE(em_roi[s], gm_roi_s).mean(-1).mean(-1) # mean over h,w
                    batch_CE += loss_weight[s] * CE_s

        batch_CE = batch_CE * self.is_there_scribble(tp, tn)


        # get final output via inverse warping
        em = F.grid_sample(F.softmax(em_roi[0], dim=1), bw_grid)[:,1]
        # return em, batch_CE, [tr5, tr4, tr3, tr2]
        return em, batch_CE, tr5