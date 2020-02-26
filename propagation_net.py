from __future__ import division
import torch

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# general libs
import math

print('Propagation Network: initialising')

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # gray scale
        self.conv1_gray = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=True)
        # previous round frame
        self.conv1_prev_r = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=True)
        # previous time frame
        self.conv1_prev_t = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=True)

        # initialisation methods in Oh's paper
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
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

        # freeze BNs
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                for p in m.parameters():
                    p.requires_grad = False

    def forward(self, in_gray, in_frame_r, in_frame_t):
        # f = (in_gray - Variable(self.mean)) / Variable(self.std)
        f = torch.unsqueeze(in_gray, dim=0).float()  # 1*c*h*w
        r = torch.unsqueeze(in_frame_r, dim=0).float()
        t = torch.unsqueeze(in_frame_t, dim=0)

        x = self.conv1_gray(f) + self.conv1_prev_r(r) + self.conv1_prev_t(t)
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
        m = s + F.upsample(pm, scale_factor=self.scale_factor, mode='bilinear')
        m = self.ResMM(m)
        return m

class Decoder(nn.Module):
    def __init__(self, mdim):
        super(Decoder, self).__init__()
        self.ResFM1 = ResBlock(4096, 1024)
        self.ResFM2= ResBlock(1024, mdim)
        self.RF4 = Refine(1024, mdim) # 1/16 -> 1/8
        self.RF3 = Refine(512, mdim) # 1/8 -> 1/4
        self.RF2 = Refine(256, mdim) # 1/4 -> 1

        self.pred5 = nn.Conv2d(mdim, 2, kernel_size=(3,3), padding=(1,1), stride=1)
        self.pred4 = nn.Conv2d(mdim, 2, kernel_size=(3,3), padding=(1,1), stride=1)
        self.pred3 = nn.Conv2d(mdim, 2, kernel_size=(3,3), padding=(1,1), stride=1)
        self.pred2 = nn.Conv2d(mdim, 2, kernel_size=(3,3), padding=(1,1), stride=1)

    def forward(self, rr5, r5, r4, r3, r2):
        m5 = torch.cat([rr5, r5], dim=1)
        m5 = self.ResFM1(m5)   
        m5 = self.ResFM2(m5)   
        m4 = self.RF4(r4, m5)  # out: 1/16, 256
        m3 = self.RF3(r3, m4)  # out: 1/8, 256
        m2 = self.RF2(r2, m3)  # out: 1/4, 256

        p2 = self.pred2(F.relu(m2))
        p3 = self.pred3(F.relu(m3))
        p4 = self.pred4(F.relu(m4))
        p5 = self.pred5(F.relu(m5))
        
        p = F.upsample(p2, scale_factor=4, mode='bilinear')
        
        return p, p2, p3, p4, p5


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


class Pnet(nn.Module):
    def __init__(self):
        super(Pnet, self).__init__()
        mdim = 256
        self.Encoder = Encoder() # inputs:: ref: rf, rm / tar: tf, tm 
        self.SEFA = SEFA(2048, r=2)
        self.Decoder = Decoder(mdim) # input: m5, r4, r3, r2 >> p
        self.cnt = 0

    def forward(self, in_gray, in_frame_r, in_frame_t, gt, fam_ref=None):
        tr5, tr4, tr3, tr2 = self.Encoder(in_gray, in_frame_r, in_frame_t)

        # c_ref is tr5
        if fam_ref is None:
            a_ref = tr5.detach()
        else:
            a_ref = self.SEFA(tr5.detach(), fam_ref.detach())
        em_ab = self.Decoder(a_ref, tr5, tr4, tr3, tr2)
        # how to estimate the pixel value ---- softmax? or .....

        CE = nn.CrossEntropyLoss(reduce=False)
        loss = CE(em_ab, gt)
        return em_ab, loss

    # def forward1(self, c_ref, p_ref, tf, tm, tx, gm, loss_weight):  # b,c,h,w // b,4 (y,x,h,w)
    #     # if first target frame (no tb)
    #     if tm is None:
    #         tm = ToCudaVariable([0.5*torch.ones(gm.size())], requires_grad=False)[0]
    #     tb = self.masks2yxhw(tm, tx, scale=1.5)

    #     oh, ow = tf.size()[2], tf.size()[3] # original size
    #     fw_grid, bw_grid, theta = self.get_ROI_grid(tb, src_size=(oh, ow), dst_size=(256,256), scale=1.0)

    #     #  Sample target frame
    #     tf_roi = F.grid_sample(tf, fw_grid)
    #     tm_roi = F.grid_sample(torch.unsqueeze(tm, dim=1).float(), fw_grid)[:,0]
    #     tx_roi = F.grid_sample(torch.unsqueeze(tx, dim=1).float(), fw_grid)[:,0]

    #     # run Siamese Encoder
    #     tr5, tr4, tr3, tr2 = self.Encoder(tf_roi, tm_roi, tx_roi)
    #     if p_ref is None:
    #         a_ref = c_ref.detach()
    #     else:
    #         a_ref = self.SEFA(c_ref.detach(), p_ref.detach())
    #     em_roi = self.Decoder(a_ref, tr5, tr4, tr3, tr2)

    #     # Losses are computed within ROI
    #     # CE loss
    #     gm_roi = F.grid_sample(torch.unsqueeze(gm, dim=1).float(), fw_grid)[:,0]
    #     gm_roi = gm_roi.detach()
    #     # CE loss
    #     CE = nn.CrossEntropyLoss(reduce=False)
    #     batch_CE = ToCudaVariable([torch.zeros(gm_roi.size()[0])])[0] # batch sized loss container 
    #     sizes=[(256,256), (64,64), (32,32), (16,16), (8,8)]
    #     for s in range(5):
    #         if s == 0:
    #             CE_s = CE(em_roi[s], torch.round(gm_roi).long()).mean(-1).mean(-1) # mean over h,w
    #             batch_CE += loss_weight[s] * CE_s
    #         else:
    #             if loss_weight[s]:
    #                 gm_roi_s = torch.round(F.upsample(torch.unsqueeze(gm_roi, dim=1), size=sizes[s], mode='bilinear')[:,0]).long()
    #                 CE_s = CE(em_roi[s], gm_roi_s).mean(-1).mean(-1) # mean over h,w
    #                 batch_CE += loss_weight[s] * CE_s

    #     # get final output via inverse warping
    #     em = F.grid_sample(F.softmax(em_roi[0], dim=1), bw_grid)[:,1]
    #     return em, batch_CE, a_ref