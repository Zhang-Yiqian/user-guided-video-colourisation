from __future__ import division
import torch
from torch import optim
# general libs
import numpy as np
import copy

# my libs
from utils.utils import *
import models.interaction_net as Inet
import models.propagation_net as Pnet


class model():
    def __init__(self, opt):
        self.Inet = Inet.Inet(opt)
        self.Pnet = Pnet.Pnet(opt)
        self.opt = opt
        if opt.gpu_ids is not None:
            self.Inet.cuda()
            self.Pnet.cuda()
            
    def setup(self):
        self.Inet.setup(self.opt)
        self.Pnet.setup(self.opt)
                   
    def train(self):
        self.Inet.train()
        self.Pnet.train()
        
    def val(self):
        self.Inet.eval()
        self.Pnet.eval()          

    def prop_forward(self, gray, fake_ab, real_ab, target, end, crt_fam, prev_fam=None):
        fam = prev_fam
        temp_fake = torch.zeros_like(fake_ab)
        for n in range(target+1, end+1):  #[1,2,...,N-1]
            print('[MODEL: propagation network] >>>>>>>>> {} to {}'.format(n-1, n))
            temp_fake[n, :, :, :], fam = self.Pnet(gray[n, :, :, :], fake_ab[n, :, :, :], fake_ab[n-1, :, :, :], crt_fam, prev_fam)
            if self.Pnet.training:
                loss = self.Pnet.calc_loss(real_ab[n, :, :, :], temp_fake[n, :, :, :])
                self.Pnet.optimizer.zero_grad() 
                loss.backward(retain_graph=True) 
                self.Pnet.optimizer.step() 
                self.total_loss += loss.detach().cpu().numpy()
        return temp_fake, fam

    def prop_backward(self, gray, fake_ab, real_ab, target, end, crt_fam, prev_fam=None):
        fam = prev_fam
        temp_fake = torch.zeros_like(fake_ab)
        for n in reversed(range(end, target)): #[N-2,N-3,...,0]
            print('[MODEL: propagation network] {} to {} <<<<<<<<<'.format(n+1, n))
            temp_fake[n, :, :, :], fam = self.Pnet(gray[n, :, :, :], fake_ab[n, :, :, :], fake_ab[n+1, :, :, :], crt_fam, prev_fam)
            if self.Pnet.training:
                loss = self.Pnet.calc_loss(real_ab[n, :, :, :], temp_fake[n, :, :, :])
                self.Pnet.optimizer.zero_grad() 
                loss.backward(retain_graph=True) 
                self.Pnet.optimizer.step()  
                self.total_loss += loss.detach().cpu().numpy()
        return temp_fake, fam
        
    def run_propagation(self, data, target, crt_fam, prev_fam=None):
        # determine the left and right end
        self.total_loss = 0
        left_end, right_end = get_ends(data['marks'], target)
        fake_ab, fam = self.prop_forward(data['gray'], data['prev'], data['ab'], target, right_end, crt_fam, prev_fam)
        fake_ab, fam = self.prop_backward(data['gray'], data['prev'], data['ab'], target, left_end, crt_fam, prev_fam)
        
        print('[MODEL] Propagation finished.')
        
        return fake_ab, fam

    def run_interaction(self, gray, clicks, prev, target=None):
        clicks = clicks.unsqueeze(0)  # [1, 224, 224, 3]
        gray = gray.unsqueeze(0)    # [1, 224, 224, 1]
        prev = prev.unsqueeze(0)    # [1, 224, 224, 2]
        print('[MODEL: interaction network] User Interaction on ' + str(target))   
        
        return self.Inet(gray, clicks, prev) 

    def run_auto_colour(self, data):
        
        return self.Inet(data['gray'], data['clicks'], data['prev'])
        
        
        