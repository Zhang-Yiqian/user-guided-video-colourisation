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
        self.model_I = Inet.Inet(opt)
        self.model_P = Pnet.Pnet(opt)
        self.opt = opt
        if opt.gpu_ids is not None:
            self.model_I.cuda()
            self.model_P.cuda()
            
    def setup(self):
        self.model_I.setup(self.opt)
        self.model_P.setup(self.opt)
                   
    def train(self):
        self.model_I.train()
        self.model_P.train()
        
    def eval(self):
        self.model_I.eval()
        self.model_P.eval()          

    def prop_forward(self, fake_ab, real_ab, target, end, crt_fam, prev_fam=None):
        fam = prev_fam
        for n in range(target+1, end+1):  #[1,2,...,N-1]
            print('[MODEL: propagation network] >>>>>>>>> {} to {}'.format(n-1, n))
            fake_ab[n, :, :, :], fam = self.model_P(gray[n, :, :, :], fake_ab[n, :, :, :], fake_ab[n-1, :, :, :], crt_fam, prev_fam)
            if self.model_P.training:
                loss = self.model_P.calc_loss(real_ab[n, :, :, :], fake_ab[n, :, :, :])
                model_P.optimizer.zero_grad() 
                loss.backward() 
                self.optimizer.step()  
        return fake_ab, fam

    def prop_backward(self, fake_ab, real_ab, target, end, crt_fam, prev_fam=None):
        fam = prev_fam
        for n in reversed(range(end, target)): #[N-2,N-3,...,0]
            print('[MODEL: propagation network] {} to {} <<<<<<<<<'.format(n+1, n))
            fake_ab[n, :, :, :], fam = self.model_P(gray[n, :, :, :], fake_ab[n, :, :, :], fake_ab[n+1, :, :, :], crt_fam, prev_fam)
            if self.model_P.training:
                loss = self.model_P.calc_loss(real_ab[n, :, :, :], fake_ab[n, :, :, :])
                model_P.optimizer.zero_grad() 
                loss.backward() 
                self.optimizer.step()  
        return fake_ab, fam
        
    def run_propagation(self, data, tr5, target):
        # determine the left and right end
        left_end, right_end = get_ends(data['marks'], target)
        fake_ab, fam = self.prop_forward(data, target, right_end)
        fake_ab, fam = self.prop_backward(data, target, left_end)

        print('[MODEL] Propagation finished.')
        
        return data, fam

    def run_interaction(self, gray, clicks, prev):
        clicks = clicks.unsqueeze(0)  # [1, 224, 224, 3]
        gray = gray.unsqueeze(0)    # [1, 224, 224, 1]
        print('[MODEL: interaction network] User Interaction')   
        
        return self.model_I(gray, clicks, prev) 

    def run_auto_colour(self, data):
        
        return self.model_I(data['gray'], data['clicks'], data['prev'])
        
        
        