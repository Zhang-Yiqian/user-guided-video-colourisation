from __future__ import division
import torch
from torch import optim
# general libs
import numpy as np
import copy

# my libs
from utils import ToCudaVariable, ToCudaPN, Dilate_mask, load_UnDP, Get_weight, overlay_davis, overlay_checker, overlay_color, overlay_fade
from interaction_net import Inet
from propagation_net import Pnet



class model():
    def __init__(self, frames, frames_gray):
        self.model_I = Inet()
        self.model_P = Pnet()
        if torch.cuda.is_available():
            print('Using GPU')
            # self.model_I = nn.DataParallel(self.model_I)
            # self.model_P = nn.DataParallel(self.model_P)
            # self.model_I.cuda()
            # self.model_P.cuda()
            # self.model_I.load_state_dict(torch.load('models/I_e290.pth'))
            # self.model_P.load_state_dict(torch.load('models/P_e290.pth'))
        else:
            print('Using CPU')
            # self.model_I.load_state_dict(load_UnDP('models/I_e290.pth'))
            # self.model_P.load_state_dict(load_UnDP('models/P_e290.pth'))
        self.model_I.train()  # turn-on BN
        self.model_P.train()  # turn-on BN
        self.optimizer_i = optim.Adam(filter(lambda p: p.requires_grad, self.model_I.parameters()))
        self.optimizer_p = optim.Adam(filter(lambda p: p.requires_grad, self.model_P.parameters()))
        
        self.frames_gray = copy.deepcopy(frames_gray)
        self.res = copy.deepcopy(frames_gray)
        self.frames = copy.deepcopy(frames)
        self.num_frames, self.height, self.width = self.frames.shape[:3]
        self.res = torch.zeros(self.num_frames, self.height, self.width, 2)  # n,h,w,2 
        # self.init_variables(self.frames)
        
    def init_variables(self, frames):
        # res need to be initializa
        self.all_F = torch.unsqueeze(torch.from_numpy(np.transpose(frames, (3, 0, 1, 2))).float() / 255., dim=0) # 1,3,t,h,w
        self.all_E = torch.zeros(1, self.num_frames, self.height, self.width)  # 1,t,h,w
        self.prev_E = torch.zeros(1, self.num_frames, self.height, self.width)  # 1,t,h,w
        self.dummy_M = torch.zeros(1, self.height, self.width).long()
        # to cuda
        self.all_F, self.all_E, self.prev_E, self.dummy_M = ToCudaVariable([self.all_F, self.all_E, self.prev_E, self.dummy_M], volatile=True)
        
        self.ref = None
        self.a_ref = None
        self.next_a_ref = None
        self.prev_targets = []

    def prop_forward(self, target, end):
        for n in range(target+1, end+1):  #[1,2,...,N-1]
            print('[MODEL: propagation network] >>>>>>>>> {} to {}'.format(n-1, n))
            self.res[n, :, :, :], loss = self.model_P(self.frames_gray[n, :, :, :], self.res[n, 1:, :, :], self.res[n-1, 1:, :, :], self.frames[n, 1:, :, :])
            self.optimizer_p.zero_grad()
            loss.backward()
            self.optimizer_p.step()

    def prop_backward(self, target, end):
        for n in reversed(range(end, target)): #[N-2,N-3,...,0]
            print('[MODEL: propagation network] {} to {} <<<<<<<<<'.format(n+1, n))
            self.res[n, :, :, :], loss = self.model_P(self.frames_gray[n, :, :], self.res[n, :, :, :], self.res[n+1, :, :, :])
            self.optimizer_p.zero_grad() 
            loss.backward()  
            self.optimizer_p.step()  
           
    def run_propagation(self, target, mode='linear', at_least=-1, std=None):
        # when new round begins
        
        if mode == 'naive':
            left_end, right_end, weight = 0, self.num_frames-1, self.num_frames*[1.0]
        elif mode == 'linear':
            left_end, right_end, weight = Get_weight(target, self.prev_targets, self.num_frames, at_least=at_least)
        else:
            raise NotImplementedError

        self.prop_forward(target, right_end)
        self.prop_backward(target, left_end)

        for f in range(self.num_frames):
            self.all_E[:, :, f] = weight[f] * self.all_E[:,:,f] + (1-weight[f]) * self.prev_E[:,:,f]

        self.prev_targets.append(target)
        print('[MODEL] Propagation finished.')    

    def run_interaction(self, scribbles):
        
        # convert davis scribbles to torch
        target = scribbles['annotated_frame']
        scribble_mask = scribbles2mask(scribbles, (self.height, self.width))[target]
        scribble_mask = Dilate_mask(scribble_mask, 1)
        self.tar_P, self.tar_N = ToCudaPN(scribble_mask)

        self.all_E[:,target], _, self.ref = self.model_I(self.all_F[:,:,target], self.all_E[:,target], self.tar_P, self.tar_N, self.dummy_M, [1,0,0,0,0]) # [batch, 256,512,2]

        print('[MODEL: interaction network] User Interaction on {}'.format(target))    
