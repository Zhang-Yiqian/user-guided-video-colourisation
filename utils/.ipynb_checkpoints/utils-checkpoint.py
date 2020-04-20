from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
from collections import OrderedDict
from IPython import embed
import random
from skimage.transform import rescale
import torch.nn.functional as F
from scipy import signal

def ToCuda(data_cpu):
    data = {}
    for key in data_cpu.keys():
        # data[key] = torch.from_numpy(data_np[key]).float().cuda()
        data[key] = data_cpu[key].float().cuda()
   
    return data

def ToCPU(data_cuda):
    data = {}
    for key in data_cpu.keys():
        data[key] = data_cuda[key].detach().float().cpu()
   
    return data

def random_crop(img, size=[224, 224]):
    assert img.shape[0] >= size[0]
    assert img.shape[1] >= size[1]
    x = random.randint(0, img.shape[0] - size[0])
    y = random.randint(0, img.shape[1] - size[1])
    img = img[x : x+size[0], y : y+size[1]]
    return img

def random_horizontal_flip(img, p=0.5):
    if random.random() > p:
        img = img[:, ::-1, :]
    return img
    
# Color conversion code
def rgb2xyz(rgb): # rgb from [0,1]
    # xyz_from_rgb = np.array([[0.412453, 0.357580, 0.180423],
        # [0.212671, 0.715160, 0.072169],
        # [0.019334, 0.119193, 0.950227]])

    mask = (rgb > .04045).type(torch.FloatTensor)
    if(rgb.is_cuda):
        mask = mask.cuda()

    rgb = (((rgb+.055)/1.055)**2.4)*mask + rgb/12.92*(1-mask)

    x = .412453*rgb[:,0,:,:]+.357580*rgb[:,1,:,:]+.180423*rgb[:,2,:,:]
    y = .212671*rgb[:,0,:,:]+.715160*rgb[:,1,:,:]+.072169*rgb[:,2,:,:]
    z = .019334*rgb[:,0,:,:]+.119193*rgb[:,1,:,:]+.950227*rgb[:,2,:,:]
    out = torch.cat((x[:,None,:,:],y[:,None,:,:],z[:,None,:,:]),dim=1)

    # if(torch.sum(torch.isnan(out))>0):
        # print('rgb2xyz')
        # embed()
    return out

def xyz2rgb(xyz):
    # array([[ 3.24048134, -1.53715152, -0.49853633],
    #        [-0.96925495,  1.87599   ,  0.04155593],
    #        [ 0.05564664, -0.20404134,  1.05731107]])

    r = 3.24048134*xyz[:,0,:,:]-1.53715152*xyz[:,1,:,:]-0.49853633*xyz[:,2,:,:]
    g = -0.96925495*xyz[:,0,:,:]+1.87599*xyz[:,1,:,:]+.04155593*xyz[:,2,:,:]
    b = .05564664*xyz[:,0,:,:]-.20404134*xyz[:,1,:,:]+1.05731107*xyz[:,2,:,:]

    rgb = torch.cat((r[:,None,:,:],g[:,None,:,:],b[:,None,:,:]),dim=1)
    rgb = torch.max(rgb,torch.zeros_like(rgb)) # sometimes reaches a small negative number, which causes NaNs

    mask = (rgb > .0031308).type(torch.FloatTensor)
    if(rgb.is_cuda):
        mask = mask.cuda()

    rgb = (1.055*(rgb**(1./2.4)) - 0.055)*mask + 12.92*rgb*(1-mask)

    # if(torch.sum(torch.isnan(rgb))>0):
        # print('xyz2rgb')
        # embed()
    return rgb

def xyz2lab(xyz):
    # 0.95047, 1., 1.08883 # white
    sc = torch.Tensor((0.95047, 1., 1.08883))[None,:,None,None]
    if(xyz.is_cuda):
        sc = sc.cuda()

    xyz_scale = xyz/sc

    mask = (xyz_scale > .008856).type(torch.FloatTensor)
    if(xyz_scale.is_cuda):
        mask = mask.cuda()

    xyz_int = xyz_scale**(1/3.)*mask + (7.787*xyz_scale + 16./116.)*(1-mask)

    L = 116.*xyz_int[:,1,:,:]-16.
    a = 500.*(xyz_int[:,0,:,:]-xyz_int[:,1,:,:])
    b = 200.*(xyz_int[:,1,:,:]-xyz_int[:,2,:,:])
    out = torch.cat((L[:,None,:,:],a[:,None,:,:],b[:,None,:,:]),dim=1)

    # if(torch.sum(torch.isnan(out))>0):
        # print('xyz2lab')
        # embed()

    return out

def lab2xyz(lab):
    y_int = (lab[:,0,:,:]+16.)/116.
    x_int = (lab[:,1,:,:]/500.) + y_int
    z_int = y_int - (lab[:,2,:,:]/200.)
    if(z_int.is_cuda):
        z_int = torch.max(torch.Tensor((0,)).cuda(), z_int)
    else:
        z_int = torch.max(torch.Tensor((0,)), z_int)

    out = torch.cat((x_int[:,None,:,:],y_int[:,None,:,:],z_int[:,None,:,:]),dim=1)
    mask = (out > .2068966).type(torch.FloatTensor)
    if(out.is_cuda):
        mask = mask.cuda()

    out = (out**3.)*mask + (out - 16./116.)/7.787*(1-mask)

    sc = torch.Tensor((0.95047, 1., 1.08883))[None,:,None,None]
    sc = sc.to(out.device)

    out = out*sc

    # if(torch.sum(torch.isnan(out))>0):
        # print('lab2xyz')
        # embed()

    return out

def rgb2lab(rgb, opt):
    lab = xyz2lab(rgb2xyz(rgb))
    l_rs = (lab[:,[0],:,:]-opt.l_cent)/opt.l_norm
    ab_rs = lab[:,1:,:,:]/opt.ab_norm
    out = torch.cat((l_rs,ab_rs),dim=1)
    # if(torch.sum(torch.isnan(out))>0):
        # print('rgb2lab')
        # embed()
    return out

def lab2rgb(lab_rs, opt):
    l = lab_rs[:,[0],:,:]*opt.l_norm + opt.l_cent
    ab = lab_rs[:,1:,:,:]*opt.ab_norm
    lab = torch.cat((l,ab),dim=1)
    out = xyz2rgb(lab2xyz(lab))
    # if(torch.sum(torch.isnan(out))>0):
        # print('lab2rgb')
        # embed()
    return out
    
def get_colorization_data(data_raw, opt, prev=None, ab_thresh=5., p=.125, num_points=None):
    data = {}
    data_lab = rgb2lab(data_raw, opt)
    data['gray'] = data_lab[:,[0,],:,:]
    data['ab'] = data_lab[:,1:,:,:]
    # data['prev'] = torch.zeros_like(data['ab'])
    if(ab_thresh > 0): # mask out grayscale images
        thresh = 1.*ab_thresh/opt.ab_norm
        mask = torch.sum(torch.abs(torch.max(torch.max(data['ab'],dim=3)[0],dim=2)[0]-torch.min(torch.min(data['ab'],dim=3)[0],dim=2)[0]),dim=1) >= thresh
        data['gray'] = data['gray'][mask,:,:,:]
        data['ab'] = data['ab'][mask,:,:,:]
        # print('Removed %i points'%torch.sum(mask==0).numpy())
        if(torch.sum(mask)==0):
            return None
    
    if prev is not None:
        data['prev'] = prev
        samp='l2'
        # data['prev'] = torch.zeros_like(data['ab'])
    else:
        data['prev'] = torch.zeros_like(data['ab'])
        N,C,H,W = data['ab'].shape
        data['clicks'] = torch.zeros([N, C+1, H, W])
        # data['clicks'] = torch.cat((torch.ones_like(data['gray'])- 0.5, data['ab']), dim=1)
        samp='normal'
        return data
    
    return add_color_patches_rand_gt(data, opt, prev, p=p, num_points=num_points, samp=samp)

def add_color_patches_rand_gt(data, opt, prev, p=.125, num_points=None, use_avg=True, samp='normal'):
# Add random color points sampled from ground truth based on:
#   Number of points
#   - if num_points is 0, then sample from geometric distribution, drawn from probability p
#   - if num_points > 0, then sample that number of points
#   Location of points
#   - if samp is 'normal', draw from N(0.5, 0.25) of image
#   - otherwise, draw from U[0, 1] of image
    
    N,C,H,W = data['ab'].shape
    data['hint_B'] = torch.zeros_like(data['ab'])
    data['mask_B'] = torch.zeros_like(data['gray'])
    if prev is not None:
        l2_dist = torch.sum(torch.pow(data['ab']-prev, 2), dim=1).detach().numpy()
        k = 1/opt.mean_kernel**2 * np.ones([opt.mean_kernel, opt.mean_kernel])
        l2_mean = np.zeros([l2_dist.shape[0], int(opt.fineSize/opt.mean_kernel), int(opt.fineSize/opt.mean_kernel)])
        for i in range(l2_dist.shape[0]):
            l2_mean[i, :, :] = signal.convolve(l2_dist[i,:,:], k, mode = "valid")[::opt.mean_kernel, ::opt.mean_kernel] # 7x7 mean 
    
    for nn in range(N):
        pp = 0
        cont_cond = True
        while cont_cond:
            if num_points is None: # draw from geometric
                # embed()
                cont_cond = np.random.rand() < (1-p)
            else: # add certain number of points
                cont_cond = pp < num_points
            if not cont_cond: # skip out of loop if condition not met
                continue

            P = np.random.choice(opt.sample_Ps) # patch size

            # sample location
            if samp == 'normal': # geometric distribution
                h = int(np.clip(np.random.normal( (H-P+1)/2., (H-P+1)/4.), 0, H-P))
                w = int(np.clip(np.random.normal( (W-P+1)/2., (W-P+1)/4.), 0, W-P))
            elif samp == 'uniform': # uniform distribution
                h = np.random.randint(H-P+1)
                w = np.random.randint(W-P+1)
            # sample location - L2 distance method
            else:
                area_h_, area_w_ = np.where(l2_mean[nn, :, :]==np.max(l2_mean[nn, :, :]))
                area_h, area_w = area_h_[0]*opt.mean_kernel, area_w_[0]*opt.mean_kernel
                # set to 0 in case of repeating
                l2_mean[nn, area_h_[0], area_w_[0]] = 0
                max_area = l2_dist[nn, area_h: area_h+opt.mean_kernel, area_w: area_w+opt.mean_kernel]
                h_, w_ = np.where(max_area == np.max(max_area))
                h, w = h_[0] + area_h, w_[0]+ area_w

            # add color point
            if use_avg:
                # embed()
                data['hint_B'][nn,:,h:h+P,w:w+P] = torch.mean(torch.mean(data['ab'][nn,:,h:h+P,w:w+P],dim=2,keepdim=True),dim=1,keepdim=True).view(1,C,1,1)
            else:
                data['hint_B'][nn,:,h:h+P,w:w+P] = data['ab'][nn,:,h:h+P,w:w+P]

            data['mask_B'][nn,:,h:h+P,w:w+P] = 1

            # increment counter
            pp+=1
    
    data['mask_B'] -= opt.mask_cent
    data['clicks'] = torch.cat((data['mask_B'], data['hint_B']),dim=1)

    return data

def save_model(model, opt, epoch, model_index, psnr):
    file_name = opt.save_root+opt.phase+'_'+'ep'+str(epoch)+'_'+'val'+'_'+str(model_index)+'_'+str(psnr)+'.pkl'
    if os.path.exists(file_name):
        file_name = 'backup_' + file_name
        warnings.warn("The model file already exits!", Warning)
    if opt.gpu_ids is not None:
        model.cpu()
        torch.save(model, file_name)
        model.cuda()
    else:
        torch.save(model, file_name)

def encode_ab_ind(data_ab, opt):
    # Encode ab value into an index
    # INPUTS
    #   data_ab   Nx2xHxW \in [-1,1]
    # OUTPUTS
    #   data_q    Nx1xHxW \in [0,Q)

    data_ab_rs = torch.round((data_ab*opt.ab_norm + opt.ab_max)/opt.ab_quant) # normalized bin number
    data_q = data_ab_rs[:,[0],:,:]*opt.A + data_ab_rs[:,[1],:,:]
    return data_q

def decode_ind_ab(data_q, opt):
    # Decode index into ab value
    # INPUTS
    #   data_q      Nx1xHxW \in [0,Q)
    # OUTPUTS
    #   data_ab     Nx2xHxW \in [-1,1]

    data_a = data_q/opt.A
    data_b = data_q - data_a*opt.A
    data_ab = torch.cat((data_a,data_b),dim=1)

    if(data_q.is_cuda):
        type_out = torch.cuda.FloatTensor
    else:
        type_out = torch.FloatTensor
    data_ab = ((data_ab.type(type_out)*opt.ab_quant) - opt.ab_max)/opt.ab_norm

    return data_ab

def decode_max_ab(data_ab_quant, opt):
    # Decode probability distribution by using bin with highest probability
    # INPUTS
    #   data_ab_quant   NxQxHxW \in [0,1]
    # OUTPUTS
    #   data_ab         Nx2xHxW \in [-1,1]

    data_q = torch.argmax(data_ab_quant,dim=1)[:,None,:,:]
    return decode_ind_ab(data_q, opt)

def decode_mean(data_ab_quant, opt):
    # Decode probability distribution by taking mean over all bins
    # INPUTS
    #   data_ab_quant   NxQxHxW \in [0,1]
    # OUTPUTS
    #   data_ab_inf     Nx2xHxW \in [-1,1]

    (N,Q,H,W) = data_ab_quant.shape
    a_range = torch.range(-opt.ab_max, opt.ab_max, step=opt.ab_quant).to(data_ab_quant.device)[None,:,None,None]
    a_range = a_range.type(data_ab_quant.type())

    # reshape to AB space
    data_ab_quant = data_ab_quant.view((N,int(opt.A),int(opt.A),H,W))
    data_a_total = torch.sum(data_ab_quant,dim=2)
    data_b_total = torch.sum(data_ab_quant,dim=1)

    # matrix multiply
    data_a_inf = torch.sum(data_a_total * a_range,dim=1,keepdim=True)
    data_b_inf = torch.sum(data_b_total * a_range,dim=1,keepdim=True)

    data_ab_inf = torch.cat((data_a_inf,data_b_inf),dim=1)/opt.ab_norm

    return data_ab_inf

def calc_batch_psnr(lightness, real_ab, fake_ab, opt, avg=True):
    psnr = 0
    if not opt.is_regression:
        fake_ab = decode_max_ab(fake_ab, opt)
        fake_ab = F.interpolate(fake_ab, scale_factor=4)
    lightness = lightness.cpu()
    fake_ab = fake_ab.cpu()
    real_ab = real_ab.cpu()
    fake_img = torch.cat((lightness, fake_ab), 1) 
    real_img = torch.cat((lightness, real_ab), 1) 
    fake_rgb = lab2rgb(fake_img, opt)
    real_rgb = lab2rgb(real_img, opt)
    fake_rgb[fake_rgb > 1] = 1.0
    for idx in range(lightness.shape[0]):
        # print(lightness[idx,:,:,:].shape, fake_ab.shape)
        # fake_img = torch.cat((lightness[idx,:,:,:], fake_ab[idx,:,:,:]), 0) 
        # real_img = torch.cat((lightness[idx,:,:,:], real_ab[idx,:,:,:]), 0) 
        # fake_rgb = lab2rgb(torch.unsqueeze(fake_img, 0), opt)[0, :, :, :]
        # real_rgb = lab2rgb(torch.unsqueeze(real_img, 0), opt)[0, :, :, :]
        
        mse = torch.mean( (fake_rgb[idx, :, :, :] - real_rgb[idx, :, :, :]) ** 2 )
        psnr += 10 * torch.log10(1.0 / mse)  
        
    if avg:
        return (psnr / lightness.shape[0]).cpu().numpy()
    else:
        return psnr.cpu().numpy()