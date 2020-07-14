#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 18:15:33 2020

@author: yiqian
"""

###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data
# from utils import rgb2lab
from PIL import Image
import os
import os.path
import numpy as np
from skimage.color import rgb2lab
from skimage.io import imread
from skimage.transform import resize
from utils.utils import random_crop, random_horizontal_flip
from joblib import Parallel, delayed
import torch
import random
import copy

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(root):
    videos = []
    for file in os.listdir(root):
        path = os.path.join(root, file)
        if (os.path.isdir(path)):
            videos.append(path)
        
    return videos


class ImageFolder(data.Dataset):

    def __init__(self, root, num_frames, transform=None):
        videos = make_dataset(root)
        if len(videos) == 0:
            raise(RuntimeError("Found 0 videos in: " + root + "\n"))
        
        self.root = root
        self.videos = videos
        self.transform = transform
        self.num_frames = num_frames

    def __getitem__(self, index):
    
        self.path = self.videos[index]
        try:
            start = np.random.randint(0, len(os.listdir(self.path)) - self.num_frames)
        except:
            start = 0
        
        idx_list = list(range(start, start+self.num_frames))
        l1 = copy.deepcopy(idx_list)
        l2 = copy.deepcopy(idx_list)
        random.shuffle(l1)
        random.shuffle(l2)
        
        with Parallel(n_jobs=6) as parallel:
            output1 = parallel(delayed(self.index_loader)(i) for i in l1)
        
        with Parallel(n_jobs=6) as parallel:
            output2 = parallel(delayed(self.index_loader)(i) for i in l2)
        
        return torch.stack(output1, axis=0), torch.stack(output2, axis=0)
        
    def __len__(self):
        return len(self.videos)
        
    def index_loader(self, n):
        nfile = "/%05d.jpg" % n
        
        return self.transform(Image.open(self.path+nfile).convert('RGB'))
    



        