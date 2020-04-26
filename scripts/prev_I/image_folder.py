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

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


# def default_loader(path, opt):
#   img = imread(path) # / 255
#    img = resize(img, [opt.fineSize, opt.fineSize])
#    img = random_horizontal_flip(img)
#    return rgb2lab(img) #/ opt.norm_factor

def default_loader(path):
    return Image.open(path).convert('RGB')

class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
       
        return img

    def __len__(self):
        return len(self.imgs)




        