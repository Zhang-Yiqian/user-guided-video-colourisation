#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 17:42:01 2020

@author: zhangyiqian
"""

import os
import shutil
import random
src = "/home/zhangy23/dataset/coco/val/" 
dst = "/home/zhangy23/dataset/coco/subval/"
train_dst = dst + "train/"
val_dst = dst + "val/"
test_dst = dst + "test/"
train = []
val = []
test = []

files= os.listdir(src) 
random.shuffle(files)
train = files[0 : int(0.7*len(files))]
val = files[int(0.7*len(files)) : int(0.8*len(files))]
test = files[int(0.8*len(files)) : ]

for f in train:
    shutil.copyfile(src+f, train_dst+f)
    
for f in test:
    shutil.copyfile(src+f, test_dst+f)

for f in val:
    shutil.copyfile(src+f, val_dst+f)
    
    
    
    
    
    
    
    
    
    
    
    