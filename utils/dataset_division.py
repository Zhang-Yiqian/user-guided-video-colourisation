#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 17:42:01 2020

@author: zhangyiqian
"""

import os
import shutil
import random
src = "/home/zhangy23/dataset/coco/" 
dst = "/home/zhangy23/dataset/coco/"
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
    
import os
def make_dataset(root):
    videos = []
    for file in os.listdir(root):
        path = os.path.join(root, file)
        if (os.path.isdir(path)):
            videos.append(path)

    return videos


root = '/Users/zhangyiqian/Downloads/valid/JPEGImages/'
videos = make_dataset(root)

for v in videos:
    i = 0
    frames = os.listdir(v)
    for n in range(100):
        nfile = "/%05d.jpg" % (n*5)
        old_name = v + '/' + nfile
        if os.path.exists(old_name):
            #print('yes')
            new_name = v + '/' + "/%05d.jpg" % i
            os.rename(old_name, new_name)
            i += 1 
            
            
            
    
    
    
    
    
    
    
    
    
    
    
    
    