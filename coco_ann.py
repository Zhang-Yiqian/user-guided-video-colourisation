#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 16:58:58 2020
@author: zhangyiqian
"""
from cocoapi.PythonAPI.pycocotools.coco import COCO
import numpy as np
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
from skimage import draw
import random


annFile = '/home/yiqian/Documents/dataset/COCO/annotations_valstuff/stuff_val2017.json'
coco=COCO(annFile)
imgIds = coco.getImgIds()
img = coco.loadImgs(imgIds[0])[0]
I = imread(img['coco_url'])       
plt.figure()
imshow(I)
# catIds = coco.getCatIds(catNms=['person']);
annIds = coco.getAnnIds(imgIds=img['id'], catIds=coco.getCatIds(), iscrowd=False)
# annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=False)
anns = coco.loadAnns(annIds)

coco.showAnns(anns)

# sample scribble areas based on uniform distribution
# sample_area_num = np.random.randint(1, len(anns)+1)
sample_area_num = 10

area_list = np.random.permutation(len(anns))[0 : sample_area_num]
scribbles = np.zeros([I.shape[0], I.shape[1]])

for i in area_list:
    area_scribbles = np.zeros([I.shape[0], I.shape[1]])
    mask = coco.annToMask(anns[i])
    
    # 1-D random sample 
    flat_mask = mask.reshape([1, -1])
    index = np.arange(flat_mask.size)
    
    # keep those pixels that are in the target area 
    index_mask = np.multiply(flat_mask, index)
    index_mask = index_mask[index_mask != 0]
    l = np.random.choice(index_mask, 3)
    
    # convert to 2D coordinates 
    X = np.floor_divide(l, I.shape[1]) + 1
    Y = np.remainder(l, I.shape[1])
    
    # fit bezier curve 
    rr, cc = draw.bezier_curve(X[0], Y[0], X[1], Y[1], X[2], Y[2], \
                    weight=random.uniform(0, 5),shape=(I.shape[0], I.shape[1]))
    area_scribbles[rr, cc] = 1    
    area_scribbles = np.multiply(area_scribbles, mask)
    
    scribbles = np.add(scribbles, area_scribbles)

    # draw.set_color(mask, [rr, cc], 1)
    # r, c = draw.circle(X[0], Y[0], 5)
    # draw.set_color(mask, [r, c], 1)
    # r, c = draw.circle(X[1], Y[1], 5)
    # draw.set_color(mask, [r, c], 1)
    # r, c = draw.circle(X[2], Y[2], 5)
    # draw.set_color(mask, [r, c], 1)
    # i  = np.multiply(I, (1 - scribbles)) + scribbles

plt.figure()
imshow(scribbles)






