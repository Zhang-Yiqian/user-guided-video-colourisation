#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 16:58:58 2020
@author: zhangyiqian
"""
from cocoapi.PythonAPI.pycocotools.coco import COCO
import numpy as np
from skimage.io import imread, imshow, imsave
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, thin, medial_axis
from skimage import morphology
from skimage import draw


annFile = '/home/yiqian/Documents/dataset/COCO/annotations_valstuff/stuff_val2017.json'
coco=COCO(annFile)
imgIds = coco.getImgIds()
img = coco.loadImgs(imgIds[0])[0]
I = imread(img['coco_url'])       
# plt.figure()
# imshow(I)
catIds = coco.getCatIds(catNms=['person']);
# annIds = coco.getAnnIds(imgIds=img['id'], catIds=coco.getCatIds(), iscrowd=False)
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=False)
anns = coco.loadAnns(annIds)
# plt.figure()
# coco.showAnns(anns)

# sample scribble areas based on uniform distribution
# sample_number = np.random.randint(1, len(anns)+1)
sample_area_num = 4

area_list = np.random.permutation(len(anns)+1)[0 : sample_area_num+1]
kernel = morphology.square(5)
scribbles = np.zeros([I.shape[0], I.shape[1]])
plt.figure()
for i in range(len(anns)):
    mask = coco.annToMask(anns[i])
    # 1-D random sample
    flat_mask = mask.reshape([1, -1])
    index = np.arange(0, len(flat_mask))
    # keep those pixels that are in the target area
    index_mask = np.multiply(flat_mask, index)
    index_mask = index_mask[index_mask != 0]
    l = np.random.choice(index_mask, 3)
    # convert to 2D coordinates
    X = np.divide(l, I.shape[1]) + 1
    Y = np.remainder(l, I.shape[1])
    # plot bezier curve
    rr, cc = draw.bezier_curve(X[0], Y[0], X[1], Y[1], X[2], Y[2], weight=0.5, \
                      shape=(I.shape[0], I.shape[1]))
    scribbles[rr, cc] = 1
    
plt.figure()
imshow(scribbles)