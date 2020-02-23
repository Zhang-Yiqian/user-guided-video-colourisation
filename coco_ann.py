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
import pylab
from skimage.morphology import skeletonize

annFile = '/home/yiqian/Documents/dataset/COCO/annotations_valstuff/stuff_val2017.json'
coco=COCO(annFile)
imgIds = coco.getImgIds()
img = coco.loadImgs(imgIds[0])[0]
I = imread(img['coco_url'])       
plt.figure()
imshow(I)
catIds = coco.getCatIds(catNms=['person']);
# annIds = coco.getAnnIds(imgIds=img['id'], catIds=coco.getCatIds(), iscrowd=False)
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=False)
anns = coco.loadAnns(annIds)
plt.figure()
coco.showAnns(anns)
scribbles = np.zeros([I.shape[0], I.shape[1]])
for i in range(len(anns)):
    mask = coco.annToMask(anns[i]) * 1.0
    scribbles += skeletonize(mask)
plt.figure()
imshow(scribbles)












