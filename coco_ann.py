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
import pylab
from skimage.morphology import skeletonize, thin, medial_axis
from skimage import morphology
from skimage import filters
import skimage.filters.rank as sfr
from skimage.morphology import disk

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
sample_number = 4
area_list = np.random.permutation(len(anns)+1)[0 : sample_number+1]
kernel = morphology.square(5)
scribbles = np.zeros([I.shape[0], I.shape[1]])
plt.figure()
for i in range(len(anns)):
    mask = coco.annToMask(anns[i])
    skel = skeletonize(mask) * 1.0
    dst = sfr.gradient(skel, disk(5))
    scribbles += (dst > 0.5) * 1.0
    # scribbles += skel
    # scribbles += morphology.opening(skel, kernel) 

plt.figure()
imshow((scribbles> 0.5) * 1.0)












