#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 16:58:58 2020

@author: zhangyiqian
"""
from pycocotools.coco import COCO
import numpy as np
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
import pylab

annFile = '~/Documents/dataset/COCO/annotations_trainval/instances_val2014.json'
coco=COCO(annFile)
imgIds = coco.getImgIds()
img = coco.loadImgs(imgIds[0])[0]
I = imread(img['coco_url'])       
imshow(I)
annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=True)
anns = coco.loadAnns(annIds)
coco.showAnns(anns)








