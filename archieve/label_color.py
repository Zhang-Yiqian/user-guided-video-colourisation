#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 11:23:48 2020

@author: zhangyiqian
"""

from skimage import data, segmentation, color
from skimage.future import graph
from matplotlib import pyplot as plt
from skimage.io import imshow, imread


img = imread('test.jpg')
# img = data.coffee()

labels1 = segmentation.slic(img, compactness=20, n_segments=600)
out1 = color.label2rgb(labels1, img, kind='avg')

g = graph.rag_mean_color(img, labels1, mode='similarity')
labels2 = graph.cut_normalized(labels1, g)
out2 = color.label2rgb(labels2, img, kind='avg')

fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True, figsize=(6, 8))

ax[0].imshow(out1)
ax[1].imshow(out2)
ax[2].imshow(img)

for a in ax:
    a.axis('off')

plt.tight_layout()








