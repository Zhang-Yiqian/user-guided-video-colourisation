#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 17:26:29 2020

@author: zhangyiqian
"""
import numpy as np       
from sklearn.cluster import MeanShift, estimate_bandwidth
from skimage.color import rgb2lab
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage import data, segmentation, color

#Loading original image
originImg = imread('test.jpg')
image = rgb2lab(originImg)

# Shape of original image    
originShape = image.shape

# Converting image into array of dimension [nb of pixels in originImage, 3]
# based on r g b intensities    
flatImg=np.reshape(image, [-1, 3])

# Estimate bandwidth for meanshift algorithm    
bandwidth = estimate_bandwidth(flatImg, quantile=0.1, n_samples=200)    
ms = MeanShift(bandwidth = bandwidth, bin_seeding=True)

# Performing meanshift on flatImg    
ms.fit(flatImg)

# (r,g,b) vectors corresponding to the different clusters after meanshift    
labels=ms.labels_

# Remaining colors after meanshift    
cluster_centers = ms.cluster_centers_    

# Finding and diplaying the number of clusters    
labels_unique = np.unique(labels)    
n_clusters_ = len(labels_unique)    
print("number of estimated clusters : %d" % n_clusters_)    

segmentedImg = cluster_centers[np.reshape(labels, originShape[:2])]
out2 = color.label2rgb(segmentedImg, originImg, kind='avg')

#plt.figure(dpi=200)
#plt.subplot(121)
#plt.imshow(originImg), plt.axis('off'),plt.title('original image')
#plt.subplot(122)
#plt.imshow(np.reshape(labels, image.shape[:2])), plt.axis('off'), plt.title('segmented image with Meanshift')
#plt.show()

#  show each segmentation result
label = 1
seg_img = (labels == label) * 1.0
seg_img = seg_img.reshape(image.shape[:2])
# sum((labels == label) * 1.0)
plt.figure()
imshow(seg_img)
plt.figure()
skeleton = skeletonize(seg_img, method='lee')
imshow(skeleton)
plt.figure()
imshow(out2)














 

