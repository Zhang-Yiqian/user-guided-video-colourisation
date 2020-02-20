#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 16:43:33 2020

@author: zhangyiqian
"""

import cv2 as cv
import numpy as np
from skimage.io import imread, imshow

src = imread('test.jpg')

h, w = src.shape[:2]
dst = cv.pyrMeanShiftFiltering(src, 25, 30, termcrit=(cv.TERM_CRITERIA_MAX_ITER+cv.TERM_CRITERIA_EPS, 5, 1))
result = np.hstack((src,dst))
imshow(result)





