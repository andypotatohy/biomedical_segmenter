#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 18:04:53 2019

@author: deeperthought
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

nii = nib.load('/home/deeperthought/Projects/Data/T1_left_post.nii')
img = nii.get_data()
img.shape

img = resize(image=img, order=1, 
                     output_shape=(58, 512, 512), 
                     anti_aliasing=True, preserve_range=True )
context = 66
detail = 17
myslice = 30
point = 200


img[myslice, point-context/2:(point+context/2+1), point-context/2:(point+context/2+1)]  = img[myslice, point-context/2:(point+context/2+1), point-context/2:(point+context/2+1)]  + 1000
img[myslice, point-detail/2:(point+detail/2+1), point-detail/2:(point+detail/2+1)] = img[myslice, point-detail/2:(point+detail/2+1), point-detail/2:(point+detail/2+1)] + 1000


plt.imshow(img[myslice])

context = 133
detail = 61
point = 330


img[myslice, point-context/2:(point+context/2+1), point-context/2:(point+context/2+1)]  = img[myslice, point-context/2:(point+context/2+1), point-context/2:(point+context/2+1)]  + 1000
img[myslice, point-detail/2:(point+detail/2+1), point-detail/2:(point+detail/2+1)] = img[myslice, point-detail/2:(point+detail/2+1), point-detail/2:(point+detail/2+1)] + 1000


plt.imshow(img[myslice])


