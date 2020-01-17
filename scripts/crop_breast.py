#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 11:17:11 2019

@author: deeperthought
"""

import nibabel as nib
import numpy as np
from scipy.ndimage.measurements import label
import matplotlib.pyplot as plt
from skimage.transform import resize
import os

OUTPUT_PATH = '/home/deeperthought/Projects/MultiPriors_MSKCC/training_sessions/BreastSegmentor_v1_breastMask_configFile_BreastMask-Model_2019-10-21_1932/predictions/crop_breast/'
BREASTMASK_PATH = '/home/deeperthought/Projects/MultiPriors_MSKCC/training_sessions/BreastSegmentor_v1_breastMask_configFile_BreastMask-Model_2019-10-21_1932/predictions/test/'
MRI_PATH = '/home/deeperthought/kirby_MSK/alignedNii-Aug2019/'


def keep_largest_connected_component(data):
  structure = np.ones((3,3,3), dtype=np.int)
                
  labeled, ncomponents = label(data, structure)  
  sizes = {}
  for i in range(1,ncomponents+1):
    sizes[i] = len(np.argwhere(labeled == i))
  
  components = sizes.values()
  components.sort(reverse=1)
  
  largest_component = components[0]

  for i in sizes.keys():
      if sizes[i] < largest_component:
          labeled[labeled == i] = 0    
  labeled[labeled > 0] = 1             
  return labeled          


for bm in os.listdir(BREASTMASK_PATH):
    print(bm)
    exam = bm.split('_T1')[0]
    side = bm.split('T1_')[1].split('_post')[0]
    nii = MRI_PATH + exam + '/T1_' + side + '_post.nii'   

    bm = nib.load(BREASTMASK_PATH + bm).get_data()
    img_nii = nib.load(nii)
    img = img_nii.get_data()

    
    if not (np.equal(bm.shape, img.shape)).all():
        bm = resize(bm, output_shape=img.shape, order=1, preserve_range=True, anti_aliasing=True)
        
    plt.imshow(img[20] + bm[20]*1000)
    
    bm[bm > 0.6] = 1
    bm[bm < 1] = 0
    img_stripped = img * bm
    
    plt.imshow(img_stripped[20])
    out = nib.Nifti1Image(img_stripped, img_nii.affine)
    nib.save(out, OUTPUT_PATH + exam + '_' + side + '_test1.nii')

    bm = keep_largest_connected_component(bm)
    img_stripped = img * bm
    plt.imshow(img_stripped[20])
    out = nib.Nifti1Image(img_stripped, img_nii.affine)
    nib.save(out, OUTPUT_PATH + exam + '_' + side + '_test2.nii')



