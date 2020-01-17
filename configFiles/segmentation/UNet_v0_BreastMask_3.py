#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 10:34:06 2018

@author: lukas
"""

import os
wd = os.getcwd()

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list="3"
tf.keras.backend.set_session(tf.Session(config=config))
###################   parameters // replace with config files ########################


#availabledatasets :'ATLAS17','CustomATLAS', 'BRATS15', 'BRATS15_TEST', 'BRATS15_wholeNormalized' ,BRATS15_ENTIRE', 'CustomBRATS15' (for explicitly giving channels)
dataset = 'breastMask'

############################## Load dataset #############################
 
TPM_channel = ''

segmentChannels = ['/CV_folds/CV_nov/SEGMENTER/test_t1post.txt']

segmentLabels = ''

output_classes = 2
    
#-------------------------------------------------------------------------------------------------------------

# Parameters 

######################################### MODEL PARAMETERS
# Models : 'CNN_TPM' , 'DeepMedic'

model = 'UNet_v0_BreastMask' 
use_coordinates = True
dpatch=[3,256,256]
segmentation_dpatch = [3,256,256] 
model_patch_reduction = [2,0,0]
model_crop = 0

using_unet_breastMask = True
resolution = 'low'

path_to_model = '/home/deeperthought/Projects/MultiPriors_MSKCC/training_sessions/UNet_v0_BreastMask_breastMask_UNet_v0_BreastMask_2019-12-23_1658/models/stMask_breastMask_UNet_v0_BreastMask_2019-12-23_1658.log_epoch45.h5'
session =  path_to_model.split('/')[-3]

percentile_normalization = True
########################################### TEST PARAMETERS
quick_segmentation = True
output_probability = True
OUTPUT_PATH = '/home/deeperthought/kirby_MSK/BreastMasks/predictions/'
save_as_nifti = True  
dice_compare = False
full_segmentation_patches = True
test_subjects = 80000
n_fullSegmentations = 80000
list_subjects_fullSegmentation = range(20000,40000)
size_test_minibatches = 16
saveSegmentation = True

import numpy as np
penalty_MATRIX = np.array([[ 1,  0],
			   [ 0,  1]], dtype='float32')

comments = ''

