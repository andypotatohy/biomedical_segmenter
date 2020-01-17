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
config.gpu_options.visible_device_list="1"
tf.keras.backend.set_session(tf.Session(config=config))

###################   parameters // replace with config files ########################

dataset = 'MSKCC'

############################## Load dataset #############################
 
TPM_channel = ''

segmentChannels = ['/CV_folds/CV_nov/DGNS/converts_benigns_t1post.txt',
		 '/CV_folds/CV_nov/DGNS/converts_benigns_sub1.txt',
		 '/CV_folds/CV_nov/DGNS/converts_benigns_sub2.txt']
segmentLabels = ''

output_classes = 2
    
#-------------------------------------------------------------------------------------------------------------

# Parameters 

######################################### MODEL PARAMETERS

model = 'UNet_v4_TPM' 
dpatch= [19,75,75]
segmentation_dpatch = [19,75,75]
model_patch_reduction = [18,38,38]
model_crop = 0 # 40 for normal model.

using_unet_breastMask = False
resolution = 'high'

path_to_model = '/home/deeperthought/Projects/MultiPriors_MSKCC/training_sessions/UNet_3D_v4_MSKCC_configFile_UNet_3D_v4_Segmenter_DGNS_2020-01-15_1823/models/best_model.h5'
session =  path_to_model.split('/')[-3]

########################################### TEST PARAMETERS
use_coordinates = False
quick_segmentation = True
output_probability = True 
dice_compare = False
save_as_nifti = True  
OUTPUT_PATH = '/home/deeperthought/kirby_MSK/DGNS_INPUT_Jan2020/UNet_3D_v4_MSKCC_configFile_UNet_3D_v4_Segmenter_DGNS_2020-01-15_1823/converts/'
percentile_normalization = True
full_segmentation_patches = True
test_subjects = 774
n_full_segmentations = 774
list_subjects_fullSegmentation = range(774)
size_test_minibatches = 16
saveSegmentation = True

import numpy as np
penalty_MATRIX = np.array([[ 1,  0],
			   [ 0,  1]], dtype='float32')

comments = ''

