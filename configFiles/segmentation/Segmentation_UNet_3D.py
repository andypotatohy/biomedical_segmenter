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

segmentChannels = ['/CV_folds/CV_alignedNii-Aug2019_actual-F4-training/test_t1post_1.txt',
		 '/CV_folds/CV_alignedNii-Aug2019_actual-F4-training/test_sub_1.txt']
segmentLabels = ''

output_classes = 2
    
#-------------------------------------------------------------------------------------------------------------

# Parameters 

######################################### MODEL PARAMETERS
# Models : 'CNN_TPM' , 'DeepMedic'

model = 'UNet_3D' 
# Rules for patch sizes:
# Anything that is divisible by 3 (on positions 2 and 3). They have to be > 12, > 66 , > 66, respectively
dpatch=[8,64,64]
segmentation_dpatch = [8,64,64]
model_patch_reduction = [7,8,8] # [12,66,66]  for normal model.
model_crop = 0 # 40 for normal model.

path_to_model = '/home/deeperthought/Projects/MultiPriors_MSKCC/training_sessions/UNet_3D_MSKCC_configFile_UNet_3D_2019-12-12_1304/models/C_configFile_UNet_3D_2019-12-12_1304.log_epoch16.h5'
session =  path_to_model.split('/')[-3]

########################################### TEST PARAMETERS
use_coordinates = False

quick_segmentation = True
output_probability = True 
dice_compare = False
save_as_nifti = True  
percentile_normalization = True
full_segmentation_patches = True
test_subjects = 278
n_full_segmentations = 278
list_subjects_fullSegmentation = range(278)
size_test_minibatches = 32
saveSegmentation = True

import numpy as np
penalty_MATRIX = np.array([[ 1,  0],
			   [ 0,  1]], dtype='float32')

comments = ''

