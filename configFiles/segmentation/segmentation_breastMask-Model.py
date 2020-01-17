#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 10:34:06 2018

@author: lukas
"""

import os
wd = os.getcwd()

###################   parameters // replace with config files ########################


#availabledatasets :'ATLAS17','CustomATLAS', 'BRATS15', 'BRATS15_TEST', 'BRATS15_wholeNormalized' ,BRATS15_ENTIRE', 'CustomBRATS15' (for explicitly giving channels)
dataset = 'breastMask'

############################## Load dataset #############################
 
TPM_channel = ''

segmentChannels = ['/CV_folds/CV_alignedNii-Aug2019_actual-F4-training/train_t1post_FIRST_HALF.txt']
#segmentChannels = ['/CV_folds/CV_alignedNii-Aug2019_actual-F4-training/val_t1post.txt']

segmentLabels = ''

output_classes = 2
    
#-------------------------------------------------------------------------------------------------------------

# Parameters 

######################################### MODEL PARAMETERS
# Models : 'CNN_TPM' , 'DeepMedic'

model = 'BreastSegmentor_v1' 
use_coordinates = True
dpatch=[13,75,75]
segmentation_dpatch = [23,135,135] #[29,145,145]
model_patch_reduction = [12,66,66]
model_crop = 40

path_to_model = '/home/deeperthought/Projects/MultiPriors_MSKCC/training_sessions/BreastSegmentor_v1_breastMask_configFile_BreastMask-Model_2019-10-21_1932/models/tor_v1_breastMask_configFile_BreastMask-Model_2019-10-21_1932.log_epoch499.h5'
session =  path_to_model.split('/')[-3]

percentile_normalization = True
########################################### TEST PARAMETERS
quick_segmentation = True
output_probability = False
save_as_nifti = True  
dice_compare = False
full_segmentation_patches = True
test_subjects = 80000
n_fullSegmentations = 80000
list_subjects_fullSegmentation = range(test_subjects)
size_test_minibatches = 16
saveSegmentation = True

import numpy as np
penalty_MATRIX = np.array([[ 1,  0],
			   [ 0,  1]], dtype='float32')

comments = ''

