#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 10:34:06 2018

@author: lukas
"""

import os
wd = os.getcwd()


###################   parameters // replace with config files ########################

dataset = 'MSKCC'

############################## Load dataset #############################
 
TPM_channel = ''

segmentChannels = ['/CV_folds/CV_alignedNii-Aug2019_actual-F4-training/train_t1post_1_malignants.txt',
	           '/CV_folds/CV_alignedNii-Aug2019_actual-F4-training/train_sub_1_malignants.txt']

segmentLabels = '/CV_folds/CV_alignedNii-Aug2019_actual-F4-training/train_labels_1_malignants.txt'

output_classes = 2
    
#-------------------------------------------------------------------------------------------------------------

# Parameters 

######################################### MODEL PARAMETERS
# Models : 'CNN_TPM' , 'DeepMedic'

model = 'MultiPriors_v2' 
dpatch=[13,75,75]
segmentation_dpatch = [25,99,99]

path_to_model = '/home/deeperthought/Projects/MultiPriors_MSKCC/training_sessions/MultiPriors_v2_MSKCC_configFile_MultiPriors_v2_F4_parallel_classBalanced_2019-11-23_1839/models/v2_MSKCC_configFile_MultiPriors_v2_F4_parallel_classBalanced_2019-11-23_1839.log_epoch70.h5'
session =  path_to_model.split('/')[-3]

########################################### TEST PARAMETERS
quick_segmentation = True
dice_compare=True
output_probability = True 
percentile_normalization = True
use_coordinates = True
full_segmentation_patches = True
test_subjects = 140
n_full_segmentations = 140
list_subjects_fullSegmentation = range(140)
size_test_minibatches = 500
saveSegmentation = True

import numpy as np
penalty_MATRIX = np.array([[ 1,  0],
			   [ 0,  1]], dtype='float32')

comments = ''

