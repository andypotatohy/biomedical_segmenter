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

segmentChannels = ['/CV_folds/Liz_Mary/With_1st_batch/test_t1post.txt',
	           '/CV_folds/Liz_Mary/With_1st_batch/test_sub.txt']


segmentLabels = ''

output_classes = 2
    
#-------------------------------------------------------------------------------------------------------------

# Parameters 

######################################### MODEL PARAMETERS
# Models : 'CNN_TPM' , 'DeepMedic'

model = 'MultiPriors_v1' 
dpatch=[13,75,75]
segmentation_dpatch = [25,99,99]

path_to_model = '/home/deeperthought/Projects/MultiPriors_MSKCC/training_sessions/MultiPriors_v1_MSKCC_configFile_MultiPriors_MSKCC_2019-10-23_1613/models/v1_MSKCC_configFile_MultiPriors_MSKCC_2019-10-23_1613.log_epoch6.h5'
session =  path_to_model.split('/')[-3]

########################################### TEST PARAMETERS
quick_segmentation = True
output_probability = False 
use_coordinates = True
full_segmentation_patches = True
test_subjects = 278
n_full_segmentations = 278
list_subjects_fullSegmentation = range(278)
size_test_minibatches = 500
saveSegmentation = True

import numpy as np
penalty_MATRIX = np.array([[ 1,  0],
			   [ 0,  1]], dtype='float32')

comments = ''

