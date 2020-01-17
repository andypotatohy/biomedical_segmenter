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
dataset = 'fullHeadSegmentation'

############################## Load dataset #############################
 
TPM_channel = '/log_TPM_gaussian.nii'

segmentChannels = ['/CV_folds/test_t1post.txt',
	           '/CV_folds/test_sub.txt',
		   '/CV_folds/test_t1pre.txt']


segmentLabels = ''

output_classes = 2
    
#-------------------------------------------------------------------------------------------------------------

# Parameters 

######################################### MODEL PARAMETERS
# Models : 'CNN_TPM' , 'DeepMedic'

model = 'MultiPriors_MSKCC_MultiScale' 
dpatch=[13,75,75]
segmentation_dpatch = [20,100,100]

path_to_model = '/home/deeperthought/Projects/MultiPriors_MSKCC/training_sessions/MultiPriors_MSKCC_MultiScale_fullHeadSegmentation_configFile_MultiScale_DataAugmentation_StandNormalization_2019-09-23_1410/models/MSKCC_MultiScale_fullHeadSegmentation_configFile_MultiScale_DataAugmentation_StandNormalization_2019-09-23_1410.log_epoch99.h5'
session =  path_to_model.split('/')[-3]

########################################### TEST PARAMETERS
quick_segmentation = True
output_probability = True 
full_segmentation_patches = True
test_subjects = 278
n_fullSegmentations = 278
list_subjects_fullSegmentation = range(278)
size_test_minibatches = 500
saveSegmentation = True

import numpy as np
penalty_MATRIX = np.array([[ 1,  0],
			   [ 0,  1]], dtype='float32')

comments = ''

