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
 
#TPM_channel = '/home/hirsch/Documents/projects/TPM/correct_labels_TPM_padded.nii'
    
TPM_channel = '/log_TPM_gaussian_reflected.nii'
    
trainChannels = ['/CV_folds/Malignant_segmentor_nr2/train_t1post.txt',
		 '/CV_folds/Malignant_segmentor_nr2/train_sub.txt',
		 '/CV_folds/Malignant_segmentor_nr2/train_t2.txt']

trainLabels   = '/CV_folds/Malignant_segmentor_nr2/train_labels.txt'
    
testChannels  = ['/CV_folds/Malignant_segmentor_nr2/val_t1post.txt',
		 '/CV_folds/Malignant_segmentor_nr2/val_sub.txt',
		 '/CV_folds/Malignant_segmentor_nr2/val_t2.txt']

testLabels = '/CV_folds/Malignant_segmentor_nr2/val_labels.txt'

validationChannels = testChannels
validationLabels = testLabels
    
output_classes = 2
test_subjects = 12
    
#-------------------------------------------------------------------------------------------------------------
 
# Parameters 

######################################### MODEL PARAMETERS
# Models : 'MultiPriors_MSKCC' ,'MultiPriors_MSKCC_MultiScale' 
model = 'MultiPriors_MSKCC' #'MultiPriors_MSKCC_MultiScale' 
dpatch=[13,75,75]
segmentation_dpatch = [25,100,100]#[20,100,100]
L2 = 0.0001
# Loss functions: 'Dice', 'wDice', 'Multinomial'
loss_function = 'Dice'

load_model = False
path_to_model = '/home/hirsch/Documents/projects/Breast_segmentation/DeepPriors_package/training_sessions/MultiPriors_MSKCC_MultiScale_fullHeadSegmentation_configFile_DM_malignant_segmentor_2019-06-27_1544/models/MSKCC_MultiScale_fullHeadSegmentation_configFile_DM_malignant_segmentor_2019-06-27_1544.log_epoch0.h5'
session =  path_to_model.split('/')[-3]

num_channels = len(trainChannels)
dropout = [0,0]  # dropout for last two fully connected layers
learning_rate = 2e-05
optimizer_decay = 0

########################################## TRAIN PARAMETERS
num_iter = 2
epochs = 30
samplingMethod_train = 1
samplingMethod_val = 1
use_coordinates = False

n_patches = 6000 
n_subjects = 200 
size_minibatches = 16
proportion_malignants_to_sample = 0.3
percentile_voxel_intensity_sample_benigns = 0
data_augmentation = False 
proportion_to_flip = 0
percentile_normalization = False

verbose = False 
quickmode = False # Train without validation. Full segmentation often but only report dice score (whole)
n_patches_val = 200
n_subjects_val = 50
size_minibatches_val = 16 

########################################### TEST PARAMETERS
output_probability = False   # not thresholded network output for full scan segmentation
quick_segmentation = True
n_fullSegmentations = 12
full_segmentation_patches = True
size_test_minibatches = 500
list_subjects_fullSegmentation = []  # Leave empty if random
epochs_for_fullSegmentation = [1,5,10,15,20,25,29]
saveSegmentation = True
proportion_malignants_fullSegmentation = 0.8

threshold_EARLY_STOP = 0

import numpy as np
penalty_MATRIX = np.array([[ 1,  0],
			   [ 0,  1]], dtype='float32')


comments = ''

