#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 10:34:06 2018

@author: lukas
"""
import numpy as np
import os
wd = os.getcwd()

###################   parameters // replace with config files ########################


#availabledatasets :'ATLAS17','CustomATLAS', 'BRATS15', 'BRATS15_TEST', 'BRATS15_wholeNormalized' ,BRATS15_ENTIRE', 'CustomBRATS15' (for explicitly giving channels)
dataset = 'MSKCC'

############################## Load dataset #############################
 
#TPM_channel = '/home/hirsch/Documents/projects/TPM/correct_labels_TPM_padded.nii'
    
TPM_channel = ''
    
trainChannels = ['/CV_folds/Liz_Mary/With_1st_batch/train_t1post.txt',
		 '/CV_folds/Liz_Mary/With_1st_batch/train_sub.txt']

trainLabels   = '/CV_folds/Liz_Mary/With_1st_batch/train_labels.txt'
    
testChannels  = ['/CV_folds/Liz_Mary/With_1st_batch/val_t1post.txt',
		 '/CV_folds/Liz_Mary/With_1st_batch/val_sub.txt']

testLabels = '/CV_folds/Liz_Mary/With_1st_batch/val_labels.txt'

validationChannels = testChannels
validationLabels = testLabels
    
output_classes = 2
test_subjects = 12
    
#-------------------------------------------------------------------------------------------------------------
 
# Parameters 

######################################### MODEL PARAMETERS
# Models : 'MultiPriors_MSKCC' ,'MultiPriors_MSKCC_MultiScale' 
model = 'MultiPriors_v2' 
dpatch=[13,75,75]
segmentation_dpatch = [25,99,99]
L2 = 0.0001
# Loss functions: 'Dice', 'wDice', 'Multinomial'
loss_function = 'Dice'

load_model = False
path_to_model = ''
if load_model:
	session =  path_to_model.split('/')[-3]

num_channels = len(trainChannels)
dropout = [0,0]  # dropout for last two fully connected layers
learning_rate = 2e-05
optimizer_decay = 0

########################################## TRAIN PARAMETERS
num_iter = 1
epochs = 500

#---- Dataset/Model related parameters ----
samplingMethod_train = 1
samplingMethod_val = 1
use_coordinates = True

merge_breastMask_model = True
path_to_breastMask_model = '/home/deeperthought/Projects/MultiPriors_MSKCC/training_sessions/BreastSegmentor_v1_breastMask_configFile_BreastMask-Model_2019-10-21_1932/models/tor_v1_breastMask_configFile_BreastMask-Model_2019-10-21_1932.log_epoch499.h5'
Context_parameters_trainable = False

sample_intensity_based = True 
percentile_voxel_intensity_sample_benigns = 0

balanced_sample_subjects = True 		# SET TO FALSE WHEN TRAINING DATA HAS NO MALIGNANT/BENGING LABEL (breast mask model)
proportion_malignants_to_sample = 0.25
#------------------------------------------
n_subjects = 500
n_patches = n_subjects*32
size_minibatches = 32

data_augmentation = True 
proportion_to_flip = 0.5
percentile_normalization = True
verbose = False 
quickmode = False # Train without validation. Full segmentation often but only report dice score (whole)
n_subjects_val = 50
n_patches_val = n_subjects_val*32
size_minibatches_val = 32

########################################### TEST PARAMETERS
output_probability = False   # not thresholded network output for full scan segmentation
quick_segmentation = True
n_full_segmentations = 10
full_segmentation_patches = True
size_test_minibatches = 500
list_subjects_fullSegmentation = []  # Leave empty if random
epochs_for_fullSegmentation = np.arange(1,epochs+1,10) # [1,5,10,15,20,25,29]
saveSegmentation = True
proportion_malignants_fullSegmentation = 0.5

threshold_EARLY_STOP = 0

penalty_MATRIX = np.array([[ 1,  0],
			   [ 0,  1]], dtype='float32')


comments = ''

