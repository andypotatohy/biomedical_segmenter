#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 10:34:06 2018

@author: lukas
"""
import numpy as np
import os
wd = os.getcwd()

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list="0"
tf.keras.backend.set_session(tf.Session(config=config))

###################   parameters // replace with config files ########################


#availabledatasets :'ATLAS17','CustomATLAS', 'BRATS15', 'BRATS15_TEST', 'BRATS15_wholeNormalized' ,BRATS15_ENTIRE', 'CustomBRATS15' (for explicitly giving channels)
dataset = 'breastMask'

############################## Load dataset #############################
 
#TPM_channel = '/home/hirsch/Documents/projects/TPM/correct_labels_TPM_padded.nii'
    
TPM_channel = ''
    
trainChannels = ['/CV_folds/CV_Replicate_Aug2019/train_t1post.txt',
		 '/CV_folds/CV_Replicate_Aug2019/train_sub1.txt',
		 '/CV_folds/CV_Replicate_Aug2019/train_sub2.txt',
		 '/CV_folds/CV_Replicate_Aug2019/train_TPM.txt']

trainLabels   = '/CV_folds/CV_Replicate_Aug2019/train_labels.txt'
    
testChannels  = ['/CV_folds/CV_Replicate_Aug2019/val_t1post.txt',
		 '/CV_folds/CV_Replicate_Aug2019/val_sub1.txt',
		 '/CV_folds/CV_Replicate_Aug2019/val_sub2.txt',
		 '/CV_folds/CV_Replicate_Aug2019/val_TPM.txt']

testLabels = '/CV_folds/CV_Replicate_Aug2019/val_labels.txt'

validationChannels = testChannels
validationLabels = testLabels
    
output_classes = 2
test_subjects = 10
    
#-------------------------------------------------------------------------------------------------------------
 
# Parameters 

######################################### MODEL PARAMETERS
# Models : 'MultiPriors_MSKCC' ,'MultiPriors_MSKCC_MultiScale' 
model = 'UNet_v0_TumorSegmenter' 
dpatch= [3,512,512]
segmentation_dpatch = [3,512,512]
model_patch_reduction = [2,0,0]
model_crop = 0 # 40 for normal model.

using_unet_breastMask = False
resolution = 'high'

L2 = 0
# Loss functions: 'Dice', 'wDice', 'Multinomial'
loss_function = 'Dice'

load_model = True
path_to_model = '/home/deeperthought/Projects/MultiPriors_MSKCC/training_sessions/UNet_v0_TumorSegmenter_breastMask_UNet_v0_TumorSegmenter_2019-12-27_1850/models/rSegmenter_breastMask_UNet_v0_TumorSegmenter_2019-12-27_1850.log_epoch23.h5'
if load_model:
	session =  path_to_model.split('/')[-3]

num_channels = len(trainChannels)
dropout = [0,0]  # dropout for last two fully connected layers
learning_rate = 2e-05
optimizer_decay = 0

########################################## TRAIN PARAMETERS
num_iter = 10
epochs = 50

#---- Dataset/Model related parameters ----
samplingMethod_train = 0
samplingMethod_val = 0
use_coordinates = True

using_unet_breastMask = False
resolution = 'high'

merge_breastMask_model = False
path_to_breastMask_model = ''
Context_parameters_trainable = False

sample_intensity_based = False 
percentile_voxel_intensity_sample_benigns = 0

balanced_sample_subjects = True # SET TO FALSE WHEN TRAINING DATA HAS NO MALIGNANT/BENGING LABEL (breast mask model)
proportion_malignants_to_sample_train = 0.5
proportion_malignants_to_sample_val = 0.5
#------------------------------------------
n_subjects = 50
n_patches = n_subjects*2
size_minibatches = 8

data_augmentation = True 
proportion_to_flip = 0.5
percentile_normalization = True
verbose = False 
quickmode = False # Train without validation. Full segmentation often but only report dice score (whole)
n_subjects_val = 10
n_patches_val = n_subjects_val*2
size_minibatches_val = 8


########################################### TEST PARAMETERS
output_probability = True   # not thresholded network output for full scan segmentation
quick_segmentation = True
OUTPUT_PATH = ''
n_full_segmentations = 10
full_segmentation_patches = True
size_test_minibatches = 8
list_subjects_fullSegmentation = np.arange(0,25,1)  # Leave empty if random
epochs_for_fullSegmentation = np.arange(1,epochs+1) # [1,5,10,15,20,25,29]
saveSegmentation = True
proportion_malignants_fullSegmentation = 0.5

threshold_EARLY_STOP = 0

penalty_MATRIX = np.array([[ 1,  0],
			   [ 0,  1]], dtype='float32')


comments = ''

