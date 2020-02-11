#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 18:47:01 2020

@author: deeperthought
"""

import tensorflow as tf
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#config.gpu_options.visible_device_list="0"
#tf.keras.backend.set_session(tf.Session(config=config))

import os
import numpy as np
import nibabel as nib
from keras.models import load_model
from keras.models import Model
import sys
sys.path.append('/home/deeperthought/Projects/MultiPriors_MSKCC/scripts/')
import matplotlib.pyplot as plt
from lib import *
from skimage.transform import resize
from MultiPriors_Models_Collection import Generalised_dice_coef_multilabel2, dice_coef_multilabel_bin0,dice_coef_multilabel_bin1
import scipy.ndimage as ndimage

#%%    USER INPUT
     
PATH_TO_MODEL = '/home/deeperthought/Projects/MultiPriors_MSKCC/training_sessions/UNet_3D_v4_MSKCC_configFile_UNet_3D_v4_Segmenter_DGNS_2020-01-24_1528/models/best_model.h5'#'/home/deeperthought/Projects/MultiPriors_MSKCC/training_sessions/UNet_3D_v4_MSKCC_configFile_UNet_3D_v4_2020-01-23_1412/models/best_model.h5'

WD = '/home/deeperthought/Projects/MultiPriors_MSKCC'


segmentChannels = [WD + '/CV_folds/CV_nov/SEGMENTER/test_t1post_malignant_Converts.txt',
            		 WD + '/CV_folds/CV_nov/SEGMENTER/test_sub1_malignant_Converts.txt',
            		 WD + '/CV_folds/CV_nov/SEGMENTER/test_sub2_malignant_Converts.txt']



VISUALIZE_FEATURES = False

OUTPUT_FOLDER = '/home/deeperthought/kirby_MSK/DGNS_INPUT_Jan2020/UNet_3D_v4_MSKCC_configFile_UNet_3D_v4_Segmenter_DGNS_2020-01-24_1528/Malignants/'

SEG_PATCH = (27, 139, 139, 3)

INDEXES = range(1000)

#%%  Define functions
#
#TPM_channel=[]
#model=model1
#testChannels=segmentChannels
#testLabels=''
#output_classes=2 
#subjectIndex=6
#segmentation_dpatch=[19, 75, 75]
#size_minibatches=16
#use_coordinates=False
#percentile_normalization=True
#model_patch_reduction=[18, 38, 38]
             
def get_features(TPM_channel, model, testChannels, testLabels, output_classes, subjectIndex, segmentation_dpatch, size_minibatches, use_coordinates, percentile_normalization, model_patch_reduction):    
   
    output_dpatch = segmentation_dpatch[0] - model_patch_reduction[0], segmentation_dpatch[1] - model_patch_reduction[1], segmentation_dpatch[2] - model_patch_reduction[2]     
    subjectIndex = [subjectIndex]
    num_channels = len(testChannels)
    firstChannelFile = open(testChannels[0],"r")   
    ch = firstChannelFile.readlines()
    subjectGTchannel = ch[subjectIndex[0]][:-1]
    subID = subjectGTchannel.split('/')[-2] + '_' + subjectGTchannel.split('/')[-1].split('.nii')[0]
    print('SEGMENTATION : Segmenting subject: ' + str(subID))  

    firstChannelFile.close()      
    proxy_img = nib.load(subjectGTchannel)
    shape = proxy_img.shape
    affine = proxy_img.affine      
    res = proxy_img.header['pixdim'][1:4]

    if res[1] > 0.6:    
      target_res = [res[0],res[1]/2.,res[2]/2.]
      shape = [int(x) for x in np.floor([float(s)*r1/r2 for s,r1,r2 in zip(shape, res, target_res)])]
    else:
      target_res = res
      shape=list(shape)
    
          
    print('SEGMENTATION : Sampling data..')  
    TPM_patches, labels, voxelCoordinates, spatial_coordinates, shape = sampleTestData(TPM_channel, testChannels, testLabels, subjectIndex, output_classes, 
                                                                                       output_dpatch, shape, use_coordinates)    
    affine = np.diag(list(target_res) + [0])        
    n_minibatches = 0 # min(0,len(voxelCoordinates)/size_minibatches) 
    total_number_of_patches = (len(voxelCoordinates)-n_minibatches*size_minibatches)  
    
    #########################################################################
    print('SEGMENTATION : Extracting {} image patches..'.format(total_number_of_patches))
    patches = np.zeros((total_number_of_patches,segmentation_dpatch[0],segmentation_dpatch[1],segmentation_dpatch[2],num_channels),dtype='float32')
    for i in range(len(testChannels)):
        patches[:,:,:,:,i] = extractImagePatch_parallelization(testChannels[i], subjectIndex[0], voxelCoordinates, shape, segmentation_dpatch, percentile_normalization, fullSegmentationPhase=True)    

    print('SEGMENTATION : Finished sampling data.')
    INPUT_DATA = []   
    INPUT_DATA.append(patches)

    if len(spatial_coordinates) > 0:
        INPUT_DATA.append(spatial_coordinates)            

    print("SEGMENTATION : Finished preprocessing data for segmentation.")
    #########################################################################
    
    segmentation = model.predict(INPUT_DATA, verbose=1, batch_size=size_minibatches)
    #return features, segmentation       

    shape[-1] = segmentation
    output2 = np.ones(shape, dtype=np.float32)  # same size as input head, start index for segmentation start at 26,26,26, rest filled with zeros....
    i = 0
    for x,y,z in voxelCoordinates:
        patch_shape = output2[x-output_dpatch[0]/2:min(x+(output_dpatch[0]/2+output_dpatch[0]%2), shape[0]),
                           y-output_dpatch[1]/2:min(y+(output_dpatch[1]/2+output_dpatch[1]%2), shape[1]),
                           z-output_dpatch[2]/2:min(z+(output_dpatch[2]/2+output_dpatch[2]%2), shape[2])].shape
        #print(np.array(indexes[i])[0:patch_shape[0], 0:patch_shape[1],0:patch_shape[2]])
        output2[x-output_dpatch[0]/2:min(x+(output_dpatch[0]/2+output_dpatch[0]%2), shape[0]),
             y-output_dpatch[1]/2:min(y+(output_dpatch[1]/2+output_dpatch[1]%2), shape[1]),
             z-output_dpatch[2]/2:min(z+(output_dpatch[2]/2+output_dpatch[2]%2), shape[2])] = np.array(segmentation[i])[0:patch_shape[0], 
                                                                                                                      0:patch_shape[1],
                                                                                                                      0:patch_shape[2]]
        i = i+1

    
    img = nib.Nifti1Image(output2[:,:,:,1], affine)
    return img, subID

from skimage.feature import peak_local_max


#%%    Load Model, get intermediate layer, get features
my_custom_objects = {'Generalised_dice_coef_multilabel2':Generalised_dice_coef_multilabel2,
                                 'dice_coef_multilabel_bin0':dice_coef_multilabel_bin0,
                                 'dice_coef_multilabel_bin1':dice_coef_multilabel_bin1}

model = load_model(PATH_TO_MODEL, custom_objects = my_custom_objects)

from Unet_3D_Class import UNet_v4
new_model = UNet_v4(input_shape=(None,None,None,3), pool_size=(2, 2, 2), n_labels=2, initial_learning_rate=0.00001, deconvolution=True,
      depth=4, n_base_filters=32, include_label_wise_dice_coefficients=False, 
      batch_normalization=True, activation_name="softmax", bilinear_upsampling=True)


for i in range(len(model.layers)):
    print(i)
    new_model.layers[i].set_weights(model.layers[i].get_weights())



for SUBJECT_INDEX in INDEXES:
    subjectIndex = [SUBJECT_INDEX]
    num_channels = len(segmentChannels)
    firstChannelFile = open(segmentChannels[0],"r")   
    ch = firstChannelFile.readlines()
    subjectGTchannel = ch[subjectIndex[0]][:-1]
    subID = subjectGTchannel.split('/')[-2] + '_' + subjectGTchannel.split('/')[-1].split('.nii')[0]
    print('SEGMENTATION : Segmenting subject: ' + str(subID))  
    scanID = subID.replace('_T1','').replace('_02_01','_features')
    if os.path.exists(OUTPUT_FOLDER + 'ROI/' + '{}'.format(scanID) + '.npy'):
        print('Already done. Skip.')
        continue
    
    segmentation, subID = get_features(TPM_channel=[], model=new_model, testChannels=segmentChannels, testLabels='', output_classes=2, subjectIndex=SUBJECT_INDEX, segmentation_dpatch=SEG_PATCH,#[19, 75, 75],
                 size_minibatches=8, use_coordinates=False, percentile_normalization=True, model_patch_reduction=[18, 38, 38])
    

    nib.save(segmentation, OUTPUT_FOLDER + '/' + scanID)



 
    

