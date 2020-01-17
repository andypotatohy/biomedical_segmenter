#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 11:35:14 2017

@author: lukas
"""
import os
from shutil import copy
import sys
import nibabel as nib
import numpy as np
import time
import random
from numpy.random import seed
from tensorflow import set_random_seed
import matplotlib.pyplot as plt
from random import sample
import keras 
import keras.backend as K
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import pandas as pd
from keras.models import load_model   
from skimage.transform import resize
from keras.utils import multi_gpu_model

seed(1)
set_random_seed(2)
        
import multiprocessing        
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool        
        
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))

from keras_radam import RAdam

################################## AUXILIARY FUNCTIONS #####################################

  
def flip_random(patches, labels, TPM_patches, proportion_to_flip=0.5):

  malignant_indexes = np.argwhere(np.sum(np.sum(labels, axis=-1), axis=-1) > 0)[:,0]
  indx_toflip = np.random.choice(malignant_indexes, int(len(malignant_indexes)*proportion_to_flip), replace=False)
  axis = np.random.choice(range(0,3),size=len(indx_toflip))
  for i in range(len(indx_toflip)):
    if axis[i] == 0:
      # SAGITTAL FLIP
      for ch in range(patches.shape[-1]):
          patches[indx_toflip[i],:,:,:,ch] = patches[indx_toflip[i],::-1,:,:,ch]
#      patches[indx_toflip[i],:,:,:,1] = patches[indx_toflip[i],::-1,:,:,1]
#      patches[indx_toflip[i],:,:,:,2] = patches[indx_toflip[i],::-1,:,:,2]  
      #labels[INDEX,0,:] = np.flip(labels[INDEX,0,:],FLIP_AXIS)
    elif axis[i] == 1:
      # AXIAL? FLIP
      for ch in range(patches.shape[-1]):
          patches[indx_toflip[i],:,:,:,ch] = patches[indx_toflip[i],:,::-1,:,ch]
      #patches[indx_toflip[i],:,:,:,1] = patches[indx_toflip[i],:,::-1,:,1]
      #patches[indx_toflip[i],:,:,:,2] = patches[indx_toflip[i],:,::-1,:,2]  
      labels[indx_toflip[i],0,:] = np.flip(labels[indx_toflip[i],0,:],0)
      if len(TPM_patches) != 0:
          TPM_patches[indx_toflip[i],0,:] = np.flip(TPM_patches[indx_toflip[i],0,:],0)
    elif axis[i] == 2:
      # CORONAL? FLIP
      for ch in range(patches.shape[-1]):
          patches[indx_toflip[i],:,:,:,ch] = patches[indx_toflip[i],:,:,::-1,ch]
#      patches[indx_toflip[i],:,:,:,1] = patches[indx_toflip[i],:,:,::-1,1]
#      patches[indx_toflip[i],:,:,:,2] = patches[indx_toflip[i],:,:,::-1,2]  
      labels[indx_toflip[i],0,:] = np.flip(labels[indx_toflip[i],0,:],1)        
      if len(TPM_patches) != 0:
          TPM_patches[indx_toflip[i],0,:] = np.flip(TPM_patches[indx_toflip[i],0,:],1)        
  return patches, labels , TPM_patches
  
def subtract_MRI(t1post,t1pre):
    t1post_flat = t1post.flatten()
    t1pre_flat = t1pre.flatten()
    w,b = np.polyfit(x=t1post_flat, y=t1pre_flat, deg=1)
    t1post_flat_transformed = w*t1post_flat + b      
    sub = t1post_flat_transformed - t1pre_flat
    sub = sub.reshape(t1post.shape)  
    return sub

def percentile95_normalizeMRI(data):
    p95 = np.percentile(data,95)
    data1 = data/p95
    return(data1)


def getVarFromFile(filename):
    import imp
    print('import using {}'.format(filename))
    f = open(filename)
    global cfg
    cfg = imp.load_source('cfg', '', f)
    f.close()


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def normalizeMRI(data, mean=0, std=1):
    if (mean == 0) and (std == 1):
        mean = np.mean(data)
        std = np.std(data)
    data1 = (data - mean)/std
    return(data1)


def classesInSample(minibatch_labels, output_classes):
	label_numbers = []
	print(minibatch_labels.shape)	
	minibatch_labels = np.argmax(minibatch_labels, axis=3)
	for c in range(output_classes):
		label_numbers.append(np.sum(minibatch_labels == c))
	#return label_numbers
	return np.sum(minibatch_labels, axis=-1)
 
################################## SAMPLING FUNCTIONS #####################################


def generateRandomIndexesSubjects(n_subjects, total_subjects):
    indexSubjects = random.sample(xrange(total_subjects), n_subjects)
    return indexSubjects

def getSubjectChannels(subjectIndexes, channel):
    "With the channels (any modality) and the indexes of the selected subjects, return the addresses of the subjects channels"
    fp = open(channel)
    # read file, per subject index extract patches given the indexesPatch
    lines = fp.readlines()
    selectedSubjects = [lines[i][:-1] for i in subjectIndexes]
    fp.close()
    return selectedSubjects


def getSubjectShapes_parallelization(subjectChannel):
    # Need to open every nifty file and get the shapes
    #scan_ID = subjectChannel.split('/')[-2] + '_' + subjectChannel.split('T1_')[-1][0]
    proxy_img = nib.load(subjectChannel)
    res = proxy_img.header['pixdim'][1:4]
    shape = proxy_img.shape
    if res[1] > 0.6:    
        target_res = [res[0],res[1]/2.,res[2]/2.]
        out_shape = np.floor([float(s)*r1/r2 for s,r1,r2 in zip(shape, res, target_res)])
    else:
        out_shape = shape
    return out_shape    

def getSubjectShapes(subjectIndexes, n_patches, channelList):
    # Need to open every nifty file and get the shapes
    fp = open(channelList)
    # read file, per subject index extract patches given the indexesPatch
    lines = fp.readlines()
    selectedSubjects = [lines[i] for i in subjectIndexes]
    fp.close()
    shapes = []
    # Get shapes of all subjects to sample from. Can be a separate function (cause apparently I am needing this everywhere)
    for subjectChannel in selectedSubjects:
        subjectChannel = str(subjectChannel)[:-1]
        proxy_img = nib.load(subjectChannel)
        res = proxy_img.header['pixdim'][1:4]
        shape = proxy_img.shape
        if res[1] > 0.6:    
            target_res = [res[0],res[1]/2.,res[2]/2.]
            out_shape = np.floor([float(s)*r1/r2 for s,r1,r2 in zip(shape, res, target_res)])
        else:
            out_shape = shape
        shapes.append(out_shape)
    return shapes      

def generateVoxelIndexes_wrapper_parallelization(DATA_INPUT):
    subjectIndexes = DATA_INPUT[0]
    target_shape = DATA_INPUT[1]
    patches_per_subject = DATA_INPUT[2]
    channels = DATA_INPUT[3]
    channel_mri = DATA_INPUT[4]     
    dpatch = DATA_INPUT[5]  
    n_patches = DATA_INPUT[6]    
    samplingMethod = DATA_INPUT[7] 
    output_classes = DATA_INPUT[8]  
    percentile_voxel_intensity_sample_benigns = DATA_INPUT[9]
    percentile_normalization = DATA_INPUT[10]
    CV_FOLDS_ARRAYS_PATH = DATA_INPUT[11]
    return generateVoxelIndexes_parallel(subjectIndexes,CV_FOLDS_ARRAYS_PATH, target_shape, patches_per_subject, dpatch, n_patches, channels, channel_mri, samplingMethod, output_classes, percentile_voxel_intensity_sample_benigns,percentile_normalization, allForegroundVoxels = "", verbose=False)

def generateVoxelIndexes_parallel(subjectIndexes,CV_FOLDS_ARRAYS_PATH, target_shape, patches_per_subject, dpatch, n_patches, channels, channel_mri, samplingMethod, output_classes, percentile_voxel_intensity_sample_benigns ,percentile_normalization , allForegroundVoxels = "", verbose=False):
    allVoxelIndexes = {} #{a:None for a in subjectIndexes} 

    #--------------------------------------------------------------------------------------------------------------------------------------------------
    if samplingMethod == 0:
        "Sample voxels from random locations in a SLICE. This allows sampling from any class present in the SLICE."
        scanVoxels = ((np.random.randint(0+dpatch[0]/2, target_shape[0]-(dpatch[0]/2)-1),
                      np.random.randint(0+dpatch[1]/2, target_shape[1]-(dpatch[1]/2)-1),
                      np.random.randint(0+dpatch[2]/2, target_shape[2]-(dpatch[2]/2)-1)))

        # only one 2D slice is labeled with the breast mask, so on the sagittal dimension there is no freedom of choice.
        data_label = nib.load(channels).get_data()

        foreground_voxels = np.argwhere(data_label>0)
        mySlice = foreground_voxels[0][0]  
        #background_voxels = np.argwhere(data_label[mySlice] == 0)
        
        scanVoxels = []
        for _ in range(patches_per_subject):
            x = mySlice 
            y = random.choice(xrange(dpatch[1]/2,int(target_shape[1])-(dpatch[1]/2)-1))
            z = random.choice(xrange(dpatch[2]/2,int(target_shape[2])-(dpatch[2]/2)-1))
            scanVoxels.append([x,y,z])
            
        allVoxelIndexes[subjectIndexes] = scanVoxels
        
    #--------------------------------------------------------------------------------------------------------------------------------------------------
    elif samplingMethod == 1:
        "Only for binary classes. Sample only foreground voxels when present. Sample background voxels only from scans that have NO foreground voxels."
        assert os.path.exists(channels), 'Generating voxel-index for samplig: ERROR: path doesnt exist {}'.format(channels)           
                
        #Check if previously stored arrays indicating sampling locations:
        exam = channel_mri.split('/')[-2]
        side = channel_mri.split('T1_')[-1][0]
        scan_ID = exam + '_' + side
        if 'BENIGN' in channels:
            benign_scan = True
            voxel_locations_array = CV_FOLDS_ARRAYS_PATH + scan_ID + '_Background_voxel_locations_{}_percentile.npz'.format(percentile_voxel_intensity_sample_benigns)
        else:
            benign_scan = False
            voxel_locations_array = CV_FOLDS_ARRAYS_PATH + scan_ID + '_Foreground_voxel_locations.npz'
            
        if os.path.exists(voxel_locations_array):
            #print('Found previously stored voxel locations..')
            candidate_voxels_for_sampling = np.load(voxel_locations_array)
            candidate_voxels_for_sampling = candidate_voxels_for_sampling[candidate_voxels_for_sampling.keys()[0]]
            if benign_scan:
                #print('Scan benign. Sampling half from voxel-intensity > {}'.format(percentile_voxel_intensity_sample_benigns))
                # Half from the intensity-based sampling:
                scanVoxels = candidate_voxels_for_sampling[random.sample(xrange(0,len(candidate_voxels_for_sampling)), min( len(candidate_voxels_for_sampling), patches_per_subject/2) )].tolist()   
                # Half from random locations:
                for _ in range(patches_per_subject/2):
                    x = random.choice(xrange(dpatch[0]/2,int(target_shape[0])-(dpatch[0]/2)-1)) 
                    y = random.choice(xrange(dpatch[1]/2,int(target_shape[1])-(dpatch[1]/2)-1))
                    z = random.choice(xrange(dpatch[2]/2,int(target_shape[2])-(dpatch[2]/2)-1))
                    scanVoxels.append([x,y,z])      
            else:
                #print('Scan malignant, sampling from labeled region.')
                scanVoxels = candidate_voxels_for_sampling[random.sample(xrange(0,len(candidate_voxels_for_sampling)), min(len(candidate_voxels_for_sampling),patches_per_subject))]
               
                
        else:
            # No previously stored voxel coordinates for candidate sampling ############
            ############################################################################    
            bV = 0
            fg = 0      
            nifti_label = nib.load(channels)
            data_label = nifti_label.get_data()
            if np.sum(data_label) > 0:
                # Target label contains segmentation of a tumor. Scan is malignant.
                fg = getForegroundBackgroundVoxels(nifti_label, data_label, target_shape) # This function returns only foreground voxels based on labels.
                if len(fg) == 0:
                  print('resize function removed the foreground voxels...')
                  print(channels)
                  print('Target shape = {}'.format(target_shape))
                  print('Original shape = {}'.format(data_label.shape))
                  sys.exit(0)
                np.savez_compressed(CV_FOLDS_ARRAYS_PATH + scan_ID + '_Foreground_voxel_locations',fg)
                scanVoxels = fg[random.sample(xrange(0,len(fg)), min(len(fg),patches_per_subject))]
            else:  
                # Only getting non-tumor voxels from benign scans:
                if percentile_voxel_intensity_sample_benigns > 0:
                    # Sample high-intensity voxels but also random from the scan.
                    bV = getBodyVoxels(channel_mri, percentile_voxel_intensity_sample_benigns, percentile_normalization)
                    np.savez_compressed(CV_FOLDS_ARRAYS_PATH + scan_ID + '_Background_voxel_locations_{}_percentile'.format(percentile_voxel_intensity_sample_benigns),bV)                    
                    # Half from the intensity-based sampling:
                    scanVoxels = bV[random.sample(xrange(0,len(bV)), min( len(bV), patches_per_subject/2) )].tolist()   
                    # Half from random locations:
                    for _ in range(patches_per_subject/2):
                        x = random.choice(xrange(dpatch[0]/2,int(target_shape[0])-(dpatch[0]/2)-1)) 
                        y = random.choice(xrange(dpatch[1]/2,int(target_shape[1])-(dpatch[1]/2)-1))
                        z = random.choice(xrange(dpatch[2]/2,int(target_shape[2])-(dpatch[2]/2)-1))
                        scanVoxels.append([x,y,z])          
                else:
                    scanVoxels = []
                    for _ in range(patches_per_subject):
                        x = random.choice(xrange(dpatch[0]/2,int(target_shape[0])-(dpatch[0]/2)-1))  
                        y = random.choice(xrange(dpatch[1]/2,int(target_shape[1])-(dpatch[1]/2)-1))
                        z = random.choice(xrange(dpatch[2]/2,int(target_shape[2])-(dpatch[2]/2)-1))
                        scanVoxels.append([x,y,z])
            del fg
            del bV   
            
    #--------------------------------------------------------------------------------------------------------------------------------------------------        
    allVoxelIndexes[subjectIndexes] = scanVoxels
    del scanVoxels
    return allVoxelIndexes    


def getAllForegroundClassesVoxels(groundTruthChannel, dpatch, output_classes):
    '''Get vector of voxel coordinates for all voxel values for all freground classes'''
    "e.g. groundTruthChannel = '/home/hirsch/Documents/projects/ATLASdataset/native_part2/c0011/c0011s0006t01/c0011s0006t01_LesionSmooth_Binary.nii.gz'"
    "NOTE: img in MRICRON starts at (1,1,1) and this function starts at (0,0,0), so points do not match when comparing in MRICRON. Add 1 to all dimensions to match in mricron. Function works properly though"
    img = nib.load(groundTruthChannel)
    data = np.array(img.dataobj[dpatch[0]/2:img.shape[0]-(dpatch[0]/2)-1, dpatch[1]/2:img.shape[1]-(dpatch[1]/2)-1, dpatch[2]/2:img.shape[2]-(dpatch[2]/2)-1],dtype='int16') # Get a cropped image, to avoid CENTRAL foreground voxels that are too near to the border. These will still be included, but not as central voxels. As long as they are in the 9x9x9 volume (-+ 4 voxels from the central, on a segment size of 25x25x25) they will still be included in the training.
    img.uncache()    
    voxels = []
    for c in range(1,output_classes):
        coords = np.argwhere(data==c)
        coords = [sum(x) for x in zip(coords , [x/2 for x in dpatch])]
        voxels.append(coords)
    return voxels  # This is a List! Use totuple() to convert if this makes any trouble
            
def getSubjectsToSample(channelList, subjectIndexes):
    "Actually returns channel of the subjects to sample"
    fp = open(channelList)
    lines = fp.readlines()
    subjects = [lines[i] for i in subjectIndexes]
    fp.close()
    return subjects



def extractLabels_parallelization_wrapper(DATA_INPUT_EXTRACT_LABELS_PATCH):
    subject_label_channel = DATA_INPUT_EXTRACT_LABELS_PATCH[0]
    voxelCoordinates = DATA_INPUT_EXTRACT_LABELS_PATCH[1]
    output_dpatch = DATA_INPUT_EXTRACT_LABELS_PATCH[2]
    output_shape = DATA_INPUT_EXTRACT_LABELS_PATCH[3]    
    return extractLabels_parallelization(subject_label_channel, voxelCoordinates, output_dpatch, output_shape)

def extractLabels_parallelization(subject_label_channel, voxelCoordinates, output_dpatch, output_shape):
    if len(voxelCoordinates) == 0:
      print('Within extractLabels_parallelization: \n len(voxelCoordinates) == 0, subject_label_channel = {}'.format(subject_label_channel))
      sys.exit(0)
    labels = []       
    subject = str(subject_label_channel)[:-1]
    proxy_label = nib.load(subject)
    label_data = np.array(proxy_label.get_data(),dtype='int8')

    label_data = resize(label_data, order=1, output_shape=output_shape, preserve_range=True, anti_aliasing=True, mode='reflect') 
    label_data[label_data > 0] = 1
     
    #DEBUG
    #out = nib.Nifti1Image(label_data, np.diag([1,1,1,0]))
    #nib.save(out, '/home/deeperthought/Projects/MultiPriors_MSKCC/DEBUG/' + 'Label_' + subject.split('MSKCC')[-1].replace('/','_label_'))     
     
    label_padded = np.pad(label_data,((0,60),(0,100),(0,100)),'constant')  # need to pad for segmentation with huge patches that go outside (only the end - ascending coordinates) boundaries. Scale stays the same, as the origin is not modified. 
    if np.sum(label_data) == 0:
      for j in range(len(voxelCoordinates)):
        labels.append(np.zeros((output_dpatch[0],output_dpatch[1],output_dpatch[2]),dtype='int8'))
    else:
      for j in range(len(voxelCoordinates)):
        D1,D2,D3 = voxelCoordinates[j]
        labels.append(label_padded[D1-output_dpatch[0]/2:D1+(output_dpatch[0]/2)+output_dpatch[0]%2,
                                   D2-output_dpatch[1]/2:D2+(output_dpatch[1]/2)+output_dpatch[1]%2,
                                   D3-output_dpatch[2]/2:D3+(output_dpatch[2]/2)+output_dpatch[2]%2])
    proxy_label.uncache()
    del label_data
    return labels


def extractLabels(groundTruthChannel_list, subjectIndexes, voxelCoordinates, output_dpatch, shapes):
    #print('extracting labels from ' + str(len(subjectIndexes))+ ' subjects.')    
    subjects = getSubjectsToSample(groundTruthChannel_list,subjectIndexes)
    labels = []       
    for i in range(len(subjects)):
        subject = str(subjects[i])[:-1]
        #print('extracting labels from subject index [{}] with path : {}'.format(subjectIndexes[i],subject))
        proxy_label = nib.load(subject)
        label_data = np.array(proxy_label.get_data(),dtype='int8')
        # WHERE AM I RESIZING THE SEGMENTATION LABEL???        
        label_data = resize(label_data, shapes[i], order=0, preserve_range=True, anti_aliasing=True)        
        #DEBUG
        #out = nib.Nifti1Image(label_data, np.diag([1,1,1,0]))
        #nib.save(out, '/home/deeperthought/Projects/MultiPriors_MSKCC/DEBUG/' + 'Label_' + subject.split('MSKCC')[-1].replace('/','_label_'))
        label_padded = np.pad(label_data,((0,60),(0,100),(0,100)),'constant')  # need to pad for segmentation with huge patches that go outside (only the end - ascending coordinates) boundaries. Scale stays the same, as the origin is not modified. 
        if np.sum(label_data) == 0:
          for j in range(len(voxelCoordinates[i])):
            labels.append(np.zeros((output_dpatch[0],output_dpatch[1],output_dpatch[2]),dtype='int8'))
        else:
          for j in range(len(voxelCoordinates[i])):
            D1,D2,D3 = voxelCoordinates[i][j]
            #print('Extracting labels from \n subject {} with shape {} and coords {},{},{}'.format(subjects[i], label_data.shape ,D1,D2,D3))
            labels.append(label_padded[D1-output_dpatch[0]/2:D1+(output_dpatch[0]/2)+output_dpatch[0]%2,
                                       D2-output_dpatch[1]/2:D2+(output_dpatch[1]/2)+output_dpatch[1]%2,
                                       D3-output_dpatch[2]/2:D3+(output_dpatch[2]/2)+output_dpatch[2]%2])
            #if len(labels[-1])==0:
            #  labels[-1] = np.zeros((9,9),dtype='int8')
        proxy_label.uncache()
        del label_data
    return labels

#shapes = [shape]
#voxelCoordinates = [voxelCoordinates]
   
def extractCoordinates(shapes, voxelCoordinates, output_dpatch):
    """ Given a list of voxel coordinates, it returns the absolute location coordinates for a given patch size (output 1x9x9) """
    #print('extracting coordinates from ' + str(len(subjectIndexes))+ ' subjects.')
    #subjects = getSubjectsToSample(channel, subjectIndexes)
    
    all_coordinates = []
    for i in xrange(len(shapes)):
        #subject = str(subjects[i])[:-1]
        #img = nib.load(subject)
        img_shape = shapes[i]
        for j in xrange(len(voxelCoordinates[i])):     
            D1,D2,D3 = voxelCoordinates[i][j]
            #all_coordinates.append(get_Coordinates_from_target_patch(img.shape,D1,D2,D3))                 
            all_coordinates.append(get_Coordinates_from_target_patch(img_shape,D1,D2,D3, output_dpatch))                    

        #img.uncache()
    return np.array(all_coordinates)    


def get_Coordinates_from_target_patch(img_shape,D1,D2,D3, output_dpatch) :

    x_ = range(D1-(output_dpatch[0]//2),D1+((output_dpatch[0]//2)+1))
    y_ = range(D2-(output_dpatch[1]//2),D2+((output_dpatch[1]//2)+1))
    z_ = range(D3-(output_dpatch[2]//2),D3+((output_dpatch[2]//2)+1))
    
    x_norm = np.array(x_)/float(img_shape[0])  
    y_norm = np.array(y_)/float(img_shape[1])  
    z_norm = np.array(z_)/float(img_shape[2])  
    
    x, y, z = np.meshgrid(x_norm, y_norm, z_norm, indexing='ij')    
    coords = np.stack([x,y,z], axis=-1)
    return coords
    
       
def get_patches_per_subject( n_patches, n_subjects):
    patches_per_subject = [n_patches/n_subjects]*n_subjects
    randomAdd = random.sample(range(0,len(patches_per_subject)),k=n_patches%n_subjects)
    randomAdd.sort()
    for index in randomAdd:
        patches_per_subject[index] = patches_per_subject[index] + 1
    return patches_per_subject



def extract_TPM_patches_parallelization_wrapper(TPM_INPUT_DATA):
    TPM_channel = TPM_INPUT_DATA[0]
    subjectIndexes = TPM_INPUT_DATA[1]
    voxelCoordinates_subject = TPM_INPUT_DATA[2]
    output_dpatch = TPM_INPUT_DATA[3]
    output_shape = TPM_INPUT_DATA[4]
    return extract_TPM_patches_parallelization(TPM_channel, subjectIndexes, voxelCoordinates_subject, output_dpatch, output_shape)

def extract_TPM_patches_parallelization(TPM_channel, subjectIndexes, voxelCoordinates_subject, output_dpatch, output_shape):    
    vol = np.zeros((len(voxelCoordinates_subject),output_dpatch[0],output_dpatch[1],output_dpatch[2]),dtype='float16')
    proxy_label = nib.load(TPM_channel)
    TPM_data = np.array(proxy_label.get_data())#,dtype='float32')  
    #print('Resizing TPM from {} to {}'.format(TPM_data.shape, output_shape))
    TPM_data = resize(TPM_data, output_shape, order=1, preserve_range=True, anti_aliasing=True, mode='reflect') 
    
    #DEBUG
    #out = nib.Nifti1Image(TPM_data, np.diag([1,1,1,0]))
    #nib.save(out, '/home/deeperthought/Projects/MultiPriors_MSKCC/DEBUG/' + str(subjectIndexes) + '_TPM.nii' )          
    
    padding_border = 100    # why so large?              
    TPM_data = np.pad(TPM_data, padding_border,'minimum')
    for j in range(len(voxelCoordinates_subject)):     
        D1,D2,D3 = voxelCoordinates_subject[j]
        # Scale these coordinates to fit the shape of the Breast Tumor TPM! 
        D1 = D1 + padding_border
        D2 = D2 + padding_border
        D3 = D3 + padding_border        

        try:
          vol[j] = TPM_data[D1-output_dpatch[0]/2:D1+(output_dpatch[0]/2)+output_dpatch[0]%2,
                                       D2-output_dpatch[1]/2:D2+(output_dpatch[1]/2)+output_dpatch[1]%2,
                                       D3-output_dpatch[2]/2:D3+(output_dpatch[2]/2)+output_dpatch[2]%2]
        except:
          print('FAILED TO EXTRACT TPM PATCH')
          print('Coordinates: {}'.format([D1,D2,D3]))
          print('From subject index {}'.format(subjectIndexes))
          print('TPM shape after padding: {}'.format(TPM_data.shape))
          
    proxy_label.uncache()
    del proxy_label
    del TPM_data
    return vol

    
def extract_TPM_patches(TPM_channel, subjectIndexes, voxelCoordinates, output_dpatch, shapes):
    print('extracting TPM patches from ' + str(len(subjectIndexes))+ ' subjects.')          
    n_patches = 0
    k = 0
    for i in range(len(voxelCoordinates)):
        n_patches += len(voxelCoordinates[i])
    vol = np.zeros((n_patches,output_dpatch[0],output_dpatch[1],output_dpatch[2]),dtype='float32')             # CHANGE THIS TO CHANNELS OF TPM!!
    print('TPM patches shape: {}'.format(vol.shape))
    for i in range(len(subjectIndexes)):            
        proxy_label = nib.load(TPM_channel)
        TPM_data = np.array(proxy_label.get_data(),dtype='float32')  
        # 'align' the TPM to the input image
        TPM_data = resize(TPM_data, shapes[i], order=1, preserve_range=True, anti_aliasing=True)
        
        #DEBUG
        #out = nib.Nifti1Image(TPM_data, np.diag([1,1,1,0]))
        #nib.save(out, '/home/deeperthought/Projects/MultiPriors_MSKCC/DEBUG/' + str(subjectIndexes) + '_TPM.nii' )         
                
        
        padding_border = 100                  
        label_data_padded = np.pad(TPM_data[:,:,:],
                                   ((padding_border,padding_border), 
                                    (padding_border,padding_border), 
                                    (padding_border,padding_border)),
                                   'minimum')
        for j in range(len(voxelCoordinates[i])):     
            D1,D2,D3 = voxelCoordinates[i][j]
            # Scale these coordinates to fit the shape of the Breast Tumor TPM! 
            D1 = D1 + padding_border
            D2 = D2 + padding_border
            D3 = D3 + padding_border        

            try:
              vol[k,:,:,:] = label_data_padded[D1-output_dpatch[0]/2:D1+(output_dpatch[0]/2)+output_dpatch[0]%2,
              D2-output_dpatch[1]/2:D2+(output_dpatch[1]/2)+output_dpatch[1]%2,
              D3-output_dpatch[2]/2:D3+(output_dpatch[2]/2)+output_dpatch[2]%2]
            except:
              print('FAILED TO EXTRACT TPM PATCH')
              print('Coordinates: {}'.format([D1,D2,D3]))
              print('From subject index {}'.format(subjectIndexes[i]))
              print('TPM shape after padding: {}'.format(label_data_padded.shape))
                                               
            k=k+1
        proxy_label.uncache()
        
    return vol

# Need one function that takes in a list of subject indexes and outputs patch volumes.
# Need one wrapper function that extracts all the arguments needed out of a single list/matrix.


def extractImagePatch_parallelization_wrapper(DATA_INPUT_EXTRACT_IMAGE_PATCH):
  # This has to have all voxel coordinates for each subject... 
  subjectIndex = DATA_INPUT_EXTRACT_IMAGE_PATCH[0]
  channel = DATA_INPUT_EXTRACT_IMAGE_PATCH[1]
  dpatch = DATA_INPUT_EXTRACT_IMAGE_PATCH[2]
  subject_channel_voxelCoordinates = DATA_INPUT_EXTRACT_IMAGE_PATCH[3]
  output_shape = DATA_INPUT_EXTRACT_IMAGE_PATCH[4]
  percentile_normalization = DATA_INPUT_EXTRACT_IMAGE_PATCH[5]
  return extractImagePatch_parallelization(channel, subjectIndex, subject_channel_voxelCoordinates, output_shape, dpatch, percentile_normalization)

def extractImagePatch_parallelization(channel, subjectIndex, subject_channel_voxelCoordinates, output_shape, dpatch, percentile_normalization, preprocess_image_data=True):   
    subject_channel = getSubjectsToSample(channel, [subjectIndex])
    n_patches = len(subject_channel_voxelCoordinates)
    subject = str(subject_channel[0])[:-1]
    proxy_img = nib.load(subject)            
    img_data = np.array(proxy_img.get_data(),dtype='float32')
    if preprocess_image_data:   
      if np.array(img_data.shape != output_shape).any():
        #print('Resizing training data: \nInput_shape = {}, \nOutput_shape = {}. \nSubject = {}'.format(img_data.shape, output_shape, subject))
        img_data = resize(img_data, output_shape=output_shape, preserve_range=True, anti_aliasing=True, mode='reflect')
      if np.any(np.isnan(img_data)):
        print('Nans found in scan {}'.format(subject))
        print('Nans replace by value: {}'.format(np.nanmin(img_data)))
        img_data[np.isnan(img_data)] = np.nanmin(img_data)
      
      if percentile_normalization:
        img_data = percentile95_normalizeMRI(img_data)
      else:
        img_data = normalizeMRI(img_data)      
      if not np.isfinite(img_data).all():
      #if np.any(np.isnan(img_data)):
        print('Normalization: Nans found in scan {}'.format(subject))
        print('Nans replace by value: {}'.format(0.0))
        img_data[np.isfinite(img_data)] = 0.0
        
    #DEBUG
    #out = nib.Nifti1Image(img_data, np.diag([1,1,1,0]))
    #nib.save(out, '/home/deeperthought/Projects/MultiPriors_MSKCC/DEBUG/' + str(subject_channel[0].split('MSKCC')[-1].replace('/','_')[:-1]))         
            
        
    padding_border = 101 #50 #250  just needs to be larger than the largest dpatch/2  (= 37.5)
    img_data_padded = np.pad(img_data, padding_border,'reflect')    
    
    vol = np.zeros((n_patches,dpatch[0],dpatch[1],dpatch[2]),dtype='float16') 
    for j in range(n_patches):      
        D1,D2,D3 = subject_channel_voxelCoordinates[j]           
        D1 = D1 + padding_border#dpatch[0]/2
        D2 = D2 + padding_border#dpatch[1]/2
        D3 = D3 + padding_border#dpatch[2]/2
        try:
          vol[j,:,:,:] = img_data_padded[D1-(dpatch[0]/2):D1+(dpatch[0]/2)+dpatch[0]%2,
                                         D2-(dpatch[1]/2):D2+(dpatch[1]/2)+dpatch[1]%2,
                                         D3-(dpatch[2]/2):D3+(dpatch[2]/2)+dpatch[2]%2]
        except:
          print('Failed to extract image data into shape... This is: \n{}, \nimg_data_padded.shape = {}, \nCoords = {}, \nCoords+Padding = {}'.format(subject_channel,img_data_padded.shape, subject_channel_voxelCoordinates[j] , [D1,D2,D3] ))
          sys.exit(0)
    proxy_img.uncache()
    del img_data
    del img_data_padded
    return vol


#channel = testChannels[i]    
#subjectIndexes = subjectIndex 
#voxelCoordinates = [minibatch_voxelCoordinates]
#dpatch = segmentation_dpatch

def extractImagePatch(channel, subjectIndexes, patches, voxelCoordinates, dpatch, debug=False, preprocess_image_data=True):
    subjects = getSubjectsToSample(channel, subjectIndexes)
    n_patches = 0   
    # Replace this thing. No need to compute. Have this information in list patches_per_subject!
    for i in range(len(voxelCoordinates)):
        n_patches += len(voxelCoordinates[i])
    #print('Starting extraction of {} patches from {} subjects.'.format(n_patches,len(voxelCoordinates)))
    vol = np.ones((n_patches,dpatch[0],dpatch[1],dpatch[2]),dtype='float32')
    k = 0
    
    for i in range(len(subjectIndexes)):   # REPLACE THIS LOOP WITH A MULTIPROCESSING MAP
        #if i%20==0:
        #  print('{}%'.format(round(i*100./len(voxelCoordinates),2)))
        subject = str(subjects[i])[:-1]
        #print('Subject with path: {}'.format(subject))
        proxy_img = nib.load(subject)            
        img_data = np.array(proxy_img.get_data(),dtype='float32')

        if preprocess_image_data:
          # Change resolution:
          res = proxy_img.header['pixdim'][1:4]
          shape = img_data.shape
          if res[1] > 0.6: 
              'Upsampling image..'
              target_res = [res[0],res[1]/2.,res[2]/2.]
              out_shape = np.floor([float(s)*r1/r2 for s,r1,r2 in zip(shape, res, target_res)])
              img_data = resize(img_data, output_shape=out_shape, preserve_range=True, anti_aliasing=True)
          else:
              out_shape = shape
          if np.any(np.isnan(img_data)):
            print('Nans found in scan {}'.format(subject))
            print('Nans replace by value: {}'.format(np.nanmin(img_data)))
            img_data[np.isnan(img_data)] = np.nanmin(img_data)
          
          shape = img_data.shape
          #print('Shape after normalization: {}'.format(shape))
          # Standardize image.
          #for i in range(img_data.shape[0]):
          #  img_data[i,:,:] = percentile95_normalizeMRI(img_data[i,:,:])

          # NEED TO ADD CONDITION SET FROM CONFIG FILE !! BUG
          img_data = percentile95_normalizeMRI(img_data)
          #img_data = normalizeMRI(img_data)

        #DEBUG
        #out = nib.Nifti1Image(img_data, np.diag([1,1,1,0]))
        #nib.save(out, '/home/deeperthought/Projects/MultiPriors_MSKCC/DEBUG/'  +  subject.split('MSKCC')[-1].replace('/','_')[:-1])         
                

        padding_border = np.max(dpatch)/2 + 10#550
        # Here just pad img_data to not to worry about borders...
        img_data_padded = np.pad(img_data, 
                                 padding_border,
                                 'reflect')
        
        # Loop over voxelCoordinates tuples of subject i
        for j in range(len(voxelCoordinates[i])):   
            #print(voxelCoordinates[i][j] )     
            D1,D2,D3 = voxelCoordinates[i][j]           

            D1 = D1 + padding_border#dpatch[0]/2
            D2 = D2 + padding_border#dpatch[1]/2
            D3 = D3 + padding_border#dpatch[2]/2

            vol[k,:,:,:] = img_data_padded[D1-(dpatch[0]/2):D1+(dpatch[0]/2)+dpatch[0]%2,
                                           D2-(dpatch[1]/2):D2+(dpatch[1]/2)+dpatch[1]%2,
                                           D3-(dpatch[2]/2):D3+(dpatch[2]/2)+dpatch[2]%2]

            k = k+1  
        
        proxy_img.uncache()
        del img_data
        if debug: print('extracted [' + str(len(voxelCoordinates[i])) + '] patches from subject ' + str(i) +'/'+ str(len(subjectIndexes)) +  ' with index [' + str(subjectIndexes[i]) + ']')        
    #print('In this batch found {} Bad Coordinates \n'.format(badCoords))
    #print('From subject(s): {}'.format(list(set(badCoords_subj))))
    #raw_input("Press Enter to continue...")
    return vol


#
#trainChannels = cfg.trainChannels
#trainLabels = cfg.trainLabels
#TPM_channel = cfg.TPM_channel
#n_patches = 100 #cfg.n_patches
#n_subjects = 500 #cfg.n_subjects
#dpatch = cfg.dpatch
#output_classes = cfg.output_classes
#samplingMethod = 1
#use_coordinates = cfg.use_coordinates
#balanced_sample_subjects = cfg.balanced_sample_subjects
#verbose=False
#debug=False
#proportion_malignants_to_sample = cfg.proportion_malignants_to_sample
#percentile_voxel_intensity_sample_benigns = cfg.percentile_voxel_intensity_sample_benigns
#percentile_normalization = cfg.percentile_normalization
#proportion_to_flip = cfg.proportion_to_flip

def sampleTrainData(trainChannels,CV_FOLDS_ARRAYS_PATH, trainLabels, TPM_channel, n_patches, n_subjects, dpatch, output_classes, samplingMethod, use_coordinates, proportion_malignants_to_sample, percentile_voxel_intensity_sample_benigns, data_augmentation, proportion_to_flip, percentile_normalization, balanced_sample_subjects=True, verbose=False, debug=False, logfile=None):
    num_channels = len(trainChannels)
    start_time = time.time()
    output_dpatch = dpatch[0] - 12, dpatch[1] - 66, dpatch[2] - 66
    patches_per_subject = get_patches_per_subject( n_patches, n_subjects)    
    labelsFile = open(trainLabels).readlines()    
    total_subjects = len(labelsFile)

    if balanced_sample_subjects:     
      proportion_malignants = int(np.ceil(n_subjects*proportion_malignants_to_sample))
      malignant_subjects_index = [labelsFile.index(x) for x in labelsFile if not 'BENIGN' in x]
      benign_subjects_index = list(set(range(total_subjects)) - set(malignant_subjects_index))
      subjectIndexes = random.sample(malignant_subjects_index, min(len(malignant_subjects_index), proportion_malignants))
      print('sampling {} malignants from partition'.format(len(subjectIndexes)))
      try:
        subjectIndexes.extend(random.sample(benign_subjects_index, n_subjects - len(subjectIndexes)))
      except:
        # if not enough or only malignants in set.
        subjectIndexes.extend(random.sample(malignant_subjects_index, n_subjects - len(subjectIndexes)))
      random.shuffle(subjectIndexes)
    else:
      print('Extracting data from randomly selected subjects.. [breast mask model]')  
      subjectIndexes = generateRandomIndexesSubjects(n_subjects, total_subjects)  
    
    #------------- Parallelization ----------------
    
    #CV_FOLDS_SHAPES_PATH = '/home/deeperthought/Projects/MultiPriors_MSKCC/CV_folds/CV_alignedNii-Aug2019/shapes/'
    
    channel_mri = getSubjectChannels(subjectIndexes, trainChannels[0]) 
    #GET_SHAPES_INPUT = zip(channel_mri, [CV_FOLDS_SHAPES_PATH]*len(channel_mri))
    pool = Pool(multiprocessing.cpu_count() - 1)#mp.cpu_count() -1)
    time1 = time.time()
    shapes = pool.map(getSubjectShapes_parallelization, channel_mri)
    print('Getting scan shapes took {} s'.format(round(time.time() - time1,2)))
    #pool.close()
    #pool.join()     
   
#    # ------------ Non Parallelized --------------- 
#    time1 = time.time()
#    shapes = getSubjectShapes(subjectIndexes, n_patches, trainChannels[0])
#    print('-Non parallelized- Getting scan shapes took {} s'.format(round(time.time() - time1,2)))
      
    ############ Generating Voxel Coordinates For training ##############    
    print('------------ Generating List of Voxel Indexes for sampling  ------------')
    #------------ Non parallelized -------------------   
#    time1 = time.time()    
#    voxelCoordinates = generateVoxelIndexes(subjectIndexes, shapes, patches_per_subject, dpatch, n_patches, trainLabels, trainChannels[0], samplingMethod, output_classes, percentile_voxel_intensity_sample_benigns)    
#    print('generating voxel coordinates for extraction took {} s'.format(round(time.time() - time1,2)))


    #------------ Parallelization --------------------
    channels = getSubjectChannels(subjectIndexes, trainLabels)
    channel_mri = getSubjectChannels(subjectIndexes, trainChannels[0]) 
    DATA_INPUT_WRAPPER = zip(subjectIndexes, shapes, patches_per_subject, channels, channel_mri, [dpatch] * len(patches_per_subject), [n_patches]*len(patches_per_subject), [samplingMethod]*len(patches_per_subject), [output_classes]*len(patches_per_subject), [percentile_voxel_intensity_sample_benigns]*len(patches_per_subject), [percentile_normalization]*len(patches_per_subject), [CV_FOLDS_ARRAYS_PATH]*len(patches_per_subject))
    DATA_INPUT_WRAPPER = list(DATA_INPUT_WRAPPER)
    #pool = Pool(multiprocessing.cpu_count() - 1)#mp.cpu_count() -1)
    time1 = time.time()
    try:      
      voxelCoordinates_full = pool.map(generateVoxelIndexes_wrapper_parallelization, DATA_INPUT_WRAPPER)
    except IndexError:
      print('...IndexError ')
      print(DATA_INPUT_WRAPPER)
    print('Generating List of Voxel Indexes for sampling took {} s'.format(round(time.time() - time1,2)))
    #pool.close()
    #pool.join()  
    subjectIndexes_check = []
    voxelCoordinates = []
    for i in range(len(voxelCoordinates_full)):
      subjectIndexes_check.append(voxelCoordinates_full[i].keys()[0])
      voxelCoordinates.extend(voxelCoordinates_full[i].values())
    assert subjectIndexes_check == subjectIndexes, 'Subject Indexes got out of order through multiprocessing...'
    del subjectIndexes_check
    del voxelCoordinates_full
    
    if debug:
        affine = np.diag((1,1,1,0))
        data = np.zeros((65,400,400))
        for coord_set in voxelCoordinates:
            for coord in coord_set:
                D1,D2,D3 = coord
                data[D1-4:D1+5,D2-4:D2+5,D3-4:D3+5] += 1
        img = nib.Nifti1Image(data, affine)
        nib.save(img,'/home/hirsch/Documents/projects/Breast_segmentation/DeepPriors_package/debug_voxel_sampling_location.nii'.format(n_subjects, n_patches))

    real_n_patches = 0
    for i in range(len(voxelCoordinates)):
        if len(voxelCoordinates[i]) == 0:
          print('Empty voxelCoordinates for i = {}, channel_mri[i] = {}, \n\n DATA_INPUT_WRAPPER[i] = {}, \n\n patches_per_subject = {},\n patches_per_subject[i] = {}'.format(i, channel_mri[i], DATA_INPUT_WRAPPER[i], patches_per_subject, patches_per_subject[i]))
          sys.exit(0)
        real_n_patches += len(voxelCoordinates[i])        
    print('------  Extracting {} image patches from {} subjects, for each of {} channels --------'.format(real_n_patches,len(voxelCoordinates), len(trainChannels) ))

    ############## Parallelization Extract Image Patches ####################
    # channel is the MRI modality... should also expand this so I dont have to loop over channels...
    # PER subject, i need to include all MRI channels, and all voxelCoordinates...
    # --> subjectindex, [mri_channel1, mri_channel2, ...] , [voxelCoord1, voxelCoord2, ...], dpatch

    patches = np.zeros((real_n_patches,dpatch[0],dpatch[1],dpatch[2],num_channels),dtype='float16')       
    for i in range(len(trainChannels)):
        print('Extracting image patches from channel: {}'.format(trainChannels[i]))
        DATA_INPUT_EXTRACT_IMAGE_PATCH = zip(subjectIndexes, [trainChannels[i]]*len(subjectIndexes), [dpatch] * len(subjectIndexes), voxelCoordinates, shapes, [percentile_normalization]*len(subjectIndexes))
        DATA_INPUT_EXTRACT_IMAGE_PATCH = list(DATA_INPUT_EXTRACT_IMAGE_PATCH)
        #pool = Pool(multiprocessing.cpu_count() -1) #-1 )
        time1 = time.time()
        channel_patches = pool.map(extractImagePatch_parallelization_wrapper, DATA_INPUT_EXTRACT_IMAGE_PATCH)
        print('Extracting image patches took {} s'.format(round(time.time() - time1,2))) 
        
        channel_patches_flattened = np.zeros((real_n_patches,dpatch[0],dpatch[1],dpatch[2]))
        start = 0
        for ii in range(len(channel_patches)):
          channel_patches_flattened[start:start+len(channel_patches[ii])] = channel_patches[ii]
          start = start+len(channel_patches[ii])        
        del channel_patches
        patches[:,:,:,:,i] = channel_patches_flattened
    #pool.close()
    #pool.join()       
        
     ########### NO PARALLELIZATION ################# 
#    patches = np.zeros((real_n_patches,dpatch[0],dpatch[1],dpatch[2],num_channels),dtype='float32')    
#    for i in xrange(len(trainChannels)):
#        print('Extracting image patches from channel: {}'.format(trainChannels[i]))
#        time1 = time.time()
#        patches[:,:,:,:,i] = extractImagePatch(trainChannels[i], subjectIndexes, patches, voxelCoordinates, dpatch, debug=False)    
#        print('-No parallelization- Extracting image patches took {} s'.format(round(time.time() - time1,2)))    
    ##########################################################################
        
    print('------  Extracting {} target-label patches from {} subjects --------'.format(real_n_patches,len(voxelCoordinates) ))
    ############# Parallelization Extract Label Patches ###################### 
        
    subjects_label_channels = getSubjectsToSample(trainLabels,subjectIndexes)          
    DATA_INPUT_EXTRACT_LABELS_PATCH = zip(subjects_label_channels, voxelCoordinates, [output_dpatch] * len(subjectIndexes), shapes)
    DATA_INPUT_EXTRACT_LABELS_PATCH = list(DATA_INPUT_EXTRACT_LABELS_PATCH)
    #pool = Pool(multiprocessing.cpu_count() -1) #-1 )
    time1 = time.time()
    labels_list_unflattened = pool.map(extractLabels_parallelization_wrapper, DATA_INPUT_EXTRACT_LABELS_PATCH)
    print('Extracting target label patches took {} s'.format(round(time.time() - time1,2)))   
    #pool.close()
    #pool.join() 
    labels = np.zeros(([real_n_patches] + list(output_dpatch) ), dtype='int8')
    start = 0
    try:
      for ii in range(len(labels_list_unflattened)):
        labels[start:start+len(labels_list_unflattened[ii])] = labels_list_unflattened[ii]
        start = start+len(labels_list_unflattened[ii])        
    except ValueError:
      print('ValueError... ii = {}, start={}, len(labels_list_unflattened[ii]) = {},  subjects_label_channels[ii]: {}'.format(ii, start,len(labels_list_unflattened[ii]), subjects_label_channels[ii] ))
    del labels_list_unflattened    
         
        
    ########### NO PARALLELIZATION #################    
#    time1 = time.time()        
#    labels_list = extractLabels(trainLabels, subjectIndexes, voxelCoordinates, output_dpatch, shapes)  
#    print('-No parallelization - Extracting target label patches took {} s'.format(round(time.time() - time1,2)))
    ################################################    

    labels = np.array(labels,dtype='int8')
    labels_list = np.array(labels)
    labels = np.array(to_categorical(labels.astype(int),output_classes),dtype='int8')
    if(samplingMethod == 2):
        patches = patches[0:len(labels)]  # when using equal sampling (samplingMethod 2), because some classes have very few voxels in a head, there are fewer patches as intended. Patches is initialized as the maximamum value, so needs to get cut to match labels.

    ###########################################################################
    if use_coordinates:
      all_coordinates = extractCoordinates(shapes, voxelCoordinates, output_dpatch)
      if debug:
        y_coords = all_coordinates[:,2,:,:]
        plt.imshow(y_coords[1])
        center_y = y_coords[:,5,1]
        plt.hist(center_y,  200)
        plt.xlabel('Normalized Y coordinate')

    else:
      all_coordinates = []
      
    ############ TPM Patches #############################  
    if len(TPM_channel) > 0:
      print('------  Extracting {} TPM patches from {} subjects --------'.format(real_n_patches,len(voxelCoordinates) ))
      #output_dpatch = [dpatch[0]-12,dpatch[1]-66,dpatch[2]-66 ]   this is only needed if doing full segmentation on bigger patches.
#      time1 = time.time()
#      TPM_patches = np.array(extract_TPM_patches(TPM_channel, subjectIndexes, voxelCoordinates, output_dpatch, shapes),dtype='float32')    
#      print('-No parallelization- Extracting target label patches took {} s'.format(round(time.time() - time1,2)))

      # Parallelization
      
      TPM_INPUT_DATA = zip([TPM_channel]*len(subjectIndexes), subjectIndexes, voxelCoordinates, [output_dpatch] * len(subjectIndexes), shapes)
      TPM_INPUT_DATA = list(TPM_INPUT_DATA)
      #pool = Pool(multiprocessing.cpu_count()-1) #-1 )
      time1 = time.time()
      TPM_patches_unflattened = pool.map(extract_TPM_patches_parallelization_wrapper, TPM_INPUT_DATA)
      print('Extracting TPM patches took {} s'.format(round(time.time() - time1,2)))
      
      TPM_patches = np.zeros(([real_n_patches] + list(output_dpatch)),dtype='float16')
      start = 0
      for ii in range(len(TPM_patches_unflattened)):
        TPM_patches[start:start+len(TPM_patches_unflattened[ii])] = TPM_patches_unflattened[ii]
        start = start+len(TPM_patches_unflattened[ii])        
      del TPM_patches_unflattened              
         
    else:
      TPM_patches = []  
      
    pool.close()
    pool.join() 
      
    if data_augmentation:
      
      print('Data augmentation: Randomly flipping {}% patches..'.format(proportion_to_flip*100))
      patches, labels, TPM_patches = flip_random(patches, labels, TPM_patches, proportion_to_flip)
     
      
    if debug:
        patches.shape
        display_index = 5
        #y_coords = all_coordinates[:,2,:,:]
        labels_img = np.array(labels_list,dtype='int8')
        plt.figure(figsize=(12,8))
        plt.subplot(231)
        plt.imshow(patches[display_index,5,:,:,0], cmap='gray')
        plt.title('T1post')
        plt.subplot(232)
        plt.imshow(patches[display_index,5,:,:,1], cmap='gray')
        plt.title('T1pre')
        plt.subplot(233)
        plt.imshow(patches[display_index,5,:,:,2], cmap='gray')
        plt.title('T2')
        plt.subplot(234)
        plt.imshow(labels_img[display_index,0], cmap='gray')
        plt.title('Target label')
        plt.subplot(235)
        #plt.imshow(y_coords[display_index])
        #plt.title('Y-Coords')        
        plt.savefig('/home/andy/projects/mskProj/DeepPriors_package/One_Patch_example_MSKCC_16-328_1_09687_20100716-l_{}.png'.format(display_index))
    if logfile != None:  
      my_logger("Finished extracting " + str(real_n_patches) + " patches, from "  + str(n_subjects) + " subjects and " + str(num_channels) + " channels. Time: " + str(round(time.time()-start_time,2)) + "s", logfile)
    return patches, labels, all_coordinates, TPM_patches
    
  
def sampleTestData(TPM_channel, testChannels, testLabels, subjectIndex, output_classes, output_dpatch, shape, use_coordinates,logfile):
    labelsFile = open(testChannels[0],"r")   
    ch = labelsFile.readlines()
    subjectGTchannel = ch[subjectIndex[0]][:-1]
    my_logger('Segmenting subject with channel: ' + str(subjectGTchannel), logfile)
    labelsFile.close()      
    xend = output_dpatch[0] * int(round(float(shape[0])/output_dpatch[0] + 0.5)) 
    yend = output_dpatch[1] * int(round(float(shape[1])/output_dpatch[1] + 0.5)) 
    zend = output_dpatch[2] * int(round(float(shape[2])/output_dpatch[2] + 0.5))
    voxelCoordinates = []
    affine = []
    for x in range(output_dpatch[0]/2,xend,output_dpatch[0]): 
        for y in range(output_dpatch[1]/2,yend,output_dpatch[1]):
            for z in range(output_dpatch[2]/2,zend,output_dpatch[2]):
                voxelCoordinates.append([x,y,z])
    
    if len(TPM_channel) > 0:
      TPM_patches = extract_TPM_patches(TPM_channel, subjectIndex, [voxelCoordinates], output_dpatch, [shape])
    else:
      TPM_patches = []
    if len(testLabels) > 0:
      labels = np.array(extractLabels(testLabels, subjectIndex, [voxelCoordinates], output_dpatch,[shape]))
      labels = to_categorical(labels.astype(int),output_classes)
    else:
      labels = []
    if use_coordinates:
      spatial_coordinates = extractCoordinates([shape], [voxelCoordinates], output_dpatch) 
    else:
      spatial_coordinates = []
    #print("Finished extracting " + str(n_patches) + " patches, from "  + str(n_subjects) + " subjects and " + str(num_channels) + " channels. Timing: " + str(round(end-start,2)) + "s")
    return TPM_patches, labels, voxelCoordinates, spatial_coordinates, shape, affine        



def getForegroundBackgroundVoxels(nifti_label, data_label, target_shape):
    "NOTE: img in MRICRON starts at (1,1,1) and this function starts at (0,0,0), so points do not match when comparing in MRICRON. Add 1 to all dimensions to match in mricron. Function works properly though"
    shape = nifti_label.shape
    if np.array(shape != target_shape).any():
      data = resize(data_label, order=0, output_shape=target_shape, preserve_range=True, anti_aliasing=True, mode='reflect') 
    else:
      data = data_label
    if np.sum(data) == 0:
      data = resize(data_label,order=1, output_shape=target_shape, preserve_range=True, anti_aliasing=True, mode='reflect') 
      data[data > 0] = 1
    nifti_label.uncache()    
    foregroundVoxels = np.argwhere(data>0)
    return foregroundVoxels
      
def getBodyVoxels(channel, percentile_voxel_intensity_sample_benigns, percentile_normalization):
    '''Get vector of voxel coordinates for all voxel values > 0'''
    "e.g. groundTruthChannel = '/home/hirsch/Documents/projects/ATLASdataset/native_part2/c0011/c0011s0006t01/c0011s0006t01_LesionSmooth_Binary.nii.gz'"
    "NOTE: img in MRICRON starts at (1,1,1) and this function starts at (0,0,0), so points do not match when comparing in MRICRON. Add 1 to all dimensions to match in mricron. Function works properly though"
    img = nib.load(channel)
    data = img.get_data()  
    res = img.header['pixdim'][1:4]
    shape = img.shape 
    if res[1] > 0.6:    
      target_res = [res[0],res[1]/2.,res[2]/2.]
      out_shape = np.floor([float(s)*r1/r2 for s,r1,r2 in zip(shape, res, target_res)])
      data = resize(data,order=1, output_shape=out_shape, preserve_range=True, anti_aliasing=True, mode='reflect')

    if percentile_normalization:
      data = percentile95_normalizeMRI(data)
    else:  
      data = normalizeMRI(data)
    if np.any(np.isnan(data)):
      print('NaN found on getBodyVoxels Function')
      sys.exit(0)
    img.uncache()    
    bodyVoxels = np.argwhere(data > np.percentile(data, percentile_voxel_intensity_sample_benigns))
    return bodyVoxels

####################################### METRIC FUNCTIONS #################################################

def weighted_generalized_dice_completeImages(img1,img2,penalty_MATRIX):
    classes = np.array(range(0,len(penalty_MATRIX)), dtype='int8')   
    dice = []
    
    for i in classes:
        dice_2 = []
        #DICE = 2*np.sum(np.multiply(img1==i,img2==i))/float(np.sum(img1==i)+np.sum(img2==i))
        for j in classes:
            wDice = 2*np.sum(np.multiply(img1==i,img2==j) * penalty_MATRIX[i,j] )/float(np.sum(img1==i)+np.sum(img2==j))
            dice_2.append(wDice)
        dice.append(np.sum(dice_2)) 
    return np.sum(dice)/len(classes), [round(x,2) for x in dice]

def dice_completeImages(img1,img2):
    return(2*np.sum(np.multiply(img1>0,img2>0))/float(np.sum(img1>0)+np.sum(img2>0)))
       
def generalized_dice_completeImages(img1,img2):
    assert img1.shape == img2.shape, 'Images of different size!'
    #assert (np.unique(img1) == np.unique(img2)).all(), 'Images have different classes!'
    classes = np.array(np.unique(img1), dtype='int8')   
    if len(classes) < len(np.array(np.unique(img2), dtype='int8')   ):
      classes = np.array(np.unique(img2), dtype='int8')   
    dice = []
    for i in classes:
        dice.append(2*np.sum(np.multiply(img1==i,img2==i))/float(np.sum(img1==i)+np.sum(img2==i)))   
    return np.sum(dice)/len(classes), [round(x,2) for x in dice]

################################## DOCUMENTATION FUNCTIONS ################################################


def plot_training(session,losses, metrics,val_performance,full_segm_DICE, smooth=50, loss_name = ['Multiclass Dice'], class_names = ['Air','GM','WM','CSF','Bone','Skin']):

    losses_df = pd.DataFrame(losses)
    losses_df.columns=loss_name
    
    losses_mv_avg = losses_df.rolling(smooth,center=False).mean()
    metrics_df = pd.DataFrame(metrics)
    metrics_df.columns = class_names
    color_dict = {'Air':'black','GM':'blue','WM':'green','CSF':'yellow','Bone':'orange','Skin':'red'}
    metrics_mv_avg = metrics_df.rolling(smooth,center=False).mean()
    
    n_plots = 2 + np.sum([int(x) for x in [2*(len(val_performance) > 0), len(full_segm_DICE) > 0]])
            
    f, axarr = plt.subplots(n_plots, sharex=False, figsize=(8,10))
    losses_mv_avg.plot(ax=axarr[0])
    axarr[0].set_title(session)
    metrics_mv_avg.plot(ax=axarr[1], color=[color_dict.get(x, '#333333') for x in metrics_mv_avg.columns])
    #axarr[1].plot(metrics_mv_avg)
    #axarr[1].set_title('Single Class Dice Loss')
    axarr[1].set_xlabel('Training Iterations')
    axarr[1].legend(loc='upper left')
       
    if len(val_performance) > 0  :
    
        loss_val = [x[0] for x in val_performance]
        metrics_val = [x[1:len(x)] for x in val_performance]
        
        loss_val_df = pd.DataFrame(loss_val)
        loss_val_df.columns=loss_name
        #loss_val_df = loss_val_df.rolling(smooth,center=False).mean()
        metrics_val_df = pd.DataFrame(metrics_val)
        metrics_val_df.columns = class_names
        #metrics_val_df = metrics_val_df.rolling(smooth,center=False).mean()
        loss_val_df.plot(ax=axarr[2])
        #axarr[2].set_title(loss_name[0])
        metrics_val_df.plot(ax=axarr[3], color=[color_dict.get(x, '#333333') for x in metrics_mv_avg.columns])
        #axarr[1].plot(metrics_mv_avg)
        #axarr[3].set_title('Single Class Dice Loss')
        #axarr[3].set_xlabel('Training Iterations')
        
        axarr[3].legend(loc='upper left')
    
    if len(full_segm_DICE) > 0:
        
        full_segm_DICE = pd.DataFrame(full_segm_DICE)
        full_segm_DICE.columns=['Full Segmentation DICE']
        full_segm_DICE.plot(ax=axarr[n_plots-1],style='-o',color='green')
        axarr[n_plots-1].legend(loc='lower right')
        

def my_logger(string, logfile):
    f = open(logfile,'a')
    f.write('\n' + str(string))
    f.close()
    print(string)
    

def start_training_session_logger(logfile,threshold_EARLY_STOP, TPM_channel, load_model,saveSegmentation,path_to_model,model,dropout, trainChannels, trainLabels, validationChannels, validationLabels, testChannels, testLabels, num_iter, epochs, n_patches, n_patches_val, n_subjects, samplingMethod_train, size_minibatches, n_full_segmentations, epochs_for_fullSegmentation, size_test_minibatches):
    my_logger('#######################################  NEW TRAINING SESSION  #######################################', logfile)    
    my_logger(trainChannels, logfile)
    my_logger(trainLabels, logfile)
    my_logger(validationChannels, logfile)        
    my_logger(validationLabels, logfile)  
    my_logger(testChannels, logfile) 
    my_logger(testLabels, logfile)
    my_logger('TPM channel (if given):', logfile)
    my_logger(TPM_channel, logfile)
    my_logger('Session parameters: ', logfile)
    my_logger('[num_iter, epochs, n_patches, n_patches_val, n_subjects, samplingMethod_train, size_minibatches, n_full_segmentations, epochs_for_fullSegmentation, size_test_minibatches]', logfile)
    my_logger([num_iter, epochs, n_patches, n_patches_val, n_subjects, samplingMethod_train, size_minibatches, n_full_segmentations, epochs_for_fullSegmentation, size_test_minibatches], logfile)
    my_logger('Dropout for last two fully connected layers: ' + str(dropout), logfile)
    my_logger('Model loss function: ' + str(model.loss), logfile)
    my_logger('Model number of parameters: ' + str(model.count_params()), logfile)
    my_logger('Optimizer used: ' +  str(model.optimizer.from_config), logfile)
    my_logger('Optimizer parameters: ' + str(model.optimizer.get_config()), logfile)
    my_logger('Save full head segmentation of subjects: ' + str(saveSegmentation), logfile)
    my_logger('EARLY STOP Threshold last 3 epochs: ' + str(threshold_EARLY_STOP), logfile)
    if load_model:
        my_logger("USING PREVIOUSLY SAVED MODEL -  Model retrieved from: " + path_to_model, logfile)


class LossHistory_multiDice6(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.dice = []
        self.metrics = []

    def on_batch_end(self, batch, logs={}):
        self.dice = []
        self.losses.append(logs.get('loss'))
        self.dice.append(logs.get('dice_coef_multilabel0'))
        self.dice.append(logs.get('dice_coef_multilabel1'))
        self.dice.append(logs.get('dice_coef_multilabel2'))
        self.dice.append(logs.get('dice_coef_multilabel3'))
        self.dice.append(logs.get('dice_coef_multilabel4'))
        self.dice.append(logs.get('dice_coef_multilabel5'))
        self.metrics.append(self.dice)



class LossHistory_multiDice2(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.dice = []
        self.metrics = []

    def on_batch_end(self, batch, logs={}):
        self.dice = []
        self.losses.append(logs.get('loss'))
        self.dice.append(logs.get('dice_coef_multilabel0'))
        self.dice.append(logs.get('dice_coef_multilabel1'))
        self.metrics.append(self.dice)

        
        
################################### SEGMENTATION FUNCTIONS ##################################################


def fullSegmentation_Flexible(wd, penalty_MATRIX, dice_compare, dsc, model, testChannels, testLabels,TPM_channel, subjectIndex, output_classes, dpatch, size_minibatches,logfile, epoch, use_coordinates, saveSegmentation = False, full_evaluation = False,debug=False):    

    subjectIndex = [subjectIndex]
    subject_channels = []
    for modality in testChannels:
      channelsFile = open(modality,"r")   
      ch = channelsFile.readlines()
      subject_channels.append(ch[subjectIndex[0]][:-1])#
      channelsFile.close()
    my_logger('Segmenting subject with channels: ' + str(subject_channels), logfile)  
    
    subID = '_'.join(subject_channels[0].split('/')[-2:])
    
    images = []
    for channel in subject_channels: 
      proxy_img = nib.load(channel)
      X = proxy_img.get_data()
      images.append(X)
    X = np.stack(images, axis=3)

    shape = X.shape
    original_shape = shape

    tpm_nii = nib.load(TPM_channel)      
    TPM_data = tpm_nii.get_data()  
    # Because the TPM has a different size, have to resize it to have the same size as the input MRI.
    # Actually I need to do this too for extracting patches. Else the center is right, but the dimensions are wrong!
    # Need to resize the TPM for each MRI input! 
    TPM_data.shape
    TPM_data = resize(TPM_data, original_shape[:-1], order=1, preserve_range=True, anti_aliasing=True)

    Res = X[:,:,:,0] + TPM_data*10
    img = nib.Nifti1Image(Res, proxy_img.affine)
    nib.save(img, '/home/andy/projects/mskProj/DeepPriors_package/test_t1-TPM.nii')
    
    affine = proxy_img.affine
    if shape[0]*shape[1]*shape[2] > 59*211*211:  # if shape exceeeds 55*261*261
      # Set boundaries for maximum allowed shape
      a = np.max([0,(shape[0] - 59)])/2   
      b = np.max([0,(shape[1] - 211)])/2
      c = np.max([0,(shape[2] - 211)])/2    
      X = X[a:shape[0]-a,:,:,:]
      X = X[:,b:shape[1]-b,:,:]
      X = X[:,:,c:shape[1]-c,:]
  
      TPM_data = TPM_data[a:shape[0]-a,:,:]
      TPM_data = TPM_data[:,b:shape[1]-b,:]
      TPM_data = TPM_data[:,:,c:shape[1]-c]
           
    shape = X.shape
    X = X.reshape( (1,) + shape)
    TPM_data = TPM_data[6:TPM_data.shape[0]-6, 33:TPM_data.shape[1]-33, 33:TPM_data.shape[2]-33]
      
    if use_coordinates:
      coords_shape = [X.shape[1] - 12 ,X.shape[2] - 66, X.shape[3] - 66]
      y_coords = np.tile(np.array([range(6,coords_shape[2]+6)]).transpose(), (1,coords_shape[1]))
      y_coords = y_coords/float(X.shape[2])
      #z_coords = np.tile(range(33,coords_shape[1]+33), (coords_shape[2],1))
      #z_coords = z_coords/float(X.shape[3])
      y_coords = np.repeat(y_coords[np.newaxis, :,: ], coords_shape[0], axis=0)    
      #z_coords = np.repeat(z_coords[np.newaxis, :,: ], coords_shape[0], axis=0)   
      y_coords = y_coords.reshape((1,) + y_coords.shape + (1,))
      #z_coords = z_coords.reshape((1,)+ z_coords.shape + (1,))  
      #print('X shape: {} , y_coords shape {} , z_coords shape {}'.format(X_padded.shape, y_coords.shape, z_coords.shape))
      
    T1post = X[:,:,:,:,0].reshape(X[:,:,:,:,0].shape + (1,))
    T1pre = X[:,:,:,:,1].reshape(X[:,:,:,:,1].shape + (1,))
    T2 = X[:,:,:,:,2].reshape(X[:,:,:,:,2].shape + (1,))
    TPM_data = TPM_data.reshape((1,) + TPM_data.shape + (1,))
    
    if debug:
      img = nib.Nifti1Image(y_coords[0,:,:,:,0], np.diag((1,1,1,0)))
      nib.save(img,'/home/andy/projects/mskProj/DeepPriors_package/y_coords_fullSegm.nii' )
      img = nib.Nifti1Image(T1post[0,:,:,:,0], np.diag((1,1,1,0)))
      nib.save(img,'/home/andy/projects/mskProj/DeepPriors_package/T1post_fullSegm.nii' )
      img = nib.Nifti1Image(T1pre[0,:,:,:,0], np.diag((1,1,1,0)))
      nib.save(img,'/home/andy/projects/mskProj/DeepPriors_package/T1pre_fullSegm.nii' )
      img = nib.Nifti1Image(T2[0,:,:,:,0], np.diag((1,1,1,0)))
      nib.save(img,'/home/andy/projects/mskProj/DeepPriors_package/T2_fullSegm.nii' )
          
      
    yhat = model.predict([T1post, T1pre, T2, TPM_data])

    #y = np.argmax(yhat, axis=4)   # For classification output
    y = yhat[:,:,:,:,1]            # For logits for class 2
    print('y shape: {}'.format(y.shape))
    y = y.reshape(y.shape[1],y.shape[2],y.shape[3])
    #y = y.reshape(shape[0]-24,shape[1]-78,shape[2]-78)
    

    y_out = np.zeros((original_shape[0],original_shape[1],original_shape[2]))

    try:
        y_out[  :, abs(original_shape[1] -y.shape[1])/2:original_shape[1] - abs(original_shape[1] -y.shape[1])/2,
                   abs(original_shape[2] -y.shape[2])/2:original_shape[2] - abs(original_shape[2] -y.shape[2])/2] = y[abs(original_shape[0] -y.shape[0])/2:y.shape[0] - abs(original_shape[0] -y.shape[0])/2,:,:]
    
    except:
        y_out[abs(original_shape[0] -y.shape[0])/2:original_shape[0] - abs(original_shape[0] -y.shape[0])/2, 
              abs(original_shape[1] -y.shape[1])/2:original_shape[1] - abs(original_shape[1] -y.shape[1])/2,
              abs(original_shape[2] -y.shape[2])/2:original_shape[2] - abs(original_shape[2] -y.shape[2])/2] = y
    
    img = nib.Nifti1Image(y_out, affine)
    segmentationName = '/predictions/' + subID + str(epoch)
    output = wd +'/' + segmentationName + '.nii'
    nib.save(img, output)
    #my_logger('Saved segmentation of subject at: ' + output, logfile)
      
#    
#penalty_MATRIX = cfg.penalty_MATRIX
#TPM_channel = cfg.TPM_channel
#testChannels = cfg.testChannels
#testLabels = cfg.testLabels
#output_classes = cfg.output_classes
#segmentation_dpatch = cfg.segmentation_dpatch
#size_test_minibatches =  cfg.size_test_minibatches
#output_probability =  cfg.output_probability
#saveSegmentation =  cfg.saveSegmentation    
#size_minibatches = cfg.size_test_minibatches
#subjectIndex = 0
#use_coordinates= cfg.use_coordinates

def fullSegmentation(wd, penalty_MATRIX, TPM_channel, dice_compare, dsc,foreground_percent_list, model, testChannels, testLabels, subjectIndex, output_classes, segmentation_dpatch, size_minibatches,output_probability, use_coordinates,logfile, epoch, saveSegmentation = False):    
    output_dpatch = segmentation_dpatch[0] - 12, segmentation_dpatch[1] - 66, segmentation_dpatch[2] - 66
    if len(testLabels) == 0:
        dice_compare = False  
    if output_probability:
        dice_compare = False        
    subjectIndex = [subjectIndex]
    num_channels = len(testChannels)
    firstChannelFile = open(testChannels[0],"r")   
    ch = firstChannelFile.readlines()
    subjectGTchannel = ch[subjectIndex[0]][:-1]
    subID = subjectGTchannel.split('/')[-2] + '_' + subjectGTchannel.split('/')[-1].split('.nii')[0]
    my_logger('Segmenting subject with channel: ' + str(subjectGTchannel), logfile)  
    segmentationName = '/predictions/' + subID + '_epoch' + str(epoch)
    output = wd +'/' + segmentationName + '.nii.gz'
    
    if os.path.exists(output):
      print('Segmentation already done. Skip')
      return None
    
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
    TPM_patches, labels, voxelCoordinates, spatial_coordinates, shape, _ = sampleTestData(TPM_channel, testChannels, testLabels, subjectIndex, output_classes, output_dpatch, shape, use_coordinates,logfile)    
    affine = np.diag(list(target_res) + [0])        
    n_minibatches = 0 # min(0,len(voxelCoordinates)/size_minibatches) 
    
    
    total_number_of_patches = (len(voxelCoordinates)-n_minibatches*size_minibatches)  
    
    #########################################################################
    patches = np.zeros((total_number_of_patches,segmentation_dpatch[0],segmentation_dpatch[1],segmentation_dpatch[2],num_channels),dtype='float16')
    for i in range(len(testChannels)):
        patches[:,:,:,:,i] = extractImagePatch(testChannels[i], subjectIndex, patches, [voxelCoordinates], segmentation_dpatch, debug=False)         
    
    INPUT_DATA = []  
    
    # Context
    context = np.array(patches[:,:,:,:,0],'float')
    context = resize(image=context, order=1, 
                         output_shape=(context.shape[0],context.shape[1],context.shape[2]/3,context.shape[3]/3), 
                         anti_aliasing=True, preserve_range=True )
    INPUT_DATA.append(context.reshape(context.shape + (1,)))        
    
    for jj in range(patches.shape[-1]):
        INPUT_DATA.append(patches[:,:,:,:,jj].reshape(patches[:,:,:,:,jj].shape + (1,)))  
    if len(TPM_patches) > 0:
        INPUT_DATA.append(TPM_patches[:,:,:,:].reshape(TPM_patches[:,:,:,:].shape + (1,)))   
    if len(spatial_coordinates) > 0:
        INPUT_DATA.append(spatial_coordinates)    

    print("Extracted patches for full image segmentation.")
    #########################################################################
      
    prediction = model.predict(INPUT_DATA, verbose=1)
    indexes = []
    if output_probability:  
        class_pred = prediction[:,:,:,:,1]
    else:  
        class_pred = np.argmax(prediction, axis=4)
    indexes.extend(class_pred)     
    del patches        
           
    head = np.ones(shape, dtype=np.float32)  # same size as input head, start index for segmentation start at 26,26,26, rest filled with zeros....
    i = 0
    for x,y,z in voxelCoordinates:
        patch_shape = head[x-output_dpatch[0]/2:min(x+(output_dpatch[0]/2+output_dpatch[0]%2), shape[0]),
                           y-output_dpatch[1]/2:min(y+(output_dpatch[1]/2+output_dpatch[1]%2), shape[1]),
                           z-output_dpatch[2]/2:min(z+(output_dpatch[2]/2+output_dpatch[2]%2), shape[2])].shape
        #print(np.array(indexes[i])[0:patch_shape[0], 0:patch_shape[1],0:patch_shape[2]])
        head[x-output_dpatch[0]/2:min(x+(output_dpatch[0]/2+output_dpatch[0]%2), shape[0]),
             y-output_dpatch[1]/2:min(y+(output_dpatch[1]/2+output_dpatch[1]%2), shape[1]),
             z-output_dpatch[2]/2:min(z+(output_dpatch[2]/2+output_dpatch[2]%2), shape[2])] = np.array(indexes[i])[0:patch_shape[0], 
                                                                                                                   0:patch_shape[1],
                                                                                                                   0:patch_shape[2]]
        i = i+1
    img = nib.Nifti1Image(head, affine)
        
    if dice_compare:
      LABEL_CHANNEL = open(testLabels).readlines()[subjectIndex[0]][:-1]
      print(LABEL_CHANNEL)
      if 'BENIGN' in LABEL_CHANNEL:
        dice_compare = False
        label_data = np.zeros((img.shape))
        foreground_percent = np.sum(head)/float(head.size)    
        my_logger('foreground_percent {}'.format(foreground_percent), logfile)
        foreground_percent_list.append(foreground_percent)    
      elif np.sum(nib.load(LABEL_CHANNEL).get_data()) == 0:    
        label_data = np.zeros((img.shape))
      else:
        label_data = nib.load(LABEL_CHANNEL).get_data()    
        if label_data.shape != shape:
          label_data = resize(label_data, output_shape=shape, preserve_range=True, anti_aliasing=True, order=0)
        
        if np.any(np.isnan(label_data)):
          label_data[np.isnan(label_data)] = np.nanmin(label_data)
        # Get only segmented slice
        try:  
          slice_of_interest = np.argwhere(label_data>0)[0][0]
          label_data = label_data[slice_of_interest, :, :]
          head = head[slice_of_interest,:,:]
          
        except:
          my_logger('Target label removed on resizing, subject: {}'.format(LABEL_CHANNEL),logfile)
          dice_compare = False
                
        score = generalized_dice_completeImages(head, label_data)
        dsc.append(score[0])
        print(dsc[-1])
        print('per class dice score: {}'.format(score[1]))
        print('mean DCS so far:' + str(np.mean(dsc)))
        my_logger('DCS ' + str(dsc[-1]),logfile)
    
    if(saveSegmentation):
        nib.save(img, output)
        my_logger('Saved segmentation of subject at: ' + output, logfile)
    
    

def segment(configFile,workingDir):

    path = '/'.join(configFile.split('/')[:-1])
    configFileName = configFile.split('/')[-1][:-3]   
    sys.path.append(path)
    cfg = __import__(configFileName)
           
    model_epoch = int(cfg.path_to_model.split('.')[-2][cfg.path_to_model.split('.')[-2].find('epoch') + 5 : ]) + 1
        
    os.chdir(workingDir + '/training_sessions/')
    session = cfg.session
    wd = workingDir + '/training_sessions/' +session
    print('\n CURRENTLY IN SESSION {} \n'.format(session))
    if not os.path.exists(wd):    
        os.mkdir(session)
        os.mkdir(session + '/models')
        os.mkdir(session + '/predictions')
    os.chdir(wd)
    
    logfile = 'segmentations.log'
    dice_compare = False
    #cfg.TPM_channel = workingDir + cfg.TPM_channel
    cfg.segmentChannels = [workingDir + x for x in cfg.segmentChannels]
    if len(cfg.segmentLabels) > 0:
      cfg.segmentLabels = workingDir + cfg.segmentLabels 
      dice_compare = True
    if len(cfg.TPM_channel) != 0:
      cfg.TPM_channel = workingDir + cfg.TPM_channel
      
    if cfg.output_classes == 6:
	try:
	    from MultiPriors_MSKCC_MultiScale import dice_coef_multilabel6, dice_coef_multilabel0,dice_coef_multilabel1,dice_coef_multilabel2,dice_coef_multilabel3,dice_coef_multilabel4,dice_coef_multilabel5
	    my_custom_objects = {'dice_coef_multilabel6':dice_coef_multilabel6,
				     'dice_coef_multilabel0':dice_coef_multilabel0,
				     'dice_coef_multilabel1':dice_coef_multilabel1,
				     'dice_coef_multilabel2':dice_coef_multilabel2,
				     'dice_coef_multilabel3':dice_coef_multilabel3,
				     'dice_coef_multilabel4':dice_coef_multilabel4,
				     'dice_coef_multilabel5':dice_coef_multilabel5}
		#custom_metrics =[dice_coef_multilabel6,dice_coef_multilabel0,dice_coef_multilabel1,dice_coef_multilabel2,dice_coef_multilabel3,dice_coef_multilabel4,dice_coef_multilabel5]
		#my_custom_objects = dict(zip(np.sort(my_custom_objects.keys()), custom_metrics))
	    model = load_model(cfg.path_to_model, custom_objects = my_custom_objects )
	except:
	    from MultiPriors_MSKCC_MultiScale import w_dice_coef_multilabel6, dice_coef_multilabel0,dice_coef_multilabel1,dice_coef_multilabel2,dice_coef_multilabel3,dice_coef_multilabel4,dice_coef_multilabel5
	    my_custom_objects = {'w_dice_coef_multilabel6':w_dice_coef_multilabel6,
					     'dice_coef_multilabel0':dice_coef_multilabel0,
					     'dice_coef_multilabel1':dice_coef_multilabel1,
					     'dice_coef_multilabel2':dice_coef_multilabel2,
					     'dice_coef_multilabel3':dice_coef_multilabel3,
					     'dice_coef_multilabel4':dice_coef_multilabel4,
					     'dice_coef_multilabel5':dice_coef_multilabel5}
    elif cfg.output_classes == 2:
        try:
            from MultiPriors_MSKCC_MultiScale import Generalised_dice_coef_multilabel2, dice_coef_multilabel0,dice_coef_multilabel1
            my_custom_objects = {'Generalised_dice_coef_multilabel2':Generalised_dice_coef_multilabel2,
				     'dice_coef_multilabel0':dice_coef_multilabel0,
				     'dice_coef_multilabel1':dice_coef_multilabel1}
        except:
            from MultiPriors_MSKCC_MultiScale import w_dice_coef_multilabel2, dice_coef_multilabel0,dice_coef_multilabel1
            my_custom_objects = {'w_dice_coef_multilabel2':w_dice_coef_multilabel2,
				     'dice_coef_multilabel0':dice_coef_multilabel0,
				     'dice_coef_multilabel1':dice_coef_multilabel1}
        model = load_model(cfg.path_to_model, custom_objects = my_custom_objects )

    full_segm_DICE = []
    np.set_printoptions(precision=3)

    print("------------------------------------------------------")
    print("                 WHOLE SCAN SEGMENTATION")
    print("------------------------------------------------------")
    dsc = []
    foreground_percent_list = []
    with open(cfg.segmentChannels[0]) as vl:
        n_segmentSubjects = len(vl.readlines())
    if cfg.test_subjects > n_segmentSubjects:
        print("Given number of subjects for test set (" + str(cfg.test_subjects) +") is larger than the amount of \
        subjects in test set (" +str(n_segmentSubjects)+ ")")
        cfg.test_subjects = n_segmentSubjects
        print('Using {} number of test subjects'.format(n_segmentSubjects))
    if len(cfg.list_subjects_fullSegmentation) == 0:
      list_subjects_fullSegmentation = range(cfg.test_subjects)
    else:
      list_subjects_fullSegmentation = cfg.list_subjects_fullSegmentation
    for subjectIndex in list_subjects_fullSegmentation: 
        
        if cfg.full_segmentation_patches:
            fullSegmentation(wd, cfg.penalty_MATRIX, cfg.TPM_channel, dice_compare, dsc, foreground_percent_list, model, cfg.segmentChannels, cfg.segmentLabels, subjectIndex, \
            cfg.output_classes, cfg.segmentation_dpatch, cfg.size_test_minibatches,cfg.output_probability, cfg.use_coordinates, logfile, model_epoch, cfg.saveSegmentation)
        else:
            fullSegmentation_Flexible(wd, cfg.penalty_MATRIX, dice_compare, dsc, model, cfg.segmentChannels, cfg.segmentLabels,cfg.TPM_channel, subjectIndex, \
            cfg.output_classes, cfg.dpatch, cfg.size_test_minibatches, logfile, model_epoch,cfg.use_coordinates, cfg.saveSegmentation)    
        
        my_logger('--------------- TEST EVALUATION ---------------', logfile)
        my_logger('          Full segmentation evaluation of subject' + str(subjectIndex), logfile)
        if dice_compare: my_logger('DCS ' + str(dsc[-1]),logfile)

    my_logger('         FULL SEGMENTATION SUMMARY STATISTICS ', logfile)
    if dice_compare: 
      full_segm_DICE.append(np.mean(dsc))    
      my_logger('Overall DCS:   ' + str(full_segm_DICE[-1]),logfile)
      my_logger('foreground_percent {}'.format(np.mean(foreground_percent_list)), logfile)
      plt.hist(dsc, edgecolor = 'black')
      #plt.axvline(np.mean(dsc), color = 'red', linewidth = 3)
      #plt.axvline(0.89, color = 'b', linestyle='dashed', linewidth = 3)
      plt.xlabel('Dice score')
      plt.ylabel('Frequency')
      plt.title('Dice score distribution')

############################# MODEL TRAINING AND VALIDATION FUNCTIONS ############################################


def train_validate_model_on_batch(model,batch,labels,TPM_patches,coords,size_minibatches,history,losses,metrics,output_classes,logfile=0, TRAINING_FLAG=True, verbose=False):
    batch_performance = []   
    INPUT_DATA = []
    
    # Context
    context = np.array(batch[:,:,:,:,0],'float')
    context = resize(image=context, order=1, 
                         output_shape=(context.shape[0],context.shape[1],context.shape[2]/3,context.shape[3]/3), 
                         anti_aliasing=True, preserve_range=True )
    INPUT_DATA.append(context.reshape(context.shape + (1,)))    
    
    for jj in range(batch.shape[-1]):
      INPUT_DATA.append(batch[:,:,:,:,jj].reshape(batch[:,:,:,:,jj].shape + (1,)))
      
    if len(TPM_patches) > 0:
      INPUT_DATA.append(TPM_patches[:,:,:,:].reshape(TPM_patches[:,:,:,:].shape + (1,)))   

    if len(coords) > 0:
      #coords = coords.reshape( (coords.shape[0],) + (1,) + coords.shape[1:] )  
      INPUT_DATA.append(coords)

    ######### TRAINING ###########
    if TRAINING_FLAG:
        print('Training..')
        model.fit(INPUT_DATA, labels, verbose = 1, callbacks = [history], batch_size = size_minibatches)

        if verbose:
          freq = classesInSample(labels, output_classes)
          if logfile != 0:
            my_logger("Sampled following number of classes in training MINIBATCH: " + str(freq), logfile)
      
        if logfile != 0:
            output_results = zip(['Train cost and metrics     ']*len(history.losses), history.losses, history.metrics)
            for line in output_results:
                my_logger(' '.join(map(str, line)),logfile)
            
    ######### VALIDATION ###########
    else:
        print('Validation..')
        batch_performance.append(model.evaluate(INPUT_DATA, labels, verbose=1, batch_size = size_minibatches))
        
    del batch
    del labels
    if TRAINING_FLAG:
        return history.losses, history.metrics
    else:    
        val_performance = np.mean(batch_performance, 0)
        my_logger('Validation cost and accuracy ' + str(val_performance),logfile)            
        return list(val_performance)   


################################ MAIN TRAINING FUNCTION ###########################################
#configFile = '/home/deeperthought/Projects/MultiPriors_MSKCC/configFiles/configFile_MultiPriors_v2_F4.py'
#workingDir = '/home/deeperthought/Projects/MultiPriors_MSKCC/'

def train_test_model(configFile, workingDir):
    print(configFile)
    path = '/'.join(configFile.split('/')[:-1])
    print(path)
    configFileName = configFile.split('/')[-1][:-3]   
    sys.path.append(path)
    #sys.path.append(path.replace('configFiles','scripts'))
    cfg = __import__(configFileName)
    if len(cfg.TPM_channel) != 0:
      cfg.TPM_channel = workingDir + cfg.TPM_channel
    cfg.trainChannels = [workingDir + x for x in cfg.trainChannels]
    cfg.trainLabels = workingDir +cfg.trainLabels 
    cfg.testChannels = [workingDir + x for x in cfg.testChannels]
    cfg.testLabels = workingDir + cfg.testLabels
    cfg.validationChannels = [workingDir + x for x in cfg.validationChannels]
    cfg.validationLabels = workingDir +cfg.validationLabels
    
    if cfg.load_model == False:

        if cfg.model == 'MultiPriors_v0':
            from MultiPriors_Models_Collection import MultiPriors_v0
            mp = MultiPriors_v0(cfg.output_classes, cfg.num_channels, cfg.L2, cfg.dropout, cfg.learning_rate, cfg.optimizer_decay, cfg.loss_function)
            model = mp.createModel()            
            model.summary()                             

        elif cfg.model == 'MultiPriors_v1':
            from MultiPriors_Models_Collection import MultiPriors_v1
            mp = MultiPriors_v1(cfg.output_classes, cfg.num_channels, cfg.L2, cfg.dropout, cfg.learning_rate, cfg.optimizer_decay, cfg.loss_function)
            model = mp.createModel()            
            model.summary()           

        elif cfg.model == 'MultiPriors_v2':
            from MultiPriors_Models_Collection import MultiPriors_v2
            mp = MultiPriors_v2(cfg.output_classes, cfg.num_channels, cfg.L2, cfg.dropout, cfg.learning_rate, cfg.optimizer_decay, cfg.loss_function)
            model = mp.createModel()            
            model.summary()        

        elif cfg.model == 'MultiPriors_TEST':
            from MultiPriors_Models_Collection import MultiPriors_v2_ContextOutput
            mp = MultiPriors_v2_ContextOutput(cfg.output_classes, cfg.num_channels, cfg.L2, cfg.dropout, cfg.learning_rate, cfg.optimizer_decay, cfg.loss_function)
            model = mp.createModel()            
            model.summary()     

        elif cfg.model == 'BreastSegmentor_v0':
            from BreastMask_Models_Collection import BreastSegmentor_v0
            mp = BreastSegmentor_v0(cfg.output_classes, cfg.num_channels, cfg.L2, cfg.dropout, cfg.learning_rate, cfg.optimizer_decay, cfg.loss_function)
            model = mp.createModel()            
            model.summary()  
            
        elif cfg.model == 'BreastSegmentor_v1':
            from BreastMask_Models_Collection import BreastSegmentor_v1
            mp = BreastSegmentor_v1(cfg.output_classes, cfg.num_channels, cfg.L2, cfg.dropout, cfg.learning_rate, cfg.optimizer_decay, cfg.loss_function)
            model = mp.createModel()            
            model.summary()              
     
        else: 
            print('ERROR: No model selected.')
            return 0          
        
        
        if cfg.merge_breastMask_model:

            from keras.models import load_model  
            from MultiPriors_Models_Collection import Generalised_dice_coef_multilabel2, dice_coef_multilabel0,dice_coef_multilabel1
            my_custom_objects = {'Generalised_dice_coef_multilabel2':Generalised_dice_coef_multilabel2,
                                 'dice_coef_multilabel0':dice_coef_multilabel0,
                                 'dice_coef_multilabel1':dice_coef_multilabel1}            
            bm_model = load_model(cfg.path_to_breastMask_model, custom_objects = my_custom_objects )            
            preTrained_convLayers = [x for x in bm_model.layers if 'Conv3D' in str(x)]
            preTrained_batchNormLayers = [x for x in bm_model.layers if 'BatchNorm' in str(x)]
            if cfg.Context_parameters_trainable:
                # If fine-tuning allowed, then skip bottleneck of last layer of the breastMask model 
                preTrained_convLayers = preTrained_convLayers[:-1]
                
            #preTrained_convLayers = [x for x in bm_model.layers if 'T1post_Context' in x.name]
            #preTrained_batchNormLayers = [x for x in bm_model.layers if 'BatchNorm' in x.name]   
            newModel_convLayers = [x for x in model.layers if 'T1post_Context' in x.name]           
            newModel_batchNormLayers = [x for x in model.layers if 'BatchNorm' in x.name]            
            
            assert len(preTrained_convLayers ) == len(newModel_convLayers), 'Models have incompatible architecture..'
            assert len(preTrained_batchNormLayers) == len(newModel_batchNormLayers), 'Models have incompatible architecture..'
            print('Transfering weights from breastMask model {} '.format(cfg.path_to_breastMask_model))       
#            for i in range(len(newModel_convLayers)):
#                print('Equal layer {}: {}'.format(newModel_convLayers[i].name, (model.get_layer(newModel_convLayers[i].name).get_weights()[0] == preTrained_convLayers[i].get_weights()[0]).all()))
            for i in range(len(newModel_convLayers)):
                print('Weight transfer of layer : {}'.format(newModel_convLayers[i].name))
                model.get_layer(newModel_convLayers[i].name).set_weights(preTrained_convLayers[i].get_weights())        
                model.get_layer(newModel_convLayers[i].name).trainable = cfg.Context_parameters_trainable
            for i in range(len(newModel_batchNormLayers)):
                print('Weight transfer of layer : {}'.format(newModel_batchNormLayers[i].name))
                model.get_layer(newModel_batchNormLayers[i].name).set_weights(preTrained_batchNormLayers[i].get_weights())        
                
            # Need to re-compile when changing the TRAINABLE attribute:   
            model = multi_gpu_model(model, gpus=4)
            model.compile(loss=Generalised_dice_coef_multilabel2, optimizer=RAdam(lr=cfg.learning_rate), metrics=['acc', dice_coef_multilabel0, dice_coef_multilabel1])
            model.summary()        
                
        start_epoch = 0
        os.chdir(workingDir + '/training_sessions/')
        session = cfg.model + '_' + cfg.dataset + '_' + configFileName + '_' + time.strftime("%Y-%m-%d_%H%M") 
        wd = workingDir + '/training_sessions/' +session
        if not os.path.exists(wd):    
            os.mkdir(session)
            os.mkdir(session + '/models')
            os.mkdir(session + '/predictions')
        os.chdir(wd) 

        CV_FOLDS_ARRAYS_PATH = '/'.join(cfg.trainChannels[0].split('/')[:-1]) + '/arrays/'
        if not os.path.exists(CV_FOLDS_ARRAYS_PATH):
            os.mkdir(CV_FOLDS_ARRAYS_PATH)
        
        #copy(workingDir + configFile[1:], wd)
        copy(configFile, wd)
        logfile = session +'.log'            
        print(model.summary())
        val_performance = []
        from keras.utils import plot_model
        plot_model(model, to_file=wd+'/multiscale_TPM.png', show_shapes=True)
        with open(wd+'/model_summary.txt','w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            model.summary(print_fn=lambda x: fh.write(x + '\n'))
        if len(cfg.comments) > 0:
            f = open('Comments.txt','w')
            f.write(str(cfg.comments))
            f.close()
        
    elif cfg.load_model == True:
        from keras.models import load_model  
        if cfg.loss_function == 'Dice6':
            from MultiPriors_Models_Collection import dice_coef_multilabel6, dice_coef_multilabel0,dice_coef_multilabel1,dice_coef_multilabel2,dice_coef_multilabel3,dice_coef_multilabel4,dice_coef_multilabel5
            my_custom_objects = {'dice_coef_multilabel6':dice_coef_multilabel6,
                                 'dice_coef_multilabel0':dice_coef_multilabel0,
                                 'dice_coef_multilabel1':dice_coef_multilabel1,
                                 'dice_coef_multilabel2':dice_coef_multilabel2,
                                 'dice_coef_multilabel3':dice_coef_multilabel3,
                                 'dice_coef_multilabel4':dice_coef_multilabel4,
                                 'dice_coef_multilabel5':dice_coef_multilabel5}
        elif cfg.loss_function == 'wDice6':
            from MultiPriors_Models_Collection import w_dice_coef_multilabel6, dice_coef_multilabel0,dice_coef_multilabel1,dice_coef_multilabel2,dice_coef_multilabel3,dice_coef_multilabel4,dice_coef_multilabel5
            my_custom_objects = {'w_dice_coef_multilabel6':w_dice_coef_multilabel6,
                                 'dice_coef_multilabel0':dice_coef_multilabel0,
                                 'dice_coef_multilabel1':dice_coef_multilabel1,
                                 'dice_coef_multilabel2':dice_coef_multilabel2,
                                 'dice_coef_multilabel3':dice_coef_multilabel3,
                                 'dice_coef_multilabel4':dice_coef_multilabel4,
                                 'dice_coef_multilabel5':dice_coef_multilabel5}
        elif cfg.loss_function == 'Dice':
            from MultiPriors_Models_Collection import Generalised_dice_coef_multilabel2, dice_coef_multilabel0,dice_coef_multilabel1
            my_custom_objects = {'Generalised_dice_coef_multilabel2':Generalised_dice_coef_multilabel2,
                                 'dice_coef_multilabel0':dice_coef_multilabel0,
                                 'dice_coef_multilabel1':dice_coef_multilabel1}
        elif cfg.loss_function == 'wDice2':
            from DM_MSKCC_Atrous_model import w_dice_coef_multilabel2, dice_coef_multilabel0,dice_coef_multilabel1
            my_custom_objects = {'w_dice_coef_multilabel2':w_dice_coef_multilabel2,
                                 'dice_coef_multilabel0':dice_coef_multilabel0,
                                 'dice_coef_multilabel1':dice_coef_multilabel1}
        model = load_model(cfg.path_to_model, custom_objects = my_custom_objects )
        print('LOADED MODEL FROM SESSION {}'.format(cfg.session))
        session = cfg.session
        start_epoch = int(cfg.path_to_model.split('.')[-2][cfg.path_to_model.split('.')[-2].find('epoch') + 5 : ]) + 1
        #cfg.epochs_for_fullSegmentation = range(start_epoch+1, cfg.epochs)
        os.chdir(workingDir + '/training_sessions/')
        wd = workingDir + '/training_sessions/' +session
        if not os.path.exists(wd):    
            os.mkdir(session)
            os.mkdir(session + '/models')
            os.mkdir(session + '/predictions')
        os.chdir(wd)
        logfile = session +'.log'
        CV_FOLDS_ARRAYS_PATH = '/'.join(cfg.trainChannels[0].split('/')[:-1]) + '/arrays/'
        if not os.path.exists(CV_FOLDS_ARRAYS_PATH):
            os.mkdir(CV_FOLDS_ARRAYS_PATH)
    #################################################################################################
    #                                                                                               #
    #                                         START SESSION                                         #
    #                                                                                               #
    #################################################################################################
    
    # OUTCOMMENTED SO I CAN KEEP USING SAME TRAINING DATA FOR SAME MODEL.
    val_performance = []
    full_segm_DICE = []
    epoch_foreground_percent = []
    losses = []
    metrics = []
    np.set_printoptions(precision=3)

    start_training_session_logger(logfile, cfg.threshold_EARLY_STOP, cfg.TPM_channel, cfg.load_model, cfg.saveSegmentation, cfg.path_to_model, model, \
        cfg.dropout, cfg.trainChannels, cfg.trainLabels, cfg.validationChannels, cfg.validationLabels, \
        cfg.testChannels, cfg.testLabels, cfg.num_iter, cfg.epochs, cfg.n_patches, cfg.n_patches_val, cfg.n_subjects, cfg.samplingMethod_train, \
        cfg.size_minibatches, cfg.n_full_segmentations, cfg.epochs_for_fullSegmentation, cfg.size_test_minibatches)
    # Callback history    
    if cfg.output_classes == 2:
        history = LossHistory_multiDice2() 
    elif cfg.output_classes == 6:
        history = LossHistory_multiDice6()


    
    EARLY_STOP = False    
    
    for epoch in xrange(start_epoch,cfg.epochs):
      t1 = time.time()
      my_logger("######################################################",logfile)
      my_logger("                   TRAINING EPOCH " + str(epoch) + "/" + str(cfg.epochs),logfile)
      my_logger("######################################################",logfile)
              
      ####################### FULL HEAD SEGMENTATION ##############################
                
      if epoch in cfg.epochs_for_fullSegmentation:
        my_logger("------------------------------------------------------", logfile)
        my_logger("                 FULL HEAD SEGMENTATION", logfile)
        my_logger("------------------------------------------------------", logfile)
        dice_compare = True
        dsc = []
        foreground_percent_list = []
        subjectIndex = 0

        with open(cfg.validationLabels) as vl:
            n_valSubjects = len(vl.readlines())
        if cfg.test_subjects > n_valSubjects:
            print("Given number of subjects for test set (" + str(cfg.test_subjects) +") is larger than the amount of \
            subjects in test set (" +str(n_valSubjects)+ ")")
            cfg.test_subjects = n_valSubjects
            cfg.n_full_segmentations = n_valSubjects
            print('Using {} number of test subjects'.format(n_valSubjects))
        if len(cfg.list_subjects_fullSegmentation) == 0:
            #list_subjects_fullSegmentation = sample(range(cfg.test_subjects), cfg.n_full_segmentations)
            if cfg.balanced_sample_subjects:
                proportion_malignants = int(np.ceil(cfg.n_full_segmentations * cfg.proportion_malignants_fullSegmentation))
                labelsFile = [x[:-1] for x in open(cfg.testLabels).readlines()]
                malignant_subjects_index = [labelsFile.index(x) for x in labelsFile if not 'BENIGN' in x]
                benign_subjects_index = list(set(range(len(labelsFile))) - set(malignant_subjects_index))
                list_subjects_fullSegmentation = random.sample(malignant_subjects_index, min(len(malignant_subjects_index), proportion_malignants))
                print('sampling {} malignants from partition'.format(len(list_subjects_fullSegmentation)))
                list_subjects_fullSegmentation.extend(random.sample(benign_subjects_index, cfg.n_full_segmentations - len(list_subjects_fullSegmentation)))
                random.shuffle(list_subjects_fullSegmentation)
            else:
                list_subjects_fullSegmentation = random.sample(xrange(cfg.test_subjects ), cfg.n_full_segmentations)

        else:
            list_subjects_fullSegmentation = cfg.list_subjects_fullSegmentation
            
        
        for subjectIndex in list_subjects_fullSegmentation: 
            t_segment = time.time()
            if cfg.full_segmentation_patches:
                fullSegmentation(wd, cfg.penalty_MATRIX, cfg.TPM_channel, dice_compare, dsc, foreground_percent_list, model, cfg.testChannels, cfg.testLabels, subjectIndex, \
                cfg.output_classes, cfg.segmentation_dpatch, cfg.size_test_minibatches,cfg.output_probability, cfg.use_coordinates, logfile, epoch, cfg.saveSegmentation)
            else:
                fullSegmentation_Flexible(wd, cfg.penalty_MATRIX, dice_compare, dsc, model, cfg.testChannels, cfg.testLabels, cfg.TPM_channel, subjectIndex, \
                cfg.output_classes, cfg.dpatch, cfg.size_test_minibatches, logfile, epoch, cfg.use_coordinates,cfg.saveSegmentation)

            my_logger('--------------- TEST EVALUATION ---------------', logfile)
            my_logger('          Full segmentation evaluation of subject' + str(subjectIndex), logfile)
            #if (dice_compare & len(dsc)>0):
            #  my_logger('DCS ' + str(dsc[-1]),logfile)
            print('Segmentation of subject took {} s'.format(time.time()-t_segment))
        my_logger('         FULL SEGMENTATION SUMMARY STATISTICS ', logfile)
        full_segm_DICE.append(np.mean(dsc))
        epoch_foreground_percent.append(np.mean(foreground_percent_list))            
        my_logger('Epoch_foreground_percent {}'.format(epoch_foreground_percent[-1]), logfile)

        my_logger('Overall DCS:   ' + str(full_segm_DICE[-1]),logfile)
        
        # Function to define if STOP flag goes to True or not, based on difference between last three or two segmentations.
        if len(full_segm_DICE) > 5:                        
            if np.max(np.abs(np.diff([full_segm_DICE[-3], full_segm_DICE[-2], full_segm_DICE[-1]] ))) < cfg.threshold_EARLY_STOP:
                EARLY_STOP = True
            #elif np.max(np.abs(np.diff([full_segm_DICE[-5],full_segm_DICE[-4],full_segm_DICE[-3], full_segm_DICE[-2], full_segm_DICE[-1]] ))) < 0.03:
            #    EARLY_STOP = True

        # Save model if best results achieved
        if len(full_segm_DICE) > 0:
          if np.max(full_segm_DICE) <= full_segm_DICE[-1]:
            my_logger('###### SAVING TRAINED MODEL AT : ' + wd +'/Output/models/'+logfile[12:]+'.h5', logfile)
            model.save(wd+'/models/'+logfile[12:]+'_epoch' + str(epoch) + '.h5')
        if EARLY_STOP:
          my_logger('Convergence criterium met. Stopping training.',logfile)
          break           
      #################################################################################################
      #                                                                                               #
      #                               Training and Validation                                         #
      #                                                                                               #
      #################################################################################################
      if cfg.sample_intensity_based:
#          if epoch > 20:
#            cfg.percentile_voxel_intensity_sample_benigns = 99   # This will sample only from VERY high intensity areas          
#            my_logger('Reached epoch {}, changing sampling benign voxels to voxel intensity > {}'.format(epoch, cfg.percentile_voxel_intensity_sample_benigns ), logfile)             

          if epoch > 10:
            cfg.percentile_voxel_intensity_sample_benigns = 90   # This will sample only from high intensity areas
            my_logger('Reached epoch {}, changing sampling benign voxels to voxel intensity > {}'.format(epoch, cfg.percentile_voxel_intensity_sample_benigns ), logfile)          
       
          elif epoch > 5:
            cfg.percentile_voxel_intensity_sample_benigns = 50   # This will sample only from breast
            my_logger('Reached epoch {}, changing sampling benign voxels to voxel intensity > {}'.format(epoch, cfg.percentile_voxel_intensity_sample_benigns ), logfile)

      ####################################### TRAINING ITERATIONS #################################################################    
      for i in range(0, cfg.num_iter):
          my_logger("                   Batch " + str(i+1) + "/" + str(cfg.num_iter) ,logfile)
          my_logger("###################################################### ",logfile)
                    
          ####################### VALIDATION ON BATCHES ############################      
          print('\n###################### VALIDATION ####################')                                      
          if not cfg.quickmode:   
              with open(cfg.validationLabels) as vl:
                  n_valSubjects = len(vl.readlines())
              if cfg.n_subjects_val > n_valSubjects:
                  print("Given number of subjects for test set (" + str(cfg.n_subjects_val) +") is larger than the amount of subjects in test set (" +str(n_valSubjects)+ ")")
                  cfg.n_subjects_val = n_valSubjects
                  print('Using {} number of test subjects'.format(n_valSubjects))
              valbatch, vallabels, valcoords, val_TPM_patches = sampleTrainData(cfg.validationChannels, CV_FOLDS_ARRAYS_PATH, cfg.validationLabels, cfg.TPM_channel, cfg.n_patches_val, 
                                                                                cfg.n_subjects_val, cfg.dpatch, cfg.output_classes, cfg.samplingMethod_val, cfg.use_coordinates, 
                                                                                cfg.proportion_malignants_to_sample, cfg.percentile_voxel_intensity_sample_benigns, 
                                                                                cfg.data_augmentation, cfg.proportion_to_flip, cfg.percentile_normalization, 
                                                                                cfg.balanced_sample_subjects, logfile)
              shuffleOrder = np.arange(valbatch.shape[0])
              np.random.shuffle(shuffleOrder)
              valbatch = valbatch[shuffleOrder]
              vallabels = vallabels[shuffleOrder] 
              if len(val_TPM_patches) > 0:
                  val_TPM_patches = val_TPM_patches[shuffleOrder] 
              if len(valcoords) > 0:
                  valcoords = valcoords[shuffleOrder]        
              val_performance.append(train_validate_model_on_batch(model, valbatch,  vallabels, val_TPM_patches, valcoords, cfg.size_minibatches_val, history, losses,  metrics, 
                                                                   cfg.output_classes, logfile, TRAINING_FLAG=False))

              del valbatch, vallabels, val_TPM_patches
              
          ####################### TRAINING ON BATCHES ##############################
          print('\n###################### TRAINING ####################')                                                   
          with open(cfg.trainLabels) as vl:
                  n_trainSubjects = len(vl.readlines())                
          if cfg.n_subjects > n_trainSubjects:
              print("Given number of subjects for test set (" + str(cfg.n_subjects) +") is larger than the amount of \
              subjects in test set (" +str(n_trainSubjects)+ ")")
              cfg.n_subjects = n_trainSubjects
              print('Using {} number of test subjects'.format(n_trainSubjects))
          print('sampling {} patches'.format(cfg.n_patches))
          batch = 0
          labels = 0
                    
          my_logger('Sampling data, using intensity percentile {}'.format(cfg.percentile_voxel_intensity_sample_benigns), logfile)
          batch, labels, coords, TPM_patches = sampleTrainData(cfg.trainChannels, CV_FOLDS_ARRAYS_PATH, cfg.trainLabels,cfg.TPM_channel, cfg.n_patches, cfg.n_subjects, cfg.dpatch, cfg.output_classes, 
                                                               cfg.samplingMethod_train, cfg.use_coordinates, cfg.proportion_malignants_to_sample, 
                                                               cfg.percentile_voxel_intensity_sample_benigns,  cfg.data_augmentation, cfg.proportion_to_flip, 
                                                               cfg.percentile_normalization, cfg.balanced_sample_subjects, logfile) 
          if np.any(np.isnan(batch)):
            print('nan found in the input data batch for training..')
            print(batch[np.isnan(batch)].shape)
            batch[np.isnan(batch)] = 0.0
          assert not np.any(np.isnan(batch)), my_logger('STILL NANs!',logfile)            
          if np.any(~ np.isfinite(batch)):
            batch[~ np.isfinite(batch)] = 0.0
          assert np.all(np.isfinite(batch)), my_logger('STILL Non-Finite Values!',logfile)
          print('TRAINING BATCH')
          print('Number of class 0 samples in whole batch: {}'.format(np.sum(labels[:,:,:,:,0])))
          print('Number of class 1 samples in whole batch: {}'.format(np.sum(labels[:,:,:,:,1])))
          if cfg.verbose:
            print('batch.shape {}'.format(batch.shape))
            print('labels.shape {}'.format(labels.shape))
            print('coords.shape {}'.format(coords.shape))
          shuffleOrder = np.arange(batch.shape[0])
          np.random.shuffle(shuffleOrder)
          batch = batch[shuffleOrder]
          labels = labels[shuffleOrder]  
          if len(coords) > 0:
            coords = coords[shuffleOrder]
          if len(TPM_patches) > 0:
            TPM_patches = TPM_patches[shuffleOrder]
          epoch_loss, epoch_metrics = train_validate_model_on_batch(model,batch,labels,TPM_patches, coords,cfg.size_minibatches,history,losses,metrics,cfg.output_classes, logfile)  
          try:
              global_loss = np.concatenate([np.load(wd + '/LOSS.npy'), epoch_loss])
              global_metrics = np.concatenate([np.load(wd + '/METRICS.npy'), epoch_metrics]) 
              np.save(wd + '/LOSS.npy', global_loss)
              np.save(wd + '/METRICS.npy', global_metrics)
          except:
              np.save(wd + '/LOSS.npy', epoch_loss)
              np.save(wd + '/METRICS.npy', epoch_metrics)              
          del batch, labels
          # For large datasets, save model after every 'epoch'
          print('Saving model..')
          model.save(wd+'/models/'+logfile[12:]+'_epoch' + str(epoch) + '.h5')        
      my_logger('Total training this epoch took ' + str(round(time.time()-t1,2)) + ' seconds',logfile)

    if cfg.output_classes == 6:
    	plot_training(session,losses, metrics, val_performance, full_segm_DICE, smooth=20, loss_name = [cfg.loss_function], class_names = ['Air','GM','WM','CSF','Bone','Skin'])
    elif cfg.output_classes == 2:
    	plot_training(session,losses, metrics, val_performance, full_segm_DICE, smooth=20, loss_name = [cfg.loss_function], class_names = ['Background','Lesion'])
    plt.savefig(wd + '/' + session + '.png')
    plt.close()




