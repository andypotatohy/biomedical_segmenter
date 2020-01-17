#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 14:18:08 2019

@author: andy
"""

# Create a malignant tumor TPM

# ATTENTION: Use only labels from the TRAINING set to do the TPM! 

import nibabel as nib
import numpy as np
from skimage.transform import resize
from scipy.ndimage.filters import gaussian_filter

labels_list = open('/home/hirsch/Documents/projects/Breast_segmentation/DeepPriors_package/CV_folds/train_labels.txt').readlines()

###########################################################

labels = [x[:-1] for x in labels_list if 'BENIGN' not in x ]
TPM = np.zeros((50,100,100))
n=0
tot = len(labels)

while n < tot:
    line = labels[n]
    print('\r {}%'.format(n*100./tot))
    nii = nib.load(line)
    data = nii.get_data()
    data[data>0] = 100
    shape = data.shape
    data = resize(data, (50,100,100), order=1, preserve_range=True, anti_aliasing=True)
    data = data/max(1,np.max(data))
    TPM += data
    n+=1

TPM_scaled = TPM/np.max(TPM)#/tot
np.max(TPM_scaled)
log_TPM = np.log(TPM_scaled + 1e-06)
log_TPM = gaussian_filter(log_TPM, sigma=2)

img = nib.Nifti1Image(log_TPM, np.diag((1,1,1,0)))
nib.save(img, '/home/hirsch/Documents/projects/Breast_segmentation/DeepPriors_package/log_TPM_gaussian_reflected.nii') 



'''
# test
TPM = np.zeros((50,100,100))
line = '/media/andy/RNN_training/LABELS_segmentation/rrMSKCC_16-328_1_00416_20050430_label_ras.nii'    
nii = nib.load(line)
data = nii.get_data()
data[data>0] = 100
data = resize(data, (50,100,100), order=3, preserve_range=True, anti_aliasing=True)
np.sum(data)
TPM += data
img = nib.Nifti1Image(TPM, np.diag((1,1,1,0)))
nib.save(img, '/home/andy/projects/mskProj/DeepPriors_package/test_t1-label.nii')

TPM = np.zeros((50,100,100))
line = '/media/andy/RNN_training/Normalized/MSKCC_16-328_1_00416_20050430/rt1post-l_stand.nii'
nii = nib.load(line)
data = nii.get_data()
data = resize(data, (50,100,100), order=1, preserve_range=True, anti_aliasing=True)
TPM += data
img = nib.Nifti1Image(TPM, np.diag((1,1,1,0)))
nib.save(img, '/home/andy/projects/mskProj/DeepPriors_package/test_t1.nii')






n=0
tot = len(labels)
for line in labels:
    print('\r {}%'.format(n*100./tot))
    nii = nib.load(line)
    data = nii.get_data()
    shape = data.shape
    TPM[:shape[0],:shape[1],:shape[2]] += data
    n+=1
    
    
img = nib.Nifti1Image(TPM, np.diag((1,1,1,0)))
nib.save(img, '/home/andy/projects/mskProj/DeepPriors_package/MSKCC_TPM_add.nii')


TPM[TPM>0] = 1
img = nib.Nifti1Image(TPM, np.diag((1,1,1,0)))
nib.save(img, '/home/andy/projects/mskProj/DeepPriors_package/MSKCC_TPM_congruence.nii')
'''