# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 18:57:28 2018

@author: hirsch
"""

import nibabel as nib
import numpy as np


ALL_PATH_MRI = open('/home/hirsch/Documents/projects/MSKCC/all_t1_post.txt').readlines()
ALL_PATH_LABEL = open('/home/hirsch/Documents/projects/MSKCC/all_labels.txt').readlines()


for i in range(len(ALL_PATH_LABEL)):

    PATH_MRI = ALL_PATH_MRI[i][0:-1]
    PATH_LABEL = ALL_PATH_LABEL[i][0:-1]
    
    br = nib.load(PATH_MRI)
    gt = nib.load(PATH_LABEL)
    
    brd = br.get_data()
    gtd = gt.get_data()
    
    '''
    if len(brd.shape) == 4:
        brd = brd[:,:,:,0]
        img = nib.Nifti1Image(brd, br.affine)
        out = '/'.join(PATH_MRI.split('/')[0:-1])
        nib.save(img,out + '/First_post.nii')
    '''
    
    #assert all(np.unique(gtd.shape) == np.unique(brd.shape)), 'Label and MRI do not have the same shape. Likely due to inclusion of both breasts.'
    
    if gtd.shape != brd.shape:
        gtd = np.swapaxes(gtd,1,2)
        gtd = np.swapaxes(gtd,0,1)
    
    #gtd = np.flip(gtd,0)
    gtd = np.flip(gtd,1)
    #gtd = np.flip(gtd,2)
    
    if np.sum(np.unique(gtd.shape) == np.unique(brd.shape)):
        back = np.zeros((brd.shape))
        back[0:gtd.shape[0], 0:gtd.shape[1], 0:gtd.shape[2]] = gtd
        gtd = back
        
    else:
        continue
    
    out = '/'.join(PATH_LABEL.split('/')[0:-1])
    img = nib.Nifti1Image(gtd, br.affine)
    nib.save(img,out+'/GT.nii')




back = np.zeros((brd.shape))

back[gtd.shape]

np.sum(back)

img = nib.Nifti1Image(brd + back, br.affine)

nib.save(img,'/home/hirsch/Documents/projects/Breast_segmentation/Data/162150_5/sum.nii')



# Create one training volume: Will have to modify the voxel coordinate generation function: For positive examples will need to take
# central voxels that are positive. This is actually already implemented.

