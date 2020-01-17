#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 12:27:50 2019

@author: deeperthought
"""

import os
import nibabel as nib
import numpy as np
import pandas as pd
from skimage.transform import resize
import matplotlib.pyplot as plt

LABELS = pd.read_csv('/home/deeperthought/Projects/MultiPriors_MSKCC/CV_folds/CV_alignedNii-Aug2019/val_labels.txt', header=None)

#PATH = '/home/deeperthought/Projects/MultiPriors_MSKCC/training_sessions/MultiPriors_v2_MSKCC_configFile_MultiPriors_v2_2019-10-25_1657/predictions/'

#PATH = '/home/deeperthought/Projects/MultiPriors_MSKCC/training_sessions/MultiPriors_v0_MSKCC_configFile_MultiPriors_v0_2019-10-28_1740/predictions/'

PATH = '/home/deeperthought/Projects/MultiPriors_MSKCC/training_sessions/MultiPriors_v2_MSKCC_configFile_MultiPriors_v2_F4_2019-11-04_1014/predictions/'

def generalized_dice_completeImages(img1,img2):
    assert img1.shape == img2.shape, 'Images of different size!'
    #assert (np.unique(img1) == np.unique(img2)).all(), 'Images have different classes!'
    classes = np.array(np.unique(img1), dtype='int8')   
    if len(classes) < len(np.array(np.unique(img2), dtype='int8')):
      classes = np.array(np.unique(img2), dtype='int8')   
    dice = []
    for i in classes:
        dice.append(2*np.sum(np.multiply(img1==i,img2==i))/float(np.sum(img1==i)+np.sum(img2==i)))   
    return np.sum(dice)/len(classes), [round(x,2) for x in dice]


LABELS[0][0]


EPOCHS = os.listdir(PATH)
EPOCHS = [x for x in EPOCHS if 'epoch' in x and not x.endswith('.nii.gz')]
EPOCHS.sort()

myDict = {}


for epoch in EPOCHS:
    myDict[epoch] = []
    print(epoch)
    scans = os.listdir(PATH + epoch)
    scans.sort()
    for scan in scans:
        print(scan)
        exam = 'MSKCC' + scan.split('MSKCC')[-1][:24]
        side = scan.split('T1_')[-1][:5].replace('_','')
        target_label = LABELS.loc[LABELS[0].str.contains(exam) * LABELS[0].str.contains(side)].values[0][0]
        gt = nib.load(target_label).get_data()
        seg = nib.load(PATH + epoch +'/' + scan).get_data()
        
        
        slice_indx1 = np.argwhere(gt > 0)[0][0]  
    
        if seg.shape != gt.shape:
          seg = resize(seg, gt.shape,order=1, anti_aliasing=True, preserve_range=True)

        thr = np.mean(seg[slice_indx1][gt[slice_indx1] == 1])
        seg[seg < thr] = 0
        seg[seg >= thr] = 1         
        dice = generalized_dice_completeImages(gt[slice_indx1], seg[slice_indx1])[0]
        myDict[epoch].append(dice)
        #plt.imshow(seg[slice_indx1] + gt[slice_indx1])


df = pd.DataFrame(myDict)



np.mean(df, axis=0)
df.T.plot(color='lightblue', legend=False, figsize=(8,4))
df.mean().plot()
plt.savefig(PATH.replace('predictions/' , 'val_fullScanSegmentation'), dpi=200)
