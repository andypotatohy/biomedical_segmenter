#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 15:30:20 2019

@author: deeperthought
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import os 
import pandas as pd

import matplotlib.backends.backend_pdf



SESSION_PATH = '/home/deeperthought/Projects/MultiPriors_MSKCC/training_sessions/BreastSegmentor_v1_breastMask_configFile_BreastMask-Model_2019-10-21_1932/'#'/home/deeperthought/Projects/MultiPriors_MSKCC/training_sessions/BreastSegmentor_v0_breastMask_configFile_BreastMask-Model_2019-10-19_1654_Training-6/predictions/'

NII_PATH = '/home/deeperthought/kirby_MSK/alignedNii-Aug2019/'

pdf = matplotlib.backends.backend_pdf.PdfPages(SESSION_PATH + "/SEGMENTATION_RESULTS.pdf")


SEGMENTATIONS_PATH = SESSION_PATH + 'predictions/'
masks = [x for x in os.listdir(SEGMENTATIONS_PATH) if x.endswith('.nii.gz')]
results = pd.DataFrame(masks)
results.columns = ['File']
results['Exam'] = results['File'].apply(lambda x : x.split('_T1')[0])
results['Scan'] = results['File'].apply(lambda x : 'T1' + x.split('T1')[-1].split('_epoch')[0] + '.nii')

RESULTS_IMG = []

myFigures = []

for indx, row in results.iterrows():
    print(indx)
    mask = SEGMENTATIONS_PATH + row['File']
    nii = NII_PATH + row['Exam'] + '/' + row['Scan']

    img = nib.load(nii).get_data()
    seg = nib.load(mask).get_data()


    
    if img.shape != seg.shape:
        img = resize(img, seg.shape, anti_aliasing=True, preserve_range=True)

    
    mySlice = np.random.choice(xrange(img.shape[0]))
    img = np.rot90(img[mySlice])
    seg = np.rot90(seg[mySlice])
        
    fig,ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.imshow(seg, cmap='Reds', alpha=0.5)
    
    myFigures.append(fig)
    #plt.imshow(np.rot90(img + seg*np.max(img)), cmap='gray')
    #plt.savefig(SESSION_PATH + 'Result_{}-{}.png'.format(row['Exam'],row['Scan']), dpi=200)
    #plt.close()
    
for fig in myFigures: ## will open an empty extra figure :(
    pdf.savefig( fig )
pdf.close()    
    