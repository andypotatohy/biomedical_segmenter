# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 10:57:48 2019

@author: hirsch
"""
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

FEATURES_PATH = '/media/hirsch/RNN_training/Features_Segmentor/'

feats = [x for x in os.listdir(FEATURES_PATH) if x.endswith('nii.gz')]

df = pd.read_csv('/media/hirsch/RNN_training/All_Triples.csv')
df['scan_ID'] = df.file.apply(lambda x : x.split('triplet')[0] + 'features_' + x.split('triplet')[1][1:2]) + '.nii.gz'


myDictionary = dict((x, df.loc[df['scan_ID'] == x, 'Pathology'].values[0]) for x in feats)

malign_scan = [x for x in myDictionary.keys() if myDictionary[x] == 'Malignant'][:3]
benign_scan = [x for x in myDictionary.keys() if myDictionary[x] != 'Malignant'][:3]

feats = malign_scan
feats.extend(benign_scan)

all_feats = []
for f in feats:
  data = nib.load(FEATURES_PATH + f).get_data()
  all_feats.append(data)

all_feats[0].shape
all_feats[1].shape

#
plt.figure(figsize=(10,16))
n = 6
for i in np.arange(3*n, step=3):
  if i/3 < 3:
    title = 'Malignant'
  else:
    title = 'Benign'
  print(i/3)
  plt.subplot(n,3,i+1)
  plt.ylabel(title)
  plt.scatter(all_feats[i/3][:,:,:,0], all_feats[i/3][:,:,:,1], s=0.05)
  i += 1
  plt.subplot(n,3,i+1)
  plt.scatter(all_feats[i/3][:,:,:,1], all_feats[i/3][:,:,:,2], s=0.05)
  i += 1
  plt.subplot(n,3,i+1)
  plt.scatter(all_feats[i/3][:,:,:,2], all_feats[i/3][:,:,:,3], s=0.05)

  plt.show()
  
  
  
X =   all_feats[-1]
X.shape

img = nib.Nifti1Image(X, np.diag([3,0.75,0.75,0]))
nib.save(img, '/media/hirsch/RNN_training/Features_Segmentor/MSKCC_16-328_1_09878_20081210_features_l.nii')