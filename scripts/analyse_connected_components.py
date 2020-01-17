# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 14:40:42 2019

@author: hirsch
"""

import os
import numpy as np
import nibabel as nib
from scipy.ndimage.measurements import label
import pandas as pd
import sys



RESULTS_SHEET = 'SEGMENTATIONS_SUMMARY_epoch65_likelihoods.csv'

PATH = '/home/hirsch/Documents/projects/Breast_segmentation/DeepPriors_package/training_sessions/MultiPriors_MSKCC_fullHeadSegmentation_configFile_DM_subs_2019-04-25_1322/predictions/'

RESULT_TABLE_PATH = '/home/hirsch/Documents/projects/Breast_segmentation/DeepPriors_package/training_sessions/MultiPriors_MSKCC_fullHeadSegmentation_configFile_DM_subs_2019-04-25_1322/connected_components_regression/'

run_test_debug = False

def largest_connected_components(data, threshold = 100, save = False, edges=False):
  structure = np.ones((3,3,3), dtype=np.int)
  if edges:
    data[data.shape[0]/2 - 10 : data.shape[0]/2 + 10, 
         data.shape[1]/2 - 50 : data.shape[1]/2 + 50,
         data.shape[2]/2 - 50 : data.shape[2]/2 + 50] = 0
                
  labeled, ncomponents = label(data, structure)  
  sizes = {}
  for i in range(1,ncomponents+1):
    sizes[i] = len(np.argwhere(labeled == i))
  
  components = sizes.values()
  components.sort(reverse=1)
  n_connected_components = len([x for x in components if x > threshold])   
  if len(components) > 0:
    size_largest_connected_component = components[0]
  else:
    size_largest_connected_component = 0
  # Remove blobs under a threshold:  
  if save:
    for key in sizes:
      if sizes[key] < threshold:
        labeled[labeled == key] = 0
  return labeled, n_connected_components, size_largest_connected_component  

if run_test_debug:
  nifti_file = '/home/hirsch/Documents/projects/Breast_segmentation/DeepPriors_package/training_sessions/MultiPriors_MSKCC_fullHeadSegmentation_configFile_DM_2019-03-11_1846/predictions/MSKCC_16-328_1_00113_20031116_l_epoch174.nii'
  nii = nib.load(nifti_file)
  data = nii.get_data()
  labeled, n_connected_components, size_largest_connected_component = largest_connected_components(data, threshold = 100, save = True, edges=False)
  img = nib.Nifti1Image(labeled, nii.affine)
  nib.save(img, nifti_file.split('.nii')[0] + '_CC.nii')
  sys.exit(0)
  
#-----------------------------------------------------------------------------------------

#open all_triples to store target pathology.
#pathology = pd.read_csv('/media/hirsch/RNN_training/All_Triples.csv') 

pathology = pd.read_csv('/media/hirsch/RNN_training/All_Triples_cleaned_present.csv') 
#pathology['Side'] = pathology['file'].apply(lambda x : x.split('.npy')[0][-1])
pathology['segmentation'] = pathology['ID_date'] + '_' + pathology['Side_inferred']


segmentations = [x for x in os.listdir(PATH) if x.endswith('.nii.gz')]

if not os.path.exists(RESULT_TABLE_PATH + RESULTS_SHEET):
  df = pd.DataFrame(columns=['scan_ID', 'n_total_pixels','n_connected_components','size_largest_component','BIRADS', 'pathology'])
else:
  df = pd.read_csv(RESULT_TABLE_PATH + RESULTS_SHEET)
  segmentations = [x for x in segmentations if not x.split('_epoch')[0] in list(df['scan_ID'])]  
  
  
tot = len(segmentations)
print('Found {} new segmentations'.format(tot))
n = 0
for nifti in segmentations:
  
  if n%20==0:
    print(str(round(n*100./tot,2)) + ' %')
  n += 1
  nii = nib.load(PATH + nifti)
  data = nii.get_data()
  data = data[1:data.shape[0]-1,1:data.shape[1]-1,1:data.shape[2]-1]
  data[data>0.99] = 1
  data[data<=0.99] = 0
  data = np.array(data, dtype=int)
  labeled, n_connected_components, size_largest_connected_component   = largest_connected_components(data, threshold = 100, edges=False)
  n_pixels_total = np.sum(data)
  scan_ID = nifti.split('_epoch')[0]
  target_label = pathology.loc[pathology['segmentation'] == scan_ID, 'Pathology'].values[0]
  BIRADS = pathology.loc[pathology['segmentation'] == scan_ID, 'BIRADS'].values[0]
  i = len(df)
  df.loc[i] = [scan_ID, n_pixels_total, n_connected_components, size_largest_connected_component,BIRADS, target_label ]
  
  
  
df.to_csv(RESULT_TABLE_PATH + RESULTS_SHEET, index=False)  

print('Malignants: {}'.format(len(df.loc[df['pathology'] == 'Malignant'])))
print('Benigns: {}'.format(len(df.loc[df['pathology'] != 'Malignant'])))

