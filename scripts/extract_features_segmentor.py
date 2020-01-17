# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 17:51:02 2019

@author: hirsch
"""
import os
os.chdir('/home/hirsch/Documents/projects/Breast_segmentation/DeepPriors_package/scripts')
from matplotlib import pyplot as plt
import numpy as np
import nibabel as nib
from keras.utils import print_summary
from keras.models import Model
from keras.models import load_model   
from DM_MSKCC_Atrous_model import Generalised_dice_coef_multilabel2, dice_coef_multilabel0,dice_coef_multilabel1
from skimage.transform import resize
import sys
import time

name = 'MultiPriors-MSKCC-174'
model_path = '/home/hirsch/Documents/projects/Breast_segmentation/DeepPriors_package/training_sessions/MultiPriors_MSKCC_fullHeadSegmentation_configFile_DM_2019-03-11_1846/models/MSKCC_fullHeadSegmentation_configFile_DM_2019-03-11_1846.log_epoch173.h5'

TPM_channel = '/home/hirsch/Documents/projects/Breast_segmentation/DeepPriors_package/log_TPM_gaussian.nii'


FEATURES_OUTPUT_PATH = '/media/hirsch/RNN_training/Features_Segmentor/'

INPUT_MRIs_PATH = '/media/hirsch/RNN_training/alignedNii/'

layer_name = 'Feature_extraction_layer'

FEATURES_IMAGE_OUTPUT_PATH = '/home/hirsch/Documents/projects/Breast_segmentation/DeepPriors_package/training_sessions/MultiPriors_MSKCC_fullHeadSegmentation_configFile_DM_2019-03-11_1846/Results/'

plot_features = False
get_full_model_output = False
padding = False
save_figure = False

  
  

def normalizeMRI(data, mean=0, std=1):
    if (mean == 0) and (std == 1):
        mean = np.mean(data)
        std = np.std(data)
    data1 = (data - mean)/std
    return(data1)  

def get_features(subject_channels, TPM_channel, model, intermediate_layer_model, plot_features):
  images = []
  for channel in subject_channels: 
    proxy_img = nib.load(channel)
    img_data = proxy_img.get_data()
    #plt.imshow(img_data[19])
    
    # RESAMPLE SCANS AND NORMALIZE INTENSITIES
    res = proxy_img.header['pixdim'][1:4]
    shape = img_data.shape
    target_res = [3, 0.75, 0.75]
    out_shape = np.floor([float(s)*r1/r2 for s,r1,r2 in zip(shape, res, target_res)])
    img_data = resize(img_data, output_shape=out_shape, preserve_range=True, anti_aliasing=True)
    if np.any(np.isnan(img_data)):
      img_data[np.isnan(img_data)] = np.nanmin(img_data)
    
    shape = img_data.shape
    #print('Shape after normalization: {}'.format(shape))
    # Standardize image.
    #for i in range(img_data.shape[0]):
    #  img_data[i,:,:] = percentile95_normalizeMRI(img_data[i,:,:])
  
    #img_data = percentile95_normalizeMRI(img_data)
    img_data = normalizeMRI(img_data)
    #plt.imshow(img_data[10,:,:])
    # Normalize inputs!
    #p95 = np.percentile(X,95)
    #X = X/p95

    images.append(img_data)
  X = np.stack(images, axis=3)
  shape = X.shape
  original_shape = shape
  
  tpm_nii = nib.load(TPM_channel)      
  TPM_data = tpm_nii.get_data()  
  TPM_data = resize(TPM_data, original_shape[:-1], order=1, preserve_range=True, anti_aliasing=True)

#  if shape[0]*shape[1]*shape[2] > 59*221*221:  # if shape exceeeds 55*261*261
#    # Set boundaries for maximum allowed shape
#    a = np.max([0,(shape[0] - 59)])/2   
#    b = np.max([0,(shape[1] - 221)])/2
#    c = np.max([0,(shape[2] - 221)])/2    
#    X = X[a:shape[0]-a,:,:,:]
#    X = X[:,b:shape[1]-b,:,:]
#    X = X[:,:,c:shape[1]-c,:]    
#    TPM_data = TPM_data[a:shape[0]-a,:,:]
#    TPM_data = TPM_data[:,b:shape[1]-b,:]
#    TPM_data = TPM_data[:,:,c:shape[1]-c]  
#  
  
  
  shape = X.shape
  X = X.reshape( (1,) + shape)
  TPM_data = TPM_data[6:TPM_data.shape[0]-6, 33:TPM_data.shape[1]-33, 33:TPM_data.shape[2]-33]
  
  T1post = X[:,:,:,:,0].reshape(X[:,:,:,:,0].shape + (1,))
  T1pre = X[:,:,:,:,1].reshape(X[:,:,:,:,1].shape + (1,))
  T2 = X[:,:,:,:,2].reshape(X[:,:,:,:,2].shape + (1,))
  TPM_data = TPM_data.reshape((1,) + TPM_data.shape + (1,))
  prediction = []
  #prediction = model.predict([T1post, T1pre, T2, TPM_data])
  #prediction = prediction[0,:,:,:,1]
  features = intermediate_layer_model.predict([T1post, T1pre, T2, TPM_data])
  #features = []
  #img = nib.Nifti1Image(features[0,:,:,:,0], np.diag([3, 0.75, 0.75,0]))  
  #nib.save(img, '/home/hirsch/Documents/presentations_and_results/features_segmentor_test.nii')
  
  #img = nib.Nifti1Image(T1post[0,:,:,:,0], np.diag([3, 0.75, 0.75,0]))  
  #nib.save(img, '/home/hirsch/Documents/presentations_and_results/t1post_segmentor_test.nii')  
  
  if plot_features:
    print('Plotting features, then stopping script.')
    slice_number = T1post.shape[1]/2
    for feature_index in range(features.shape[-1]):
      plot_interm_layer(slice_number, features, T1post, T1pre, T2, feature_index)     
      if save_figure:
          plt.savefig(FEATURES_IMAGE_OUTPUT_PATH + 'Features_{}_{}_slice{}_feature{}.png'.format(layer_name,subject_channels[0].split('/')[-2],slice_number, feature_index))      
          plt.clf()
          plt.close()    
  return features, prediction

def plot_interm_layer(slice_number, intermediate_output, T1post, T1pre, T2, feature_index):
    plt.figure(figsize=(12,8))
    plt.subplot(221)
    plt.imshow(T1post[0,slice_number,:,:,0], aspect='equal')#, vmin=0, vmax=2)
    plt.title('T1 post')
    plt.subplot(222)
    plt.imshow(T1pre[0,slice_number,:,:,0], aspect='equal')#, vmin=0, vmax=2)
    plt.title('T1 pre')
    plt.subplot(223)
    plt.imshow(T2[0,slice_number,:,:,0], aspect='equal')#, vmin=0, vmax=2)
    plt.title('T2')
    plt.subplot(224)
    plt.imshow(intermediate_output[0,slice_number-6,:,:,feature_index], aspect='equal')
    plt.colorbar()
    plt.xlabel(layer_name)

print('Loading model..')
my_custom_objects = {'Generalised_dice_coef_multilabel2':Generalised_dice_coef_multilabel2,
                                 'dice_coef_multilabel0':dice_coef_multilabel0,
                                 'dice_coef_multilabel1':dice_coef_multilabel1}
model = load_model(model_path, custom_objects = my_custom_objects )
print_summary(model, positions=[.45, .75, .82, 1])
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)


if visualize_weights:
  weights = model.get_layer(layer_name).get_weights()[0]
  weights.shape
  bias = model.get_layer(layer_name).get_weights()[1]
  plt.scatter(weights[:,:,:,:,0],weights[:,:,:,:,1])



if not os.path.exists(FEATURES_OUTPUT_PATH):
  os.mkdir(FEATURES_OUTPUT_PATH)

exams = os.listdir(INPUT_MRIs_PATH)

tot = len(exams)

print('Loading data and extracting features from {} exams..'.format(tot))

import pandas as pd
df = pd.read_csv('/media/hirsch/RNN_training/All_Triples.csv')

df = df.loc[df.Pathology == 'Malignant']
malign_exams = [x for x in df.ID_date if x in exams]
exams = malign_exams

n = 0
work_step = 0
for exam in exams:
  n+=1
  scans = os.listdir(INPUT_MRIs_PATH  +  exam)
  print(exam)
  left_breast = [INPUT_MRIs_PATH + exam + '/' + x for x in scans if 'l.nii' in x]
  right_breast = [INPUT_MRIs_PATH + exam + '/' + x for x in scans if 'r.nii' in x]
  
  if len(left_breast) == 3:
    if os.path.exists(FEATURES_OUTPUT_PATH + exam + '_l.npy'): 
      print('skip')
      continue
    else:
      left_breast = np.sort(left_breast)
      features, prediction = get_features(left_breast, TPM_channel,model, intermediate_layer_model, plot_features)
      np.save(FEATURES_OUTPUT_PATH + exam + '_l.npy', features)
      img = nib.Nifti1Image(features[0], np.diag([3, 0.75, 0.75, 0]))
      nib.save(img, FEATURES_OUTPUT_PATH + exam + '_features_l.nii.gz')
#
#      img = nib.Nifti1Image(prediction, np.diag([3, 0.75, 0.75, 0]))
#      nib.save(img, FEATURES_OUTPUT_PATH + exam + '_l.nii.gz')
      
  if len(right_breast) == 3:
    if os.path.exists(FEATURES_OUTPUT_PATH + exam + '_r.npy'): 
      print('skip')
      continue
    else:
      right_breast = np.sort(right_breast)
      features, prediction = get_features(right_breast, TPM_channel, model, intermediate_layer_model, plot_features)
      np.save(FEATURES_OUTPUT_PATH + exam + 'features_r.npy', features)
      img = nib.Nifti1Image(features[0], np.diag([3, 0.75, 0.75, 0]))
      nib.save(img, FEATURES_OUTPUT_PATH + exam + '_features_r.nii.gz')
      
      #img = nib.Nifti1Image(prediction, np.diag([3, 0.75, 0.75, 0]))
      #nib.save(img, FEATURES_OUTPUT_PATH + exam + '_r.nii.gz')      
      
  work_step += 1

  if plot_features:
    sys.exit(0)
	
  print(work_step)
#  if work_step%20 == 0:
#    print('{}%'.format(n*100./tot))
#    print("Attempt to cool off processor units..")
#    time.sleep(40)    # pause 60 seconds
#    print("Back to work.")

  
  
#  


#import pandas as pd
#pddf = pd.DataFrame([f0,f1,f2,f3])
#pddf = pddf.transpose()
#import seaborn as sns
#sns.pairplot(pddf)
#
