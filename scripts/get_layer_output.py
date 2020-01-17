#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 12:08:51 2019

@author: lukas
"""
from matplotlib import pyplot as plt
import numpy as np
import nibabel as nib
from keras.utils import print_summary
from keras.models import Model
from keras.models import load_model   
from DM_MSKCC_Atrous_model import Generalised_dice_coef_multilabel2, dice_coef_multilabel0,dice_coef_multilabel1
from skimage.transform import resize

name = 'model34'
model_path = '/home/andy/projects/mskProj/DeepPriors_package/training_sessions/MultiPriors_MSKCC_MSKCC_configFile_DM_2019-02-06_1900/models/MSKCC_MSKCC_configFile_DM_2019-02-06_1900.log_epoch14.h5'
test_image = '/home/andy/projects/mskProj/DeepPriors_package/dummy_image.nii'

TPM_channel = '/home/andy/projects/mskProj/DeepPriors_package/TPM_gaussian.nii'

#subject_channels = ['/media/andy/RNN_training/Normalized/MSKCC_16-328_1_00416_20050430/rt1post-l_stand.nii',
#                    '/media/andy/RNN_training/Normalized/MSKCC_16-328_1_00416_20050430/rt1pre-l_stand.nii',
#                    '/media/andy/RNN_training/Normalized/MSKCC_16-328_1_00416_20050430/rt2-l_stand.nii']
#                    
subject_channels = ['/media/andy/RNN_training/Normalized/MSKCC_16-328_1_07351_20051207/rt1post-l_stand.nii',
        '/media/andy/RNN_training/Normalized/MSKCC_16-328_1_07351_20051207/rt1pre-l_stand.nii',
'/media/andy/RNN_training/Normalized/MSKCC_16-328_1_07351_20051207/rt2-l_stand.nii'
]


layer_name = 'Softmax'
slice_number = 25
load_model_flag = True
modify_coordinates_processing_weights = False
new_model_flag = False
get_full_model_output = False
padding = True
save_figure = True


def plot_interm_layer(slice_number, intermediate_output, T1post, T1pre, T2, y_coords, coord_intermediate_output):
    layer_output = intermediate_output[0,slice_number - 12,:,:,-1]
    plt.figure(figsize=(12,8))
    plt.subplot(231)
    plt.imshow(T1post[0,slice_number,:,:,0], aspect='equal', vmin=0, vmax=1)
    plt.title('T1 post')
    plt.subplot(232)
    plt.imshow(T1pre[0,slice_number,:,:,0], aspect='equal', vmin=0, vmax=1)
    plt.title('T1 pre')
    plt.subplot(233)
    plt.imshow(T2[0,slice_number,:,:,0], aspect='equal', vmin=0, vmax=1)
    plt.title('T2')
    plt.subplot(234)
    plt.imshow(layer_output, aspect='equal')
    plt.colorbar()
    plt.xlabel(layer_name)
    plt.subplot(235)
    plt.imshow(y_coords[0,0,:,:,0], aspect='equal')
    plt.colorbar()    
    plt.xlabel('Second Input')
    plt.subplot(236)
    plt.imshow(coord_intermediate_output[0,0,:,:,0], aspect='equal')
    plt.xlabel('Coordinate processing')
    plt.colorbar()

if load_model_flag:
    my_custom_objects = {'Generalised_dice_coef_multilabel2':Generalised_dice_coef_multilabel2,
                                     'dice_coef_multilabel0':dice_coef_multilabel0,
                                     'dice_coef_multilabel1':dice_coef_multilabel1}
    model = load_model(model_path, custom_objects = my_custom_objects )

elif new_model_flag:
    from MultiPriors_MSKCC import DeepMedic
    dm = DeepMedic([13,75,75], 2, 3, 0.0001, [0], 0.001, 0,'Multinomial')
    model = dm.createModel()    

print_summary(model, positions=[.45, .75, .82, 1])

if len(layer_name) == 0:
    layer_name = input("Input layer name to get output from: ")

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

TPM_data = resize(TPM_data, original_shape[:-1], order=1, preserve_range=True, anti_aliasing=True)

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

T1post = X[:,:,:,:,0].reshape(X[:,:,:,:,0].shape + (1,))
T1pre = X[:,:,:,:,1].reshape(X[:,:,:,:,1].shape + (1,))
T2 = X[:,:,:,:,2].reshape(X[:,:,:,:,2].shape + (1,))
TPM_data = TPM_data.reshape((1,) + TPM_data.shape + (1,))

print_summary(model, positions=[.45, .75, .82, 1])


layer_name = 'conv3d_80'
model.get_layer(layer_name).get_weights()
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict([T1post, T1pre, T2, TPM_data])
intermediate_output.shape

model.get_layer('conv3d_81').get_weights()
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('activation_80').output)
coord_intermediate_output = intermediate_layer_model.predict([T1post, T1pre, T2, TPM_data])
coord_intermediate_output.shape

plot_interm_layer(22, intermediate_output, T1post, T1pre, T2, TPM_data, coord_intermediate_output)

if save_figure:
    plt.savefig('/home/andy/projects/mskProj/DeepPriors_package/Coordinate_threshold_{}_{}_{}.png'.format(name,subject_channels[0].split('/')[-2], layer_name))

if modify_coordinates_processing_weights:
    # Set weights for coordinate processing:  
    model.get_layer('Y_Processing').get_weights()
    model.get_layer('Y_Processing').set_weights([np.array([[[[[25]]]]]), np.array([-9.5 ])])
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('Y_Processing').output)
    intermediate_output = intermediate_layer_model.predict([T1post, T1pre, T2, TPM_data])
    intermediate_output.shape
    plt.subplot(131)
    plt.imshow(TPM_data[0,0,:,:,0])
    plt.subplot(132)
    plt.imshow(intermediate_output[0,0,:,:,0])
    model.get_layer('Classification_layer').get_weights()
    model.get_layer('Classification_layer').set_weights([np.array([[[[[ 1, 1],
                                                           [ 1, 1],
                                                           [ 10, 10]]]]]),
                                                np.array([0,0])])   


if get_full_model_output:
 
    yhat = model.predict([T1post, T1pre, T2, TPM_data])
    
    plt.figure(figsize=(12,8))
    plt.subplot(231)
    plt.imshow(T1post[0,slice_number,:,:,0], aspect='equal', vmin=0, vmax=1)
    plt.title('T1 post')
    plt.subplot(232)
    plt.imshow(T1pre[0,slice_number,:,:,0], aspect='equal', vmin=0, vmax=1)
    plt.title('T1 pre')
    plt.subplot(233)
    plt.imshow(T2[0,slice_number,:,:,0], aspect='equal', vmin=0, vmax=1)
    plt.title('T2')
    plt.subplot(234)
    plt.imshow(yhat[0,slice_number - 12,:,:,-1], aspect='equal')
    plt.colorbar()
    plt.xlabel(layer_name)
    plt.subplot(235)
    plt.imshow(TPM_data[0,0,:,:,0], aspect='equal', vmin=0, vmax=1)
    plt.xlabel('Coordinates Input')
    plt.subplot(236)
    plt.imshow(coord_intermediate_output[0,0,:,:,0], aspect='equal', vmin=0, vmax=1)
    plt.xlabel('Coordinate processing')
    plt.colorbar()
    
    y = yhat[:,:,:,:,1]            # For logits for class 2
    print('y shape: {}'.format(y.shape))
    y = y.reshape(y.shape[1],y.shape[2],y.shape[3])
    #y = y.reshape(shape[0]-24,shape[1]-78,shape[2]-78)
    

    y_out = np.zeros((shape[0],shape[1],shape[2]))

    try:
        y_out[  :, abs(shape[1] -y.shape[1])/2:shape[1] - abs(shape[1] -y.shape[1])/2,abs(shape[2] -y.shape[2])/2:shape[2] - abs(shape[2] -y.shape[2])/2] = y[abs(shape[0] -y.shape[0])/2:y.shape[0] - abs(shape[0] -y.shape[0])/2,:,:]
    except:
        y_out[abs(shape[0] -y.shape[0])/2:shape[0] - abs(shape[0] -y.shape[0])/2, 
              abs(shape[1] -y.shape[1])/2:shape[1] - abs(shape[1] -y.shape[1])/2,
              abs(shape[2] -y.shape[2])/2:shape[2] - abs(shape[2] -y.shape[2])/2] = y


    img = nib.Nifti1Image(y_out, proxy_img.affine)
    segmentationName = '/home/andy/projects/mskProj/DeepPriors_package/Coordinate_threshold_{}_{}.nii'.format(name,subject_channels[0].split('/')[-2])
    nib.save(img, segmentationName)
    
    plt.figure(figsize=(12,8))
    plt.imshow(y_out[:,26,:], aspect='equal')
    plt.colorbar()
    plt.xlabel(layer_name)

    y_out[:,26,23]
    
    plt.figure(figsize=(12,8))
    plt.imshow(yhat[0,:,26,:,0], aspect='equal')
    plt.colorbar()
    plt.xlabel(layer_name)

    yhat[0,:,26,23,0]

    
'''
#softmax tests
myX = np.arange(0,1,0.01)
w = 10
b = -5
lin = w*myX+b 
softmax = 1/(1 + np.exp(-1*lin))
plt.plot(myX,softmax )



myX = intermediate_output[0,0,:,:,0]
w = -1
b = 0
lin = w*myX+b 
plt.plot(myX, 1/(1 + np.exp(lin)))
'''
