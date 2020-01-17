# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 14:52:46 2019

@author: hirsch
"""

import numpy as np
import nibabel as nib
from keras.layers import Input
import keras.backend as K
import matplotlib.pyplot as plt
from keras.models import load_model, Model
# Build the VGG16 network with ImageNet weights
model = load_model('/home/hirsch/Documents/projects/Breast_segmentation/DeepPriors_package/training_sessions/MultiPriors_MSKCC_fullHeadSegmentation_configFile_DM_2019-03-11_1846/Results/model.h5')
print('Model loaded.')
model.summary()
# The name of the layer we want to visualize
# (see model definition in vggnet.py)

layer_name = 'activation_1'  # 'activation_1'

layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])

input_img = np.random.binomial(1, 0, (1, 18, 200, 200,1))
input_img = np.array(input_img, dtype='float')
#input_img = np.random.random((1, 18, 200, 200,1))
#inter_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
#inter_model.predict(input_img)

layer_output = layer_dict[layer_name].output

loss = K.mean(layer_output)

K.gradients(loss, model.input)

grads = K.gradients(loss, model.input)
grads = grads[0]
grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-7)
iterate = K.function([model.input], [loss, grads])

for i in range(2000):
    loss_value, grads_value = iterate([input_img])
    input_img += grads_value# * i
    
plt.imshow(input_img[0,4,:,:,0])   

img = nib.Nifti1Image(input_img[0,:,:,:,0], np.diag([3, 0.75, 0.75, 0]))
nib.save(img, '/home/hirsch/Documents/projects/Breast_segmentation/DeepPriors_package/training_sessions/MultiPriors_MSKCC_fullHeadSegmentation_configFile_DM_2019-03-11_1846/AUC95/InputMax.nii')