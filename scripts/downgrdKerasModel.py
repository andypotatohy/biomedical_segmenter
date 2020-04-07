#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 17:13:19 2020

@author: andy
"""
#import numpy as np
import keras
from keras.optimizers import Adam
from MultiPriors_Models_Collection import Generalised_dice_coef_multilabel2, dice_coef_multilabel_bin0,dice_coef_multilabel_bin1

#trained_model = keras.models.load_model('/home/andy/projects/lukasSegmenter/biomedical_segmenter/fastSegmenter.h5', compile=False)
trained_model = keras.models.load_model('/home/andy/projects/lukasSegmenter/biomedical_segmenter/stMask_breastMask_UNet_v0_BreastMask_2019-12-23_1658.log_epoch45.h5', compile=False)

#from Unet_3D_Class import UNet_v4
#model = UNet_v4(input_shape=(19,75,75,3), pool_size=(2, 2, 2), n_labels=2, initial_learning_rate=0.00001, deconvolution=True,
#      depth=4, n_base_filters=32, include_label_wise_dice_coefficients=False, 
#      batch_normalization=True, activation_name="softmax", bilinear_upsampling=True) 

#from Unet_3D_Class import UNet_v4
#model = UNet_v4(input_shape=(None,None,None,3), pool_size=(2, 2, 2), n_labels=2, initial_learning_rate=0.00001, deconvolution=True,
#      depth=4, n_base_filters=32, include_label_wise_dice_coefficients=False, 
#      batch_normalization=True, activation_name="softmax", bilinear_upsampling=True)

from Unet_3D_Class import UNet_v0_BreastMask
model = UNet_v0_BreastMask(input_shape = (3,256,256,1), pool_size=(1, 2, 2), n_labels=1, initial_learning_rate=0.0001, deconvolution=False,
       depth=4, n_base_filters=32, include_label_wise_dice_coefficients=False, 
       batch_normalization=False, activation_name="sigmoid")

len(trained_model.layers)

for i in range(len(trained_model.layers)):    
    w_trained = trained_model.layers[i].get_weights()
#    w_random = model.layers[i].get_weights()
#    if len(w_trained) > 0:
#        print(np.equal(w_trained[0], w_random[0]))
    model.layers[i].set_weights(w_trained)


model.compile(loss=Generalised_dice_coef_multilabel2, optimizer=Adam(lr=0.001), metrics=['acc', dice_coef_multilabel_bin0, dice_coef_multilabel_bin1])

#model.save('/home/andy/projects/lukasSegmenter/biomedical_segmenter/fastSegmenter_for_old_keras.h5')
model.save('/home/andy/projects/lukasSegmenter/biomedical_segmenter/stMask_breastMask_UNet_v0_BreastMask_2019-12-23_1658.log_epoch45_for_old_keras.h5')


##my_custom_objects = {'Generalised_dice_coef_multilabel2':Generalised_dice_coef_multilabel2,
##                                 'dice_coef_multilabel0':dice_coef_multilabel0,
##                                 'dice_coef_multilabel1':dice_coef_multilabel1}
#my_custom_objects = {'Generalised_dice_coef_multilabel2':Generalised_dice_coef_multilabel2,
#                                 'dice_coef_multilabel_bin0':dice_coef_multilabel_bin0,
#                                 'dice_coef_multilabel_bin1':dice_coef_multilabel_bin1}  
#keras.models.load_model('/home/andy/projects/lukasSegmenter/biomedical_segmenter/fastSegmenter_for_old_keras.h5',my_custom_objects)
