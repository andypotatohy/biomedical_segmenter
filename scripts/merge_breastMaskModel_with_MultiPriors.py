#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 12:47:57 2019

@author: deeperthought
"""

from MultiPriors_MSKCC_MultiScale import MultiPriors_v1
mp = MultiPriors_v1(2, 3, 0.001, [0], 0.01, 0, 'Dice' )
multipriors_model = mp.createModel()            
multipriors_model.summary()                        

from keras.models import load_model  
from MultiPriors_MSKCC_MultiScale import Generalised_dice_coef_multilabel2, dice_coef_multilabel0,dice_coef_multilabel1
my_custom_objects = {'Generalised_dice_coef_multilabel2':Generalised_dice_coef_multilabel2,
                     'dice_coef_multilabel0':dice_coef_multilabel0,
                     'dice_coef_multilabel1':dice_coef_multilabel1}

bm_model = load_model('/home/deeperthought/Projects/MultiPriors_MSKCC/training_sessions/BreastSegmentor_v1_breastMask_configFile_BreastMask-Model_2019-10-21_1932/models/tor_v1_breastMask_configFile_BreastMask-Model_2019-10-21_1932.log_epoch499.h5', custom_objects = my_custom_objects )

convLayers = [x for x in bm_model.layers if 'Conv3D' in str(x)]
context_layers = [x for x in multipriors_model.layers if 'T1post_Context' in x.name]

assert len(context_layers) == len(convLayers), 'Models have incompatible architecture..'

for i in range(len(context_layers)):
    print(i)
    print(context_layers[i].name)
    multipriors_model.get_layer(context_layers[i].name).set_weights(convLayers[i].get_weights())


