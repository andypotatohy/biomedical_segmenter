#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 11:55:32 2019

@author: deeperthought
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model   
from MultiPriors_Models_Collection import Generalised_dice_coef_multilabel2, dice_coef_multilabel0,dice_coef_multilabel1

### Check how much weights have changed on breast-mask pathway ###

# Compare with v1-non-trainable weights  --> Should be the same
# Compare with v2 -trainable weights --> Show change after first iteration (little change), to last iteration (larger change)

path_to_breastMask_model = '/home/deeperthought/Projects/MultiPriors_MSKCC/training_sessions/BreastSegmentor_v1_breastMask_configFile_BreastMask-Model_2019-10-21_1932/models/tor_v1_breastMask_configFile_BreastMask-Model_2019-10-21_1932.log_epoch499.h5'

model_v2_initial_path = '/home/deeperthought/Projects/MultiPriors_MSKCC/training_sessions/MultiPriors_v2_MSKCC_configFile_MultiPriors_v2_F4_2019-11-04_1014/models/v2_MSKCC_configFile_MultiPriors_v2_F4_2019-11-04_1014.log_epoch0.h5'
model_v2_last_path = '/home/deeperthought/Projects/MultiPriors_MSKCC/training_sessions/MultiPriors_v2_MSKCC_configFile_MultiPriors_v2_F4_2019-11-04_1014/models/v2_MSKCC_configFile_MultiPriors_v2_F4_2019-11-04_1014.log_epoch90.h5'

model_v1_last_path = '/home/deeperthought/Projects/MultiPriors_MSKCC/training_sessions/MultiPriors_v1_MSKCC_configFile_MultiPriors_v1_2019-10-31_1830/models/v1_MSKCC_configFile_MultiPriors_v1_2019-10-31_1830.log_epoch79.h5'

layer_index = 13

OUTPUT_PATH = '/home/deeperthought/Projects/MSKCC/MSKCC/Segmenter_HumanPerformance/F4_Final/weight_changes_visualization/'

#%%
my_custom_objects = {'Generalised_dice_coef_multilabel2':Generalised_dice_coef_multilabel2,
                     'dice_coef_multilabel0':dice_coef_multilabel0,
                     'dice_coef_multilabel1':dice_coef_multilabel1}      

def compare_weights(layer_index, path_to_breastMask_model, path_to_model, my_custom_objects):
    model = load_model(path_to_model, custom_objects = my_custom_objects )
    print('loaded first model..')          
    bm_model = load_model(path_to_breastMask_model, custom_objects = my_custom_objects )            
    print('loaded second model..')
    preTrained_convLayers = [x for x in bm_model.layers if 'Conv3D' in str(x)]
    #preTrained_batchNormLayers = [x for x in bm_model.layers if 'BatchNorm' in str(x)]
    preTrained_convLayers = preTrained_convLayers[:-1]
    newModel_convLayers = [x for x in model.layers if 'T1post_Context' in x.name]           
    #newModel_batchNormLayers = [x for x in model.layers if 'BatchNorm' in x.name]            
    
    #assert len(preTrained_convLayers ) == len(newModel_convLayers), 'Models have incompatible architecture..'
    #assert len(preTrained_batchNormLayers) == len(newModel_batchNormLayers), 'Models have incompatible architecture..'

    w1 = model.get_layer(newModel_convLayers[layer_index].name).get_weights()[0] 
    w2 = preTrained_convLayers[layer_index].get_weights()[0]
    w1 = w1.reshape(np.prod(w1.shape))
    w2 = w2.reshape(np.prod(w2.shape))
    print('finished extracting weights from layer {}'.format(layer_index))
    return w1, w2


w1_v2_ini,  w2 =  compare_weights(layer_index, path_to_breastMask_model, model_v2_initial_path, my_custom_objects)
w1_v2_last, w2 =  compare_weights(layer_index, path_to_breastMask_model, model_v2_last_path, my_custom_objects)
w1_v1_last, w2 =  compare_weights(layer_index, path_to_breastMask_model, model_v1_last_path, my_custom_objects)


fig, ax = plt.subplots(1,3)
ax[0].set_title('trainable - epoch0')
ax[0].plot([0,1],[w2, w1_v2_ini], linewidth=0.5, alpha=0.8)
ax[0].set_xticks([0,1])
ax[0].set_xticklabels(['BreastMask','Context-Model'], rotation=45)
ax[1].set_title('trainable - epoch80')
ax[1].plot([0,1],[w2, w1_v2_last], linewidth=0.5, alpha=0.8)
ax[1].set_xticks([0,1])
ax[1].set_xticklabels(['BreastMask','Context-Model'], rotation=45)
ax[2].set_title('non-trainable - epoch80')
ax[2].plot([0,1],[w2, w1_v1_last], linewidth=0.5, alpha=0.8)
ax[2].set_xticks([0,1])
ax[2].set_xticklabels(['BreastMask','Context-Model'], rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_PATH + '/BreastMask_Pathway_Weights_Change_plot_layer{}.png'.format(layer_index),dpi=200)


fig, ax = plt.subplots(1,3, sharex=False, figsize=(8,5))
fig.suptitle('Histogram of differences')
#ax[0].set_xlim([-0.025,0.025])
ax[0].set_title('trainable - epoch0')
ax[0].hist(w1_v2_ini -  w2)
ax[1].set_title('trainable - epoch80')
ax[1].hist(w1_v2_last - w2)
ax[2].set_title('non-trainable - epoch80')
ax[2].hist(w1_v1_last - w2, 100)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(OUTPUT_PATH + '/BreastMask_Pathway_Weights_Change_differences_layer{}.png'.format(layer_index),dpi=200)




