#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 17:18:17 2019

@author: lukas
"""

from keras.models import Model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.convolutional import Conv3D
from keras.initializers import he_normal
from keras.initializers import Orthogonal, RandomNormal, RandomUniform
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1_l2
from keras.layers import Input, Flatten, Reshape, Permute
from keras.layers.merge import Concatenate
from keras.layers import MaxPooling3D
from keras.layers import AveragePooling3D
from keras.layers.convolutional import Cropping3D
from keras.layers import UpSampling3D
from keras.layers import concatenate
from keras.layers.advanced_activations import PReLU
from keras.layers import LeakyReLU
from keras.utils import print_summary
from keras import regularizers
from keras.optimizers import RMSprop
import keras.backend as K
from keras.optimizers import Adam
import numpy as np
from keras.activations import softmax

#------------------------------------------------------------------------------------------

def dice_coef(y_true, y_pred):
    smooth = 1e-6
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f**2) + K.sum(y_pred_f**2) + smooth)

def Generalised_dice_coef_multilabel2(y_true, y_pred, numLabels=2):
    """This is the loss function to MINIMIZE. A perfect overlap returns 0. Total disagreement returns numeLabels"""
    dice=0
    for index in range(numLabels):
        dice -= dice_coef(y_true[:,:,:,:,index], y_pred[:,:,:,:,index])
    return numLabels + dice

def dice_coef_multilabel6(y_true, y_pred, numLabels=6):
    dice=0
    for index in range(numLabels):
        dice -= dice_coef(y_true[:,:,:,:,index], y_pred[:,:,:,:,index])
    return numLabels + dice
def w_dice_coef(y_true, y_pred, PENALTY):
    smooth = 1e-6
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f) * PENALTY
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_multilabel0(y_true, y_pred):
    index = 0
    dice = dice_coef(y_true[:,:,:,:,index], y_pred[:,:,:,:,index])
    return dice
def dice_coef_multilabel1(y_true, y_pred):
    index = 1
    dice = dice_coef(y_true[:,:,:,:,index], y_pred[:,:,:,:,index])
    return dice
def dice_coef_multilabel2(y_true, y_pred):
    index = 2
    dice = dice_coef(y_true[:,:,:,:,index], y_pred[:,:,:,:,index])
    return dice


class DeepMedic():
    
    def __init__(self, output_classes, num_channels, L2, dropout, learning_rate, optimizer_decay, loss_function):
        
        self.output_classes = output_classes
        self.conv_features = [20,20,20,20,30,30,30,30,30,30,30,50,50,50,50] #[50,50,50,50,50,50,50,70,70,70,70,100,100]
        self.fc_features = [60,60,80,100]#[50,50,50,50]
        self.num_channels = num_channels
        self.L2 = L2
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.optimizer_decay = optimizer_decay
        self.loss_function = loss_function
    
    def createModel(self):
    
        T1post = Input((None,None,None, 1),name = 'T1post_input')
        T1pre = Input((None,None,None, 1),name = 'T1pre_input')
        T2 = Input((None,None,None, 1),name = 'T2_input')
       
        ########################  T1 post pathway #########################
        x1        = Conv3D(filters = 20, 
                               kernel_size = (3,3,3), 
                               
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(T1post)
        x1        = LeakyReLU()(x1)
        x1        = BatchNormalization()(x1)
        
        for feature in self.conv_features[0:5]:    # reduce all dimensions in -24
            x1        = Conv3D(filters = feature, 
                               kernel_size = (3,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x1)
            x1        = LeakyReLU()(x1)
            x1        = BatchNormalization()(x1)
        
        for feature in (self.conv_features[0:9]):    # reduce in -36
            x1        = Conv3D(filters = feature, 
                               kernel_size = (1,5,5), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x1)
            x1        = LeakyReLU()(x1)
            x1        = BatchNormalization()(x1)
        
        for feature in (self.conv_features[0:9]):    
            x1        = Conv3D(filters = feature, 
                               kernel_size = (1,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x1)
            x1        = LeakyReLU()(x1)
            x1        = BatchNormalization()(x1)
        
        
        ########################  T1 pre pathway #########################
        x2        = Conv3D(filters = 20, 
                               kernel_size = (3,3,3), 
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(T1pre)
        x2        = LeakyReLU()(x2)
        x2        = BatchNormalization()(x2)
        
        for feature in self.conv_features[0:5]:    # reduce all dimensions in -24
            x2        = Conv3D(filters = feature, 
                               kernel_size = (3,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x2)
            x2        = LeakyReLU()(x2)
            x2        = BatchNormalization()(x2)     
            
        for feature in (self.conv_features[0:9]):    # reduce in -36
            x2        = Conv3D(filters = feature, 
                               kernel_size = (1,5,5), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x2)
            x2        = LeakyReLU()(x2)
            x2        = BatchNormalization()(x2)     
        
        for feature in (self.conv_features[0:9]):    
            x2        = Conv3D(filters = feature, 
                               kernel_size = (1,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x2)
            x2        = LeakyReLU()(x2)
            x2        = BatchNormalization()(x2)     
          

        ########################  T2 pathway #########################
        x3        = Conv3D(filters = 20, 
                               kernel_size = (3,3,3), 
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(T2)
        x3        = LeakyReLU()(x3)
        x3        = BatchNormalization()(x3)        

        for feature in self.conv_features[0:5]:    # reduce all dimensions in -24
            x3        = Conv3D(filters = feature, 
                               kernel_size = (3,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x3)
            x3        = LeakyReLU()(x3)
            x3        = BatchNormalization()(x3)
            
        for feature in (self.conv_features[0:9]):    # reduce in -36
            x3        = Conv3D(filters = feature, 
                               kernel_size = (1,5,5), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x3)
            x3        = LeakyReLU()(x3)
            x3        = BatchNormalization()(x3)
        
        for feature in (self.conv_features[0:9]):    
            x3        = Conv3D(filters = feature, 
                               kernel_size = (1,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x3)
            x3        = LeakyReLU()(x3)
            x3        = BatchNormalization()(x3)
                            

	
	########################  Merge Modalities #########################

        x = concatenate([x1,x2,x3])        
            
        for feature in (self.fc_features[0:2]):  
            x        = Conv3D(filters = feature, 
                               kernel_size = (1,1,1), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x)
            x        = LeakyReLU()(x)
            x        = BatchNormalization()(x)
        
        
        x        = Conv3D(filters = self.output_classes, 
                           kernel_size = (1,1,1), 
                           #kernel_initializer=he_normal(seed=seed),
                           kernel_initializer=Orthogonal(),
                           kernel_regularizer=regularizers.l2(self.L2))(x)
        
        TPM = Input((None,None,None,1))
        
        x        = concatenate([x,TPM])  #  MIXING ONLY CHANNELS + CHANNELS. 
        
        x        = Conv3D(filters = 100, 
                   kernel_size = (1,1,1), 
                   #kernel_initializer=he_normal(seed=seed),
                   kernel_initializer=Orthogonal(),
                   kernel_regularizer=regularizers.l2(self.L2))(x)
        x        = LeakyReLU()(x)
        x        = BatchNormalization()(x)

        #x        = concatenate([x,TPM])

#        x        = Conv3D(filters = 4, 
#                   kernel_size = (1,1,1), 
#                   #kernel_initializer=he_normal(seed=seed),
#                   kernel_initializer=Orthogonal(),
#                   name = 'Feature_extraction_layer',
#                   kernel_regularizer=regularizers.l2(self.L2))(x)
#        x        = LeakyReLU()(x)
#        x        = BatchNormalization()(x)


        x        = Conv3D(filters = 2, 
                   kernel_size = (1,1,1), 
                   kernel_initializer=Orthogonal(),
                   kernel_regularizer=regularizers.l2(self.L2))(x)
        x        = Activation('sigmoid')(x)
        
        model     = Model(inputs=[T1post,T1pre,T2,TPM], outputs=x)
	if self.loss_function == 'Dice':
        	model.compile(loss=Generalised_dice_coef_multilabel2, optimizer=Adam(lr=self.learning_rate, clipnorm=1.0), metrics=['acc', dice_coef_multilabel0, dice_coef_multilabel1])
    	elif self.loss_function == 'Multinomial':
		model.compile(loss='binary_crossentropy', optimizer=Adam(lr=self.learning_rate), metrics=['acc', dice_coef_multilabel0, dice_coef_multilabel1])
        return model


