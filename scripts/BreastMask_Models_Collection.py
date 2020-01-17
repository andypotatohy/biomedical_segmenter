# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 13:22:55 2019

@author: hirsch
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

from keras.utils import multi_gpu_model
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))
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


class BreastSegmentor_v0():
    
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
       
        ########################  T1 post pathway #########################
            
        x = AveragePooling3D(pool_size=(1, 3, 3), name='T1post_Context')(T1post)    
        # (13, 25, 25)        
        for _ in range(6):    
            x        = Conv3D(filters = 30, 
                               kernel_size = (3,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x)
            x        = LeakyReLU()(x)                              
            x        = BatchNormalization()(x) 
 
        for _ in range(5):    
            x        = Conv3D(filters = 30, 
                               kernel_size = (1,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x)
            x        = LeakyReLU()(x)                              
            x        = BatchNormalization()(x)     
     
        x   =  UpSampling3D(size=(1,3,3))(x)
      

        ######################## FC Parts #############################
          
        for feature in (self.fc_features[0:2]):  
            x        = Conv3D(filters = 60, 
                               kernel_size = (1,1,1), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x)
            x        = LeakyReLU()(x)
            x        = BatchNormalization()(x)
        
           
        x        = Conv3D(filters = 100, 
                   kernel_size = (1,1,1), 
                   #kernel_initializer=he_normal(seed=seed),
                   kernel_initializer=Orthogonal(),
                   kernel_regularizer=regularizers.l2(self.L2))(x)
        x        = LeakyReLU()(x)
        x        = BatchNormalization()(x)
        

        x        = Conv3D(filters = 2, 
                   kernel_size = (1,1,1), 
                   kernel_initializer=Orthogonal(),
                   kernel_regularizer=regularizers.l2(self.L2))(x)
        x        = Activation('sigmoid')(x)
        
        model     = Model(inputs=[T1post], outputs=x)

    	#model = multi_gpu_model(model, gpus=4)
    
        if self.loss_function == 'Dice':
            model.compile(loss=Generalised_dice_coef_multilabel2, optimizer=Adam(lr=self.learning_rate), metrics=['acc', dice_coef_multilabel0, dice_coef_multilabel1])
        elif self.loss_function == 'Multinomial':
            model.compile(loss='binary_crossentropy', optimizer=Adam(lr=self.learning_rate), metrics=['acc', dice_coef_multilabel0, dice_coef_multilabel1])
        return model




class BreastSegmentor_v1():
    
    def __init__(self, output_classes, num_channels, L2, dropout, learning_rate, optimizer_decay, loss_function):
        
        self.output_classes = output_classes
        self.conv_features = [20,20,20,20,30,30,30,30,30,30,30,50,50,50,50] 
        self.fc_features = [60,60,80,100]#[50,50,50,50]
        self.num_channels = num_channels
        self.L2 = L2
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.optimizer_decay = optimizer_decay
        self.loss_function = loss_function
    
    def createModel(self):
    
        T1post = Input((None,None,None, 1),name = 'T1post_input')
        Coords = Input((None,None,None, 3), name = 'Spatial_coordinates')
        
        ########################  T1 post pathway #########################
            
        x = AveragePooling3D(pool_size=(1, 3, 3), name='T1post_Context')(T1post)    
        # (13, 25, 25)        
        for _ in range(6):    
            x        = Conv3D(filters = 30, 
                               kernel_size = (3,3,3), 
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x)
            x        = LeakyReLU()(x)                              
            x        = BatchNormalization()(x) 
 
        for _ in range(5):    
            x        = Conv3D(filters = 30, 
                               kernel_size = (1,3,3), 
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x)
            x        = LeakyReLU()(x)                              
            x        = BatchNormalization()(x)     
     
        x   =  UpSampling3D(size=(1,3,3))(x)

        ######################## FC Parts #############################
          
        x = concatenate([x, Coords])
        
        for feature in (self.fc_features[0:2]):  
            x        = Conv3D(filters = 60, 
                               kernel_size = (1,1,1), 
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x)
            x        = LeakyReLU()(x)
            x        = BatchNormalization()(x)
        
           
        x        = Conv3D(filters = 100, 
                   kernel_size = (1,1,1), 
                   kernel_initializer=Orthogonal(),
                   kernel_regularizer=regularizers.l2(self.L2))(x)
        x        = LeakyReLU()(x)
        x        = BatchNormalization()(x)
        

        x        = Conv3D(filters = 2, 
                   kernel_size = (1,1,1), 
                   kernel_initializer=Orthogonal(),
                   kernel_regularizer=regularizers.l2(self.L2))(x)
        x        = Activation('sigmoid')(x)
        
        model     = Model(inputs=[T1post, Coords], outputs=x)
    
        if self.loss_function == 'Dice':
            model.compile(loss=Generalised_dice_coef_multilabel2, optimizer=Adam(lr=self.learning_rate), metrics=['acc', dice_coef_multilabel0, dice_coef_multilabel1])
        elif self.loss_function == 'Multinomial':
            model.compile(loss='binary_crossentropy', optimizer=Adam(lr=self.learning_rate), metrics=['acc', dice_coef_multilabel0, dice_coef_multilabel1])
        return model



        
#dm = BreastSegmentor_v1(2, 3, 0.001, [0], 0.01, 0, 'Dice' )
#model = dm.createModel()            
#model.summary()  
#from keras.utils import plot_model
#plot_model(model, to_file='/home/deeperthought/Projects/MultiPriors_MSKCC/' +'BreastSegmentor_v1.png', show_shapes=True)    
#
#
#X = np.random.randn(1,13,75,75,1)
#y = np.random.binomial(n=1, p=0.5,size=81*2).reshape(1,1,9,9,2)
#y.shape
#
#coords = np.random.randn(1,1,9,9,3)
#
#yhat = model.predict([X, coords])
#yhat.shape
#
#model.fit([X, coords], y, epochs=10)
