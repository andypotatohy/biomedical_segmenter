# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 15:27:52 2019

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
        #############   High res pathway   ##################         
        x11        = Cropping3D(cropping = ((0,0),(20,20),(20,20)), input_shape=(None,None,None, self.num_channels),name = 'T1post_Detail')(T1post)
        
        # reduced original input by -40    : 13,35,35     
        for feature in (self.conv_features[0:7]):    # reduce in -36
            x11        = Conv3D(filters = feature, 
                               kernel_size = (1,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x11)
            x11        = LeakyReLU()(x11)                              
            x11        = BatchNormalization()(x11)         
            
        for feature in self.conv_features[0:6]:    # reduce all dimensions in -12
            x11        = Conv3D(filters = feature, 
                               kernel_size = (3,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x11)
            x11        = LeakyReLU()(x11)                              
            x11        = BatchNormalization()(x11)   
        # 1, 23, 23

            
        # output of pathway should be 1,9,9                
        #############   Context pathway   ##################  
            
#        x12        = Conv3D(filters = 20, 
#                               kernel_size = (3,3,3),
#                               dilation_rate=(3,3,3),
#                               padding='same', 
#                               kernel_initializer=Orthogonal(),
#                               kernel_regularizer=regularizers.l2(self.L2))(T1post)
#        x12        = LeakyReLU()(x12)                              
#        x12        = BatchNormalization()(x12)             
            
        x12 = AveragePooling3D(pool_size=(1, 3, 3), strides=(1,1,1), name='T1post_Context')(T1post)    
        # (13, 25, 25)        

        for _ in range(2):    
            x12        = Conv3D(filters = 30, 
                               kernel_size = (1,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x12)
            x12        = LeakyReLU()(x12)                              
            x12        = BatchNormalization()(x12) 
        
        # (13, 21, 21)        
        
        for _ in range(6):    
            x12        = Conv3D(filters = 30, 
                               kernel_size = (3,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x12)
            x12        = LeakyReLU()(x12)                              
            x12        = BatchNormalization()(x12) 
        # Result: (1,9,9)
        for _ in range(4):    
            x12        = Conv3D(filters = 30, 
                               kernel_size = (1,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x12)
            x12        = LeakyReLU()(x12)                              
            x12        = BatchNormalization()(x12)       

        x12 = UpSampling3D(size=(1,9,9))(x12)


        x1 = concatenate([x11,x12])

        
#        x1        = Conv3D(filters = 50, 
#                               kernel_size = (1,1,1),
#                               #padding='same', 
#                               kernel_initializer=Orthogonal(),
#                               kernel_regularizer=regularizers.l2(self.L2))(x1)
#        x1        = LeakyReLU()(x1)                              
#        x1        = BatchNormalization()(x1) 
        
        
        
        ########################  T1 pre pathway #########################
        #############   High res pathway   ##################         
        x21        = Cropping3D(cropping = ((0,0),(20,20),(20,20)), input_shape=(None,None,None, self.num_channels),name = 'T1pre_Detail')(T1pre)
        
        # reduced original input by -40    : 13,35,35     

        for feature in (self.conv_features[0:7]):    # reduce in -14
            x21        = Conv3D(filters = feature, 
                               kernel_size = (1,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x21)
            x21        = LeakyReLU()(x21)                              
            x21        = BatchNormalization()(x21)  
        
        for feature in self.conv_features[0:6]:    # reduce all dimensions in -12
            x21        = Conv3D(filters = feature, 
                               kernel_size = (3,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x21)
            x21        = LeakyReLU()(x21)                              
            x21        = BatchNormalization()(x21) 
        # 1, 23, 23
     
        # output of pathway should be 1,9,9                
            
        #############   Context pathway   ##################  

                    
        x22 = AveragePooling3D(pool_size=(1, 3, 3),name='T1pre_Context')(T1pre)    
        # (13, 25, 25)        

        for _ in range(2):    
            x22        = Conv3D(filters = 30, 
                               kernel_size = (1,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x22)
            x22        = LeakyReLU()(x22)                              
            x22        = BatchNormalization()(x22) 
        
        for _ in range(6):    
            x22        = Conv3D(filters = 30, 
                               kernel_size = (3,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x22)
            x22        = LeakyReLU()(x22)                              
            x22        = BatchNormalization()(x22) 
        for _ in range(4):    
            x22        = Conv3D(filters = 30, 
                               kernel_size = (1,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x22)
            x22        = LeakyReLU()(x22)                              
            x22        = BatchNormalization()(x22)       

        x22 = UpSampling3D(size=(1,9,9))(x22)

        x2 = concatenate([x21,x22])

        ########################  T2 pathway #########################
        #############   High res pathway   ##################         
        x31        = Cropping3D(cropping = ((0,0),(20,20),(20,20)), input_shape=(None,None,None, self.num_channels),name = 'T2_Detail')(T2)
        
        # reduced original input by -40    : 13,35,35     

        for feature in (self.conv_features[0:7]):    # reduce in -36
            x31        = Conv3D(filters = feature, 
                               kernel_size = (1,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x31)
            x31        = LeakyReLU()(x31)                              
            x31        = BatchNormalization()(x31)     
        
        for feature in self.conv_features[0:6]:    # reduce all dimensions in -12
            x31        = Conv3D(filters = feature, 
                               kernel_size = (3,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x31)
            x31        = LeakyReLU()(x31)                              
            x31        = BatchNormalization()(x31)    
        # 1, 23, 23

        # output of pathway should be 1,9,9                
        #############   Context pathway   ##################  
        # starting from (min) 13,75,75              

        x32 = AveragePooling3D(pool_size=(1, 3, 3),name='T2_Context')(T2)    
        # (13, 25, 25)        
        
        for _ in range(2):    
            x32        = Conv3D(filters = 30, 
                               kernel_size = (1,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x32)
            x32        = LeakyReLU()(x32)                              
            x32        = BatchNormalization()(x32)         
        
        
        for _ in range(6):    
            x32        = Conv3D(filters = 30, 
                               kernel_size = (3,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x32)
            x32        = LeakyReLU()(x32)                              
            x32        = BatchNormalization()(x32) 

        for _ in range(4):    
            x32        = Conv3D(filters = 30, 
                               kernel_size = (1,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x32)
            x32        = LeakyReLU()(x32)                              
            x32        = BatchNormalization()(x32)       

        x32 = UpSampling3D(size=(1,9,9))(x32)

        x3 = concatenate([x31,x32])

	########################  Merge Modalities #########################

        TPM = Input((None,None,None,1), name='TPM')

        x = concatenate([x1,x2,x3, TPM])   
          
        for feature in (self.fc_features[0:2]):  
            x        = Conv3D(filters = 100, 
                               kernel_size = (1,1,1), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x)
            x        = LeakyReLU()(x)
            x        = BatchNormalization()(x)
        
        
#        x        = Conv3D(filters = self.output_classes, 
#                           kernel_size = (1,1,1), 
#                           #kernel_initializer=he_normal(seed=seed),
#                           kernel_initializer=Orthogonal(),
#                           kernel_regularizer=regularizers.l2(self.L2))(x)
        
        
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
        	model.compile(loss=Generalised_dice_coef_multilabel2, optimizer=Adam(lr=self.learning_rate), metrics=['acc', dice_coef_multilabel0, dice_coef_multilabel1])
    	elif self.loss_function == 'Multinomial':
		model.compile(loss='binary_crossentropy', optimizer=Adam(lr=self.learning_rate), metrics=['acc', dice_coef_multilabel0, dice_coef_multilabel1])
        return model




# Debug
##        
#dm = DeepMedic(2, 3, 0.001, [0], 0.01, 0, 'Dice' )
#model = dm.createModel()            
#model.summary()  
#from keras.utils import plot_model
#plot_model(model, to_file='/home/hirsch/Documents/projects/Breast_segmentation/DeepPriors_package' +'/multiscale_TPM_withPooling.png', show_shapes=True)    
#
##
#X = np.random.randn(1,13,75,75,3,1)
#y = np.random.binomial(n=1, p=0.5,size=81*2).reshape(1,1,9,9,2)
#y.shape
#
#TPM = np.random.randn(1,1,9,9,1)
#
#yhat = model.predict([X[:,:,:,:,0],X[:,:,:,:,0],X[:,:,:,:,0],TPM])
#yhat.shape
#
#model.fit([X[:,:,:,:,0],X[:,:,:,:,0],X[:,:,:,:,0],TPM], y)
