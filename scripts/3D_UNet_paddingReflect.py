#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:26:02 2019

@author: deeperthought
"""
import sys
sys.path.append('/home/deeperthought/Projects/MultiPriors_MSKCC/scripts/')

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))


import numpy as np
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, Deconvolution3D
from keras.optimizers import Adam
from MultiPriors_Models_Collection import Generalised_dice_coef_multilabel2, dice_coef_multilabel_bin0, dice_coef_multilabel_bin1

#from unet3d.metrics import dice_coefficient_loss, get_label_dice_coefficient_function, dice_coefficient

#K.set_image_data_format("channels_first")

try:
    from keras.engine import merge
except ImportError:
    from keras.layers.merge import concatenate

from functools import partial

def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)


def weighted_dice_coefficient(y_true, y_pred, axis=(-3, -2, -1), smooth=0.00001):
    """
    Weighted dice coefficient. Default axis assumes a "channels first" data structure
    :param smooth:
    :param y_true:
    :param y_pred:
    :param axis:
    :return:
    """
    return K.mean(2. * (K.sum(y_true * y_pred,
                              axis=axis) + smooth/2)/(K.sum(y_true,
                                                            axis=axis) + K.sum(y_pred,
                                                                               axis=axis) + smooth))


def weighted_dice_coefficient_loss(y_true, y_pred):
    return -weighted_dice_coefficient(y_true, y_pred)


def label_wise_dice_coefficient(y_true, y_pred, label_index):
    return dice_coefficient(y_true[:, label_index], y_pred[:, label_index])


def get_label_dice_coefficient_function(label_index):
    f = partial(label_wise_dice_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'label_{0}_dice_coef'.format(label_index))
    return f


dice_coef = dice_coefficient
dice_coef_loss = dice_coefficient_loss


#input_shape=(8,64,64,2)    
#pool_size=(2, 2, 2)
#n_labels=2
#initial_learning_rate=0.00001
#deconvolution=True
#depth=4
#n_base_filters=32
#include_label_wise_dice_coefficients=False
#metrics=dice_coefficient
#batch_normalization=True
#activation_name="softmax"

def unet_model_3d(input_shape, pool_size=(2, 2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=False,
                  depth=4, n_base_filters=32, include_label_wise_dice_coefficients=False, metrics=dice_coefficient,
                  batch_normalization=True, activation_name="softmax"):
    """
    Builds the 3D UNet Keras model.f
    :param metrics: List metrics to be calculated during model training (default is dice coefficient).
    :param include_label_wise_dice_coefficients: If True and n_labels is greater than 1, model will report the dice
    coefficient for each label as metric.
    :param n_base_filters: The number of filters that the first layer in the convolution network will have. Following
    layers will contain a multiple of this number. Lowering this number will likely reduce the amount of memory required
    to train the model.
    :param depth: indicates the depth of the U-shape for the model. The greater the depth, the more max pooling
    layers will be added to the model. Lowering the depth may reduce the amount of memory required for training.
    :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size). The x, y, and z sizes must be
    divisible by the pool size to the power of the depth of the UNet, that is pool_size^depth.
    :param pool_size: Pool size for the max pooling operations.
    :param n_labels: Number of binary labels that the model is learning.
    :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
    :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of up-sampling. This
    increases the amount memory required during training.
    :return: Untrained 3D UNet Model
    """
    
    inputs = Input(input_shape)
    current_layer = inputs
    levels = list()


    # add levels with max pooling
    for layer_depth in range(depth):
        if layer_depth == depth-1:
            PADDING_BORDERS=[[0,0],[0,0],[1,1],[1,1], [0,0]]
            kernel = (1,3,3)
        else:
            PADDING_BORDERS=[[0,0],[1,1],[1,1],[1,1], [0,0]]
            kernel = (3,3,3)            
        layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_base_filters*(2**layer_depth),
                                          batch_normalization=batch_normalization, PADDING_BORDERS=PADDING_BORDERS, kernel=kernel)
        layer2 = create_convolution_block(input_layer=layer1, n_filters=n_base_filters*(2**layer_depth)*2,
                                          batch_normalization=batch_normalization, PADDING_BORDERS=PADDING_BORDERS, kernel=kernel)
        if layer_depth < depth - 1:
            current_layer = MaxPooling3D(pool_size=pool_size)(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])

    # add levels with up-convolution or up-sampling
    for layer_depth in range(depth-2, -1, -1):
        if layer_depth == depth-2:
            PADDING_BORDERS=[[0,0],[0,0],[1,1],[1,1], [0,0]]
            kernel = (1,3,3)
        else:
            PADDING_BORDERS=[[0,0],[1,1],[1,1],[1,1], [0,0]]
            kernel = (3,3,3)                    
        up_convolution = get_up_convolution(pool_size=pool_size, deconvolution=deconvolution,
                                            n_filters=current_layer._keras_shape[1])(current_layer)
        concat = concatenate([up_convolution, levels[layer_depth][1]], axis=-1)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                 input_layer=concat, batch_normalization=batch_normalization, PADDING_BORDERS=PADDING_BORDERS, kernel=kernel)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                 input_layer=current_layer,
                                                 batch_normalization=batch_normalization, PADDING_BORDERS=PADDING_BORDERS, kernel=kernel)


    # Convert from 3D to 2D, for MSK Labels:
    #current_layer = Conv3D(4, (1, 1, 1))(current_layer, activation='relu')
    
    final_convolution = Conv3D(n_labels, (8, 1, 1))(current_layer)
    act = Activation(activation_name)(final_convolution)
    model = Model(inputs=inputs, outputs=act)

    if not isinstance(metrics, list):
        metrics = [metrics]

    if include_label_wise_dice_coefficients and n_labels > 1:
        label_wise_dice_metrics = [get_label_dice_coefficient_function(index) for index in range(n_labels)]
        if metrics:
            metrics = metrics + label_wise_dice_metrics
        else:
            metrics = label_wise_dice_metrics

    #model.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coefficient_loss, metrics=metrics)
    model.compile(loss=Generalised_dice_coef_multilabel2, optimizer=Adam(lr=initial_learning_rate), metrics=['acc', dice_coef_multilabel_bin0, dice_coef_multilabel_bin1])
    return model


def create_convolution_block(input_layer, n_filters, batch_normalization=False, kernel=(3,3,3), activation=None,
                             padding='same', PADDING_BORDERS=[[0,0],[1,1],[1,1],[1,1], [0,0]], strides=(1, 1, 1), 
                             instance_normalization=False):
    """
    :param strides:
    :param input_layer:
    :param n_filters:
    :param batch_normalization:
    :param kernel:
    :param activation: Keras activation layer to use. (default is 'relu')
    :param padding:
    :return:
    """
    if padding=='same':
        input_layer = Lambda(lambda x: tf.pad(x, PADDING_BORDERS, 'REFLECT'))(input_layer)        
    layer = Conv3D(n_filters, kernel, padding='valid', strides=strides)(input_layer)
    
    if batch_normalization:
        layer = BatchNormalization()(layer)
    elif instance_normalization:
        try:
            from keras_contrib.layers.normalization import InstanceNormalization
        except ImportError:
            raise ImportError("Install keras_contrib in order to use instance normalization."
                              "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
        layer = InstanceNormalization(axis=-1)(layer)
    if activation is None:
        return Activation('relu')(layer)
    else:
        return activation()(layer)


def compute_level_output_shape(n_filters, depth, pool_size, image_shape):
    """
    Each level has a particular output shape based on the number of filters used in that level and the depth or number 
    of max pooling operations that have been done on the data at that point.
    :param image_shape: shape of the 3d image.
    :param pool_size: the pool_size parameter used in the max pooling operation.
    :param n_filters: Number of filters used by the last node in a given level.
    :param depth: The number of levels down in the U-shaped model a given node is.
    :return: 5D vector of the shape of the output node 
    """
    output_image_shape = np.asarray(np.divide(image_shape, np.power(pool_size, depth)), dtype=np.int32).tolist()
    return tuple([None, n_filters] + output_image_shape)


def get_up_convolution(n_filters, pool_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                       deconvolution=False):
    if deconvolution:
        return Deconvolution3D(filters=n_filters, kernel_size=kernel_size,
                               strides=strides)
    else:
        return UpSampling3D(size=pool_size)

    
#The x, y, and z sizes must be divisible by the pool size to the power of the depth of the UNet, that is pool_size^depth.
    
import tensorflow as tf
from keras.layers import Lambda

input_shape=(8,64,64,2)    

model =  unet_model_3d(input_shape, pool_size=(2, 2, 2), n_labels=2, initial_learning_rate=0.00001, deconvolution=True,
                  depth=4, n_base_filters=32, include_label_wise_dice_coefficients=False, metrics=dice_coefficient,
                  batch_normalization=False, activation_name="softmax")  

#model.summary()
#model.input
#model.output
#X = np.ones((32,8,64,64,2))
#y_pred = model.predict([X])
#y_pred.shape
#
#model.fit([X],y_pred, epochs=5)

#from keras.utils import plot_model
#plot_model(unet_3d, to_file='/home/deeperthought/Projects/MultiPriors_MSKCC/models/UNet_3D.png', show_shapes=True)
