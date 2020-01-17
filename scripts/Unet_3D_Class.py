#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 15:19:39 2019

@author: deeperthought
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:26:02 2019

@author: deeperthought
"""
#import tensorflow as tf
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#tf.keras.backend.set_session(tf.Session(config=config))

import sys
sys.path.append('/home/deeperthought/Projects/MultiPriors_MSKCC/scripts/')
import numpy as np

#import tensorflow as tf
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#config.gpu_options.visible_device_list="0"
#tf.keras.backend.set_session(tf.Session(config=config))


from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, AveragePooling3D, UpSampling3D, Activation, BatchNormalization, Deconvolution3D
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, UpSampling2D, Deconvolution2D
from keras.layers.convolutional import Cropping3D
from keras.optimizers import Adam
from MultiPriors_Models_Collection import Generalised_dice_coef_multilabel2, dice_coef_multilabel_bin0, dice_coef_multilabel_bin1

#from unet3d.metrics import dice_coefficient_loss, get_label_dice_coefficient_function, dice_coefficient

#K.set_image_data_format("channels_first")

from keras.layers.merge import concatenate
from functools import partial
import tensorflow as tf
from keras.layers import Lambda


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


def create_convolution_block_reflectPadding(input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3), activation=None,
                             padding='same', PADDING_BORDERS=[[0,0],[1,1], [1,1], [1,1], [0,0]], strides=(1, 1, 1), 
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

def create_convolution_block(input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3), activation=None,
                             padding='same', strides=(1, 1, 1), instance_normalization=False):
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
    layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
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


def get_up_convolution(n_filters, pool_size=(2,2,2), kernel_size=(2,2,2), strides=(2, 2, 2),
                       deconvolution=True, bilinear_upsampling=False):
    if deconvolution:
        if bilinear_upsampling:
            return Deconvolution3D(filters=n_filters, kernel_size=(3,3,3),
                                   strides=strides, trainable=False)#, kernel_initializer=make_bilinear_filter_5D(shape=(3,3,3,n_filters,n_filters)), trainable=False)
        else:
            return Deconvolution3D(filters=n_filters, kernel_size=(2,2,2),
                                   strides=strides)            
    else:
        return UpSampling3D(size=pool_size)

def my_init(shape, dtype=None):
    return K.random_normal(shape, dtype=dtype)

def make_bilinear_filter_5D(N=10,dtype='float32'):
    r = np.array([0.5, 1., 0.5], dtype=dtype)
    x, y, z = np.meshgrid(r, r, r, indexing='ij')   
    D = x*y*z
    w = np.zeros((3,3,3,N,N), dtype=dtype)
    for i in range(N):
        w[:,:,:,i,i] = D
    return w


def UNet_v0(input_shape =  (8,64,64,2), pool_size=(2, 2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=False,
                      depth=4, n_base_filters=32, include_label_wise_dice_coefficients=False, 
                      batch_normalization=False, activation_name="sigmoid"):
        """ Simple version, padding 'same' on every layer, output size is equal to input size. Has border artifacts and checkerboard artifacts """
        inputs = Input(input_shape)
        current_layer = inputs
        levels = list()
    
        # add levels with max pooling
        for layer_depth in range(depth):
            layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_base_filters*(2**layer_depth),
                                              batch_normalization=batch_normalization, padding='same')
            layer2 = create_convolution_block(input_layer=layer1, n_filters=n_base_filters*(2**layer_depth)*2,
                                              batch_normalization=batch_normalization, padding='same')
            if layer_depth < depth - 1:
                current_layer = MaxPooling3D(pool_size=(2,2,2))(layer2)
                levels.append([layer1, layer2, current_layer])
            else:
                current_layer = layer2
                levels.append([layer1, layer2])

        for layer_depth in range(depth-2, -1, -1):
            
            up_convolution = get_up_convolution(pool_size=pool_size, deconvolution=deconvolution,
                                                n_filters=current_layer._keras_shape[-1])(current_layer)

            concat = concatenate([up_convolution, levels[layer_depth][1]] , axis=-1)
            current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[-1],
                                                     input_layer=concat, batch_normalization=batch_normalization, padding='same')
            current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[-1],
                                                     input_layer=current_layer, batch_normalization=batch_normalization, padding='same')
            #L += 1


        #current_layer = Cropping3D([(0,0),(4,4),(4,4)])(current_layer)
    
        final_convolution = Conv3D(2, (8, 1, 1))(current_layer)
        
       
        act = Activation(activation_name)(final_convolution)
        model = Model(inputs=inputs, outputs=act)
        model.compile(loss=Generalised_dice_coef_multilabel2, optimizer=Adam(lr=initial_learning_rate), metrics=['acc', dice_coef_multilabel_bin0, dice_coef_multilabel_bin1])

        return model


def UNet_v0_A(input_shape=(8,64,64,2)  , pool_size=(2, 2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=True,
                  depth=4, n_base_filters=32, include_label_wise_dice_coefficients=False, metrics=dice_coefficient,
                  batch_normalization=True, activation_name="sigmoid"):
    """
    Like model3, but padding is using reflect. A little less border effect
    """
    inputs = Input(input_shape)
    current_layer = inputs
    levels = list()


    # add levels with max pooling
    for layer_depth in range(depth):
        layer1 = create_convolution_block_reflectPadding(input_layer=current_layer, n_filters=n_base_filters*(2**layer_depth),
                                          batch_normalization=batch_normalization)
        layer2 = create_convolution_block_reflectPadding(input_layer=layer1, n_filters=n_base_filters*(2**layer_depth)*2,
                                          batch_normalization=batch_normalization)
        if layer_depth < depth - 1:
            current_layer = MaxPooling3D(pool_size=pool_size)(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])

    # add levels with up-convolution or up-sampling
    for layer_depth in range(depth-2, -1, -1):
        up_convolution = get_up_convolution(pool_size=pool_size, deconvolution=deconvolution,
                                            n_filters=current_layer._keras_shape[-1])(current_layer)
        concat = concatenate([up_convolution, levels[layer_depth][1]], axis=-1)
        current_layer = create_convolution_block_reflectPadding(n_filters=levels[layer_depth][1]._keras_shape[-1],
                                                 input_layer=concat, batch_normalization=batch_normalization)
        current_layer = create_convolution_block_reflectPadding(n_filters=levels[layer_depth][1]._keras_shape[-1],
                                                 input_layer=current_layer,
                                                 batch_normalization=batch_normalization)


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

def UNet_v1(input_shape=(12,76,76,2)   , pool_size=(2, 2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=True,
                  depth=4, n_base_filters=32, include_label_wise_dice_coefficients=False, 
                  batch_normalization=True, activation_name="sigmoid"):
    """
    Only valid padding, no border artifact. Upsampling with transposed convolution -learnable- so checkerboard artifacts.
    """
    inputs = Input(input_shape)
    current_layer = inputs
    levels = list()

    # add levels with max pooling
    kernel = [(3,3,3),(1,3,3),(1,1,1),(1,1,1)]
    padding = ['valid','valid','valid','valid']
    pool_size = [(2,2,2),(2,2,2),(2,2,2)]
    for layer_depth in range(depth):
        layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_base_filters*(2**layer_depth),
                                          batch_normalization=batch_normalization, padding=padding[layer_depth], kernel=kernel[layer_depth])
        layer2 = create_convolution_block(input_layer=layer1, n_filters=n_base_filters*(2**layer_depth)*2,
                                          batch_normalization=batch_normalization, padding=padding[layer_depth], kernel=kernel[layer_depth])
        if layer_depth < depth - 1:
            current_layer = MaxPooling3D(pool_size=pool_size[layer_depth])(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])

    # add levels with up-convolution or up-sampling
    CROP_BORDER = [(0,8,8),(0,0,0),(0,0,0)]
    
    for layer_depth in range(depth-2, -1, -1):
        
        up_convolution = get_up_convolution(pool_size=(2,2,2), deconvolution=deconvolution,
                                            n_filters=current_layer._keras_shape[-1]/2)(current_layer)
        # NEED TO CROP levels[layer_depth]
        levels[layer_depth][1] = Cropping3D([(CROP_BORDER[layer_depth][0],CROP_BORDER[layer_depth][0]),
                                             (CROP_BORDER[layer_depth][1],CROP_BORDER[layer_depth][1]),
                                             (CROP_BORDER[layer_depth][2],CROP_BORDER[layer_depth][2])])(levels[layer_depth][1])
        
        concat = concatenate([up_convolution, levels[layer_depth][1]] , axis=-1)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[-1], kernel=kernel[layer_depth],
                                                 input_layer=concat, batch_normalization=batch_normalization, padding=padding[layer_depth])
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[-1], kernel=kernel[layer_depth],
                                                 input_layer=current_layer,
                                                 batch_normalization=batch_normalization, padding=padding[layer_depth])
    
    final_convolution = Conv3D(n_labels, (4, 1, 1))(current_layer)
    act = Activation(activation_name)(final_convolution)
    model = Model(inputs=inputs, outputs=act)

    model.compile(loss=Generalised_dice_coef_multilabel2, optimizer=Adam(lr=initial_learning_rate), metrics=['acc', dice_coef_multilabel_bin0, dice_coef_multilabel_bin1])
    return model

def UNet_v1_A(input_shape=(8,64,64,2)   , pool_size=(2, 2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=False,
                  depth=4, n_base_filters=32, include_label_wise_dice_coefficients=False, 
                  batch_normalization=False, activation_name="sigmoid"):
    """
    Like the first version, but here in the bottom layers I have 3,3,3 convolutions + 'same' padding.
    A lot more parameters, but still doesnt introduce boundary artifacts
    """
    inputs = Input(input_shape)
    current_layer = inputs
    levels = list()

    # add levels with max pooling
    kernel = [(3,3,3),(1,3,3),(3,3,3),(3,3,3)]#(1,1,1),(1,1,1)]
    padding = ['valid','valid','same','same']
    pool_size = [(2,2,2),(2,2,2),(2,2,2)]
    for layer_depth in range(depth):
        layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_base_filters*(2**layer_depth),
                                          batch_normalization=batch_normalization, padding=padding[layer_depth], kernel=kernel[layer_depth])
        layer2 = create_convolution_block(input_layer=layer1, n_filters=n_base_filters*(2**layer_depth)*2,
                                          batch_normalization=batch_normalization, padding=padding[layer_depth], kernel=kernel[layer_depth])
        if layer_depth < depth - 1:
            current_layer = MaxPooling3D(pool_size=pool_size[layer_depth])(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])

    # add levels with up-convolution or up-sampling
    CROP_BORDER = [(0,8,8),(0,0,0),(0,0,0)]
    
    for layer_depth in range(depth-2, -1, -1):
        
        up_convolution = get_up_convolution(pool_size=(2,2,2), deconvolution=deconvolution,
                                            n_filters=current_layer._keras_shape[-1]/2)(current_layer)
        # NEED TO CROP levels[layer_depth]
        levels[layer_depth][1] = Cropping3D([(CROP_BORDER[layer_depth][0],CROP_BORDER[layer_depth][0]),
                                             (CROP_BORDER[layer_depth][1],CROP_BORDER[layer_depth][1]),
                                             (CROP_BORDER[layer_depth][2],CROP_BORDER[layer_depth][2])])(levels[layer_depth][1])
        
        concat = concatenate([up_convolution, levels[layer_depth][1]] , axis=-1)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[-1], kernel=kernel[layer_depth],
                                                 input_layer=concat, batch_normalization=batch_normalization, padding=padding[layer_depth])
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[-1], kernel=kernel[layer_depth],
                                                 input_layer=current_layer,
                                                 batch_normalization=batch_normalization, padding=padding[layer_depth])


    # Convert from 3D to 2D, for MSK Labels:
    #current_layer = Conv3D(4, (1, 1, 1))(current_layer, activation='relu')
    
    final_convolution = Conv3D(n_labels, (4, 1, 1))(current_layer)
    act = Activation(activation_name)(final_convolution)
    model = Model(inputs=inputs, outputs=act)

    model.compile(loss=Generalised_dice_coef_multilabel2, optimizer=Adam(lr=initial_learning_rate), metrics=['acc', dice_coef_multilabel_bin0, dice_coef_multilabel_bin1])
    return model


def UNet_v2(input_shape=(12,76,76,2), pool_size=(2, 2, 2), n_labels=2, initial_learning_rate=0.00001, deconvolution=True,
                  depth=4, n_base_filters=32, include_label_wise_dice_coefficients=False, 
                  batch_normalization=True, activation_name="softmax", bilinear_upsampling=False):
    """
    Like model v1 but with bilinear upsampling using transposed Convolution 3D with fixed weights of size 3,3,3
    """
    inputs = Input(input_shape)
    current_layer = inputs
    levels = list()
    kernel = [(3,3,3),(1,3,3),(1,1,1),(1,1,1)]
    
    # DOWNSAMPLING
    for layer_depth in range(depth):
        layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_base_filters*(2**layer_depth),
                                          batch_normalization=batch_normalization, padding='valid', kernel=kernel[layer_depth])
        layer2 = create_convolution_block(input_layer=layer1, n_filters=n_base_filters*(2**layer_depth)*2,
                                          batch_normalization=batch_normalization, padding='valid', kernel=kernel[layer_depth])
        if layer_depth < depth - 1:
            current_layer = MaxPooling3D(pool_size=(2,2,2), strides=2, padding='valid')(layer2)#pool_size=(2,2,2))(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])
    
    # UPSAMPLING
    for layer_depth in range(depth-2, -1, -1):
        up_convolution = get_up_convolution(kernel_size=(3, 3, 3), deconvolution=deconvolution, n_filters=current_layer._keras_shape[-1], 
                                            bilinear_upsampling=bilinear_upsampling)(current_layer)
        if bilinear_upsampling: 
            up_convolution = Cropping3D(((0,1),(0,1),(0,1)))(up_convolution)
            #up_convolution = Cropping3D(((1,1),(1,1),(1,1)))(up_convolution)
            #levels[layer_depth][1] = Cropping3D(((1,0),(1,0),(1,0)))(levels[layer_depth][1])
            
        if layer_depth == 0: 
            levels[layer_depth][1] = Cropping3D([(0,0),(8,8),(8,8)])(levels[layer_depth][1])                

        concat = concatenate([up_convolution, levels[layer_depth][1]] , axis=-1)     
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[-1], kernel=kernel[layer_depth],
                                                 input_layer=concat, batch_normalization=batch_normalization, padding='valid')     
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[-1], kernel=kernel[layer_depth],
                                                 input_layer=current_layer, batch_normalization=batch_normalization, padding='valid')
    
    final_convolution = Conv3D(n_labels, (4, 1, 1))(current_layer)
    act = Activation(activation_name)(final_convolution)
    model = Model(inputs=inputs, outputs=act)
    
    if bilinear_upsampling:
        up = [x for x in model.layers if 'Conv3DTranspose' in str(x)]
        for layer in up:
            print(layer.trainable)
            w0 = layer.get_weights()
            w0[0]
            N = w0[0].shape[-1]
            W = make_bilinear_filter_5D(N)
            b = np.zeros((N,))
            layer.set_weights([W,b])
            layer.trainable = False
    
    model.compile(loss=Generalised_dice_coef_multilabel2, optimizer=Adam(lr=initial_learning_rate), metrics=['acc', dice_coef_multilabel_bin0, dice_coef_multilabel_bin1])

    return model
    


def UNet_v3(input_shape=(12,76,76,2), pool_size=(2, 2, 2), n_labels=2, initial_learning_rate=0.00001, deconvolution=True,
                  depth=4, n_base_filters=32, include_label_wise_dice_coefficients=False, 
                  batch_normalization=True, activation_name="softmax", bilinear_upsampling=False):
    """
    Like model v1 but with bilinear upsampling using transposed Convolution 3D with fixed weights of size 3,3,3
    
    Input odd : 
    - output is same size (actually input + 1 but keras removes one border)
    """
    inputs = Input(input_shape)
    current_layer = inputs
    levels = list()
    kernel = [(3,3,3),(1,3,3),(1,1,1),(1,1,1)]
    
    # DOWNSAMPLING
    for layer_depth in range(depth):
        layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_base_filters*(2**layer_depth),
                                          batch_normalization=batch_normalization, padding='valid', kernel=kernel[layer_depth])
        layer2 = create_convolution_block(input_layer=layer1, n_filters=n_base_filters*(2**layer_depth)*2,
                                          batch_normalization=batch_normalization, padding='valid', kernel=kernel[layer_depth])
        if layer_depth < depth - 1:
            current_layer = MaxPooling3D(pool_size=(2,2,2), strides=2, padding='valid')(layer2)#pool_size=(2,2,2))(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])
    
    kernel = [(1,3,3),(1,3,3),(1,1,1),(1,1,1)]
    Cropping = [(7,0),(3,0),(1,0)] # from layers 0 to 2, inverse order!
    # UPSAMPLING
    for layer_depth in range(depth-2, -1, -1):
        up_convolution = get_up_convolution(kernel_size=(3, 3, 3), deconvolution=deconvolution, n_filters=current_layer._keras_shape[-1], 
                                            bilinear_upsampling=bilinear_upsampling)(current_layer)
        if bilinear_upsampling: 
            #up_convolution = Cropping3D(((0,1),(0,1),(0,1)))(up_convolution)
            up_convolution = Cropping3D(((1,1),(1,1),(1,1)))(up_convolution)
            levels[layer_depth][1] = Cropping3D((Cropping[layer_depth],Cropping[layer_depth],Cropping[layer_depth]))(levels[layer_depth][1])
            
        if layer_depth == 0: 
            levels[layer_depth][1] = Cropping3D([(0,0),(8,8),(8,8)])(levels[layer_depth][1])                

        concat = concatenate([up_convolution, levels[layer_depth][1]] , axis=-1)     
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[-1], kernel=kernel[layer_depth],
                                                 input_layer=concat, batch_normalization=batch_normalization, padding='valid')     
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[-1], kernel=kernel[layer_depth],
                                                 input_layer=current_layer, batch_normalization=batch_normalization, padding='valid')
    
    final_convolution = Conv3D(n_labels, (1, 1, 1))(current_layer)
    act = Activation(activation_name)(final_convolution)
    model = Model(inputs=inputs, outputs=act)
    
    if bilinear_upsampling:
        up = [x for x in model.layers if 'Conv3DTranspose' in str(x)]
        for layer in up:
            print(layer.trainable)
            w0 = layer.get_weights()
            w0[0]
            N = w0[0].shape[-1]
            W = make_bilinear_filter_5D(N)
            b = np.zeros((N,))
            layer.set_weights([W,b])
            layer.trainable = False
    
    model.compile(loss=Generalised_dice_coef_multilabel2, optimizer=Adam(lr=initial_learning_rate), metrics=['acc', dice_coef_multilabel_bin0, dice_coef_multilabel_bin1])

    return model
    

#input_shape=(19,75,75,2)
#pool_size=(2, 2, 2)
#n_labels=2
#initial_learning_rate=0.00001
#deconvolution=True
#depth=4
#n_base_filters=32
#include_label_wise_dice_coefficients=False
#batch_normalization=True
#activation_name="softmax"
#bilinear_upsampling=True


def UNet_v4(input_shape=(19,75,75,3), pool_size=(2, 2, 2), n_labels=2, initial_learning_rate=0.00001, deconvolution=True,
                  depth=4, n_base_filters=32, include_label_wise_dice_coefficients=False, 
                  batch_normalization=True, activation_name="softmax", bilinear_upsampling=True):
    """
    Like model v1 but with bilinear upsampling using transposed Convolution 3D with fixed weights of size 3,3,3
    
    Input odd : 
    - output is same size (actually input + 1 but keras removes one border)
    
    Downsampling is done using average pooling with kernels of size 3 and stride 2. This mimic the upsampling done by the transposed
    convolutiion of kernel 3 and stride. This operation leaves 1 invalid pixel per side, which need to be cropped, same with the input.
    This downsampling and upsampling is the method that best preserves a signal (compared: average and max pooling with kernel 2 stride 2,
    which makes unclear which border pixel needs to be cropped, and also compared against max pooling kernel 3 stride 2, which introduces 
    border artifacts due to the overlap max pooling).
    """
    input_shape = (19,75,75,3)
    inputs = Input(input_shape)
    current_layer = inputs
    levels = list()
    kernel = [(3,3,3),(1,3,3),(1,1,1),(1,1,1)]
    # DOWNSAMPLING
    for layer_depth in range(depth):
        layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_base_filters*(2**layer_depth),
                                          batch_normalization=batch_normalization, padding='valid', kernel=kernel[layer_depth])
        layer2 = create_convolution_block(input_layer=layer1, n_filters=n_base_filters*(2**layer_depth)*2,
                                          batch_normalization=batch_normalization, padding='valid', kernel=kernel[layer_depth])
        if layer_depth < depth - 1:
            current_layer = AveragePooling3D(pool_size=(3,3,3), strides=2, padding='valid')(layer2)#pool_size=(2,2,2))(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])
    
    kernel = [(1,3,3),(1,3,3),(1,1,1),(1,1,1)]
    Cropping_z_axis = [(7,7),(3,3),(1,1)]
    Cropping = [(15,15),(3,3),(1,1)] # from layers 0 to 2, inverse order!
    # UPSAMPLING
    for layer_depth in range(depth-2, -1, -1):
        up_convolution = get_up_convolution(kernel_size=(3, 3, 3), deconvolution=deconvolution, n_filters=current_layer._keras_shape[-1], 
                                            bilinear_upsampling=bilinear_upsampling)(current_layer)
        
        if bilinear_upsampling: 
            #up_convolution = Cropping3D(((0,1),(0,1),(0,1)))(up_convolution)
            up_convolution = Cropping3D(((1,1),(1,1),(1,1)))(up_convolution)
            levels[layer_depth][1] = Cropping3D((Cropping_z_axis[layer_depth],
                                                 Cropping[layer_depth],
                                                 Cropping[layer_depth]))(levels[layer_depth][1])
            
        #if layer_depth == 0: 
        #    levels[layer_depth][1] = Cropping3D([(0,0),(8,8),(8,8)])(levels[layer_depth][1])                

        concat = concatenate([up_convolution, levels[layer_depth][1]] , axis=-1)     
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[-1], kernel=kernel[layer_depth],
                                                 input_layer=concat, batch_normalization=batch_normalization, padding='valid')     
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[-1], kernel=kernel[layer_depth],
                                                 input_layer=current_layer, batch_normalization=batch_normalization, padding='valid')
    
    final_convolution = Conv3D(n_labels, (1, 1, 1))(current_layer)
    act = Activation(activation_name)(final_convolution)
    model = Model(inputs=inputs, outputs=act)
    
    if bilinear_upsampling:
        up = [x for x in model.layers if 'Conv3DTranspose' in str(x)]
        for layer in up:
            print(layer.trainable)
            w0 = layer.get_weights()
            w0[0]
            N = w0[0].shape[-1]
            W = make_bilinear_filter_5D(N)
            b = np.zeros((N,))
            layer.set_weights([W,b])
            layer.trainable = False
    
    model.compile(loss=Generalised_dice_coef_multilabel2, optimizer=Adam(lr=initial_learning_rate), metrics=['acc', dice_coef_multilabel_bin0, dice_coef_multilabel_bin1])

    return model


from keras.initializers import Orthogonal
from keras.layers import LeakyReLU, Reshape
from keras import regularizers

#
#input_shape=(19,75,75,4)
#pool_size=(2, 2, 2)
#n_labels=2
#initial_learning_rate=0.00001
#deconvolution=True
#depth=4
#n_base_filters=32
#include_label_wise_dice_coefficients=False
#batch_normalization=True
#activation_name="softmax"
#bilinear_upsampling=True

def UNet_v4_TPM(input_shape=(19,75,75,4), pool_size=(2, 2, 2), n_labels=2, initial_learning_rate=0.00001, deconvolution=True,
                  depth=4, n_base_filters=32, include_label_wise_dice_coefficients=False, 
                  batch_normalization=True, activation_name="softmax", bilinear_upsampling=True):
    
    inputs = Input(input_shape)
        
    TPM = Lambda(lambda x: x[:,:,:,:,-1], output_shape= input_shape[:-1])(inputs)
    TPM = Reshape((19,75,75,1))(TPM)
    TPM = Cropping3D(((9,9),(19,19),(19,19)))(TPM)
    
    current_layer = inputs
    levels = list()
    kernel = [(3,3,3),(1,3,3),(1,1,1),(1,1,1)]
    # DOWNSAMPLING
    for layer_depth in range(depth):
        layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_base_filters*(2**layer_depth),
                                          batch_normalization=batch_normalization, padding='valid', kernel=kernel[layer_depth])
        layer2 = create_convolution_block(input_layer=layer1, n_filters=n_base_filters*(2**layer_depth)*2,
                                          batch_normalization=batch_normalization, padding='valid', kernel=kernel[layer_depth])
        if layer_depth < depth - 1:
            current_layer = AveragePooling3D(pool_size=(3,3,3), strides=2, padding='valid')(layer2)#pool_size=(2,2,2))(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])
    
    kernel = [(1,3,3),(1,3,3),(1,1,1),(1,1,1)]
    Cropping_z_axis = [(7,7),(3,3),(1,1)]
    Cropping = [(15,15),(3,3),(1,1)] # from layers 0 to 2, inverse order!
    # UPSAMPLING
    for layer_depth in range(depth-2, -1, -1):
        up_convolution = get_up_convolution(kernel_size=(3, 3, 3), deconvolution=deconvolution, n_filters=current_layer._keras_shape[-1], 
                                            bilinear_upsampling=bilinear_upsampling)(current_layer)
        
        if bilinear_upsampling: 
            #up_convolution = Cropping3D(((0,1),(0,1),(0,1)))(up_convolution)
            up_convolution = Cropping3D(((1,1),(1,1),(1,1)))(up_convolution)
            levels[layer_depth][1] = Cropping3D((Cropping_z_axis[layer_depth],
                                                 Cropping[layer_depth],
                                                 Cropping[layer_depth]))(levels[layer_depth][1])
            
        #if layer_depth == 0: 
        #    levels[layer_depth][1] = Cropping3D([(0,0),(8,8),(8,8)])(levels[layer_depth][1])                

        concat = concatenate([up_convolution, levels[layer_depth][1]] , axis=-1)     
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[-1], kernel=kernel[layer_depth],
                                                 input_layer=concat, batch_normalization=batch_normalization, padding='valid')     
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[-1], kernel=kernel[layer_depth],
                                                 input_layer=current_layer, batch_normalization=batch_normalization, padding='valid')

    Coords = Input((1,37,37,3))
    
    current_layer = concatenate([current_layer, TPM, Coords])
    
    current_layer = Conv3D(levels[layer_depth][1]._keras_shape[-1]*2, (1, 1, 1))(current_layer)

    
    final_convolution = Conv3D(n_labels, (1, 1, 1))(current_layer)
    act = Activation(activation_name)(final_convolution)
    
    model = Model(inputs=[inputs, Coords], outputs=act)
    
    if bilinear_upsampling:
        up = [x for x in model.layers if 'Conv3DTranspose' in str(x)]
        for layer in up:
            print(layer.trainable)
            w0 = layer.get_weights()
            w0[0]
            N = w0[0].shape[-1]
            W = make_bilinear_filter_5D(N)
            b = np.zeros((N,))
            layer.set_weights([W,b])
            layer.trainable = False
    
    model.compile(loss=Generalised_dice_coef_multilabel2, optimizer=Adam(lr=initial_learning_rate), metrics=['acc', dice_coef_multilabel_bin0, dice_coef_multilabel_bin1])

    return model


    
#input_shape=(3, 256,256,1)
#pool_size=(1, 2, 2)
#n_labels=2
#initial_learning_rate=0.00001
#deconvolution=False
#depth=3
#n_base_filters=32
#include_label_wise_dice_coefficients=False
#batch_normalization=True
#activation_name="softmax"
#bilinear_upsampling=True
    
def UNet_v0_BreastMask(input_shape =  (3, 256,256,1), pool_size=(1, 2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=False,
                      depth=4, n_base_filters=32, include_label_wise_dice_coefficients=False, 
                      batch_normalization=False, activation_name="sigmoid"):
        """ Simple version, padding 'same' on every layer, output size is equal to input size. Has border artifacts and checkerboard artifacts """
        inputs = Input(input_shape)
        levels = list()
    
        coords = Input((1,256,256,3))

        current_layer = Conv3D(8, (3, 1, 1))(inputs)

        current_layer = concatenate([current_layer, coords])
    
        # add levels with max pooling
        for layer_depth in range(depth):
            layer1 = create_convolution_block(input_layer=current_layer, kernel=(1,3,3), n_filters=16*(layer_depth+1),
                                              batch_normalization=batch_normalization, padding='same')
            layer2 = create_convolution_block(input_layer=layer1, kernel=(1,3,3),  n_filters=16*(layer_depth+1),
                                              batch_normalization=batch_normalization, padding='same')
            if layer_depth < depth - 1:
                current_layer = MaxPooling3D(pool_size=(1,2,2))(layer2)
                levels.append([layer1, layer2, current_layer])
            else:
                current_layer = layer2
                levels.append([layer1, layer2])

        for layer_depth in range(depth-2, -1, -1):
            
            up_convolution = get_up_convolution(pool_size=(1,2,2), deconvolution=deconvolution,
                                                n_filters=16*(layer_depth+1))(current_layer)

            concat = concatenate([up_convolution, levels[layer_depth][1]] , axis=-1)
            current_layer = create_convolution_block(n_filters=16*(layer_depth+1),kernel=(1,3,3), 
                                                     input_layer=concat, batch_normalization=batch_normalization, padding='same')
            current_layer = create_convolution_block(n_filters=16*(layer_depth+1),kernel=(1,3,3), 
                                                     input_layer=current_layer, batch_normalization=batch_normalization, padding='same')
            #L += 1


        #current_layer = Cropping3D([(0,0),(4,4),(4,4)])(current_layer)
        current_layer = concatenate([current_layer, coords])
    
        final_convolution = Conv3D(2, (1, 1, 1))(current_layer)
        
       
        act = Activation(activation_name)(final_convolution)
        model = Model(inputs=[inputs, coords], outputs=act)
        model.compile(loss=Generalised_dice_coef_multilabel2, optimizer=Adam(lr=initial_learning_rate), metrics=['acc', dice_coef_multilabel_bin0, dice_coef_multilabel_bin1])

        return model
  
#input_shape=(3, 512, 512,4)
#pool_size=(1, 2, 2)
#n_labels=2
#initial_learning_rate=0.00001
#deconvolution=False
#depth=6
#n_base_filters=32
#include_label_wise_dice_coefficients=False
#batch_normalization=True
#activation_name="softmax"
#bilinear_upsampling=True

def UNet_v0_TumorSegmenter(input_shape =  (3, 512, 512,2), pool_size=(1, 2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=False,
                      depth=4, n_base_filters=32, include_label_wise_dice_coefficients=False, 
                      batch_normalization=False, activation_name="softmax"):
        """ Simple version, padding 'same' on every layer, output size is equal to input size. Has border artifacts and checkerboard artifacts """
        inputs = Input(input_shape)
        levels = list()
    
        coords = Input((1,512,512,3))

        current_layer = Conv3D(8, (3, 1, 1))(inputs)

        current_layer = concatenate([current_layer, coords])
    
        # add levels with max pooling
        for layer_depth in range(depth):
            layer1 = create_convolution_block(input_layer=current_layer, kernel=(1,3,3), n_filters=16*(layer_depth+1),
                                              batch_normalization=batch_normalization, padding='same')
            layer2 = create_convolution_block(input_layer=layer1, kernel=(1,3,3),  n_filters=16*(layer_depth+1),
                                              batch_normalization=batch_normalization, padding='same')
            if layer_depth < depth - 1:
                current_layer = MaxPooling3D(pool_size=(1,2,2))(layer2)
                levels.append([layer1, layer2, current_layer])
            else:
                current_layer = layer2
                levels.append([layer1, layer2])

        for layer_depth in range(depth-2, -1, -1):
            
            up_convolution = get_up_convolution(pool_size=(1,2,2), deconvolution=deconvolution,
                                                n_filters=16*(layer_depth+1))(current_layer)

            concat = concatenate([up_convolution, levels[layer_depth][1]] , axis=-1)
            current_layer = create_convolution_block(n_filters=16*(layer_depth+1),kernel=(1,3,3), 
                                                     input_layer=concat, batch_normalization=batch_normalization, padding='same')
            current_layer = create_convolution_block(n_filters=16*(layer_depth+1),kernel=(1,3,3), 
                                                     input_layer=current_layer, batch_normalization=batch_normalization, padding='same')
            #L += 1


        #current_layer = Cropping3D([(0,0),(4,4),(4,4)])(current_layer)
        current_layer = concatenate([current_layer, coords])
    
        final_convolution = Conv3D(2, (1, 1, 1))(current_layer)
        
       
        act = Activation(activation_name)(final_convolution)
        model = Model(inputs=[inputs, coords], outputs=act)
        model.compile(loss=Generalised_dice_coef_multilabel2, optimizer=Adam(lr=initial_learning_rate), metrics=['acc', dice_coef_multilabel_bin0, dice_coef_multilabel_bin1])

        return model
    

def UNet_v4_MultiPriors(input_shape=(19,75,75,2), pool_size=(2, 2, 2), n_labels=2, initial_learning_rate=0.00001, deconvolution=True,
                  depth=4, n_base_filters=32, include_label_wise_dice_coefficients=False, 
                  batch_normalization=True, activation_name="softmax", bilinear_upsampling=True):
    """
    Like model v1 but with bilinear upsampling using transposed Convolution 3D with fixed weights of size 3,3,3
    
    Input odd : 
    - output is same size (actually input + 1 but keras removes one border)
    
    Downsampling is done using average pooling with kernels of size 3 and stride 2. This mimic the upsampling done by the transposed
    convolutiion of kernel 3 and stride. This operation leaves 1 invalid pixel per side, which need to be cropped, same with the input.
    This downsampling and upsampling is the method that best preserves a signal (compared: average and max pooling with kernel 2 stride 2,
    which makes unclear which border pixel needs to be cropped, and also compared against max pooling kernel 3 stride 2, which introduces 
    border artifacts due to the overlap max pooling).
    """
    input_shape = (19,75,75,2)
    inputs = Input(input_shape)
    current_layer = inputs
    levels = list()
    kernel = [(3,3,3),(1,3,3),(1,1,1),(1,1,1)]
    # DOWNSAMPLING
    for layer_depth in range(depth):
        layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_base_filters*(2**layer_depth),
                                          batch_normalization=batch_normalization, padding='valid', kernel=kernel[layer_depth])
        layer2 = create_convolution_block(input_layer=layer1, n_filters=n_base_filters*(2**layer_depth)*2,
                                          batch_normalization=batch_normalization, padding='valid', kernel=kernel[layer_depth])
        if layer_depth < depth - 1:
            current_layer = AveragePooling3D(pool_size=(3,3,3), strides=2, padding='valid')(layer2)#pool_size=(2,2,2))(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])
    
    kernel = [(1,3,3),(1,3,3),(1,1,1),(1,1,1)]
    Cropping_z_axis = [(7,7),(3,3),(1,1)]
    Cropping = [(15,15),(3,3),(1,1)] # from layers 0 to 2, inverse order!
    # UPSAMPLING
    for layer_depth in range(depth-2, -1, -1):
        up_convolution = get_up_convolution(kernel_size=(3, 3, 3), deconvolution=deconvolution, n_filters=current_layer._keras_shape[-1], 
                                            bilinear_upsampling=bilinear_upsampling)(current_layer)
        
        if bilinear_upsampling: 
            #up_convolution = Cropping3D(((0,1),(0,1),(0,1)))(up_convolution)
            up_convolution = Cropping3D(((1,1),(1,1),(1,1)))(up_convolution)
            levels[layer_depth][1] = Cropping3D((Cropping_z_axis[layer_depth],
                                                 Cropping[layer_depth],
                                                 Cropping[layer_depth]))(levels[layer_depth][1])
            
        #if layer_depth == 0: 
        #    levels[layer_depth][1] = Cropping3D([(0,0),(8,8),(8,8)])(levels[layer_depth][1])                

        concat = concatenate([up_convolution, levels[layer_depth][1]] , axis=-1)     
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[-1], kernel=kernel[layer_depth],
                                                 input_layer=concat, batch_normalization=batch_normalization, padding='valid')     
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[-1], kernel=kernel[layer_depth],
                                                 input_layer=current_layer, batch_normalization=batch_normalization, padding='valid')
    
    
    ### INSERT BREAST MASK HERE ###
    
    ######################### Breast Mask Model ###########################
    #######################################################################
        
     
    x_mask = Lambda(lambda x: x[:,:,:,:,0], output_shape=(1,) + input_shape[2:])(inputs)  # only t1post
    x_mask = AveragePooling3D(pool_size=(1, 3, 3), name='Context')(x_mask)              # low res
    x_mask = Cropping3D(((3,3), (0,0), (0,0)))(x_mask)
    # (13, 25, 25)        
    #SEGMENTATION: (25,33,33)
    for iii in range(6):    
        x_mask        = Conv3D(filters = 30, 
                           kernel_size = (3,3,3), 
                           kernel_initializer=Orthogonal(),
                           name='V2_T1post_Context_{}'.format(iii))(x_mask)
        x_mask        = LeakyReLU()(x_mask)                              
        x_mask        = BatchNormalization(name='V2_BatchNorm_{}'.format(iii))(x_mask) 
    # (1, 13,13)
    #SEGMENTATION: (13,21,21)
    for iii in range(5):    
        x_mask        = Conv3D(filters = 30, 
                           kernel_size = (1,3,3), 
                           kernel_initializer=Orthogonal(),
                           name='V2_T1post_Context_{}'.format(iii+6))(x_mask)
        x_mask        = LeakyReLU()(x_mask)                              
        x_mask        = BatchNormalization(name='V2_BatchNorm_{}'.format(iii+6))(x_mask)     

    x_mask   =  UpSampling3D(size=(1,3,3))(x_mask)
    x_mask = concatenate([x_mask, Coords])
    
    for iii in range(2):  
        x_mask        = Conv3D(filters =60, 
                           kernel_size = (1,1,1), 
                           kernel_initializer=Orthogonal(),
                           name='V2_T1post_Context_{}'.format(iii+11))(x_mask)
        x_mask        = LeakyReLU()(x_mask)
        x_mask        = BatchNormalization(name='V2_BatchNorm_{}'.format(iii+11))(x_mask)
    
       
    x_mask        = Conv3D(filters = 100, 
                       kernel_size = (1,1,1), 
                       kernel_initializer=Orthogonal(),
                       name='V2_T1post_Context_13')(x_mask)
    x_mask        = LeakyReLU()(x_mask)
    x_mask        = BatchNormalization(name='V2_BatchNorm_13')(x_mask)
    
    x_mask        = Conv3D(filters = 2, 
                       kernel_size = (1,1,1), 
                       name='T1post_Context_14',
                       kernel_initializer=Orthogonal())(x_mask)
    x_mask        = Activation('sigmoid')(x_mask)
    
    # (1,9,9)
    
    
    # (1, 54, 54)
    final_convolution = Conv3D(n_labels, (1, 1, 1))(current_layer)
    act = Activation(activation_name)(final_convolution)
    model = Model(inputs=[inputs, Coords], outputs=act)
    
    if bilinear_upsampling:
        up = [x for x in model.layers if 'Conv3DTranspose' in str(x)]
        for layer in up:
            print(layer.trainable)
            w0 = layer.get_weights()
            w0[0]
            N = w0[0].shape[-1]
            W = make_bilinear_filter_5D(N)
            b = np.zeros((N,))
            layer.set_weights([W,b])
            layer.trainable = False
    
    model.compile(loss=Generalised_dice_coef_multilabel2, optimizer=Adam(lr=initial_learning_rate), metrics=['acc', dice_coef_multilabel_bin0, dice_coef_multilabel_bin1])

    return model

#%%
#The x, y, and z sizes must be divisible by the pool size to the power of the depth of the UNet, that is pool_size^depth.      
    
#input_shape=(12,76,76,2)    
#depth = 4
##
##model = UNet_v0(input_shape =  (8,64,64,2), pool_size=(2, 2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=True,
##                      depth=4, n_base_filters=32, include_label_wise_dice_coefficients=False, 
##                      batch_normalization=True, activation_name="softmax")
#
##model = UNet_v1(input_shape, pool_size=(2, 2, 2), n_labels=2, initial_learning_rate=0.00001, 
##                deconvolution=True,  depth=depth, n_base_filters=32, include_label_wise_dice_coefficients=False,
##                batch_normalization=True, activation_name="softmax")
#
#model = UNet_v2(input_shape=(12,76,76,2), pool_size=(2, 2, 2), n_labels=2, initial_learning_rate=0.00001, 
#                deconvolution=True,  depth=4, n_base_filters=32, include_label_wise_dice_coefficients=False,
#                batch_normalization=True, activation_name="softmax",bilinear_upsampling=True)
#
#
#model = UNet_v4_BreastMask(input_shape=(19,75,75,2), pool_size=(2, 2, 2), n_labels=2, initial_learning_rate=0.00001, deconvolution=True,
#                  depth=4, n_base_filters=32, include_label_wise_dice_coefficients=False, 
#                  batch_normalization=True, activation_name="softmax", bilinear_upsampling=True)

#
#
#from keras.utils import plot_model
#plot_model(model, '/home/deeperthought/Projects/MultiPriors_MSKCC/training_sessions/DEBUG_TrainingSession/UNet_v2.png', show_shapes=True)

#unet_3d.summary()
#unet_3d.input
#unet_3d.output
#X = np.ones((4,12,76,76,2))
#y_pred = unet_3d.predict([X])
#y_pred.shape
#
#unet_3d.fit([X],y_pred, epochs=5, batch_size=32)

#from keras.utils import plot_model
#plot_model(unet_3d, to_file='/home/deeperthought/Projects/MultiPriors_MSKCC/models/UNet_3D_noPadding.png', show_shapes=True)
