#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 16:10:59 2019

@author: deeperthought
"""

import sys
sys.path.append('/home/deeperthought/Projects/MultiPriors_MSKCC/scripts/')
from lib import *
#import tensorflow as tf
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#tf.keras.backend.set_session(tf.Session(config=config))
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
from keras.callbacks import History

#configFile = '/home/deeperthought/Projects/MultiPriors_MSKCC/configFiles/UNet_v0_TumorSegmenter.py'
configFile = '/home/deeperthought/Projects/MultiPriors_MSKCC/configFiles/configFile_UNet_3D_v4_TPM_NovData.py'
#'/home/deeperthought/Projects/MultiPriors_MSKCC/configFiles/UNet_v0_BreastMask.py'#
workingDir = '/home/deeperthought/Projects/MultiPriors_MSKCC/'

logfile = '/home/deeperthought/Projects/MultiPriors_MSKCC/Debug_logFile.txt'

LOAD_MODEL = 0
DEBUG_TRAINING_DATA_SAMPLING = 1
DEBUG_MODEL_TRAINING = 0
DEBUG_SEGMENTATION = 0
PLOT_PATCHES_TARGETS = 0
TRAIN_UNET = 0

n_patches = 150#40
n_subjects = 75#20

CUSTOM_PARAMETERS = False

model = 'UNet_v0_BreastMask'
dpatch= [3,512,512]
segmentation_dpatch = [3,512,512]
model_patch_reduction = [2,0,0]
size_test_minibatches = 8
model_crop = 0

#model = 'UNet_v0_BreastMask'
#dpatch= [3,256,256]
#segmentation_dpatch = [3,256,256]
#model_patch_reduction = [2,0,0]
#size_test_minibatches = 8
#model_crop = 0


#model = 'UNet_3D'
#dpatch= [19,75,75]
#segmentation_dpatch = [19,75,75]
#model_patch_reduction = [18,38,38]
#size_test_minibatches = 16
#model_crop = 0

#model = 'UNet_3D'
#dpatch= [8,64,64]
#segmentation_dpatch = [8,64,64]
#model_patch_reduction = [7,0,0]
#size_test_minibatches = 64 
#model_crop = 0

#model = 'UNet_3D'
#dpatch= [20,108,108]#[8,64,64]
#segmentation_dpatch = dpatch#[8,64,64]
#model_patch_reduction = [19,88,88]#[7,0,0]
#size_test_minibatches = 8#64 
#model_crop = 0

#model = 'MultiPriors_v2_Big_U' 
#dpatch= [13,135,135]#[13,141,141]
#segmentation_dpatch = [29,171,171] #
#model_patch_reduction = [12,111,111]  #   for normal model.
#size_test_minibatches = 16 


##model = 'MultiPriors_v2_Big_BreastMask' 
#model = 'MultiPriors_v2_Big' 
#dpatch= [13,135,135]#[13,141,141]
#segmentation_dpatch = [29,171,171] #
#model_patch_reduction = [12,132,132]  #   for normal model.
#size_test_minibatches = 16 
#model_crop = 70
 
#model = 'UNet_3D'#'MultiPriors_v2' 
#dpatch= [13,69,69]#[13,75,75] 
#segmentation_dpatch = [25,99,99]
#model_patch_reduction = [12,66,66]
#size_test_minibatches = 64 
#model_crop = 40

#%%

print(configFile)
path = '/'.join(configFile.split('/')[:-1])
print(path)
configFileName = configFile.split('/')[-1][:-3]   
sys.path.append(path)
#sys.path.append(path.replace('configFiles','scripts'))
cfg = __import__(configFileName)
if len(cfg.TPM_channel) != 0:
  cfg.TPM_channel = workingDir + cfg.TPM_channel
cfg.trainChannels = [workingDir + x for x in cfg.trainChannels]
cfg.trainLabels = workingDir +cfg.trainLabels 
cfg.testChannels = [workingDir + x for x in cfg.testChannels]
cfg.testLabels = workingDir + cfg.testLabels
cfg.validationChannels = [workingDir + x for x in cfg.validationChannels]
cfg.validationLabels = workingDir + cfg.validationLabels

if CUSTOM_PARAMETERS:
    cfg.n_patches = n_patches
    cfg.n_subjects = n_subjects
    wd = workingDir
    cfg.model = model
    cfg.dpatch = dpatch
    cfg.segmentation_dpatch = segmentation_dpatch
    cfg.model_patch_reduction = model_patch_reduction
    cfg.size_test_minibatches = size_test_minibatches
    
    cfg.percentile_voxel_intensity_sample_benigns = 90
    
    cfg.model_crop = model_crop
    
    cfg.merge_breastMask_model = False
    cfg.data_augmentation = True
    cfg.proportion_to_flip = 0.5
    
    cfg.proportion_malignants_to_sample_train = 0.3

CV_FOLDS_ARRAYS_PATH = '/home/deeperthought/Projects/MultiPriors_MSKCC/CV_folds/CV_alignedNii-Aug2019_actual-F4-training/arrays/'#'/'.join(cfg.trainLabels.split('/')[:-1]) + '/arrays/'


if LOAD_MODEL:
    if cfg.model == 'MultiPriors_v0':
        from MultiPriors_Models_Collection import MultiPriors_v0
        mp = MultiPriors_v0(cfg.output_classes, cfg.num_channels, cfg.L2, cfg.dropout, cfg.learning_rate, cfg.optimizer_decay, cfg.loss_function)
        model = mp.createModel()            
        model.summary()                             
    
    elif cfg.model == 'MultiPriors_v1':
        from MultiPriors_Models_Collection import MultiPriors_v1
        mp = MultiPriors_v1(cfg.output_classes, cfg.num_channels, cfg.L2, cfg.dropout, cfg.learning_rate, cfg.optimizer_decay, cfg.loss_function)
        model = mp.createModel()            
        model.summary()           
    
    elif cfg.model == 'MultiPriors_v2':
        from MultiPriors_Models_Collection import MultiPriors_v2
        mp = MultiPriors_v2(cfg.output_classes, cfg.num_channels, cfg.L2, cfg.dropout, cfg.learning_rate, cfg.optimizer_decay, cfg.loss_function)
        model = mp.createModel()            
        model.summary()        
    
    elif cfg.model == 'MultiPriors_v2_Big':
        from MultiPriors_Models_Collection import MultiPriors_v2_Big
        mp = MultiPriors_v2_Big(cfg.output_classes, cfg.num_channels, cfg.L2, cfg.dropout, cfg.learning_rate, cfg.optimizer_decay, cfg.loss_function)
        model = mp.createModel()            
        model.summary()      

    elif cfg.model == 'MultiPriors_v2_Big_BreastMask':
        from MultiPriors_Models_Collection import MultiPriors_v2_Big_BreastMask
        mp = MultiPriors_v2_Big_BreastMask(cfg.output_classes, cfg.num_channels, cfg.L2, cfg.dropout, cfg.learning_rate, cfg.optimizer_decay, cfg.loss_function)
        model = mp.createModel()            
        model.summary()         
        
    elif cfg.model == 'MultiPriors_v2_Big_U':
        from MultiPriors_Models_Collection import MultiPriors_v2_Big_U
        mp = MultiPriors_v2_Big_U(cfg.output_classes, cfg.num_channels, cfg.L2, cfg.dropout, cfg.learning_rate, cfg.optimizer_decay, cfg.loss_function)
        model = mp.createModel()            
        model.summary()                

    elif cfg.model == 'UNet_3D':
        from Unet_3D_Class import Unet_3D
        model = Unet_3D().create_model2((tuple(dpatch) + (2,)), pool_size=(2, 2, 2), n_labels=2, initial_learning_rate=0.00001, deconvolution=True,
                  depth=4, n_base_filters=32, include_label_wise_dice_coefficients=False, batch_normalization=True, activation_name="softmax")     
        model.summary()         

#    # DEBUG MODEL
#    import numpy as np
#    model.input
#    patches.shape
#    context.shape
#    spatial_coordinates.shape
#    X1 = np.ones((1,13,45,45,1))
#    X2 = np.ones((1,13,65,65,1))
#    X3 = np.ones((1,1,3,3,3))
#    y_pred = model.predict([X1,X2,X2,X3])    
#    y_pred.shape
 
    if cfg.merge_breastMask_model:

        from keras.models import load_model  
        from MultiPriors_Models_Collection import Generalised_dice_coef_multilabel2, dice_coef_multilabel0,dice_coef_multilabel1
        my_custom_objects = {'Generalised_dice_coef_multilabel2':Generalised_dice_coef_multilabel2,
                             'dice_coef_multilabel0':dice_coef_multilabel0,
                             'dice_coef_multilabel1':dice_coef_multilabel1}            
        bm_model = load_model(cfg.path_to_breastMask_model, custom_objects = my_custom_objects )            
        preTrained_convLayers = [x for x in bm_model.layers if 'Conv3D' in str(x)]
        preTrained_batchNormLayers = [x for x in bm_model.layers if 'BatchNorm' in str(x)]
        if cfg.Context_parameters_trainable:
            # If fine-tuning allowed, then skip bottleneck of last layer of the breastMask model 
            preTrained_convLayers = preTrained_convLayers[:-1]
            
        #preTrained_convLayers = [x for x in bm_model.layers if 'T1post_Context' in x.name]
        #preTrained_batchNormLayers = [x for x in bm_model.layers if 'BatchNorm' in x.name]   
        newModel_convLayers = [x for x in model.layers if 'T1post_Context' in x.name]           
        newModel_batchNormLayers = [x for x in model.layers if 'BatchNorm' in x.name]            
        
        assert len(preTrained_convLayers ) == len(newModel_convLayers), 'Models have incompatible architecture..'
        assert len(preTrained_batchNormLayers) == len(newModel_batchNormLayers), 'Models have incompatible architecture..'
        print('Transfering weights from breastMask model {} '.format(cfg.path_to_breastMask_model))       
#            for i in range(len(newModel_convLayers)):
#                print('Equal layer {}: {}'.format(newModel_convLayers[i].name, (model.get_layer(newModel_convLayers[i].name).get_weights()[0] == preTrained_convLayers[i].get_weights()[0]).all()))
        for i in range(len(newModel_convLayers)):
            print('Weight transfer of layer : {}'.format(newModel_convLayers[i].name))
            model.get_layer(newModel_convLayers[i].name).set_weights(preTrained_convLayers[i].get_weights())        
            model.get_layer(newModel_convLayers[i].name).trainable = cfg.Context_parameters_trainable
        for i in range(len(newModel_batchNormLayers)):
            print('Weight transfer of layer : {}'.format(newModel_batchNormLayers[i].name))
            model.get_layer(newModel_batchNormLayers[i].name).set_weights(preTrained_batchNormLayers[i].get_weights())                  
        # Need to re-compile when changing the TRAINABLE attribute:   
        #model = multi_gpu_model(model, gpus=4)
        
        from MultiPriors_Models_Collection import Generalised_dice_coef_multilabel2     , dice_coef_multilabel_bin0, dice_coef_multilabel_bin1
        model.compile(loss=Generalised_dice_coef_multilabel2, optimizer=keras.optimizers.adam(lr=cfg.learning_rate), 
                      metrics=['acc', dice_coef_multilabel_bin0, dice_coef_multilabel_bin1])
        model.summary()     


if DEBUG_TRAINING_DATA_SAMPLING or DEBUG_MODEL_TRAINING:   
     
    history = LossHistory_multiDice2()     
    losses = []
    metrics= []
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    train_data_process = Process(target=sampleTrainData_daemon, args=(return_dict, 'TRAINING', cfg.resolution, cfg.trainChannels, CV_FOLDS_ARRAYS_PATH, cfg.trainLabels,cfg.TPM_channel, 
                                                                      cfg.n_patches, cfg.n_subjects, cfg.dpatch, cfg.output_classes, 
                                                                      cfg.samplingMethod_train, cfg.use_coordinates, cfg.proportion_malignants_to_sample_train, 
                                                                      cfg.percentile_voxel_intensity_sample_benigns,  cfg.data_augmentation, cfg.proportion_to_flip, 
                                                                      cfg.percentile_normalization, cfg.model_patch_reduction, cfg.model_crop, cfg.balanced_sample_subjects))
    
    train_data_process.start() 
    train_data_process.join()
    
    context = return_dict['TRAINING'][0]
    patches = return_dict['TRAINING'][1]
    labels = return_dict['TRAINING'][2]
    spatial_coordinates = return_dict['TRAINING'][3]
    TPM_patches = return_dict['TRAINING'][4]    
    
    #print('context.shape = {}'.format(context.shape))
    print('patches.shape = {}'.format(patches.shape))
    print('labels.shape = {}'.format(labels.shape))
    print('spatial_coordinates shape = {}'.format(spatial_coordinates.shape))

    if TRAIN_UNET:
        
        part = int(len(patches)*0.9)
        x_train = patches[:part]
        coords_train = spatial_coordinates[:part]
        y_train = labels[:part]
        x_val = patches[part+1:]
        coords_val = spatial_coordinates[part+1:]
        y_val = labels[part+1:]
        
        x_train = [x_train, coords_train]
        x_val = [x_val, coords_val]

        model.fit(x_train,y_train, epochs=50, shuffle=True, batch_size=8, validation_data = [x_val, y_val])    
        
        plt.subplot(2,1,1)
        plt.plot(History.history['loss'])
        plt.plot(History.history['val_loss'])
        plt.subplot(2,1,2)
        plt.plot(History.history['acc'])
        plt.plot(History.history['val_acc'])
        
        model.evaluate(x_val, y_val)
    
        
        y_pred = model.predict(x_val)
        y_bin = np.argmax(y_pred, axis=-1)
    
        x_val[0] = np.array(x_val[0], dtype='float32')
        x_val[1] = np.array(x_val[1], dtype='float32')
    
        OUTPUT_PATH = '/home/deeperthought/Projects/MultiPriors_MSKCC/'
        pdf = matplotlib.backends.backend_pdf.PdfPages(OUTPUT_PATH + 'U-Net_v0_tumorSegmentertest_{}_{}_epochs.pdf'.format('firstTest', 2))
        INDEX = 0
        middle = patches.shape[1]/2
        for _ in range(y_pred.shape[0]/4):        
            fig, ax = plt.subplots(nrows=4,ncols=7, figsize=(20, 10))
            for i in range(4):
                ax[i][0].imshow(x_val[0][INDEX,middle,:,:,0])
                ax[i][1].imshow(x_val[0][INDEX,middle,:,:,1])
                ax[i][2].imshow(x_val[0][INDEX,middle,:,:,2])
                ax[i][3].imshow(x_val[0][INDEX,middle,:,:,3])
                #ax[i][1].imshow(x_val[0][INDEX,middle,19:-19,19:-19,0])
                ax[i][4].imshow(y_pred[INDEX,0,:,:,1], vmin=0, vmax=1)
                ax[i][5].imshow(y_bin[INDEX,0], vmin=0, vmax=1)
                ax[i][6].imshow(y_val[INDEX,0,:,:,1], vmin=0, vmax=1)
                INDEX += 1
            pdf.savefig(fig)
            plt.close()
        pdf.close()    
            
    #(patches.nbytes + context.nbytes)/float(2e9)
    (patches.nbytes)/float(2e9)

    
    if PLOT_PATCHES_TARGETS:
        
        import numpy as np
        OUTPUT_PATH = '/home/deeperthought/Projects/MultiPriors_MSKCC/'
        pdf = matplotlib.backends.backend_pdf.PdfPages(OUTPUT_PATH + 'MODEL_UNet_v4_BreastMask_INPUT_OUTPUT_{}.pdf'.format(model))
        INDEX = 0
        middle_slice = patches.shape[1]/2
        for _ in range(context.shape[0]/4):
            
            fig, ax = plt.subplots(nrows=4,ncols=3, figsize=(20, 10))
            for i in range(4):
                vmax=np.max(context[INDEX,6]*2.5)
                #ax[i][0].imshow(context[INDEX, middle_slice], cmap='gray', vmin=0, vmax=vmax )
                #ax[i][0].imshow(patches[INDEX, middle_slice,:,:,0] + labels[INDEX,0,:,:,1]*np.max(patches[INDEX])/4, cmap='gray', vmin=0, vmax=vmax )
                ax[i][0].imshow(patches[INDEX, middle_slice,:,:,0], cmap='gray', vmin=0, vmax=vmax )
                ax[i][1].imshow(spatial_coordinates[INDEX, 0,:,:,1], cmap='gray', vmin=0, vmax=vmax )    
                ax[i][2].imshow(labels[INDEX,0,:,:,1],vmin=0,vmax=1)
                INDEX += 1
            pdf.savefig(fig)
            plt.close()
        pdf.close()    

    if DEBUG_MODEL_TRAINING:

        epoch_loss, epoch_metrics = train_validate_model_on_batch(cfg.model, model,return_dict['TRAINING'][0],return_dict['TRAINING'][1],
                                                              return_dict['TRAINING'][2], return_dict['TRAINING'][3], return_dict['TRAINING'][4],
                                                              cfg.size_minibatches,history,losses,metrics,cfg.output_classes, logfile)  
        print(epoch_loss)
        print(epoch_metrics)
        plt.subplot(2,1,1)
        plt.plot(epoch_loss)
        plt.subplot(2,1,2)
        plt.plot(epoch_metrics)

    
if DEBUG_SEGMENTATION:
    epoch = 1
    subjectIndex = 1
    dice_compare = True
    dsc = []
    smooth_dice_scores = []
    foreground_percent_list = []
    
    cfg.testChannels = ['/home/deeperthought/Projects/MultiPriors_MSKCC/CV_folds/CV_alignedNii-Aug2019_actual-F4-training/test_t1post_1.txt',
                        '/home/deeperthought/Projects/MultiPriors_MSKCC/CV_folds/CV_alignedNii-Aug2019_actual-F4-training/test_sub_1.txt',
                        '/home/deeperthought/Projects/MultiPriors_MSKCC/CV_folds/CV_alignedNii-Aug2019_actual-F4-training/test_t1post_1.txt',
                        '/home/deeperthought/Projects/MultiPriors_MSKCC/CV_folds/CV_alignedNii-Aug2019_actual-F4-training/test_sub_1.txt']
    cfg.testLabels = ''
    
    #for subjectIndex in range(139):
    img_probs, output, dsc, score_smooth,foreground_percent = fullSegmentation(wd, cfg.penalty_MATRIX,cfg.resolution, cfg.OUTPUT_PATH, cfg.TPM_channel, 
                                                                                                  dice_compare, dsc, smooth_dice_scores, foreground_percent_list, 
                                                                                                  model, cfg.testChannels, cfg.testLabels, subjectIndex, 
                                                                                                  cfg.output_classes, cfg.segmentation_dpatch, cfg.size_test_minibatches,
                                                                                                  cfg.output_probability, cfg.use_coordinates, cfg.percentile_normalization,
                                                                                                  cfg.model_patch_reduction, cfg.model_crop, epoch, using_Unet=True, 
                                                                                                  using_unet_breastMask=cfg.using_unet_breastMask)
    print('--------------- TEST EVALUATION ---------------')
    print('          Full segmentation evaluation of subject' + str(subjectIndex))
    print('foreground_percent {}'.format(foreground_percent))
    print('SMOOTH_DCS ' + str(score_smooth))    
    if len(dsc) > 0:
        print('DCS ' + str(dsc[-1]))
    
    import matplotlib.pyplot as plt
    INDEX = 24
    seg = img_probs.get_data()

    #plt.imshow(seg[INDEX,:,:])

    note = 'UNet_v4_TPM_TEST'

    nib.save(img_probs, '/home/deeperthought/Projects/MultiPriors_MSKCC/training_sessions/DEBUG_TrainingSession/DEBUG/{}_{}'.format(note,output.split('/')[-1].replace('.gz','')))    
