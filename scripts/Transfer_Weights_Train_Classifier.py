# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 18:43:40 2018

@author: hirsch
"""
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import print_summary
from keras.optimizers import Adam
import sys
sys.path.append('/home/hirsch/Documents/projects/Breast_segmentation/DeepPriors_package/scripts/')
from DM_MSKCC_Atrous_model import Generalised_dice_coef_multilabel2, dice_coef_multilabel0,dice_coef_multilabel1
from keras.models import load_model  

model_path = '/home/hirsch/Documents/projects/Breast_segmentation/DeepPriors_package/training_sessions/MultiPriors_MSKCC_fullHeadSegmentation_configFile_DM_2019-03-11_1846/models/MSKCC_fullHeadSegmentation_configFile_DM_2019-03-11_1846.log_epoch198.h5'

test_model_fit = False


def load_and_prepare_data(subject_channels, set_equal_shapes=True):
    images = []       
    for channel in subject_channels: 
      proxy_img = nib.load(channel)
      X = proxy_img.get_data()
      images.append(X)
    X = np.stack(images, axis=3)
    shape = X.shape
    
    if set_equal_shapes:  # crop same center volume. For batch training of size > 1
      # Set boundaries for maximum allowed shape
      a = 20
      b = 50
      c = 50
      X = X[a:(shape[0]/2)+a,:,:,:]
      X = X[:,b:(shape[1]/2)+b,:,:]
      X = X[:,:,c:(shape[2]/2)+c,:]      
      
    elif shape[0]*shape[1]*shape[2] > 55*201*201:  # if shape exceeeds 55*261*261
      # Set boundaries for maximum allowed shape
      a = np.max([0,(shape[0] - 55)])/2   
      b = np.max([0,(shape[1] - 201)])/2
      c = np.max([0,(shape[2] - 201)])/2    
      X = X[a:shape[0]-a,:,:,:]
      X = X[:,b:shape[1]-b,:,:]
      X = X[:,:,c:shape[1]-c,:]
      

    shape = X.shape
    X = X.reshape(((1,) + shape))
    # Padding
    X_pad = []
    for m in range(shape[3]):
      #print('padding modality {}'.format(m))
      X_pad.append(np.pad(X[0,:,:,:,m],10,'minimum'))
    X_padded = np.stack(X_pad,axis=3)
    X_padded = X_padded.reshape(((1,) + X_padded.shape))
    coords_shape = [X_padded.shape[1] - 12 ,X_padded.shape[2] - 66, X_padded.shape[3] - 66]
    y_coords = np.tile(np.array([range(6,coords_shape[2]+6)]).transpose(), (1,coords_shape[1]))
    y_coords = y_coords/float(X.shape[2])
    y_coords = np.repeat(y_coords[np.newaxis, :,: ], coords_shape[0], axis=0)    
    y_coords = y_coords.reshape((1,) + y_coords.shape + (1,))
    T1post = X_padded[:,:,:,:,0].reshape(X_padded[:,:,:,:,0].shape + (1,))
    T1pre = X_padded[:,:,:,:,1].reshape(X_padded[:,:,:,:,1].shape + (1,))
    T2 = X_padded[:,:,:,:,2].reshape(X_padded[:,:,:,:,2].shape + (1,))
    return T1post,T1pre,T2,y_coords



my_custom_objects = {'Generalised_dice_coef_multilabel2':Generalised_dice_coef_multilabel2,
                                 'dice_coef_multilabel0':dice_coef_multilabel0,
                                 'dice_coef_multilabel1':dice_coef_multilabel1}
model = load_model(model_path, custom_objects = my_custom_objects )


#my_custom_objects = {'Generalised_dice_coef_multilabel2':Generalised_dice_coef_multilabel2,
#				     'dice_coef_multilabel0':dice_coef_multilabel0,
#				     'dice_coef_multilabel1':dice_coef_multilabel1}
#
#model.save_weights('/'.join(model_path.split('/')[:-1]) + '/trained_segmentor_test_weights.h5')
#model.load_weights('/'.join(model_path.split('/')[:-1]) + '/trained_segmentor_test_weights.h5')


from DeepMedic_model_flexibleInput_classifier_multiModal import DeepMedic as DM 
dm = DM(2, 3, 0.0001, [0], 2e-05)
model1 = dm.createModel()            
print_summary(model1, positions=[.33, .8, .9,1])

model1.layers[-1]

print_summary(model, positions=[.33, .8, .9,1])


# NEED TO MAKE SURE THAT BOTH MODELS HAVE THE SAME ARCHITECTURE...


len(model1.layers)
len(model.layers)

layers = len(model1.layers)
for i in range(layers - 2): #2):
  print(model.layers[i])
  weights = model.layers[i].get_weights()  
  model1.layers[i].set_weights(weights)
  model1.layers[i].trainable = False 
  
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model1.compile(optimizer=adam,loss='binary_crossentropy')      

for i in range(layers):
  print('{} : {}'.format(model1.layers[i].name, model1.layers[i].trainable)  )
  

######################### Test if computer can evaluate and train the model on real data ###########################
if test_model_fit:
    # with MRI
    subject_channels = ['/media/andy/RNN_training/Normalized/MSKCC_16-328_1_00416_20050430/rt1post-l_stand.nii',
                        '/media/andy/RNN_training/Normalized/MSKCC_16-328_1_00416_20050430/rt1pre-l_stand.nii',
                        '/media/andy/RNN_training/Normalized/MSKCC_16-328_1_00416_20050430/rt2-l_stand.nii']
    
    T1post,T1pre,T2,y_coords = load_and_prepare_data(subject_channels)
    
    T1post.shape
    
    y = np.array([[0,1]])
    model1.evaluate([T1post,T1pre,T2,y_coords],y)
    
    yhat = model1.predict([T1post,T1pre,T2,y_coords])
    yhat.shape
    
    model1.fit([T1post,T1pre,T2,y_coords],y)

######################################### Training Classifier ###################################################################3

total_epochs = 10
training_steps = 50

# TRAINING

trainChannels = ['/home/andy/projects/mskProj/DeepPriors_package/CV_folds/MRIs_train_t1post_set0.txt',
                 '/home/andy/projects/mskProj/DeepPriors_package/CV_folds/MRIs_train_t1pre_set0.txt',
                 '/home/andy/projects/mskProj/DeepPriors_package/CV_folds/MRIs_train_t2_set0.txt']

trainLabels = '/home/andy/projects/mskProj/DeepPriors_package/CV_folds/labels_train_set0.txt'

testChannels = ['/home/andy/projects/mskProj/DeepPriors_package/CV_folds/MRIs_test_t1post_set0.txt',
                '/home/andy/projects/mskProj/DeepPriors_package/CV_folds/MRIs_test_t1pre_set0.txt',
                '/home/andy/projects/mskProj/DeepPriors_package/CV_folds/MRIs_test_t2_set0.txt']
                
testLabels = '/home/andy/projects/mskProj/DeepPriors_package/CV_folds/labels_test_set0.txt'

Y_Benign = np.array([[1,0]])
Y_Malignant = np.array([[0,1]])
# Have to sort malignants from benigns. 
# Make a dictionary of both. And train on batches of balanced classes. At least batch size = 2.


subjects = open(trainChannels[0]).readlines()
order = range(len(subjects))
np.random.shuffle(order)

subjects = open(testChannels[0]).readlines()
order_test = range(len(subjects))
np.random.shuffle(order_test)


Y_Benign = np.array([[1,0]])
Y_Malignant = np.array([[0,1]])


step_training_loss = []
step_validation_loss = []
validation_accuracy = []
      

epoch = 0
modalities = len(trainChannels)


while epoch < total_epochs:
  epoch += 1
  print('------  EPOCH {} ------'.format(epoch))
  np.random.shuffle(order)
  steps = 0
  
  training_loss = []
  # when the order is complete, a full epoch has passed.
  while steps <= len(order):
  #for index in order:
    index = order[steps]
    
    subject_channels = []
    for m in range(modalities):
        subject_channels.append(open(trainChannels[m]).readlines()[index][:-1])
    
    T1post,T1pre,T2,y_coords = load_and_prepare_data(subject_channels, True) 

    label = open(trainLabels).readlines()[index][:-1]
    if 'BENIGN' in label:
      Y = Y_Benign
    else:
      Y = Y_Malignant
    
    print('{} - TRAINING on subject index {}, X.shape {} , target {}'.format(steps, index, T1post.shape, Y))
    #history = model1.train_on_batch(X,Y)
    history = model1.fit([T1post,T1pre,T2,y_coords],Y,class_weight={0:1,1:12}, verbose=1)    
    training_loss.append(history.history['loss'])
    print('loss: ', history.history['loss'])
    steps += 1

    if steps % 100 == 0:
      print('{} % of epoch'.format(steps*100./len(order)))
    
    if steps % training_steps == 0:
     
      print('VALIDATION')

      validation_loss = []
      predictions = []
      true_labels = []
      for test_index in order_test:
        test_subject_channels = []
        for m in range(modalities):
            test_subject_channels.append(open(testChannels[m]).readlines()[test_index][:-1])
        
        T1post,T1pre,T2,y_coords = load_and_prepare_data(test_subject_channels) 
    
        label = open(testLabels).readlines()[test_index][:-1]
        if 'BENIGN' in label:
          Y = Y_Benign
        else:
          Y = Y_Malignant
        print('Validation subject index {}, X.shape {} , target {}'.format(test_index, T1post.shape, Y))
        prediction = np.argmax(model1.predict([T1post,T1pre,T2,y_coords]))
        predictions.append(prediction)
        true_labels.append(np.argmax(Y))
        validation_loss.append(model1.evaluate([T1post,T1pre,T2,y_coords],Y, verbose=0))
      print('Prediction: {}'.format(predictions))
      print('True label: {}'.format(true_labels))
      
      acc = sum([m==n for m,n in zip(predictions, true_labels)])/float(len(predictions))
      print('Validation step accuracy: {}'.format(acc))
      validation_accuracy.append(acc)
      step_validation_loss.append(np.mean(validation_loss))
   
    step_training_loss.append(np.mean(training_loss))   
    

plt.figure()
plt.subplot(311)
plt.plot(range(0,len(step_training_loss)),step_training_loss,'-')
plt.legend(['train',])
plt.title('Loss ')
plt.subplot(312)
plt.plot(range(0,len(step_validation_loss)),step_validation_loss,'-', color='r')
plt.subplot(313)
plt.plot(range(0,len(validation_accuracy)),validation_accuracy,'-', color='r')
plt.legend(['val',])
plt.title('Accuracy')

plt.savefig('/media/hirsch/RNN_training/TrainTestSets_Segmentor/Results/Classifier_fine_tuning.png')




#%%  ATTEMP TO MAKE BATCHES OF SIZE 2 TO ALWAYS HAVE BALANCED TRAINING BATCHES. bUT THIS IS IMPOSSIBLE WITH INPUT DATA OF DIFFERENT SIZES. A BATCH HAS TO HAVE EQUAL SHAPED DATA.
# WITH THE CROPPING METHOD I COULD MAYBE DO THIS...



'''
labels = open(trainLabels).readlines()

targets = map(lambda x : 0 if 'BENIGN' in x else 1, labels)
len(targets)

sum(targets)

T1post = [x[:-1] for x in open(trainChannels[0]).readlines()]
T1pre = [x[:-1] for x in open(trainChannels[1]).readlines()]
T2 = [x[:-1] for x in open(trainChannels[2]).readlines()]


Malignants = {(a,b,c) : value for a,b,c, value in zip(T1post,T1pre,T2,targets) if value == 1}
Benigns = {(a,b,c) : value for a,b,c, value in zip(T1post,T1pre,T2,targets) if value == 0}

len(Malignants.keys())
len(Benigns.keys()) 

# training list
order = list(Malignants.keys())
np.random.shuffle(order)

order_benigns = list(Benigns.keys())
np.random.shuffle(order_benigns)

# validation list
subjects = open(testChannels[0]).readlines()
order_test = range(len(subjects))
np.random.shuffle(order_test)


# START TRAINING

step_training_loss = []
step_validation_loss = []
validation_accuracy = []
      
total_epochs = 10
training_steps = 100

epoch = 0

while epoch < total_epochs:
  epoch += 1
  print('------  EPOCH {} ------'.format(epoch))
  np.random.shuffle(order)
  steps = 0
  
  training_loss = []
  # when the order is complete, a full epoch has passed.
  while steps <= len(order):
  #for index in order:
  
    # get malignant channels in order. into batch size = 2
    malignant_channels = order[steps]
    benign_channels = order_benigns[steps] 
    
    # Retrieve MRIs and target label
    Data = []
    for modality in malignant_channels:
      Data.append(nib.load(modality).get_data())
    X = np.stack(Data, axis=3)
    X = X.reshape(((1,) + X.shape))
    X.shape
    
    Data = []
    for modality in benign_channels:
      Data.append(nib.load(modality).get_data())
    X2 = np.stack(Data, axis=3)
    X2 = X2.reshape(((1,) + X2.shape))
    X2.shape
    

    minibatch = np.stack([X,X2], axis=0)    
    
    print('TRAINING on subject index {}, X.shape {} , target {}'.format(index, X.shape, Y))
    history = model1.fit(X,Y, verbose=0)
    training_loss.append(history.history['loss'])
    print('loss: ', history.history['loss'])
    steps += 1

    if steps % 100 == 0:
      print('{} % of epoch'.format(steps*100./len(order)))
    
    if steps % training_steps == 0:
     
      print('VALIDATION')

      validation_loss = []
      predictions = []
      true_labels = []
      for test_index in order_test:
        print(test_index)
        Data = []
        for modality in testChannels:
          channel = open(modality).readlines()[test_index][:-1]
          Data.append(nib.load(channel).get_data())
        X = np.stack(Data, axis=3)
        X = X.reshape(((1,) + X.shape))
        X.shape
        label = open(testLabels).readlines()[test_index][:-1]
        if 'BENIGN' in label:
          Y = Y_Benign
        else:
          Y = Y_Malignant
        print('Validation subject index {}, X.shape {} , target {}'.format(test_index, X.shape, Y))
        prediction = np.argmax(model1.predict(X))
        predictions.append(prediction)
        true_labels.append(np.argmax(Y))
        validation_loss.append(model1.evaluate(X,Y, verbose=0))
      print(predictions)
      print(true_labels)
      
      acc = sum([m==n for m,n in zip(predictions, true_labels)])/float(len(predictions))
      print('Validation step accuracy: {}'.format(acc))
      validation_accuracy.append(acc)
      step_validation_loss.append(np.mean(validation_loss))
   
    step_training_loss.append(np.mean(training_loss))   
    
'''