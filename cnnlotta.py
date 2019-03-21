# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 13:04:34 2019

@author: Mieke
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, TensorBoard

# unused for now, to be used for ROC analysis
from sklearn.metrics import roc_curve, auc


# the size of the images in the PCAM dataset
IMAGE_SIZE = 96


def get_pcam_generators(base_dir, train_batch_size=512, val_batch_size=32):

     # dataset parameters
     TRAIN_PATH = os.path.join(base_dir, 'train')
     VALID_PATH = os.path.join(base_dir, 'valid')

     RESCALING_FACTOR = 1./255
     
     # instantiate data generators
     datagen = ImageDataGenerator(rescale=RESCALING_FACTOR)

     train_gen = datagen.flow_from_directory(TRAIN_PATH,
                                             target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             batch_size=train_batch_size,
                                             class_mode='binary')

     val_gen = datagen.flow_from_directory(VALID_PATH,
                                             target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             batch_size=val_batch_size,
                                             class_mode='binary',
                                             shuffle=False)
     
     return train_gen, val_gen



def get_model(kernel_size=(3,3), pool_size=(4,4), first_filters=32, second_filters=64):

     # build the model
     model = Sequential()

     model.add(Conv2D(first_filters, kernel_size, activation = 'relu', padding = 'same', input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)))
     model.add(MaxPool2D(pool_size = pool_size))
     model.add(BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.01, center = True))

     model.add(Conv2D(second_filters, kernel_size, activation = 'relu', padding = 'same'))
     model.add(MaxPool2D(pool_size = pool_size))
     model.add(BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.01, center = True))
     
     model.add(Conv2D(64,kernel_size, activation = 'relu',padding = 'same'))
     model.add(MaxPool2D(pool_size = pool_size))
     model.add(BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.01, center = True))
     
     model.add(Flatten())
     model.add(Dense(256, activation = 'relu'))
     model.add(Dense(128,activation = 'relu'))
     model.add(Dense(32,activation = 'relu'))
     model.add(Dense(1, activation = 'sigmoid'))


     # compile the model
     model.compile(SGD(lr=0.01, momentum=0.95  ), loss = 'binary_crossentropy', metrics=['accuracy'])

     return model


# get the data generators
train_gen, val_gen = get_pcam_generators('C:/Users/Mieke/Desktop/Floris/8p361-project-imaging-master')

for layer in model.layers:
    print(layer.output_shape)

#for layer in model.layers:
#    print(layer.output_shape)

# save the model and weights
model_name = 'cnnjeannot'
model_filepath = model_name + '.json'
weights_filepath = model_name + '_weights.hdf5'

model_json = model.to_json() # serialize model to JSON
with open(model_filepath, 'w') as json_file:
    json_file.write(model_json) 


# define the model checkpoint and Tensorboard callbacks
checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tensorboard = TensorBoard(os.path.join('logs', model_name))
callbacks_list = [checkpoint, tensorboard]


# train the model
train_steps = train_gen.n//train_gen.batch_size
val_steps = val_gen.n//val_gen.batch_size

history = model.fit_generator(train_gen, steps_per_epoch=train_steps, 
                    validation_data=val_gen,
                    validation_steps=val_steps,
                    epochs=3,
                    callbacks = callbacks_list)

# ROC analysis
# TODO Perform ROC analysis on the validation set

# calculate the fpr and tpr for all thresholds of the classification
probability = model.predict_generator(val_gen, steps=val_steps)
fpr, tpr, threshold = roc_curve(val_gen.labels, probability)
roc_auc = auc(fpr, tpr)

# plot ROC curve
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = {:.2f}'.format(roc_auc))
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
