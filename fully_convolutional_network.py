'''
TU/e BME Project Imaging 2019
Convolutional neural network for PCAM
Author: Suzanne Wetstein
'''

import os

import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPool2D, BatchNormalization
from keras.layers import BatchNormalization
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, TensorBoard

# unused for now, to be used for ROC analysis
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# the size of the images in the PCAM dataset
IMAGE_SIZE = 96


def get_pcam_generators(base_dir, train_batch_size=100, val_batch_size=100):

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


def get_model(kernel_size=(3,3), pool_size=(4,4), first_filters=64, second_filters=52, third_filters=40, fourth_filters=28, fifth_filters=16, sixth_filters=16):


     # build the model
     model = Sequential()

     model.add(Conv2D(first_filters, kernel_size, activation = 'relu', padding = 'same', input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)))
     model.add(BatchNormalization(axis=-1, momentum=0.99))
     
     model.add(Conv2D(second_filters, kernel_size, activation = 'relu', padding = 'same'))
     model.add(MaxPool2D(pool_size = pool_size)) 
     model.add(BatchNormalization(axis=-1, momentum=0.99))
     
     model.add(Conv2D(third_filters, kernel_size, activation = 'relu', padding = 'same'))
     model.add(BatchNormalization(axis=-1, momentum=0.99))
     
     model.add(Conv2D(fourth_filters, kernel_size, activation = 'relu', padding = 'same'))
     model.add(MaxPool2D(pool_size = pool_size))
     model.add(BatchNormalization(axis=-1, momentum=0.99))
     
     model.add(Conv2D(fifth_filters, kernel_size, activation = 'relu', padding = 'same'))
     model.add(BatchNormalization(axis=-1, momentum=0.99))
    
     model.add(Conv2D(sixth_filters, kernel_size, activation = 'relu', padding = 'same'))
     model.add(BatchNormalization(axis=-1, momentum=0.99))

     model.add(Flatten())
     model.add(Dense(1, activation = 'sigmoid'))
     

     # compile the model
     model.compile(SGD(lr=0.01, momentum=0.95), loss = 'binary_crossentropy', metrics=['accuracy'])

     return model


# get the model
model = get_model()


# get the data generators
#train_gen, val_gen = get_pcam_generators('C:/Datasets')
train_gen, val_gen = get_pcam_generators('C:/Users/s155868/OneDrive - TU Eindhoven/Vakken/2018-2019/Kwart 3/8P361 Project Imaging/Datasets')

# save the model and weights
model_name = 'fully_convolutional_cnn_model'
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
                    callbacks=callbacks_list)

#%% ROC analysis
val_gen.reset()
y_pred = model.predict_generator(val_gen, steps=val_steps, verbose=1)

#y_pred_rounded = np.rint(y_pred)

#y_true = np.append(np.zeros((int(val_gen.n/2),1)), np.ones((int(val_gen.n/2),1)), axis=0)
y_true = val_gen.classes

fpr, tpr, thresholds = roc_curve(y_true, y_pred)
auc_metric = auc(fpr, tpr)
print("AUC of model: "+str(auc_metric))

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = {:.2f}'.format(auc_metric))
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
# TODO Perform ROC analysis on the validation set