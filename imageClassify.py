#This is an image classifier trained utilizing cat and dog images
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import optimizers

#dimensions of the images used in training
img_width, img_height = 150, 150

train_data_dir = 'data/train'
validation_data_dir ='data/validation'

# Import data
# Used to rescale the pixel values from [0,255] to [0,1] interval
datagen =  ImageDataGenerator(rescale = 1./255)

#automatically retrive images and their classes for train and validation sets
train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size = (img_width, img_height)
    batch_size = 16
    class_mode = 'binary'
)

validation_generator = datagen.flow_from_directory(
    validation_data_dir,
    target_size = (img_width, img_height),
    batch_size = 32,
    class_mode = 'binary'
)

#Small CNN
#Model architecture definition
model = Sequential()

model.add(Convolution2D(32, 3, 3, input_shape=(img_width, img_height,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3,3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optomizer= 'rmsprop',
              metrics = ['accuracy'])
             
#Training
nb_epoch = 30
nb_train_samples = 2048
nb_validation_samples = 832

model.fit_generator(
    train_generator,
    samples_per_epoch = nb_train_samples,
    nb_epoch= nb_epoch,
    validation_data = validation_generator, 
    nb_val_samples = nb_validation_samples)
    




