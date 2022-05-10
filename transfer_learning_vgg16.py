#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 08:48:36 2022

@author: garimasharma
Let's try something called - transfer learning- suitable for small datasets '

"""

#%% 

import numpy as np
import keras
from keras.applications import vgg16, vgg19
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
import scipy.misc


#%%

model = vgg16.VGG16(include_top=(True), weights='imagenet')
model.summary()

#%% input our own image to see classification

import librosa
import glob
import os.path
import matplotlib.pyplot as plt 
from librosa import display

t=[]
sampling_rate=[]
path = '/Users/garimasharma/Downloads/data-to-share-covid-19-sounds/KDD_paper_data/covidandroidnocough/cough/'
for filename in glob.glob(os.path.join(path, '*.wav')):
    y, sr = librosa.load(filename) 
    #y = y / tf.int16.max
    sampling_rate.append(sr)
    t.append(y)
    y_freq_domain = librosa.stft(y,n_fft=512, hop_length=256)
    y_fft_abs = np.abs(y_freq_domain)
    # display a spectrogtam
    fig, ax = plt.subplots()
    img = librosa.display.specshow(librosa.amplitude_to_db(y_fft_abs,ref=np.max),y_axis='log', x_axis='time', ax=ax)
    file = str(filename) + '.png'
    plt.savefig(str(file))
    
#%% 
train_dir = '/Users/garimasharma/Downloads/covid_transferlearning/dataset/training_set'
test_dir = '/Users/garimasharma/Downloads/covid_transferlearning/dataset/test_set'
validation_dir= '/Users/garimasharma/Downloads/covid_transferlearning/dataset/validation_set'



train_covid_dir = '/Users/garimasharma/Downloads/covid_transferlearning/dataset/training_set/train_covid'
test_covid_dir = '/Users/garimasharma/Downloads/covid_transferlearning/dataset/test_set/test_covid'
validation_covid_dir = '/Users/garimasharma/Downloads/covid_transferlearning/dataset/validation_set/validation_covid'
train_nocovid_dir = '/Users/garimasharma/Downloads/covid_transferlearning/dataset/training_set/train_nocovid'
test_nocovid_dir = '/Users/garimasharma/Downloads/covid_transferlearning/dataset/test_set/test_nocovid'
validation_nocovid_dir = '/Users/garimasharma/Downloads/covid_transferlearning/dataset/validation_set/validation_nocovid'


num_covid_train = len(os.listdir(train_covid_dir))
num_covid_test = len(os.listdir(test_covid_dir))
num_covid_val = len(os.listdir(validation_covid_dir))

num_noncovid_train = len(os.listdir(train_nocovid_dir))
num_nocovid_test = len(os.listdir(test_nocovid_dir))
num_noncovid_val = len(os.listdir(validation_nocovid_dir))

print(f"The number of train covid: {num_covid_train}")
print(f"The number of test sampling in covid:{num_covid_test}")
print(f"The number of val samples in covid: {num_covid_val}")
print(f"The number of train sampling in no covid cough:{num_noncovid_train}")
print(f"The number of test noncovid: {num_nocovid_test}")
print(f"The number of val sampls in noncovid:{num_noncovid_val}")
#%%

image_shape = 224
batch_size = 10

#%% 

from keras.preprocessing.image import ImageDataGenerator

train_image_generation = ImageDataGenerator(rescale=(1./255) )
train_data_generation = train_image_generation.flow_from_directory(train_dir,
                                                                   target_size=(image_shape,image_shape),
                                                                  batch_size=batch_size,
                                                                  class_mode='binary')

test_image_generation = ImageDataGenerator(rescale=(1./255))
test_data_generation = test_image_generation.flow_from_directory(test_dir, 
                                                                target_size=(image_shape, image_shape),
                                                                batch_size=batch_size,
                                                                class_mode='binary')

val_image_generation = ImageDataGenerator(rescale=(1./255))
val_data_generation = val_image_generation.flow_from_directory(validation_dir,
                                                               target_size=(image_shape, image_shape),
                                                               batch_size=batch_size,
                                                               class_mode='binary')

#%% 

for layer in model.layers:
    print(layer.name)
    layer.trainable = False
    
#%% 

last_layer = model.get_layer('block5_pool')
last_output = last_layer.output
x = keras.layers.GlobalMaxPooling2D()(last_output)
x = keras.layers.Dense(512, activation='relu')(x)
x = keras.layers.Dropout(0.5)(x)
x = keras.layers.Dense(2, activation='sigmoid')(x)

#%% 
import tensorflow as tf
model = tf.keras.Model(model.input, x)

#%%

model.compile(optimizer='adam', loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['acc'])
    

#%% 

model.summary()

#%% 

total_train = num_covid_train + num_noncovid_train
total_validation = num_covid_val + num_noncovid_val

vgg_classifier = model.fit(train_data_generation,
steps_per_epoch=(total_train//batch_size),
epochs = 50,
validation_data=val_data_generation,
validation_steps=(total_validation//batch_size),
batch_size = batch_size,
verbose = 1)

#%% 

result = model.evaluate(test_data_generation,batch_size=batch_size)
print("test_loss, test accuracy",result)
    
#%% plot the traning and validation accuracy 

acc=vgg_classifier.history['acc']  ##getting  accuracy of each epochs
epochs_=range(0,50)    
plt.plot(epochs_,acc,label='training accuracy')
plt.xlabel('no of epochs')
plt.ylabel('accuracy')

acc_val=vgg_classifier.history['val_acc']  ##getting validation accuracy of each epochs
plt.scatter(epochs_,acc_val,label="validation accuracy")
plt.title("no of epochs vs accuracy")
plt.legend()

