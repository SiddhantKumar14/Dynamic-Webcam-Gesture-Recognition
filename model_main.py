from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras.layers import BatchNormalization
from keras.layers import GlobalMaxPool3D, GlobalMaxPool2D
from keras.layers import LSTM
from keras.layers import Conv2D
from keras.layers import Reshape
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils, generic_utils

import tensorflow as tf
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn import preprocessing

model = Sequential()
#`channels_last` corresponds to inputs with shape `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
strides = (1,1,1)
kernel_size = (3, 3, 3)
model.add(Conv3D(32, kernel_size, strides=strides, activation='relu', padding='same', input_shape=(32, 64, 96, 3)))
print(model.output_shape)
model.add(BatchNormalization())
model.add(MaxPooling3D(pool_size=(1, 2, 2)))
print(model.output_shape)

model.add(Conv3D(64, kernel_size, strides=strides, activation='relu',padding='same'))
print(model.output_shape)
model.add(BatchNormalization())
model.add(MaxPooling3D(pool_size=(1, 2, 2)))
print(model.output_shape)

model.add(Conv3D(128, kernel_size, strides=strides, activation='relu',padding='same'))
print(model.output_shape)
model.add(BatchNormalization())
model.add(MaxPooling3D(pool_size=(1, 2, 2)))
print(model.output_shape)

model.add(Conv3D(256, kernel_size, strides=strides, activation='relu',padding='same'))
print(model.output_shape)
model.add(BatchNormalization())

model.add(Conv3D(256, kernel_size, strides=strides, activation='relu',padding='same'))
print(model.output_shape)
model.add(BatchNormalization())

model.add(Conv3D(256, kernel_size, strides=strides, activation='relu',padding='same'))
print(model.output_shape)
model.add(BatchNormalization())

model.add(MaxPooling3D(pool_size=(1,8,12)))
print(model.output_shape)

model.add(Reshape((32, 256)))
print(model.output_shape)
model.add(LSTM(256, return_sequences=True))
print(model.output_shape)
model.add(LSTM(256))
print(model.output_shape)

model.add(Dense(256, activation='relu'))
print(model.output_shape)

model.add(Dense(nb_classes, activation='softmax'))
print(model.output_shape)

# model.add(LSTM(256))