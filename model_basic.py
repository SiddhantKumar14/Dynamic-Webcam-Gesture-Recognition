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

kernel_size = (3, 3, 3)
strides = (1, 1, 1)

model0 = Sequential()

# Conv Block 1
model0.add(Conv3D(64, kernel_size, strides=strides, activation='relu',
                 padding='same', input_shape=(176, 100, 32, 3)))
model0.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

# Conv Block 2
model0.add(Conv3D(128, kernel_size, strides=strides, activation='relu',
                 padding='same'))
model0.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

# Conv Block 3
model0.add(Conv3D(256, kernel_size, strides=strides, activation='relu',
                 padding='same'))
#model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
model0.add(Conv3D(256, kernel_size, strides=strides, activation='relu',
                 padding='same'))

model0.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
# Conv Block 4
model0.add(Conv3D(512, kernel_size, strides=strides, activation='relu',
                 padding='same'))

model0.add(Conv3D(512, kernel_size, strides=strides, activation='relu',
                 padding='same'))# Dense Block
model0.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

#Conv Block 5
model0.add(Conv3D(512, kernel_size, strides=strides, activation='relu',
                 padding='same'))
model0.add(Conv3D(512, kernel_size, strides=strides, activation='relu',
                 padding='same'))
model0.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

model0.add(Flatten())
model0.add(Dense(2048, activation='relu'))
model0.add(Dense(2048, activation='relu'))
model0.add(Dense(4, activation='softmax'))
