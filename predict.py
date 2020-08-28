from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers.core import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.layers.convolutional import Conv3D, MaxPooling3D
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import GlobalMaxPool2D
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Reshape
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.utils import np_utils, generic_utils
from tensorflow.keras.callbacks import ModelCheckpoint

import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

nb_classes = 27

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

to_predict = []
classes = ['Pushing Two Fingers Away',
 'Pushing Hand Away',
 'Doing other things',
 'Turning Hand Clockwise',
 'Zooming In With Two Fingers',
 'Sliding Two Fingers Left',
 'Stop Sign',
 'Pulling Two Fingers In',
 'Drumming Fingers',
 'Sliding Two Fingers Right',
 'Sliding Two Fingers Down',
 'No gesture',
 'Rolling Hand Backward',
 'Swiping Down',
 'Rolling Hand Forward',
 'Turning Hand Counterclockwise',
 'Thumb Down',
 'Swiping Up',
 'Zooming Out With Full Hand',
 'Shaking Hand',
 'Thumb Up',
 'Zooming In With Full Hand',
 'Swiping Right',
 'Zooming Out With Two Fingers',
 'Swiping Left',
 'Sliding Two Fingers Up',
 'Pulling Hand In']

model.load_weights('main.h5')
num_frames = 0
cap = cv2.VideoCapture(0)
cap.set(12, 50)
cap.set(6, 10)
classe = ''
import time 
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame_cp = frame
    #print(cap.get(6))
    frame_cp = cv2.resize(frame, (96, 64))
    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    to_predict.append(frame_cp)
    to_predict.append(frame_cp)
    
    if len(to_predict) == 32:
        print(".", end="")
        frame_to_predict = [[]]
        frame_to_predict[0] = np.array(to_predict, dtype=np.float32)
        #frame_to_predict = normaliz_data(frame_to_predict)
        #print(frame_to_predict)
        predict = model.predict(np.array(frame_to_predict))
        classe = classes[np.argmax(predict)]
        if np.argmax(predict)!=2:
            print('Class = ',classe, 'Precision = ', np.amax(predict)*100,'%')


        #print(frame_to_predict)
        to_predict = []
        time.sleep(0.1) # Time in seconds
        font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, classe, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0.5, 0.5),1,cv2.LINE_AA)


    # Display the resulting frame
    cv2.imshow('Hand Gesture Recognition',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
