import numpy as np
import imutils as imu
import cv2
import pandas as pd
import os

labels = pd.read_csv('/home/siddhant/Datasets/20bn Jester/jester-v1-train.csv', sep = ';')
classes = pd.read_csv('/home/siddhant/Datasets/20bn Jester/jester-v1-labels.csv')

labels = list(labels['label'])


myDir = '/home/siddhant/Datasets/20bn Jester/Doing other things/'
sizes = []
indices = []
for file in os.listdir(myDir):
    sizes.append(len(os.listdir(f'/home/siddhant/Datasets/20bn Jester/Doing other things/{file}')))
    indices.append(file)