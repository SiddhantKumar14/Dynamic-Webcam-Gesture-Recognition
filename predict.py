import cv2
import numpy as np
from tensorflow.keras import models
import os
from tensorflow.keras.models import load_model
import model_main 

to_predict = []
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
            print('Classe = ',classe, 'Precision = ', np.amax(predict)*100,'%')


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
