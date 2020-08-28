# Dynamic-Webcam-Gesture-Recognition
This is our work on the open source data 20bn Jester, which contains about 25 different hand gestures performed in front of a webcam. We will be using Optical Flow and 3 dimensional CNNs primarily with other computer vision techniques.

I have implemented the model based on [this paper](https://openaccess.thecvf.com/content_ICCVW_2019/papers/HANDS/Materzynska_The_Jester_Dataset_A_Large-Scale_Video_Dataset_of_Human_Gestures_ICCVW_2019_paper.pdf), which has a good response time while maintaining the accuracy of the model. The model I have trained has a testing accuracy of 93.7% at 15 epochs (weights - main.h5), which is where it plateaus.
I have also written a custom VideoDataGenerator that works in a similar fashion to keras' ImageDataGenerator for 3D neural networks, feel free to use it :)

Run predict.py with the necessary libraries (keras, tf, opencv, matplotlib, numpy and os) installed.
