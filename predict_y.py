
import pickle
import time 

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
prev = 0

import vlc
player = vlc.MediaPlayer("/media/siddhant/Data/Movies/Goodfellas (1990)/Goodfellas.1990.720p.BrRip.264.YIFY.mp4")

while True:
    with open('gesture.pkl', 'rb') as f:
        gesture = pickle.load(f)
    if prev!=gesture:
        print(classes[gesture])
        
        if classes[gesture] == 'Zooming In With Two Fingers':
            player.play()
        if classes[gesture] == 'Zooming Out With Full Hand':
            player.toggle_fullscreen()
        if classes[gesture] == 'Zooming Out With Two Fingers':
            player.stop()
        if classes[gesture] == 'Stop Sign':
            player.pause()
        if classes[gesture] == 'Swiping Right':
            for i in range(240):
                player.next_frame()
    prev = gesture
    time.sleep(0.5)