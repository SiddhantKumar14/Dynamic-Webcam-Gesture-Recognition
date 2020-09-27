
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

from pynput.keyboard import Key, Controller
keyboard = Controller()
def config():
    if classes[gesture] == 'Turning Hand Counterclockwise':
        keyboard.press(Key.alt)
        keyboard.tap(Key.tab)
        keyboard.release(Key.alt)
        
        
    if classes[gesture] == 'Turning Hand Clockwise':
        keyboard.press(Key.alt)
        keyboard.press(Key.shift)
        keyboard.tap(Key.tab)
        keyboard.release(Key.shift)
        keyboard.release(Key.alt)
        
        
#     if classes[gesture] == 'Zooming In With Full Hand':
        
#     if classes[gesture] == 'Zooming Out With Full Hand':
        
#     if classes[gesture] == 'Zooming In With Two Fingers':
        
#     if classes[gesture] == 'Zooming Out With Two Fingers':
        
#     if classes[gesture] == 'Stop Sign':
        
    if classes[gesture] == 'Swiping Right':
        keyboard.press(Key.ctrl)
        keyboard.tap(Key.tab)
        keyboard.release(Key.ctrl)
        
        
    if classes[gesture] == 'Swiping Left':
        keyboard.press(Key.ctrl)
        keyboard.press(Key.shift)
        keyboard.tap(Key.tab)
        keyboard.release(Key.shift)
        keyboard.release(Key.ctrl)
        
        
    if classes[gesture] == 'Swiping Up':
        keyboard.tap(Key.page_down)
        
        
    if classes[gesture] == 'Swiping Down':
        keyboard.tap(Key.page_up)
        
        
    if classes[gesture] == 'Sliding Two Fingers Up':
        keyboard.tap(Key.down)
        
    if classes[gesture] == 'Sliding Two Fingers Down':
        keyboard.tap(Key.up)
        
while True:
    with open('gesture.pkl', 'rb') as f:
        try:
            gesture = pickle.load(f)
        except:
            gesture = 11
    if prev!=gesture:
        if classes[gesture] != 'No gesture':
            print(classes[gesture])
            config()
    prev = gesture
    time.sleep(0.1)
    
