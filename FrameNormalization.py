import cv2
import os
import numpy as np

def show(img, title="img"):
    cv2.imshow(title, img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
def convertToHomogenousSample(sample):
    #here sample is a list of all frames, each frame is opencv image object
    meanFrameValue = 32 #VAR
    extraFrames = abs(len(sample) - meanFrameValue)
    c = 0
    if(len(sample) > meanFrameValue):
        indicesToDelete = np.ndarray.tolist(np.linspace(0, len(sample) - 1, extraFrames))
        indicesToDelete = list(map(int, indicesToDelete))
        
        sampleOut = sample
        for i in indicesToDelete:
            i = i - c
            sampleOut.pop(i)
            c = c + 1
            
        #print(f"Number of frames deleted - {extraFrames}. They were at frame indices - {indicesToDelete}")
        return sampleOut
    
    
    elif(len(sample) < meanFrameValue):
        indicesToCopy = np.ndarray.tolist(np.linspace(0, len(sample) - 1, extraFrames ))
        indicesToCopy = list(map(int, indicesToCopy))
        
        sampleOut = sample
        for i in indicesToCopy:
            i = i + c
            sampleOut.insert(i + 1, sample[i])
            c = c + 1
        #print(f"Number of frames added - {extraFrames}. They are copies of frame indices - {indicesToCopy}")
        return sampleOut
    
    
    elif(len(sample) == meanFrameValue):
        return sample



def readSamples(pathToSample):
    # pathToSample points to dir containing the frames
    frames = os.listdir(pathToSample)
    sample = []
    for frame in os.listdir(pathToSample):
        sample.append(cv2.imread(f'{pathToSample}/{frame}'))
        
    return sample
    
    
