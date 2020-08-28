import random
import os
import matplotlib.image as img

def picker(sample_count):
    L = []
    for cl in sample_count:
        L = L + [(f'{cl[1]}')]*cl[0]
    random.shuffle(L)
    return L

def pick_sample(shuffled_list):
    a = random.choice(shuffled_list)
    return a

def VideoDataGenerator(data_path, frame_dim, batch_size = 64):
    '''The data_path must contain folders containing samples from only that class.
    The samples should be homogenously numerated. (eg - 001.jpg, 002.jpg etc) for each frame of the video. 
    Enter frame_dim as tuple of dimensions in the format: (x-axis, y-axis)
    '''
    def resize_frame(frame, size = frame_dim):
        frame = img.imread(frame)
        frame = cv2.resize(frame, size)
        return frame
    if data_path[-1] != '/':
        data_path = f"{data_path}/"
    
    classes = os.listdir(data_path)
    print(f"{len(classes)} classes found in folder")
    
    class_target_map = [[classes[i], i] for i in range(len(classes))]
    samples_count = [[len(os.listdir(f"{data_path}{classes[i]}")), i] for i in range(len(classes))]
    picker_list = picker(samples_count)
    
    sample_names = [[] for i in range(len(classes))]
    for i in range(len(classes)):
        sample_names[i] = os.listdir(f'{data_path}{classes[i]}')
    
    while True:
        total_samples = 0
        c = 0
        for cl in sample_names:
            if not cl:
                c += 1
            if c == len(classes) - 1:
                for i in range(len(classes)):
                    sample_names[i] = os.listdir(f'{data_path}{classes[i]}')
                c = 0
                continue
        X = []
        y = []
        for cl in classes:
            total_samples = total_samples + len(os.listdir(f'{data_path}{cl}'))
            
        for batch_iter in range(batch_size):
            
            sample_class = pick_sample(picker_list)
            picker_list.remove(sample_class)
            sample_class = int(sample_class)
            if not sample_names[sample_class]: 
                batch_iter -= 1
                continue
            vid_sample = f'{data_path}{classes[sample_class]}/{sample_names[sample_class][0]}'
            
            frames = os.listdir(vid_sample)
            frames.sort()
            vid = []
            for frame in frames:
                frame = resize_frame(f'{vid_sample}/{frame}')
                try:
                    #frame = resize_frame(frame)
                    vid.append(frame)
                except:
                    print(f'Sample with tag - {sample_names[sample_class][0]} is broken and skipped.')
                    continue
                
            X.append(vid)
            y.append(sample_class)
            
            sample_names[sample_class].pop(0)
        X = np.array(X)
        y = np.array(y)
        
        yield X,y
