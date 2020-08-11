import random
import os

def picker(sample_count):
    L = []
    for cl in sample_count:
        L = L + [(f'{cl[1]}')]*cl[0]
    random.shuffle(L)
    return L

def pick_sample(shuffled_list):
    a = random.choice(shuffled_list)
    return a

def VideoDataGenerator(data_path, batch_size = 64):
    '''The data_path must contain folders containing samples from only that class.
    The samples should be homogenously numerated. (eg - 001.jpg, 002.jpg etc) for each frame of the video. 
    '''
    X = []
    y = []
    
    if data_path[-1] != '/':
        data_path = f"{data_path}/"
    
    classes = os.listdir(data_path)
    print(f"{len(classes)} classes found in folder. Namely - {classes}")
    
    class_target_map = [[classes[i], i] for i in range(len(classes))]
    samples_count = [[len(os.listdir(f"{data_path}{classes[i]}")), i] for i in range(len(classes))]
    picker_list = picker(samples_count)
    
    sample_names = [[] for i in range(len(classes))]
    for i in range(len(classes)):
        sample_names[i] = os.listdir(f'{data_path}{classes[i]}')
    
    while True:
        samples_pushed = []
        offset = 0
        total_samples = 0
        c = 0
        for cl in sample_names:
            if not cl:
                c += 1
            if c == len(classes) - 1:
                break
                
        for cl in classes:
            total_samples = total_samples + len(os.listdir(f'{data_path}{cl}'))
            
        for batch_iter in range(batch_size):
            X = []
            y = []
            sample_class = pick_sample(picker_list)
            picker_list.remove(sample_class)
            sample_class = int(sample_class)
            if not sample_names[sample_class]: 
                batch_iter -= 1
                continue
            vid_sample = f'{data_path}{classes[sample_class]}/{sample_names[sample_class][0]}'
            #fuck up hoga toh ye^ line me hoga
            frames = os.listdir(vid_sample)
            frames.sort()
            vid = []
            for frame in frames:
                try:
                    #frame = resize_frame(frame)
                    vid.append(cv2.imread(f'{vid_sample}/{frame}'))
                except:
                    print(f'Sample with tag - {sample_names[sample_class][0]} is broken and skipped.')
                    continue
                
            X.append(vid)
            y.append(sample_class)
            
            sample_names[sample_class].pop(0)
        X = np.array(X)
        y = np.array(y)
        
        yield X,y