from .education import *

import pickle
import random,os

load_test=True
load_train=True


def return_dataset(cfg,num_clients):

    img_path='../data/processed_v2'
    videos=os.listdir(cfg.data_path)

    train_videos=[]
    test_videos=[]
    undecided_videos=[]
    
    
    '''class_videos={"1":[],"2":[],"3":[],"4":[],"0":[]}
    for video in videos:
        video_labels=[]
        seqs=os.listdir(cfg.data_path+'/'+video)
        for seq in seqs:
            if not os.path.isdir(cfg.data_path+'/'+video+'/'+seq):
                continue
            label=seq[-1]
            if label not in video_labels:
                video_labels.append(label)
        
        for label in video_labels:
            class_videos[label].append(video)
    
    #print(class_videos)
    for label in class_videos.keys():
        class_videos_=[]
        for video in class_videos[label]:
            if video not in train_videos and video not in test_videos:
                class_videos_.append(video)
        random.shuffle(class_videos_)
        train_videos.append(class_videos_[0])
        test_videos.append(class_videos_[1])
    
    for video in videos:
        if video not in train_videos and video not in test_videos:
            undecided_videos.append(video)
    
    #print(train_videos)
    #print(test_videos)
    random.shuffle(undecided_videos)
    split=int(len(videos)*0.2-len(test_videos))
    train_videos.extend(undecided_videos[:-split])
    test_videos.extend(undecided_videos[-split:])
    print(len(train_videos),len(test_videos),len(undecided_videos),len(videos))

    #print(train_videos)
    #print(test_videos)
    
    test_videos=random.sample(videos,max(len(videos)//5,1))
    with open('test_samples.txt','w') as f: 
        for video in test_videos:
            f.write(video+'\n')
    
    print(len(test_videos),len(train_videos),len(videos))'''
    
    with open('test_samples.txt','r') as f: 
        test_videos=f.readlines()
    test_videos=[seq.replace('\n','') for seq in test_videos]

    for video in videos:
        if video not in test_videos:
            train_videos.append(video)

    video_per_client=len(train_videos)//num_clients

    allocated_videos=video_per_client*num_clients
    idx=0
    train_datasets=[]
    samples=[]

    '''images_test,activities_test,bboxes_test,bboxes_num_test,pos_mat_test= \
        education_read_annotations(cfg.data_path,test_videos,img_path,cfg.num_frames,cfg.out_size)'''
    
    print("test set:")

    if load_test:
        with open('val.pickle', 'rb') as f:
            validation_set = pickle.load(f)
            validation_set.image_size=cfg.image_size
        
        print("load val successful")
    else:
        validation_set=EducationDataset(cfg.data_path,test_videos,img_path,\
            cfg.num_frames,cfg.image_size,cfg.out_size,cfg.num_boxes)
        with open('val.pickle', 'wb') as f:
            pickle.dump(validation_set, f)
    

    #max_bboxes=max(bboxes_num_test)

    for i in range(num_clients):

        if allocated_videos<len(train_videos):
            cur=train_videos[idx:idx+video_per_client+1] 
            allocated_videos+=1
            idx+=video_per_client+1
        else: 
            cur=train_videos[idx:idx+video_per_client] 
            idx+=video_per_client

        '''images_train,activities_train,bboxes_train,bboxes_num_train= \
            education_read_annotations(cfg.data_path,cur,img_path,cfg.num_frames,cfg.out_size)'''
        print("train set:")

        loadname='train_'+str(i)+'_'+str(num_clients)+".pickle"

        if load_train:
            with open(loadname, 'rb') as f:
                training_set = pickle.load(f)
                training_set.image_size=cfg.image_size
            
            print(f"load train {i} successful")
        else:
            print(f"building training set {i}/{num_clients}")
            training_set=EducationDataset(cfg.data_path,cur,img_path,\
                cfg.num_frames,cfg.image_size,cfg.out_size,cfg.num_boxes)
            with open(loadname, 'wb') as f:
                pickle.dump(training_set, f)
        train_datasets.append(training_set)
        samples.append(len(training_set))
        #max_bboxes=max(max_bboxes,max(bboxes_num_train))
    
    '''train_4=[]
    for i in range(4):
        loadname='train_'+str(i)+'_4.pickle'
        with open(loadname, 'rb') as f:
            training_set = pickle.load(f)
            training_set.image_size=cfg.image_size
            train_4.append(training_set)
    
    train_4[0].copy(train_4[1])
    train_4[2].copy(train_4[3])

    train_datasets.append(train_4[0])
    train_datasets.append(train_4[2])
    samples.append(len(train_4[0]))
    samples.append(len(train_4[2]))

    with open('train_0_2.pickle', 'wb') as f:
            pickle.dump(train_4[0], f)
    
    with open('train_1_2.pickle', 'wb') as f:
            pickle.dump(train_4[2], f)
        
    #print("max bboxes:",max_bboxes)'''

    

    print('%s train samples'%str(samples))
    print('%d test samples'%len(validation_set))
                                         
    
    print('Reading dataset finished...')
    #print('%d train samples'%len(train_frames))
    #print('%d test samples'%len(test_frames))
    
    return train_datasets, validation_set
    