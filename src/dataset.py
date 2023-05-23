from .education import *

import pickle
import random,os


def return_dataset(cfg,num_clients):

    img_path='/data/extracted_new_'
    videos=os.listdir(cfg.data_path)

    train_videos=[]
    test_videos=[]
    undecided_videos=[]
    
    

    '''class_videos={"1":[],"2":[],"3":[],"4":[],"5":[]}
    for video in videos:
        video_labels=[]
        seqs=os.listdir(cfg.data_path+'/'+video)
        for seq in seqs:
            if not os.path.isdir(cfg.data_path+'/'+video+'/'+seq):
                continue
            label=seq[-1]
            if label not in video_labels:
                video_labels.append(label)
        if len(video_labels)>1:
            video_labels.remove('1')
        if len(video_labels)!=1:
            print(video_labels)
        class_videos[video_labels[0]].append(video)
    
    print(class_videos)
    for label in class_videos.keys():
        random.shuffle(class_videos[label])
        train_videos.append(class_videos[label][0])
        if len(class_videos[label])>1:
            test_videos.append(class_videos[label][1])
        if len(class_videos[label])>2:
            undecided_videos.extend(class_videos[label][2:])
    
    print(train_videos)
    print(test_videos)
    random.shuffle(undecided_videos)
    split=int(len(undecided_videos)*0.8)
    train_videos.extend(undecided_videos[:split])
    test_videos.extend(undecided_videos[split:])
    print(train_videos)
    print(test_videos)
    
    #test_videos=random.sample(seqs,max(len(seqs)//5,1))
    with open('test_samples.txt','w') as f: 
        for video in test_videos:
            f.write(video+'\n')'''
    
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
    validation_set=None

    '''images_test,activities_test,bboxes_test,bboxes_num_test,pos_mat_test= \
        education_read_annotations(cfg.data_path,test_videos,img_path,cfg.num_frames,cfg.out_size)'''
    print("test set:")
    if os.path.exists('./preprocessed/test_set.pkl'):
        with open('./preprocessed/test_set.pkl', 'rb') as f:
            validation_set = pickle.load(f)
    else:
        os.mkdir('./preprocessed/')
        validation_set = EducationDataset(cfg.data_path, test_videos, img_path,\
            cfg.num_frames, cfg.image_size, cfg.out_size, cfg.num_boxes)
        with open('./preprocessed/test_set.pkl', 'wb') as f:
            pickle.dump(validation_set, f)

    max_bboxes=max(validation_set.bboxes_num)

    if os.path.exists('./preprocessed/client_num_{}/'.format(num_clients)):
        for i in range(num_clients):
            with open('./preprocessed/client_num_{}/train_set_{}.pkl'.format(num_clients, i), 'rb') as f:
                train_datasets.append(pickle.load(f))
    else:
        os.makedirs('./preprocessed/client_num_{}/'.format(num_clients))
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
            training_set=EducationDataset(cfg.data_path,cur,img_path,\
                cfg.num_frames,cfg.image_size,cfg.out_size,cfg.num_boxes)
            train_datasets.append(training_set)
            with open('./preprocessed/client_num_{}/train_set_{}.pkl'.format(num_clients, i), 'wb') as f:
                pickle.dump(training_set, f)
            # samples.append(len(training_set))
            # max_bboxes=max(max_bboxes,max(training_set.bboxes_num))
    for i in range(num_clients):
        samples.append(len(train_datasets[i]))
        max_bboxes=max(max_bboxes,max(train_datasets[i].bboxes_num))
    print("max bboxes:",max_bboxes)

    

    print('%s train samples'%str(samples))
    print('%d test samples'%len(validation_set))
                                         
    
    print('Reading dataset finished...')
    #print('%d train samples'%len(train_frames))
    #print('%d test samples'%len(test_frames))
    
    return train_datasets, validation_set
    