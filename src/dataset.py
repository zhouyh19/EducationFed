from .education import *

import pickle
import random,os


def return_dataset(cfg,num_clients):

    img_path='/data/extracted'
    print(cfg)
    seqs=os.listdir(cfg.data_path)
    train_seqs=[]

    '''test_seqs=random.sample(seqs,max(len(seqs)//5,1))
    with open('test_samples.txt','w') as f: 
        for seq in test_seqs:
            f.write(seq+'\n')'''
    
    with open('test_samples.txt','r') as f: 
        test_seqs=f.readlines()
    test_seqs=[seq.replace('\n','') for seq in test_seqs]
    print(test_seqs)

    for seq in seqs:
        if seq not in test_seqs:
            train_seqs.append(seq)
    random.shuffle(train_seqs)
    print(train_seqs)

    seq_per_client=len(train_seqs)//num_clients

    allocated_seqs=seq_per_client*num_clients
    idx=0
    train_datasets=[]
    samples=[]

    '''images_test,activities_test,bboxes_test,bboxes_num_test,pos_mat_test= \
        education_read_annotations(cfg.data_path,test_seqs,img_path,cfg.num_frames,cfg.out_size)'''
    validation_set=EducationDataset(cfg.data_path,test_seqs,img_path,\
        cfg.num_frames,cfg.image_size,cfg.out_size,cfg.num_boxes)

    #max_bboxes=max(bboxes_num_test)

    for i in range(num_clients):

        if allocated_seqs<len(train_seqs):
            cur=train_seqs[idx:idx+seq_per_client+1] 
            allocated_seqs+=1
            idx+=seq_per_client+1
        else: 
            cur=train_seqs[idx:idx+seq_per_client] 
            idx+=seq_per_client

        '''images_train,activities_train,bboxes_train,bboxes_num_train= \
            education_read_annotations(cfg.data_path,cur,img_path,cfg.num_frames,cfg.out_size)'''
        training_set=EducationDataset(cfg.data_path,cur,img_path,\
            cfg.num_frames,cfg.image_size,cfg.out_size,cfg.num_boxes)
        train_datasets.append(training_set)
        samples.append(len(training_set))
        #max_bboxes=max(max_bboxes,max(bboxes_num_train))
        
    #print("max bboxes:",max_bboxes)

    

    print('%s train samples'%str(samples))
    print('%d test samples'%len(validation_set))
                                         
    
    print('Reading dataset finished...')
    #print('%d train samples'%len(train_frames))
    #print('%d test samples'%len(test_frames))
    
    return train_datasets, validation_set
    