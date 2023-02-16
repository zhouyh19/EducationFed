import torch
from torch.utils import data
import torchvision.models as models
import torchvision.transforms as transforms

import random
from PIL import Image
import numpy as np

import os,json

from collections import Counter


FRAMES_NUM={1: 302, 2: 347, 3: 194, 4: 257, 5: 536, 6: 401, 7: 968, 8: 221, 9: 356, 10: 302, 
            11: 1813, 12: 1084, 13: 851, 14: 723, 15: 464, 16: 1021, 17: 905, 18: 600, 19: 203, 20: 342, 
            21: 650, 22: 361, 23: 311, 24: 321, 25: 617, 26: 734, 27: 1804, 28: 470, 29: 635, 30: 356, 
            31: 690, 32: 194, 33: 193, 34: 395, 35: 707, 36: 914, 37: 1049, 38: 653, 39: 518, 40: 401, 
            41: 707, 42: 420, 43: 410, 44: 356}

 
FRAMES_SIZE={1: (480, 720), 2: (480, 720), 3: (480, 720), 4: (480, 720), 5: (480, 720), 6: (480, 720), 7: (480, 720), 8: (480, 720), 9: (480, 720), 10: (480, 720), 
             11: (480, 720), 12: (480, 720), 13: (480, 720), 14: (480, 720), 15: (450, 800), 16: (480, 720), 17: (480, 720), 18: (480, 720), 19: (480, 720), 20: (450, 800), 
             21: (450, 800), 22: (450, 800), 23: (450, 800), 24: (450, 800), 25: (480, 720), 26: (480, 720), 27: (480, 720), 28: (480, 720), 29: (480, 720), 30: (480, 720), 
             31: (480, 720), 32: (480, 720), 33: (480, 720), 34: (480, 720), 35: (480, 720), 36: (480, 720), 37: (480, 720), 38: (480, 720), 39: (480, 720), 40: (480, 720), 
             41: (480, 720), 42: (480, 720), 43: (480, 720), 44: (480, 720)}


ACTIONS=['NA','Crossing','Waiting','Queueing','Walking','Talking']
ACTIVITIES=['Crossing','Waiting','Queueing','Walking','Talking']


ACTIONS_ID={a:i for i,a in enumerate(ACTIONS)}
ACTIVITIES_ID={a:i for i,a in enumerate(ACTIVITIES)}
Action6to5 = {0:0, 1:1, 2:2, 3:3, 4:1, 5:4}
Activity5to4 = {0:0, 1:1, 2:2, 3:0, 4:3}


def education_read_annotations(path,selected_files,img_path,num_frames):
    images=[]
    activities=[]
    bboxes=[]
    bboxes_num=[]

    seqs=selected_files
    type_anno={}
    for seq in seqs:
        seq_path=path+'/'+seq+'/labels/'
        files=os.listdir(seq_path)

        anno=json.load(open(path+'/'+seq+'/annotation.json','r'))
        slices=anno["anno"]
        print(anno)

        for file in range(anno['total']):
            try:

                with open(seq_path+str(file)+'.txt',mode='r') as f:
                    bboxes_local=[]
                    for l in f.readlines():
                        values=l.replace('\n','').split(' ')
                        if values[0]!='0':
                            continue
                        x,y,w,h = (float(values[i])  for i  in range(1,5))
                        bboxes_local.append( (y,x,y+h,x+w) )
            except FileNotFoundError:
                continue
            
            if file>slices[0][1]:
                slices=slices[1:]

            if len(bboxes_local)==0:
                continue
            images.append(img_path+'/'+seq+'/'+str(file)+'.png')
            #print(slices)
            activities.append(slices[0][-1]-1)
            bboxes.append(bboxes_local)
            bboxes_num.append(len(bboxes_local))

        croped_len=len(images)-len(images)%num_frames
        images=images[:croped_len]
        activities=activities[:croped_len]
        bboxes=bboxes[:croped_len]
        bboxes_num=bboxes_num[:croped_len]
           
    return images,activities,bboxes,bboxes_num
            


class EducationDataset(data.Dataset):
    """
    Characterize collective dataset for pytorch
    """
    def __init__(self,images,activities,bboxes,bboxes_num,num_frames,image_size,feature_size,num_boxes):

        self.images=images
        self.activities=activities
        self.bboxes=bboxes
        self.bboxes_num=bboxes_num

        self.num_frames=num_frames
        self.image_size=image_size
        self.feature_size=feature_size
        self.num_boxes=num_boxes

        # self.frames_seq = np.empty((1337, 2), dtype = np.int)
        # self.flag = 0

    def __len__(self):
        """
        Return the total number of samples
        """
        return len(self.images)//self.num_frames
    
    def __getitem__(self,index):
        """
        Generate one sample of the dataset
        """
        # Save frame sequences
        # self.frames_seq[self.flag] = self.frames[index] # [0], self.frames[index][1]
        # if self.flag == 764: # 1336
        #     save_seq = self.frames_seq
        #     np.savetxt('vis/Collective/frames_seq.txt', save_seq)
        # self.flag += 1

        OH, OW=self.feature_size

        images=self.images[index*self.num_frames:(index+1)*self.num_frames]
        activities=self.activities[index*self.num_frames:(index+1)*self.num_frames]
        bboxes=self.bboxes[index*self.num_frames:(index+1)*self.num_frames]
        bboxes_num=self.bboxes_num[index*self.num_frames:(index+1)*self.num_frames]

        loaded_images=[]
        for img in images:
            img = Image.open(img)

            img=transforms.functional.resize(img,self.image_size)
            img=np.array(img)

            # H,W,3 -> 3,H,W
            img=img.transpose(2,0,1)
            loaded_images.append(img)
        
        images=loaded_images

        bboxes_aligned=[]
        for bboxes_local in bboxes:
            temp_boxes=[]
            for box in bboxes_local:
                y1,x1,y2,x2=box
                w1,h1,w2,h2 = x1*OW, y1*OH, x2*OW, y2*OH  
                temp_boxes.append((w1,h1,w2,h2))
            
            while len(temp_boxes)<self.num_boxes:
                temp_boxes.append((0,0,0,0))
            bboxes_aligned.append(temp_boxes)

        images = np.stack(images)
        activities = np.array(activities, dtype=np.int32)
        bboxes_num = np.array(bboxes_num, dtype=np.int32)
        bboxes=np.array(bboxes_aligned,dtype=np.float).reshape(-1,self.num_boxes,4)

        images=torch.from_numpy(images).contiguous()
        bboxes=torch.from_numpy(bboxes).float()
        activities=torch.from_numpy(activities).long()
        bboxes_num=torch.from_numpy(bboxes_num).int()
        
        return images, bboxes, activities, bboxes_num
    
    

    
