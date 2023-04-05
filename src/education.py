import torch
from torch.utils import data
import torchvision.models as models
import torchvision.transforms as transforms

import random,math
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

#num_person=40

def iou(box0, box1):
    """ 计算一对一交并比

    Parameters
    ----------
    box0, box1: Tensor of shape `(4, )`
        边界框
    """
    xy_max = torch.min(box0[2:], box1[2:])
    xy_min = torch.max(box0[:2], box1[:2])

    # 计算交集
    inter = torch.clamp(xy_max-xy_min, min=0)
    inter = inter[0]*inter[1]

    # 计算并集
    area_0 = (box0[2]-box0[0])*(box0[3]-box0[1])
    area_1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    union = area_0 + area_1- inter
    union = max(union,1e-8)

    return inter/union

def positional_embedding(pos1,pos2):
    '''
        pos:(x1,y1,x2,y2)
    '''
    if torch.norm(pos1)==0 or torch.norm(pos2)==0:
        pos_vec=torch.zeros(9)
    else:

        x1,y1=(pos1[0]+pos1[2])/2,(pos1[1]+pos1[3])/2
        x2,y2=(pos2[0]+pos2[2])/2,(pos2[1]+pos2[3])/2
        w1,h1=pos1[2]-pos1[0],pos1[3]-pos1[1]
        w2,h2=pos2[2]-pos2[0],pos2[3]-pos2[1]
        dx,dy=x2-x1,y2-y1


        dx=max(dx,1e-8)
        w1=max(w1,1e-8)
        h1=max(h1,1e-8)
        
        iou_=iou(pos1,pos2)
        pos_vec=torch.tensor((dx,dy,math.sqrt(dx**2+dy**2),math.atan(dy/dx),iou_,dx/w1,dy/h1,w2/w1,h2/h1))

    if True in torch.isnan(pos_vec):
        print('nan:',pos_vec)
        pos_vec=torch.nan_to_num(pos_vec)

    return pos_vec

def temporal_embedding(pos1,pos2,dt):
    if torch.norm(pos1)==0 or torch.norm(pos2)==0:
        tmp_vec=torch.zeros(10)
    else:
        x1,y1=(pos1[0]+pos1[2])/2,(pos1[1]+pos1[3])/2
        x2,y2=(pos2[0]+pos2[2])/2,(pos2[1]+pos2[3])/2
        w1,h1=pos1[2]-pos1[0],pos1[3]-pos1[1]
        w2,h2=pos2[2]-pos2[0],pos2[3]-pos2[1]
        dx,dy=x2-x1,y2-y1

        #print(dx,pos1,pos1.shape)
        dx=max(dx,1e-8)
        w1=max(w1,1e-8)
        h1=max(h1,1e-8)

        if dt==0:
            vx,vy,v=0,0,0
        else: 
            vx,vy,v=dx/dt,dy/dt,math.sqrt(dx**2+dy**2)/dt
        
        iou_=iou(pos1,pos2)
        tmp_vec=torch.tensor((dx,dy,math.sqrt(dx**2+dy**2),math.atan(dy/dx),vx,vy,v,w2/w1,h2/h1,iou_))
    return tmp_vec


def get_pos_mat(pos,num_person):
    mat=torch.zeros((num_person,num_person,9))
    for i in range(num_person):
        for j in range(num_person):
            mat[i][j]=positional_embedding(pos[i],pos[j])
    return mat

def get_tmp_mat(pos,num_timesteps):
    mat=torch.zeros((num_timesteps,num_timesteps,1,10))
    for i in range(num_timesteps):
        for j in range(num_timesteps):
            mat[i][j][0]=temporal_embedding(pos[i],pos[j],j-i)
    return mat

'''
def education_read_annotations(path,selected_files,img_path,num_frames,feature_size):
    
    OH, OW=feature_size

    
           
    return images,activities,bboxes,bboxes_num,pos_mat'''
            


class EducationDataset(data.Dataset):
    """
    Characterize collective dataset for pytorch
    """
    def __init__(self,path,selected_files,img_path,num_frames,image_size,feature_size,num_boxes):

        OH, OW=feature_size

        images=[]
        activities=[]
        bboxes=[]
        bboxes_num=[]
        pos_mat=[]
        tmp_mat=[]

        videos=selected_files
        type_anno={}
        for video in videos:
            video_path=path+'/'+video+'/'
            seqs=os.listdir(video_path)

            for seq in seqs:
                seq_path=video_path+seq+'/'
                frames=os.listdir(seq_path)
                selected_frames=[]
                person={}
                for frame in frames:
                    if frame!='annotations.txt':
                        selected_frames.append(frame.replace('.png',''))
                
                selected_frames.sort(key=lambda f: int(f))
                #print(selected_frames)

                #print('begin seq:',seq_path)
                with open(seq_path+'annotations.txt',mode='r') as f:
                    for l in f.readlines():
                        values=l.replace('\n','').split(',')
                        if values[0] not in selected_frames:
                            #print('rejected:',values[0],selected_frames)
                            continue
                        #print('frame',values[0])
                        if values[1] not in person:
                            person[values[1]]=[]
                        
                        person[values[1]].append(values)
                
                #print(person)
                person=list(person.values())
                if len(person)>num_boxes or len(person)<=1:
                    continue
                
                if len(selected_frames) != num_frames:
                    print('wrong frame number')
                    continue

                #print(len(person))
                for i in range(len(person)):
                    person[i].sort(key=lambda v:int(v[0]))

                local_pos_mat=[]
                for frame in selected_frames:
                    bboxes_local=[]
                    for idx,value in enumerate(person) :
                        #print(value)
                        if len(value)==0 or value[0][0]!=frame:
                            bboxes_local.append((0,0,0,0))
                        else: 
                            #print(value)
                            x1,y1,x2,y2 = (float(value[0][i])  for i  in range(2,6))
                            x1,y1,x2,y2 = x1*OW, y1*OH, x2*OW, y2*OH  
                            bboxes_local.append((x1,y1,x2,y2))
                            del person[idx][0]
                    
                    bboxes_num.append(len(bboxes_local))

                    while len(bboxes_local)<num_boxes:
                        bboxes_local.append((0,0,0,0))

                    bboxes_local=np.array(bboxes_local)
                    bboxes_local=torch.from_numpy(bboxes_local).float()

                    images.append(seq_path+frame+'.png')
                    activities.append(int(seq[-1])-1)
                    bboxes.append(bboxes_local.unsqueeze(0))
                    local_pos_mat.append(get_pos_mat(bboxes_local,num_boxes).unsqueeze(0))

                local_tmp_mat=[]
                for person in range(bboxes_num[-1]):
                    tmp_pos=[bbox[0][person] for bbox in bboxes[-num_frames:]]
                    local_tmp_mat.append(get_tmp_mat(tmp_pos,num_frames))
                while len(local_tmp_mat)<num_boxes:
                    local_tmp_mat.append(torch.zeros((num_frames,num_frames,1,10)))

                pos_mat.append(torch.concat(local_pos_mat,0).unsqueeze(0))
                #print(pos_mat[-1].shape)
                tmp_mat.append(torch.concat(local_tmp_mat,2).unsqueeze(0))

            croped_len=len(images)-len(images)%num_frames
            images=images[:croped_len]
            activities=activities[:croped_len]
            bboxes=bboxes[:croped_len]
            bboxes_num=bboxes_num[:croped_len]
            #pos_mat=pos_mat[:croped_len]
        
        #bboxes=np.array(bboxes,dtype=float)
        #bboxes=torch.from_numpy(bboxes).float()
        print(len(bboxes),len(pos_mat),len(videos))
        bboxes=torch.concat(bboxes,0)
        pos_mat=torch.concat(pos_mat,0)
        tmp_mat=torch.concat(tmp_mat,0)
        print("bboxes",bboxes.shape)
        print("pos_mat",pos_mat.shape)

        activity_cnt=[0,0,0,0,0]
        for i in range(len(activities)):
            activity=activities[i]
            #print(activity)
            activity_cnt[activity]+=1
        print("activity count:",activity_cnt)
        
        self.images=images
        self.activities=activities
        self.bboxes=bboxes
        self.bboxes_num=bboxes_num
        self.pos_mat=pos_mat
        self.tmp_mat=tmp_mat

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
        bboxes=self.bboxes[index*self.num_frames:(index+1)*self.num_frames].reshape(-1,self.num_boxes,4)
        bboxes_num=self.bboxes_num[index*self.num_frames:(index+1)*self.num_frames]
        pos_mat=self.pos_mat[index]
        tmp_mat=self.tmp_mat[index]

        loaded_images=[]
        for img in images:
            img = Image.open(img)

            img=transforms.functional.resize(img,self.image_size)
            img=np.array(img)

            # H,W,3 -> 3,H,W
            img=img.transpose(2,0,1)
            loaded_images.append(img)
        
        images=loaded_images

        '''bboxes_aligned=[]
        for bboxes_local in bboxes:
            temp_boxes=[]
            for box in bboxes_local:
                x1,y1,x2,y2=box
                #w1,h1,w2,h2 = x1*OW, y1*OH, x2*OW, y2*OH  
                temp_boxes.append((w1,h1,w2,h2))
            
            while len(temp_boxes)<self.num_boxes:
                temp_boxes.append((0,0,0,0))
            bboxes_aligned.append(temp_boxes)'''
        #bboxes_aligned=bboxes

        images = np.stack(images)
        activities = np.array(activities, dtype=np.int32)
        bboxes_num = np.array(bboxes_num, dtype=np.int32)
        #bboxes=np.array(bboxes_aligned,dtype=np.float)

        images=torch.from_numpy(images).contiguous()
        #bboxes=torch.from_numpy(bboxes).float()
        activities=torch.from_numpy(activities).long()
        bboxes_num=torch.from_numpy(bboxes_num).int()
        
        return images, bboxes, activities, bboxes_num,pos_mat,tmp_mat
    
    

    
