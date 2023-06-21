import gc
import pickle
import logging
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader

from .utils import *

logger = logging.getLogger(__name__)


class Client(object):
    """Class for client object having its own (private) data and resources to train a model.

    Participating client has its own dataset which are usually non-IID compared to other clients.
    Each client only communicates with the center server with its trained parameters or globally aggregated parameters.

    Attributes:
        id: Integer indicating client's id.
        data: torch.utils.data.Dataset instance containing local data.
        device: Training machine indicator (e.g. "cpu", "cuda").
        __model: torch.nn instance as a local model.
    """
    def __init__(self, client_id, local_data, device,cfg):
        """Client object is initiated by the center server."""
        self.id = client_id
        self.data = local_data
        self.device = device
        self.__model = None
        self.cfg=cfg

    @property
    def model(self):
        """Local model getter for parameter aggregation."""
        return self.__model

    @model.setter
    def model(self, model):
        """Local model setter for passing globally aggregated model parameters."""
        self.__model = model

    def __len__(self):
        """Return a total size of the client's local data."""
        return len(self.data)

    def setup(self, **client_config):
        """Set up common configuration of each client; called by center server."""
        self.dataloader = DataLoader(self.data, batch_size=client_config["batch_size"], shuffle=True)
        self.local_epoch = client_config["num_local_epochs"]
        self.criterion = client_config["criterion"]
        self.optimizer = client_config["optimizer"]
        self.optim_config = client_config["optim_config"]

    def client_update(self):
        """Update local model using local dataset."""
        self.model.train()
        self.model.to(self.device)

        activities_meter=AverageMeter()
        loss_meter=AverageMeter()
        activities_conf = ConfusionMeter(self.cfg.num_activities)
        epoch_timer=Timer()

        print('lr:',self.cfg.train_learning_rate)
        optimizer=optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), \
            lr=self.cfg.train_learning_rate,weight_decay=self.cfg.weight_decay)

        begin_time=time.time()
        report=len(self.dataloader)//5

        for e in range(self.local_epoch):
            print('epoch ',e)

            for i,batch_data in enumerate(self.dataloader):
                batch_data=[b.to(device=self.device) for b in batch_data]
                batch_size=batch_data[0].shape[0]
                num_frames=batch_data[0].shape[1]
                batch_data[0]=batch_data[0].contiguous()
                #time.sleep(1200)
                # forward
                # actions_scores,activities_scores=model((batch_data[0],batch_data[1],batch_data[4]))

                #print(batch_data[0].shape,batch_data[1].dtype,batch_data[3].dtype)
                activities_scores = self.model((batch_data[0], batch_data[1], batch_data[3],batch_data[4]))["activities"]
                activities_in = batch_data[2].reshape((batch_size,num_frames))
                bboxes_num = batch_data[3].reshape(batch_size,num_frames)
                    
                activities_in = activities_in[:,0].reshape(batch_size,)

                # Predict activities
                activities_loss=F.cross_entropy(activities_scores,activities_in)
                activities_labels=torch.argmax(activities_scores,dim=1)  #B*T,
                activities_correct=torch.sum(torch.eq(activities_labels.int(),activities_in.int()).float())
                #print(activities_correct)
                activities_accuracy=activities_correct.item()/activities_scores.shape[0]
                activities_meter.update(activities_accuracy, activities_scores.shape[0])
                activities_conf.add(activities_labels, activities_in)

                # Total loss
                total_loss = activities_loss # + cfg.actions_loss_weight*actions_loss
                if torch.isnan(total_loss):
                    print("loss nan",total_loss)
                loss_meter.update(total_loss.item(), batch_size)

                # Optim
                optimizer.zero_grad()
                total_loss.backward()

                for m in self.model.state_dict():
                    '''if self.model.state_dict()[m].grad is None:
                        print("none",m)'''
                    if self.model.state_dict()[m].grad !=None and True in torch.isnan(self.model.state_dict()[m].grad):
                        print('grad nan')
                optimizer.step()

                model_nan=False
                for m in self.model.state_dict():
                    '''if self.model.state_dict()[m].grad is None:
                        print("none",m)'''
                    if  True in torch.isnan(self.model.state_dict()[m]):
                        model_nan=True 
                if model_nan:
                    print("model nan",i)

                if i%report==100:
                    print(f"batch {i}/{len(self.dataloader)} cur:{time.time()-begin_time}  est:{(time.time()-begin_time)*(len(self.dataloader)-i-1)/(i+1)} loss:{loss_meter.avg}")

                if self.device == "cuda": torch.cuda.empty_cache()      

        train_info={
            'time':epoch_timer.timeit(),
            'loss':loss_meter.avg,
            'activities_acc':activities_meter.avg*100,
            'activities_MPCA': MPCA(activities_conf.value()),
        } #'actions_acc':actions_meter.avg*100

        print(train_info)

        self.model.to("cpu")

    def client_evaluate(self):
        """Evaluate local model using local dataset (same as training set for convenience)."""


        self.model.eval()
        self.model.to(self.device)

        test_loss, correct = 0, 0

        activities_meter=AverageMeter()
        loss_meter=AverageMeter()
        epoch_timer=Timer()

        with torch.no_grad():
            for batch_data in self.dataloader:
                # prepare batch data
                batch_data=[b.to(device=self.device) for b in batch_data]
                batch_size=batch_data[0].shape[0]
                num_frames=batch_data[0].shape[1]
                
                #actions_in=batch_data[2].reshape((batch_size,num_frames,cfg.num_boxes))
                activities_in=batch_data[2].reshape((batch_size,num_frames))
                bboxes_num=batch_data[3].reshape(batch_size,num_frames)

                # forward
                activities_scores = self.model((batch_data[0], batch_data[1], batch_data[3],batch_data[4],batch_data[5]))['activities']
                
                activities_in=activities_in[:,0].reshape(batch_size,)

                # Predict activities
                activities_loss=F.cross_entropy(activities_scores,activities_in)
                activities_labels=torch.argmax(activities_scores,dim=1)  #B,
                activities_correct=torch.sum(torch.eq(activities_labels.int(),activities_in.int()).float())
                activities_accuracy=activities_correct.item()/activities_scores.shape[0]
                activities_meter.update(activities_accuracy, activities_scores.shape[0])
                #activities_conf.add(activities_labels, activities_in)

                # Total loss
                total_loss=activities_loss # + cfg.actions_loss_weight*actions_loss
                loss_meter.update(total_loss.item(), batch_size)

                if self.device == "cuda": torch.cuda.empty_cache()
        self.model.to("cpu")

        test_info={
            'time':epoch_timer.timeit(),
            'loss':loss_meter.avg,
            'activities_acc':activities_meter.avg*100,
            #'activities_MPCA': MPCA(activities_conf.value()),
        } #'actions_acc':actions_meter.avg*100

        '''test_loss = test_loss / len(self.dataloader)
        test_accuracy = correct / len(self.data)

        message = f"\t[Client {str(self.id).zfill(4)}] ...finished evaluation!\
            \n\t=> Test loss: {test_loss:.4f}\
            \n\t=> Test accuracy: {100. * test_accuracy:.2f}%\n"
        print(message, flush=True); logging.info(message)
        del message; gc.collect()'''
        print(test_info)

        return loss_meter.avg,activities_meter.avg*100
