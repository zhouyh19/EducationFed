import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter   

from .backbone import *
from .utils import *
from roi_align.roi_align import RoIAlign

from .dynamic_infer_module import Dynamic_Person_Inference, Hierarchical_Dynamic_Inference, Multi_Dynamic_Inference
from .pctdm_infer_module import PCTDM
from .higcin_infer_module import CrossInferBlock
from .AT_infer_module import Actor_Transformer, Embfeature_PositionEmbedding
from .ARG_infer_module import GCN_Module
from .SACRF_BiUTE_infer_module import SACRF, BiUTE
from .TCE_STBiP_module import MultiHeadLayerEmbfeatureContextEncoding
from .positional_encoding import Context_PositionEmbeddingSine

#################################
# Models for federated learning #
#################################
# McMahan et al., 2016; 199,210 parameters
class TwoNN(nn.Module):
    def __init__(self, name, in_features, num_hiddens, num_classes):
        super(TwoNN, self).__init__()
        self.name = name
        self.activation = nn.ReLU(True)

        self.fc1 = nn.Linear(in_features=in_features, out_features=num_hiddens, bias=True)
        self.fc2 = nn.Linear(in_features=num_hiddens, out_features=num_hiddens, bias=True)
        self.fc3 = nn.Linear(in_features=num_hiddens, out_features=num_classes, bias=True)

    def forward(self, x):
        if x.ndim == 4:
            x = x.view(x.size(0), -1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

# McMahan et al., 2016; 1,663,370 parameters
class CNN(nn.Module):
    def __init__(self, name, in_channels, hidden_channels, num_hiddens, num_classes):
        super(CNN, self).__init__()
        self.name = name
        self.activation = nn.ReLU(True)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=(5, 5), padding=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels * 2, kernel_size=(5, 5), padding=1, stride=1, bias=False)
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=(hidden_channels * 2) * (7 * 7), out_features=num_hiddens, bias=False)
        #print('fc1:',(hidden_channels * 2) * (7 * 7),num_hiddens)
        self.fc2 = nn.Linear(in_features=num_hiddens, out_features=num_classes, bias=False)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.maxpool1(x)

        x = self.activation(self.conv2(x))
        x = self.maxpool2(x)
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

# for CIFAR10
class CNN2(nn.Module):
    def __init__(self, name, in_channels, hidden_channels, num_hiddens, num_classes):
        super(CNN2, self).__init__()
        self.name = name
        self.activation = nn.ReLU(True)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=(5, 5), padding=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels * 2, kernel_size=(5, 5), padding=1, stride=1, bias=False)
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=(hidden_channels * 2) * (8 * 8), out_features=num_hiddens, bias=False)
        self.fc2 = nn.Linear(in_features=num_hiddens, out_features=num_classes, bias=False)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.maxpool1(x)

        x = self.activation(self.conv2(x))
        x = self.maxpool2(x)
        x = self.flatten(x)
    
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        
        return x





class PositionalInferenceBlock(nn.Module):
    def __init__(self,s,t,feature_len,attn_len,posembed_len):
        super(PositionalInferenceBlock, self).__init__()
        self.s=s
        self.t=t
        self.feature_len=feature_len
        self.attn_len=attn_len
        self.posembed_len=posembed_len
        self.a_mat=nn.Linear(feature_len,attn_len)
        self.b_mat=nn.Linear(feature_len+posembed_len,attn_len)
        self.g=nn.Linear(feature_len+posembed_len,feature_len)
        self.pos=nn.Linear(9,posembed_len)
        self.tmp=nn.Linear(10,posembed_len)
        self.softmax=nn.Softmax(dim=1)

    def forward(self, batch_data, positions,temp):
        #positions=positions.to(self.device)
        batch_data=batch_data.reshape((self.t,self.s,-1))
        #pos1=self.pos(positions[0])
        #pos2=self.pos(positions[1])
        #pos=torch.concat((pos1,pos2),-1)
        pos=self.pos(positions)
        tmp=self.tmp(temp)
        #print(pos.shape)

        theta=self.a_mat(batch_data).reshape((self.t,self.s,-1))

        '''new_pos=[pos.unsqueeze(0) for i in range(self.t)]
        new_pos=torch.concat(new_pos)'''
        #batch_data_with_pos=torch.concat((batch_data,new_pos),2)
        #phi=self.b_mat(batch_data_with_pos).reshape((self.t,self.s,-1))
        #feats=self.g(batch_data_with_pos).reshape((self.t,self.s,-1))

        #phi=self.b_mat(batch_data).reshape((self.t,self.s,-1))
        results=torch.zeros((self.t,self.s,self.feature_len)).to(batch_data.device)

        phi_with_pos=[]
        feats_with_pos=[]
        for j in range(self.s):
            #print(batch_data.shape,new_pos[:,j].shape)
            local_pos=torch.concat((batch_data,pos[:,j]),-1)
            local_phi=self.b_mat(local_pos).reshape((self.t,self.s,-1))
            local_feats=self.g(local_pos).reshape((self.t,self.s,-1))
            phi_with_pos.append(local_phi)
            feats_with_pos.append(local_feats)

        for i in range(self.t):
            #phi_=torch.transpose(phi[i],0,1)
            
            #batch_result=torch.mm(theta_,phi_)
            #print(theta.shape,theta_.shape,phi_.shape,batch_result.shape)
            for j in range(self.s):
                batch_result=torch.mm(theta[i][j].unsqueeze(0),torch.transpose(phi_with_pos[j][i],0,1))
                batch_result=self.softmax(batch_result)
                results[i][j]+=torch.mm(batch_result.reshape((1,self.s)),feats_with_pos[j][i]).reshape(-1)
        
        phi_with_tmp=[]
        feats_with_tmp=[]
        for i in range(self.t):
            #print(tmp.shape)
            local_pos=torch.concat((batch_data,tmp[i]),-1)
            local_phi=self.b_mat(local_pos).reshape((self.t,self.s,-1))
            local_feats=self.g(local_pos).reshape((self.t,self.s,-1))
            phi_with_tmp.append(local_phi)
            feats_with_tmp.append(local_feats)

        for i in range(self.s):
            #print(theta.shape,theta_.shape,phi_.shape,batch_result.shape)
            for j in range(self.t):
                batch_result=torch.mm(theta[j][i].unsqueeze(0),torch.transpose(phi_with_tmp[j][:,i],0,1))
                batch_result=self.softmax(batch_result)
                results[j][i]+=torch.mm(batch_result.reshape((1,self.t)),feats_with_tmp[j][:,i]).reshape(-1)

        #print(results)
        #print(True in torch.isnan(results))
        #results/=(self.s+self.t)
        return results

class STGridModule(nn.Module):
    def __init__(self,s,t,feature_len):
        super(STGridModule, self).__init__()
        self.s=s
        self.t=t
        self.feature_len=feature_len
        self.f=nn.Linear(feature_len*2,feature_len)
    
    def forward(self,batch_data):
        batch_data=batch_data.reshape((self.t,self.s,self.feature_len))
        s_pooling,_=torch.max(batch_data,dim=1)
        t_pooling,_=torch.max(batch_data,dim=0)
        result=torch.zeros((self.t,self.s,self.feature_len)).to(batch_data.device)
        for i in range(self.s):
            for j in range(self.t):
                result[j,i]=self.f(torch.concat((s_pooling[j],t_pooling[i]),0))
        return result


class CrossInferenceBlock(nn.Module):
    def __init__(self,s,t,feature_len,attn_len):
        super(CrossInferenceBlock, self).__init__()
        self.s=s
        self.t=t
        self.feature_len=feature_len
        self.attn_len=attn_len
        self.a_mat=nn.Linear(feature_len,attn_len)
        self.b_mat=nn.Linear(feature_len,attn_len)
        self.g=nn.Linear(feature_len,feature_len)
    
    def forward(self, batch_data):
        theta=self.a_mat(batch_data).reshape((self.t,self.s,-1))
        phi=self.b_mat(batch_data).reshape((self.t,self.s,-1))
        feats=self.g(batch_data).reshape((self.t,self.s,-1))
        results=torch.zeros((self.t,self.s,self.feature_len)).to(batch_data.device)
        for i in range(self.t):
            theta_=theta[i]
            phi_=torch.transpose(phi[i],0,1)
            
            batch_result=torch.mm(theta_,phi_)
            #print(theta.shape,theta_.shape,phi_.shape,batch_result.shape)
            for j in range(self.s):
                results[i][j]+=torch.mm(batch_result[j].reshape((1,self.s)),feats[i]).reshape(-1)
        
        '''for i in range(self.s):
            theta_=theta[:,i]
            phi_=torch.transpose(phi[:,i],0,1)
            
            batch_result=torch.mm(theta_,phi_)
            #print(theta.shape,theta_.shape,phi_.shape,batch_result.shape)
            for j in range(self.t):
                results[j][i]+=torch.mm(batch_result[j].reshape((1,self.t)),feats[:,i]).reshape(-1)'''

        results/=(self.s+self.t)
        return results


class Dynamic_collective(nn.Module):
    def __init__(self, cfg):
        super(Dynamic_collective, self).__init__()
        self.cfg = cfg
        T, N = cfg.num_frames, cfg.num_boxes
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn

        if cfg.backbone == 'inv3':
            self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=True)
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=True)
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained=True)
        else:
            assert False
        # self.backbone = MyInception_v3(transform_input=False, pretrained=True)

        if not self.cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)
        self.fc_emb_1 = nn.Linear(K * K * D, NFB)
        self.nl_emb_1 = nn.LayerNorm([NFB])

        #self.gcn_list = torch.nn.ModuleList([GCN_Module(self.cfg) for i in range(self.cfg.gcn_layers)])
        if self.cfg.lite_dim:
            in_dim = self.cfg.lite_dim
            print_log(cfg.log_path, 'Activate lite model inference.')
        else:
            in_dim = NFB
            print_log(cfg.log_path, 'Deactivate lite model inference.')

        if not self.cfg.hierarchical_inference:
            self.DPI = Dynamic_Person_Inference(
                in_dim = in_dim,
                person_mat_shape = (T, N),
                stride = cfg.stride,
                kernel_size = cfg.ST_kernel_size,
                dynamic_sampling=cfg.dynamic_sampling,
                sampling_ratio = cfg.sampling_ratio, # [1,2,4]
                group = cfg.group,
                scale_factor = cfg.scale_factor,
                beta_factor = cfg.beta_factor,
                parallel_inference = cfg.parallel_inference,
                cfg = cfg)
            print_log(cfg.log_path, 'Hierarchical Inference : ' + str(cfg.hierarchical_inference))
        else:
            self.DPI = Hierarchical_Dynamic_Inference(
                in_dim = in_dim,
                person_mat_shape=(T, N),
                stride=cfg.stride,
                kernel_size=cfg.ST_kernel_size,
                dynamic_sampling=cfg.dynamic_sampling,
                sampling_ratio=cfg.sampling_ratio,  # [1,2,4]
                group=cfg.group,
                scale_factor=cfg.scale_factor,
                beta_factor=cfg.beta_factor,
                parallel_inference=cfg.parallel_inference,
                cfg = cfg,)
            print(cfg.log_path, 'Hierarchical Inference : ' + str(cfg.hierarchical_inference))
        self.dpi_nl = nn.LayerNorm([T, in_dim])
        self.dropout_global = nn.Dropout(p=self.cfg.train_dropout_prob)

        # Lite Dynamic inference
        if self.cfg.lite_dim:
            self.point_conv = nn.Conv2d(NFB, in_dim, kernel_size=1, stride=1)
            self.point_ln = nn.LayerNorm([T, N, in_dim])
            self.fc_activities = nn.Linear(in_dim, self.cfg.num_activities)
        else:
            self.fc_activities = nn.Linear(in_dim, self.cfg.num_activities)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        #self.cim=CrossInferenceBlock(40,10,1024,1024)
        self.pim1=PositionalInferenceBlock(40,10,1024,1024,512)
        self.stg=STGridModule(40,10,1024)
        #self.pim2=PositionalInferenceBlock(40,10,1024,1024,512)
        #nn.init.zeros_(self.fc_gcn_3.weight)
        #self.writer=SummaryWriter('./runs')

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)

    def forward(self, batch_data):
        images_in, boxes_in, bboxes_num_in,pos_mat,tmp_mat = batch_data

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        MAX_N = self.cfg.num_boxes
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn
        #print(B,T,MAX_N,bboxes_num_in)
        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in = boxes_in.reshape(B * T, MAX_N, 4)

        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)


        # Build multiscale features
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)
        features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW

        boxes_in_flat = torch.reshape(boxes_in, (B * T * MAX_N, 4))  # B*T*MAX_N, 4
        boxes_idx = [i * torch.ones(MAX_N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, MAX_N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * MAX_N,))  # B*T*MAX_N,

        # RoI Align
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features_all = self.roi_align(features_multiscale,
                                            boxes_in_flat,
                                            boxes_idx_flat)  # B*T*MAX_N, D, K, K,
        boxes_features_all = boxes_features_all.reshape(B, T, MAX_N, -1)  # B*T,MAX_N, D*K*K

        # Embedding
        boxes_features_all = self.fc_emb_1(boxes_features_all)  # B, T,MAX_N, NFB
        boxes_features_all = self.nl_emb_1(boxes_features_all)
        boxes_features_all = F.relu(boxes_features_all)

        if self.cfg.lite_dim:
            boxes_features_all = boxes_features_all.permute(0, 3, 1, 2)
            boxes_features_all = self.point_conv(boxes_features_all)
            boxes_features_all = boxes_features_all.permute(0, 2, 3, 1)
            boxes_features_all = self.point_ln(boxes_features_all)
            boxes_features_all = F.relu(boxes_features_all, inplace = True)
        else:
            None

        # boxes_features_all = boxes_features_all.reshape(B, T, MAX_N, NFB)
        # boxes_in = boxes_in.reshape(B, T, MAX_N, 4)

        #actions_scores = []
        activities_scores = []
        bboxes_num_in = bboxes_num_in.reshape(B, T)  # B,T,
        for b in range(B):
            #N = bboxes_num_in[b][0]
            #N=MAX_N
            N = bboxes_num_in[b][0]
            boxes_features = boxes_features_all[b, :, :MAX_N, :].reshape(1, T, MAX_N, -1)  # 1,T,N,NFB
            
            # boxes_positions = boxes_in[b, :, :N, :].reshape(T * N, 4)  # T*N, 4

            # Dynamic graph inference
            #print(boxes_features.shape)
            #boxes_features=self.cim(boxes_features).reshape(1,10,40,1024)
            graph_boxes_features=self.pim1(boxes_features,pos_mat[b],tmp_mat[b]).reshape(1,10,40,1024)
            grid_features=self.stg(boxes_features).reshape(1,10,40,1024)
            #boxes_features=self.pim2(boxes_features,pos_mat[b],tmp_mat[b]).reshape(1,10,40,1024)
            graph_boxes_features=graph_boxes_features+grid_features  #[:,:,:N]
            #print(graph_boxes_features.shape)
            #graph_boxes_features = self.DPI(boxes_features)
            torch.cuda.empty_cache()

            # cat graph_boxes_features with boxes_features
            #print(len(graph_boxes_features),boxes_features.shape)
            #print(graph_boxes_features[0].shape,graph_boxes_features[1].shape)
            boxes_states = graph_boxes_features #+ boxes_features  # 1, T, N, NFG
            boxes_states = boxes_states[:,:,:N]
            boxes_states = boxes_states.permute(0, 2, 1, 3).view(N, T, -1)
            boxes_states = self.dpi_nl(boxes_states)
            boxes_states = F.relu(boxes_states, inplace=True)
            boxes_states = self.dropout_global(boxes_states)
            NFS = NFG
            # boxes_states = boxes_states.view(T, N, -1)

            # Predict actions
            # actn_score = self.fc_actions(boxes_states)  # T,N, actn_num
            # actn_score = torch.mean(actn_score, dim=0).reshape(N, -1)  # N, actn_num
            # actions_scores.append(actn_score)
            # Predict activities
            boxes_states_pooled, _ = torch.max(boxes_states, dim = 0)  # T, NFS
            acty_score = self.fc_activities(boxes_states_pooled)  # T, acty_num
            acty_score = torch.mean(acty_score, dim=0).reshape(1, -1)  # 1, acty_num
            activities_scores.append(acty_score)

        # actions_scores = torch.cat(actions_scores, dim=0)  # ALL_N,actn_num
        activities_scores = torch.cat(activities_scores, dim=0)  # B,acty_num

        return {'activities':activities_scores}# activities_scores # actions_scores,

class Dynamic_volleyball(nn.Module):
    """
    main module of GCN for the volleyball dataset
    """
    def __init__(self, cfg):
        super(Dynamic_volleyball, self).__init__()
        self.cfg=cfg
        
        T, N=self.cfg.num_frames, self.cfg.num_boxes
        D=self.cfg.emb_features
        K=self.cfg.crop_size[0]
        NFB=self.cfg.num_features_boxes
        NFR, NFG=self.cfg.num_features_relation, self.cfg.num_features_gcn
        NG=self.cfg.num_graph
        
        
        if cfg.backbone=='inv3':
            self.backbone=MyInception_v3(transform_input=False, pretrained=True)
        elif cfg.backbone=='vgg16':
            self.backbone=MyVGG16(pretrained = True)
        elif cfg.backbone=='vgg19':
            self.backbone=MyVGG19(pretrained = True)
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained = True)
        elif cfg.backbone == 'alex':
            self.backbone = MyAlex(pretrained=True)
        else:
            assert False
        
        if not cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad=False
        
        self.roi_align = RoIAlign(*self.cfg.crop_size)
        # self.avgpool_person = nn.AdaptiveAvgPool2d((1,1))
        self.fc_emb_1 = nn.Linear(K*K*D,NFB)
        self.nl_emb_1 = nn.LayerNorm([NFB])
        
        
        #self.gcn_list = torch.nn.ModuleList([ GCN_Module(self.cfg)  for i in range(self.cfg.gcn_layers) ])
        if self.cfg.lite_dim:
            in_dim = self.cfg.lite_dim
            print_log(cfg.log_path, 'Activate lite model inference.')
        else:
            in_dim = NFB
            print_log(cfg.log_path, 'Deactivate lite model inference.')

        if not self.cfg.hierarchical_inference:
            # self.DPI = Dynamic_Person_Inference(
            #     in_dim = in_dim,
            #     person_mat_shape = (10, 12),
            #     stride = cfg.stride,
            #     kernel_size = cfg.ST_kernel_size,
            #     dynamic_sampling=cfg.dynamic_sampling,
            #     sampling_ratio = cfg.sampling_ratio, # [1,2,4]
            #     group = cfg.group,
            #     scale_factor = cfg.scale_factor,
            #     beta_factor = cfg.beta_factor,
            #     parallel_inference = cfg.parallel_inference,
            #     cfg = cfg)
            self.DPI = Multi_Dynamic_Inference(
                in_dim = in_dim,
                person_mat_shape = (10, 12),
                stride = cfg.stride,
                kernel_size = cfg.ST_kernel_size,
                dynamic_sampling=cfg.dynamic_sampling,
                sampling_ratio = cfg.sampling_ratio, # [1,2,4]
                group = cfg.group,
                scale_factor = cfg.scale_factor,
                beta_factor = cfg.beta_factor,
                parallel_inference = cfg.parallel_inference,
                num_DIM = cfg.num_DIM,
                cfg = cfg)
            print_log(cfg.log_path, 'Hierarchical Inference : ' + str(cfg.hierarchical_inference))
        else:
            self.DPI = Hierarchical_Dynamic_Inference(
                in_dim = in_dim,
                person_mat_shape=(10, 12),
                stride=cfg.stride,
                kernel_size=cfg.ST_kernel_size,
                dynamic_sampling=cfg.dynamic_sampling,
                sampling_ratio=cfg.sampling_ratio,  # [1,2,4]
                group=cfg.group,
                scale_factor=cfg.scale_factor,
                beta_factor=cfg.beta_factor,
                parallel_inference=cfg.parallel_inference,
                cfg = cfg,)
            print(cfg.log_path, 'Hierarchical Inference : ' + str(cfg.hierarchical_inference))
        self.dpi_nl = nn.LayerNorm([T, N, in_dim])
        self.dropout_global = nn.Dropout(p=self.cfg.train_dropout_prob)


        # Lite Dynamic inference
        if self.cfg.lite_dim:
            self.point_conv = nn.Conv2d(NFB, in_dim, kernel_size = 1, stride = 1)
            self.point_ln = nn.LayerNorm([T, N, in_dim])
            self.fc_activities = nn.Linear(in_dim, self.cfg.num_activities)
        else:
            self.fc_activities=nn.Linear(NFG, self.cfg.num_activities)

        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
                    
    def loadmodel(self,filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)

    def loadpart(self, pretrained_state_dict, model, prefix):
        num = 0
        model_state_dict = model.state_dict()
        pretrained_in_model = collections.OrderedDict()
        for k,v in pretrained_state_dict.items():
            if k.replace(prefix, '') in model_state_dict:
                pretrained_in_model[k.replace(prefix, '')] = v
                num +=1
        model_state_dict.update(pretrained_in_model)
        model.load_state_dict(model_state_dict)
        print(str(num)+' parameters loaded for '+prefix)


    def forward(self,batch_data):
        images_in, boxes_in = batch_data
        
        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W=self.cfg.image_size
        OH, OW=self.cfg.out_size
        N=self.cfg.num_boxes
        
        # Reshape the input data
        images_in_flat=torch.reshape(images_in,(B*T,3,H,W))  #B*T, 3, H, W
        boxes_in_flat=torch.reshape(boxes_in,(B*T*N,4))  #B*T*N, 4

        boxes_idx=[i * torch.ones(N, dtype=torch.int)   for i in range(B*T) ]
        boxes_idx=torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat=torch.reshape(boxes_idx,(B*T*N,))  #B*T*N,
        
        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build  features
        assert outputs[0].shape[2:4]==torch.Size([OH,OW])
        features_multiscale=[]
        for features in outputs:
            if features.shape[2:4]!=torch.Size([OH,OW]):
                features=F.interpolate(features,size=(OH,OW),mode='bilinear',align_corners=True)
            features_multiscale.append(features)
        
        features_multiscale=torch.cat(features_multiscale,dim=1)  #B*T, D, OH, OW
        
        
        # RoI Align
        boxes_in_flat.requires_grad=False
        boxes_idx_flat.requires_grad=False
        boxes_features=self.roi_align(features_multiscale,
                                            boxes_in_flat,
                                            boxes_idx_flat)  #B*T*N, D, K, K,
        boxes_features=boxes_features.reshape(B,T,N,-1)  #B,T,N, D*K*K

        # Embedding 
        boxes_features=self.fc_emb_1(boxes_features)  # B,T,N, NFB
        boxes_features=self.nl_emb_1(boxes_features)
        boxes_features=F.relu(boxes_features, inplace = True)

        if self.cfg.lite_dim:
            boxes_features = boxes_features.permute(0, 3, 1, 2)
            boxes_features = self.point_conv(boxes_features)
            boxes_features = boxes_features.permute(0, 2, 3, 1)
            boxes_features = self.point_ln(boxes_features)
            boxes_features = F.relu(boxes_features, inplace = True)
        else:
            None

        # Dynamic graph inference
        # graph_boxes_features = self.DPI(boxes_features)
        graph_boxes_features, ft_infer_MAD = self.DPI(boxes_features)
        torch.cuda.empty_cache()


        if self.cfg.backbone == 'res18':
            graph_boxes_features = graph_boxes_features.reshape(B, T, N, -1)
            graph_boxes_features = self.dpi_nl(graph_boxes_features)
            graph_boxes_features = F.relu(graph_boxes_features, inplace=True)
            boxes_features = boxes_features.reshape(B, T, N, -1)
            boxes_states = graph_boxes_features + boxes_features
            boxes_states = self.dropout_global(boxes_states)
        elif self.cfg.backbone == 'vgg16':
            graph_boxes_features = graph_boxes_features.reshape(B, T, N, -1)
            boxes_features = boxes_features.reshape(B, T, N, -1)
            boxes_states = graph_boxes_features + boxes_features
            boxes_states = self.dpi_nl(boxes_states)
            boxes_states = F.relu(boxes_states, inplace = True)
            boxes_states = self.dropout_global(boxes_states)


        # Predict actions
        # boxes_states_flat=boxes_states.reshape(-1,NFS)  #B*T*N, NFS
        # actions_scores=self.fc_actions(boxes_states_flat)  #B*T*N, actn_num
        
        # Predict activities
        boxes_states_pooled, _ = torch.max(boxes_states,dim=2)
        boxes_states_pooled_flat = boxes_states_pooled.reshape(B*T, -1)
        activities_scores = self.fc_activities(boxes_states_pooled_flat)  #B*T, acty_num
        
        # Temporal fusion
        # actions_scores = actions_scores.reshape(B,T,N,-1)
        # actions_scores = torch.mean(actions_scores,dim=1).reshape(B*N,-1)
        activities_scores = activities_scores.reshape(B, T, -1)
        activities_scores = torch.mean(activities_scores,dim=1).reshape(B,-1)

        return {'activities':activities_scores} # actions_scores, activities_scores


class CrossInferBlock(nn.Module):
    def __init__(self, in_dim, Temporal, Spatial):
        super(CrossInferBlock, self).__init__()
        latent_dim = in_dim//2
        field = Temporal + Spatial

        self.theta = nn.Linear(in_dim, latent_dim, bias = False)
        self.phi = nn.Linear(in_dim, latent_dim, bias = False)
        self.fun_g = nn.Linear(in_dim, latent_dim, bias = False)
        self.W = nn.Linear(latent_dim, in_dim, bias = False)
        self.bn = nn.BatchNorm2d(in_dim)
        # self.embedding = nn.Linear(in_dim, latent_dim, bias = True)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        '''
        :param x: shape [B, T, N, NFB]
        :return:
        '''
        B, T, N, NFB = x.shape
        newx = x.clone()
        for i in range(T):
            for j in range(N):
                x_ij = x[:, i, j, :] # [B, NFB]
                embed_x_ij = self.theta(x_ij).unsqueeze(dim = 2) # [B, NFB//2, 1]

                # Spatial
                spatio_x = x[:,i] # [B, N, NFB]
                g_spatial = self.fun_g(spatio_x)
                phi_spatio_x = self.phi(spatio_x) # [B, N, NFB//2]
                    # Original paper does not use softmax, thus we stick to it
                sweight = torch.bmm(phi_spatio_x, embed_x_ij).squeeze(dim = 2) # [B,N]
                n = len(sweight[0,:])
                spatio_info = torch.einsum('ij,ijk->ik', sweight/n, g_spatial)

                # Temporal
                temporal_x = x[:,:,j]
                g_temporal = self.fun_g(temporal_x)
                embed_temporal_x = self.phi(temporal_x)
                    # Original paper does not use softmax, thus we stick to it
                tweight = torch.bmm(embed_temporal_x, embed_x_ij).squeeze(dim = 2)
                n = len(tweight[0,:])
                temporal_info = torch.einsum('ij,ijk->ik', tweight/n, g_temporal)

                ST_info = (spatio_info + temporal_info)/(T+N)
                res_ST_info = self.W(ST_info) + x_ij
                newx[:,i,j,:] = res_ST_info

        newx = newx.permute(0, 3, 1, 2)
        newx = self.bn(newx)
        newx = newx.permute(0, 2, 3, 1)

        return newx

def MAC2FLOP(macs, params, module_name = ''):
    macs, params = clever_format([macs, params], "%.3f")
    print('{} MACs: {}  #Params: {}'.format(module_name, macs, params))
    if 'M' in macs:
        flops = float(macs.replace('M', '')) * 2
        flops = str(flops/1000) + 'G'
    elif 'G' in macs:
        flops = str(float(macs.replace('G', '')) * 2) + 'G'
    print('{} GFLOPs: {}  #Params: {}'.format(module_name, flops, params))


class HiGCIN_volleyball(nn.Module):
    """
    main module of GCN for the volleyball dataset
    """

    def __init__(self, cfg):
        super(HiGCIN_volleyball, self).__init__()
        self.cfg = cfg

        T, N = self.cfg.num_frames, self.cfg.num_boxes
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn
        NG = self.cfg.num_graph

        if cfg.backbone == 'inv3':
            self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=True)
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=True)
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained=True)
        elif cfg.backbone == 'alex':
            self.backbone = MyAlex(pretrained=True)
        else:
            assert False

        if not cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)
        self.person_avg_pool = nn.AvgPool2d((K**2, 1), stride = 1)
        self.BIM = CrossInferBlock(in_dim = D, Temporal = T, Spatial = K**2)
        self.PIM = CrossInferBlock(in_dim = D, Temporal = T, Spatial = N)
        self.dropout = nn.Dropout()
        self.fc_activities = nn.Linear(D, cfg.num_activities, bias = False)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        # self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)

    def forward(self, batch_data):
        images_in, boxes_in = batch_data

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        N = self.cfg.num_boxes

        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]


        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in_flat = torch.reshape(boxes_in, (B * T * N, 4))  # B*T*N, 4

        boxes_idx = [i * torch.ones(N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * N,))  # B*T*N,

        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build  features
        assert outputs[0].shape[2:4] == torch.Size([OH, OW])
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)

        features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW

        # RoI Align
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features = self.roi_align(features_multiscale,
                                        boxes_in_flat,
                                        boxes_idx_flat)  # B*T*N, D, K, K,
        boxes_features = boxes_features.view(B, T, N, D, K*K)
        boxes_features = boxes_features.permute(0, 2, 1, 4, 3).contiguous()
        boxes_features = boxes_features.view(B*N, T, K*K, D) # B*N, T, K*K, D

        # HiGCIN Inference
        boxes_features = self.BIM(boxes_features) # B*N, T, K*K, D
        boxes_features = self.person_avg_pool(boxes_features) # B*N, T, D
        boxes_features = boxes_features.view(B, N, T, D).contiguous().permute(0, 2, 1, 3) # B, T, N, D
        boxes_states = self.PIM(boxes_features) # B, T, N, D
        boxes_states = self.dropout(boxes_states)
        torch.cuda.empty_cache()

        # Predict actions
        # boxes_states_flat=boxes_states.reshape(-1,NFS)  #B*T*N, NFS
        # actions_scores=self.fc_actions(boxes_states_flat)  #B*T*N, actn_num

        # Predict activities
        boxes_states_pooled, _ = torch.max(boxes_states, dim=2)
        boxes_states_pooled_flat = boxes_states_pooled.reshape(B * T, -1)
        activities_scores = self.fc_activities(boxes_states_pooled_flat)  # B*T, acty_num

        # Temporal fusion
        # actions_scores = actions_scores.reshape(B,T,N,-1)
        # actions_scores = torch.mean(actions_scores,dim=1).reshape(B*N,-1)
        activities_scores = activities_scores.reshape(B, T, -1)
        activities_scores = torch.mean(activities_scores, dim=1).reshape(B, -1)

        return {'activities':activities_scores}