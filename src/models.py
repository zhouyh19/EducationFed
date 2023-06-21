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



class TemporalModule(nn.Module):
    def __init__(self,s,t,feature_len,attn_len):
        super(TemporalModule, self).__init__()
        self.s=s
        self.t=t
        self.pos_len=128
        self.feature_len=feature_len
        self.attn_len=attn_len
        self.a_mat=nn.Linear(feature_len,attn_len)
        self.b_mat=nn.Linear(feature_len,attn_len)
        self.g=nn.Linear(feature_len,feature_len)
        self.h=nn.Linear(2,self.pos_len)
        #self.W=nn.Linear(feature_len,feature_len)
        self.softmax=nn.Softmax(dim=-1)
    
    def forward(self, batch_data,xywh):
        batch_data=batch_data.reshape((self.t,self.s,-1))
        xywh=xywh.reshape((self.t,self.s,-1))
        avg_pos=torch.mean(xywh,dim=0)
        rel_pos=xywh[:,:,:2]-avg_pos[:,:2]
        pos_embeds=self.h(rel_pos)
        #batch_data=torch.concat((batch_data,pos_embeds),dim=-1)

        theta=self.a_mat(batch_data).reshape((self.t,self.s,-1))
        phi=self.b_mat(batch_data).reshape((self.t,self.s,-1))
        feats=self.g(batch_data).reshape((self.t,self.s,-1))
        results=torch.zeros((self.t,self.s,self.feature_len)).to(batch_data.device)
        
        for i in range(self.s):
            theta_=theta[:,i]
            phi_=torch.transpose(phi[:,i],0,1)
            
            batch_result=torch.mm(theta_,phi_)
            #batch_result=self.softmax(batch_result)
            #print(theta.shape,theta_.shape,phi_.shape,batch_result.shape)
            for j in range(self.t):
                results[j][i]+=torch.mm(batch_result[j].reshape((1,self.t)),feats[:,i]).reshape(-1)

        #results/=self.t
        #results=self.W(results)
        return results

class SpatioVisualModule(nn.Module):
    def __init__(self,s,feature_len,attn_len):
        super(SpatioVisualModule, self).__init__()
        self.s=s
        self.feature_len=feature_len
        self.attn_len=attn_len
        self.a_mat=nn.Linear(feature_len,attn_len)
        self.b_mat=nn.Linear(feature_len,attn_len)
        self.g=nn.Linear(feature_len,attn_len)
        #self.W=nn.Linear(feature_len,feature_len)
        self.softmax=nn.Softmax(dim=-1)
    
    def forward(self, batch_data):
        theta=self.a_mat(batch_data).reshape((self.s,-1))
        phi=self.b_mat(batch_data).reshape((self.s,-1))
        feats=self.g(batch_data).reshape((self.s,-1))
        results=torch.zeros((self.s,self.attn_len)).to(batch_data.device)
        

        phi=torch.transpose(phi,0,1)
        batch_weights=torch.mm(theta,phi)
        batch_weights=self.softmax(batch_weights)
        #print(theta.shape,theta_.shape,phi_.shape,batch_result.shape)
        for j in range(self.s):
            results[j]+=torch.mm(batch_weights[j].reshape((1,self.s)),feats).reshape(-1)

        
        #results=self.W(results)
        return results,batch_weights

class SpatioPositionalModule(nn.Module):
    def __init__(self,s,feature_len,attn_len):
        super(SpatioPositionalModule, self).__init__()
        self.s=s
        self.feature_len=feature_len
        self.attn_len=attn_len
        self.highlight_fc=nn.Linear(feature_len,2)
        self.gcn_fc=nn.Linear(feature_len,attn_len)

    def forward(self,batch_data,xywh,OW,OH,actor_weights,avg_pos,num_person):

        #print(xywh[:num_person])
        #print(OW,OH)
        
        highlight_pred=self.highlight_fc(batch_data)
        
        highlight_pred=torch.tanh(highlight_pred)

        sigma_x,sigma_y=4.0,1.0
        alpha=0.25

        highlight_pred_x=highlight_pred[:,0]*xywh[:,2]*sigma_x+xywh[:,0]
        highlight_pred_y=highlight_pred[:,1]*xywh[:,3]*sigma_y+xywh[:,1]

        '''highlight_pred_x=highlight_pred[:,0]*OW+avg_pos[:,0]
        highlight_pred_y=highlight_pred[:,1]*OH+avg_pos[:,1]'''
        
        '''for i in range(num_person):
            offset_x=xywh[:,0]-highlight_pred_x[i]
            offset_y=xywh[:,1]-highlight_pred_y[i]
            offset_x=actor_weights[i]*offset_x
            offset_y=actor_weights[i]*offset_y

            offset_x=torch.sum(offset_x)
            offset_y=torch.sum(offset_y)

            highlight_pred_x[i]+=alpha*offset_x
            highlight_pred_y[i]+=alpha*offset_y'''


        
        '''x_avg,y_avg=torch.mean(highlight_pred_x,dim=-1),torch.mean(highlight_pred_y,dim=-1)
        x_avg,y_avg=x_avg.unsqueeze(-1),y_avg.unsqueeze(-1)
        x_div,y_div=highlight_pred_x-x_avg,highlight_pred_y-y_avg
        dis_div=x_div**2+y_div**2
        agr_loss=torch.sum(dis_div)'''

        weights=torch.zeros((self.s,self.s)).to(batch_data.device)
        targets=self.gcn_fc(batch_data)
        targets=F.relu(targets, inplace = True)

        ent_loss=0

        for i in range(num_person):

            if torch.sum(xywh[i])<1e-8:
                    continue

            for j in range(num_person):
                
                if torch.sum(xywh[j])<1e-8:
                    continue

                dis=torch.sqrt((highlight_pred_x[i]-xywh[j][0])**2+(highlight_pred_y[i]-xywh[j][1])**2) 
                weights[i][j]=1.0/(dis+1e-4)

            weights[i]/=torch.sum(weights[i])#torch.softmax(weights[i],-1)
        
        #ent_loss+=calc_entropy(weights)

    
        targets_weighted=torch.mm(weights,targets)
 
        return targets_weighted,None,highlight_pred

class SemPosBlock(nn.Module):
    def __init__(self,s,t,feature_len,person_feature_len):
        super(SemPosBlock, self).__init__()
        self.s=s
        self.t=t
        self.feature_len=feature_len
        
       
        self.posembed_len=1024
        self.person_feature_len=person_feature_len
        self.posembed_fc=nn.Linear(4,self.posembed_len)
        #self.temp=torch.nn.GRU(self.person_feature_len,self.person_feature_len)
        #self.temp=
        #self.person_fc=nn.Linear(self.t*(self.feature_len),self.person_feature_len)

        self.SpVis=SpatioVisualModule(self.s,self.person_feature_len,self.person_feature_len)
        self.SpPos=SpatioPositionalModule(self.s,self.person_feature_len,self.person_feature_len)

    
    def forward(self,batch_data,xywh,OW,OH,num_person):
        
        #batch_data=batch_data+self.temp(batch_data)
        xywh=xywh.reshape((self.t,self.s,-1))
        num_person=num_person.reshape((-1))

        avg_pos=torch.mean(xywh,dim=0)

        #pos_embeds=self.posembed_fc(xywh)
        #batch_with_pos=torch.concat((batch_data,pos_embeds),dim=-1)
        #person_tmp_features,_=self.temp(batch_data_with_pos,torch.zeros((self.s,self.person_feature_len)).to(batch_data.device))
        
        

        #person_features=batch_data.reshape((self.s,self.person_feature_len))
        person_features=batch_data.reshape((self.t,self.s,self.person_feature_len))

        results=[]
        
        for i in range(self.t):

            vis_result,actor_weights=self.SpVis(person_features[i])

            #pos_result,agr_loss,offset=self.SpPos(person_features[i],xywh[i],OW,OH,actor_weights,avg_pos,num_person[i])
            #有时会导致梯度爆炸，仍需检查
            
            result=vis_result#+pos_result
            #result=#torch.concat((pos_result,vis_result),dim=-1) 
            results.append(result.unsqueeze(0))

        results=torch.concat(results,dim=0)
        
        #result=vis_result+person_features#+pos_result
        #result=torch.concat((vis_result,pos_result),dim=-1)
        
        #result=(vis_result,pos_result,person_features)
        #result=person_features
        agr_loss=None

        return results,agr_loss



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

class Dynamic_volleyball(nn.Module):
    """
    DIN model
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
            self.DPI = Dynamic_Person_Inference(
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
                cfg = cfg)
            '''self.DPI = Multi_Dynamic_Inference(
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
                cfg = cfg)'''
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
        images_in, boxes_in,_,_,_ = batch_data
        
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
        graph_boxes_features = self.DPI(boxes_features)
        #print(graph_boxes_features.shape,boxes_features.shape)
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
        images_in, boxes_in,_,_,_ = batch_data

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

class Dynamic_education(nn.Module):
    """
    main module of GCN for the volleyball dataset
    """
    def __init__(self, cfg):
        super(Dynamic_education, self).__init__()
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

        
        #self.dpi_nl = nn.LayerNorm([T, N, in_dim*2])
        self.person_feature_len=1024
        self.dpi_nl = nn.LayerNorm([N, self.person_feature_len])
        self.dpi_nl2 = nn.LayerNorm([T, N, in_dim])
        self.dropout_global = nn.Dropout(p=self.cfg.train_dropout_prob)

        self.temp=TemporalModule(N,T,in_dim,in_dim)
        self.sp=SemPosBlock(N,T,in_dim,self.person_feature_len)


        NFG=self.person_feature_len

        # Lite Dynamic inference
        if self.cfg.lite_dim:
            self.point_conv = nn.Conv2d(NFB, in_dim, kernel_size = 1, stride = 1)
            self.point_ln = nn.LayerNorm([T, N, in_dim])
            self.fc_activities = nn.Linear(in_dim, self.cfg.num_activities)
        else:
            self.fc_activities=nn.Linear(NFG, self.cfg.num_activities)

        self.fc_actions = nn.Linear(NFG, self.cfg.num_actions)

        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        if not cfg.train_temp:
            for p in self.backbone.parameters():
                p.requires_grad=False
            
            for p in self.person_fc.parameters():
                p.requires_grad=False
            
            for p in self.temp.parameters():
                p.requires_grad=False
            
            for p in self.person_fc.parameters():
                p.requires_grad=False

        self.FirstNan=False
    
    def savemodel(self,filepath):
        state = {
            'backbone_state_dict': self.backbone.state_dict(),
            'fc_emb_state_dict':self.fc_emb_1.state_dict(),
            'temp_state_dict':self.temp.state_dict(),
            #'person_fc_state_dict':self.person_fc.state_dict()
        }
        
        torch.save(state, filepath)
        print('base_temp model saved to:',filepath)
                    
    def loadmodel(self,filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)
    
    def loadmodel_stage2(self,filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        self.temp.load_state_dict(state['temp_state_dict'])
        self.person_fc.load_state_dict(state['person_fc_state_dict'])
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
        images_in, boxes_in, bboxes_num_in,xywh = batch_data

        #print('images',True in torch.isnan(images_in))
        
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
        '''print('outputs',True in torch.isnan(outputs[0]))
        for layer in self.backbone.state_dict():
            print("backbone",layer,True in torch.isnan(self.backbone.state_dict()[layer]))'''
        #print(outputs[0].shape)

        # Build  features
        assert outputs[0].shape[2:4]==torch.Size([OH,OW])
        features_multiscale=[]
        for features in outputs:
            if features.shape[2:4]!=torch.Size([OH,OW]):
                features=F.interpolate(features,size=(OH,OW),mode='bilinear',align_corners=True)
            features_multiscale.append(features)
        
        features_multiscale=torch.cat(features_multiscale,dim=1)  #B*T, D, OH, OW
        #print('features_multiscale',True in torch.isnan(features_multiscale))
        #print(features_multiscale.shape)
        
        # RoI Align
        boxes_in_flat.requires_grad=False
        boxes_idx_flat.requires_grad=False
        boxes_features=self.roi_align(features_multiscale,
                                            boxes_in_flat,
                                            boxes_idx_flat)  #B*T*N, D, K, K,
        boxes_features=boxes_features.reshape(B,T,N,-1)  #B,T,N, D*K*K
        
        #print(boxes_features.shape)

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

        if True in torch.isnan(boxes_features) and not self.FirstNan:
            print('vis',True in torch.isnan(boxes_features))
            self.FirstNan=True

        graph_boxes_features=self.temp(boxes_features,xywh)+boxes_features
        if True in torch.isnan(graph_boxes_features) and not self.FirstNan:
            print('temp',True in torch.isnan(graph_boxes_features))
            self.FirstNan=True

        graph_boxes_features = graph_boxes_features.reshape(B, T, N, -1)
        #graph_boxes_features = graph_boxes_features.reshape(B, N, -1)
        graph_boxes_features = self.dpi_nl(graph_boxes_features)
        graph_boxes_features = F.relu(graph_boxes_features, inplace=True)


        tmp_boxes_features,_ = self.sp(graph_boxes_features,xywh,OW,OH,bboxes_num_in)
        graph_boxes_features = tmp_boxes_features+graph_boxes_features#(graph_boxes_features,tmp_mat,OW,OH)

        if True in torch.isnan(graph_boxes_features) and not self.FirstNan:
            print('full',True in torch.isnan(graph_boxes_features))
            self.FirstNan=True

        graph_boxes_features = graph_boxes_features.reshape(B, T, N, -1)
        #graph_boxes_features = graph_boxes_features.reshape(B, N, -1)
        graph_boxes_features = self.dpi_nl2(graph_boxes_features)
        graph_boxes_features = F.relu(graph_boxes_features, inplace=True)

        #print(graph_boxes_features.shape)
        #results,agr_loss = self.hb(person_features,tmp_mat,OW,OH)
        #graph_boxes_features=person_features
        #print(graph_boxes_features.shape)
        torch.cuda.empty_cache()
        boxes_states = self.dropout_global(graph_boxes_features)

        
    
        boxes_states_pooled, _ = torch.max(boxes_states,dim=2)
        boxes_states_pooled_flat = boxes_states_pooled.reshape(B*T, -1)
        #boxes_states_pooled_flat = boxes_states_pooled.reshape(B, -1)
        

        
        #print(boxes_states.shape)
        # Predict actions
        #boxes_states_flat=boxes_states.reshape(-1,self.cfg.num_features_gcn)  #B*T*N, NFS
        #boxes_states_flat=boxes_states.reshape(-1,self.person_feature_len*2)
        #print(boxes_states.shape)
        #actions_scores=self.fc_actions(boxes_states_flat)  #B*T*N, actn_num
        #print(actions_scores.shape)
        
        # Predict activities
        #boxes_states_pooled, _ = torch.max(boxes_states,dim=2)
        
        activities_scores = self.fc_activities(boxes_states_pooled_flat)  #B*T, acty_num


        
        # Temporal fusion
        activities_scores = activities_scores.reshape(B, T, -1)
        activities_scores = torch.mean(activities_scores,dim=1).reshape(B,-1)

        return {'activities':activities_scores} # actions_scores, activities_scores 