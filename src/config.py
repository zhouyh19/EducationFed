import time
import os


class Config(object):
    """
    class to save config parameter
    """

    def __init__(self):
        
        # Dataset
        self.dataset_name='education'
        self.data_path='../processed_new'
        
        # Backbone 
        self.crop_size = 5, 5  #crop size of roi align
        
        # Activity Action
        self.actions_loss_weight = 1.0  #weight used to balance action loss and activity loss
        self.actions_weights = None

        # Sample
        self.num_before = 5
        self.num_after = 4

        # ARG params
        self.num_features_boxes = 1024
        self.num_features_relation=256
        self.num_features_gcn=self.num_features_boxes
        self.gcn_layers=1  #number of GCN layers
        self.pos_threshold=0.2  #distance mask threshold in position relation

        # Training Parameters
        self.train_random_seed = 0
        
        # Exp
        self.stage1_model_path=''  #path of the base model, need to be set in stage2
        self.test_before_train=False
        self.exp_note='Group-Activity-Recognition'
        self.exp_name=None
        self.set_bn_eval = False
        self.inference_module_name = 'dynamic_volleyball'

        # Dynamic Inference
        self.num_DIM = 1
        self.stage2model = None

        # Actor Transformer
        self.temporal_pooled_first = False

        # SACRF + BiUTE
        self.halting_penalty = 0.0001

        self.inference_module_name = 'dynamic_collective'

        self.device_list="0"
        self.training_stage=2
        self.use_gpu = True
        self.use_multi_gpu = False
        self.train_backbone = True

        # ResNet18
        '''
        self.backbone = 'res18'
        self.image_size = 480, 720
        self.out_size = 15, 23
        self.emb_features = 512
        self.stage1_model_path = 'result/basemodel_CAD_res18.pth'
        '''

        self.load_backbone_stage2=False
        self.load_stage2model=False

        # VGG16
        '''
        self.backbone = 'vgg16'
        self.image_size = 480, 720
        self.out_size = 15, 22
        self.emb_features = 512
        self.stage1_model_path = 'result/basemodel_CAD_vgg16.pth'
        '''


        
        self.backbone = 'vgg19'
        self.image_size = 480, 720
        self.out_size = 15, 22
        self.emb_features = 512

        self.num_boxes = 40
        self.num_actions = 5
        self.num_activities = 5
        self.num_frames = 10
        self.num_graph = 4
        self.tau_sqrt=True
        self.batch_size = 2
        self.test_batch_size = 8
        self.test_interval_epoch = 1
        self.train_learning_rate = 5e-5
        self.momentum = 0.9
        self.train_dropout_prob = 0.5
        self.weight_decay = 1e-4
        self.lr_plan = {}
        self.max_epoch = 50


        # Dynamic Inference setup
        self.group = 1
        self.stride = 1
        self.ST_kernel_size = (3, 3)
        self.dynamic_sampling = True
        self.sampling_ratio = [1]  # [1,2,4]
        self.lite_dim = None # 128
        self.scale_factor = True
        self.beta_factor = False
        self.hierarchical_inference = False
        self.parallel_inference = False

        self.init_type = "xavier"
        self.init_gain= 1.0
        self.gpu_ids= [0]

        self.exp_note='Dynamic_collective'
        self.init_config()
        
        
    def init_config(self, need_new_folder=True):
        if self.exp_name is None:
            time_str=time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            self.exp_name='[%s_stage%d]<%s>'%(self.exp_note,self.training_stage,time_str)
            
        self.result_path='result/%s'%self.exp_name
        self.log_path='result/%s/log.txt'%self.exp_name
            
        if need_new_folder:
            if not os.path.exists(self.result_path):
                os.makedirs(self.result_path)
