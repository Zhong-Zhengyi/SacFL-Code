# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.MyModel import MyModel
from dataset import data_process


class Config(object):

    def __init__(self, dataset, embedding, scenario):
        self.model_name = 'CNN_Cifar10'
        self.scenario = scenario
        self.train_path = dataset + '/data/train_012.csv'                               
        self.dev_path = dataset + '/data/eval_012.csv'                                  
        self.test_path = dataset + '/data/test_012.csv'                                
        self.task_number = 3
        self.num_users = 50
        if self.scenario == 'class':
            task_class_length, train_tasks, dev_tasks, user_data, user_task, proxy_dict = data_process.process_cifar10(
                self.num_users, self.task_number, 1)
        elif self.scenario == 'domain':
            task_class_length, train_tasks, dev_tasks, user_data, user_task = data_process.process_cifar10_domain(
                self.num_users, self.task_number)
        self.task_class_length = task_class_length
        self.train_tasks = train_tasks
        self.dev_tasks = dev_tasks
        self.test_tasks = dev_tasks
        self.user_data = user_data
        self.user_task = user_task
        if self.scenario == 'class':
            self.proxy_dict = proxy_dict

        self.class_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        self.vocab_path = 'dataset/vocab.pkl'
        self.save_path = 'saved_dict/' + self.model_name + '.ckpt'       
        self.log_path = 'log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load('dataset/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None                                     
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

        self.dropout = 0.5                                            
        self.require_improvement = 1000                                
        self.num_classes = len(self.class_list)                         
        # self.num_classes = 6
        self.n_vocab = 0                                                                              
        self.batch_size = 128                                        
        self.schedule_gamma = 0.9
        self.more_loss = True

        self.frac = 1
        self.local_ep = 5
        self.local_bs = 32
        self.iid = True
        self.server_distillation = False

        self.verbose = 1
        self.pad_size = 32                                           
        self.learning_rate = 0.01    #1e-3                                    
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300          
        self.filter_sizes = (2, 3, 4)                                   
        self.num_filters = 256                                       


    
class ResNet18Feature(nn.Module):
    def __init__(self, original_model):
        super(ResNet18Feature, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.fc = original_model.fc
        

    def forward(self, x):
        feature_output = self.features(x)
        mid = feature_output.view(x.size(0), -1)
        out = self.fc(mid)
        return out, mid
