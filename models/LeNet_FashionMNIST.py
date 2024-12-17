# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.MyModel import MyModel
from dataset import data_process

class Config(object):

    def __init__(self, dataset, embedding, scenario):
        self.model_name = 'LeNet_FashionMNIST'
        self.scenario = scenario
        self.task_number = 5
        self.num_users = 50
        if self.scenario == 'class':
            task_class_length, train_tasks, dev_tasks, user_data, user_task, proxy_dict = data_process.process_fashionmnist(self.num_users, self.task_number, 1.0)
        elif self.scenario == 'domain':
            task_class_length, train_tasks, dev_tasks, user_data, user_task = data_process.process_fashionmnist_domain(
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
        self.save_path = 'saved_dict/' + self.model_name + '.ckpt'        # save path of trained model
        self.log_path = 'log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load('dataset/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None                                       # pre-trained word vector
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # GPU

        self.dropout = 0.5
        self.require_improvement = 1000
        self.num_classes = len(self.class_list)
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
        self.learning_rate = 0.05
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300
        self.filter_sizes = (2, 3, 4)
        self.num_filters = 256

class Model(MyModel):
    def __init__(self, config):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(config.num_channels, 6, 5)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(config.num_filters, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc = nn.Linear(84, config.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        mid_val = F.relu(self.fc2(x))
        output = self.fc(mid_val)
        return output, mid_val