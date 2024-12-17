#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import reduce

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    @staticmethod
    def split_weight_name(name):
        if 'weight' or 'bias' in name:
            return ''.join(name.split('.')[:-1])
        return name

    def save_params(self):
        for param_name, param in self.named_parameters():
            if 'alpha' in param_name or 'beta' in param_name:
                continue
            _buff_param_name = param_name.replace('.', '__')
            self.register_buffer(_buff_param_name, param.data.clone())

    def compute_diff(self):
        diff_mean = dict()
        for param_name, param in self.named_parameters():
            layer_name = self.split_weight_name(param_name)
            _buff_param_name = param_name.replace('.', '__')
            old_param = getattr(self, _buff_param_name, default=0.0)
            diff = (param - old_param) ** 2
            diff = diff.sum()
            total_num = reduce(lambda x, y: x*y, param.shape)
            diff /= total_num
            diff_mean[layer_name] = diff
        return diff_mean

    def remove_grad(self, name=''):
        for param_name, param in self.named_parameters():
            if name in param_name:
                param.requires_grad = False

class network(nn.Module):

    def __init__(self, numclass, feature_extractor):
        super(network, self).__init__()
        self.feature = feature_extractor
        self.fc = nn.Linear(feature_extractor.fc.in_features, numclass, bias=True)

    def forward(self, input):
        _, mid = self.feature(input)
        x = self.fc(mid)
        return x, mid

    def Incremental_learning(self, numclass):
        weight = self.fc.weight.data
        bias = self.fc.bias.data
        in_feature = self.fc.in_features
        out_feature = self.fc.out_features

        self.fc = nn.Linear(in_feature, numclass, bias=True)
        self.fc.weight.data[:out_feature] = weight
        self.fc.bias.data[:out_feature] = bias

    def feature_extractor(self, inputs):
        _, x = self.feature(inputs)
        return x

    def predict(self, fea_input):
        output = self.fc(fea_input)
        return output


class NetModule:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = []
        self.heads = []
        self.decomposed_layers = {}
        self.initializer = nn.init.xavier_uniform_
        self.lid = 0
        self.adaptive_factor = 3
        
        if self.args.model_name == 'LeNet_FashionMNIST':
            self.shapes = [(self.args.num_channels, 6, 5),
                (6,16,5),
                (256,120),
                (120,84)]
        elif 'cifar' in self.args.model_name.lower():
            self.shapes = [
                (3,64,3), (64,128,1),(128,256,1),(256,512,1)
            ]
        elif self.args.model_name == 'TextCNN':
            self.shapes = [
                (768,100)
            ]
        
        self.decomposed_variables = {
            'shared': [],
            'adaptive': {},
            'mask': {},
            'bias': {},
            'atten': {},
            'from_kb': {}
        }

    def build_model(self, decomposed=False, initial_weights=None):
        if self.args.model_name == 'LeNet_FashionMNIST':
            self.model_body = self.build_lenet_body(decomposed, initial_weights)
        elif 'cifar' in self.args.model_name.lower():
            self.model_body = self.build_resnet18_body(decomposed, initial_weights)
        elif self.args.model_name == 'TextCNN':
            self.model_body = self.build_text_body(decomposed, initial_weights)
        
        for tid in range(self.args.task_number):
            self.models.append(self.add_head(self.model_body))

    def build_lenet_body(self, decomposed, initial_weights):
        self.init_decomposed_variables(initial_weights)
    
        class LeNet(nn.Module):
            def __init__(self, decomposed, net_module):
                super(LeNet, self).__init__()
                self.net_module = net_module
                self.layer1 = self.net_module.conv_decomposed(0, 0)
                self.layer2 = nn.ReLU()
                self.layer3 = nn.MaxPool2d(2,2)
                self.layer4 = self.net_module.conv_decomposed(1, 0)    
                self.layer5 = nn.MaxPool2d(2,2)
                self.layer6 = nn.Flatten()
                self.layer7 = self.net_module.dense_decomposed(2, 0)
                self.layer8 = nn.ReLU()
                self.layer9 = self.net_module.dense_decomposed(3, 0)
                self.layer10 = nn.ReLU()

            def forward(self, x):
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                x = self.layer5(x)
                x = self.layer6(x)
                x = self.layer7(x)
                x = self.layer8(x)
                x = self.layer9(x)
                x = self.layer10(x)
                return x

        return LeNet(decomposed, self)

    def build_resnet18_body(self, decomposed, initial_weights):
        self.init_decomposed_variables(initial_weights)

        class BasicBlock(nn.Module):

            def __init__(self, in_channels, out_channels, lid, net_module,stride=1):
                super(BasicBlock, self).__init__()
                self.net_module = net_module
                self.left = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                    # self.net_module.conv_decomposed(lid, 0, stride=stride, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    # self.net_module.conv_decomposed(lid, 0, stride=1, padding=1),
                    nn.BatchNorm2d(out_channels)
                )

                self.shortcut = nn.Sequential()
                if stride != 1 or in_channels != out_channels:
                    self.shortcut = nn.Sequential(
                        # nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                        self.net_module.conv_decomposed(lid, 0, stride=stride, padding=0),
                        nn.BatchNorm2d(out_channels)
                    )
                self.dropout = nn.Dropout(0.5)

            def forward(self, x):
                out = self.left(x)
                out += self.shortcut(x)
                out = F.relu(out)
                out = self.dropout(out)
                return out

        class ResNet(nn.Module):
            def __init__(self, block, net_module):
                super(ResNet, self).__init__()
                self.net_module = net_module
                self.inchannel = 64
                self.conv1 = self.net_module.conv_decomposed(0, 0, stride=1, padding=1)
                self.layer1 = self._make_layer(block, 64,  2, 0, stride=1)
                self.layer2 = self._make_layer(block, 128,  2, 1, stride=2)
                self.layer3 = self._make_layer(block, 256,  2, 2, stride=2)
                self.layer4 = self._make_layer(block, 512,  2, 3, stride=2)
                

            def _make_layer(self, block, channels, blocks, lid, stride=1):

                strides = [stride] + [1] * (blocks - 1)  # strides=[1,1]
                layers = []
                for stride in strides:
                    layers.append(block(self.inchannel, channels, lid, self.net_module, stride))
                    self.inchannel = channels
                return nn.Sequential(*layers)

            def forward(self, x):
                x = self.conv1(x)
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)

                x = F.avg_pool2d(x, 4)
                x = x.view(x.size(0), -1)

                return x

        return ResNet(BasicBlock, self)

    def build_text_body(self, decomposed, initial_weights):
        self.init_decomposed_variables(initial_weights)

        class Model(MyModel):
            def __init__(self, config,net_module):
                super(Model, self).__init__()
                self.net_module = net_module
                if config.embedding_pretrained is not None:
                    self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
                else:
                    self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
                self.convs = nn.ModuleList(
                    [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
                self.dropout = nn.Dropout(config.dropout)
                # self.fc100 = nn.Linear(config.num_filters * len(config.filter_sizes), 100)

                self.fc100 = self.net_module.dense_decomposed(0, 0)


            def conv_and_pool(self, x, conv):
                x = F.relu(conv(x)).squeeze(3)
                x = F.max_pool1d(x, x.size(2)).squeeze(2)  # one-dimension max-pooling
                return x

            def forward(self, x):
                out = self.embedding(x)
                out = out.unsqueeze(1)
                out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
                out = self.dropout(out)
                out = self.fc100(out)
                return out
            
        return Model(self.args, self)
    
    def conv_decomposed(self, lid, tid, stride=1, padding=0):
        return DecomposedConv(
            name='layer_{}'.format(lid),
            in_channels=self.shapes[lid][0],
            out_channels=self.shapes[lid][1],
            kernel_size=self.shapes[lid][2],
            stride=stride,
            padding=padding,
            lambda_l1=self.args.lambda_l1,
            lambda_mask=self.args.lambda_mask,
            shared=self.get_variable('shared', lid),
            adaptive=self.get_variable('adaptive', lid, tid),
            from_kb=self.get_variable('from_kb', lid, tid),
            atten=self.get_variable('atten', lid, tid),
            bias=self.get_variable('bias', lid, tid),
            use_bias=True,
            mask=self.generate_mask(self.get_variable('mask', lid, tid)),
            kernel_regularizer=nn.L1Loss(), args=self.args)

    def dense_decomposed(self, lid, tid):
        return DecomposedDense(
            name='layer_{}'.format(lid),
            in_features=self.shapes[lid][0],
            out_features=self.shapes[lid][1],
            lambda_l1=self.args.lambda_l1,
            lambda_mask=self.args.lambda_mask,
            shared=self.get_variable('shared', lid),
            adaptive=self.get_variable('adaptive', lid, tid),
            from_kb=self.get_variable('from_kb', lid, tid),
            atten=self.get_variable('atten', lid, tid),
            bias=self.get_variable('bias', lid, tid),
            use_bias=True,
            mask=self.generate_mask(self.get_variable('mask', lid, tid)),
            kernel_regularizer=nn.L1Loss(), args=self.args)
    
    def add_head(self, body):
        if self.args.model_name == 'LeNet_FashionMNIST':
            in_features = 84
        elif 'cifar' in self.args.model_name.lower():
            in_features = 512
        elif self.args.model_name == 'TextCNN':
            in_features = 100
        head = nn.Linear(in_features, self.args.num_classes)
        self.heads.append(head)
        class Model(nn.Module):
            def __init__(self, body, head):
                super(Model, self).__init__()
                self.body = body
                self.head = head
            def forward(self, x):
                mid = self.body(x)
                x = self.head(mid)
                return x, mid
        return Model(body, head)

    def get_model_by_tid(self, tid):
        return self.models[tid]

    def init_global_weights(self):
        global_weights = []
        for i in range(len(self.shapes)):
            if len(self.shapes[i]) > 2:
                tensor = torch.empty(self.shapes[i][1],self.shapes[i][0],self.shapes[i][2],self.shapes[i][2])
            elif len(self.shapes[i]) == 2:
                tensor = torch.empty(self.shapes[i][1],self.shapes[i][0])
            global_weights.append(self.initializer(tensor).numpy())
            
        return global_weights
    
    def init_decomposed_variables(self, initial_weights):
        self.decomposed_variables['shared'] = [torch.nn.Parameter(torch.tensor(initial_weights[i], dtype=torch.float32),
                requires_grad=True) for i in range(len(self.shapes))]
        for tid in range(self.args.task_number):
            for lid in range(len(self.shapes)):
                var_types = ['adaptive', 'bias', 'mask', 'atten', 'from_kb']
                for var_type in var_types:
                    self.create_variable(var_type, lid, tid)

    def create_variable(self, var_type, lid, tid=None):
        trainable = True 
        if tid not in self.decomposed_variables[var_type]:
            self.decomposed_variables[var_type][tid] = {}
        if var_type == 'adaptive':
            init_value = self.decomposed_variables['shared'][lid].detach().numpy() / self.adaptive_factor
        # elif var_type == 'atten':
        #     shape = (int(round(self.args.num_users * self.args.frac)),)
        #     if tid == 0:
        #         trainable = False
        #         init_value = np.zeros(shape, dtype=np.float32)
        #     else:
        #         tensor = torch.empty(shape).view(1,-1)
        #         init_value = self.initializer(tensor).detach().numpy()
        elif var_type == 'from_kb':
            if len(self.shapes[lid]) > 2:
                shape = np.concatenate([[int(round(self.args.num_users * self.args.frac))], [self.shapes[lid][1],self.shapes[lid][0],self.shapes[lid][2],self.shapes[lid][2]]], axis=0)
            else:
                shape = np.concatenate([[int(round(self.args.num_users * self.args.frac))], [self.shapes[lid][1],self.shapes[lid][0]]], axis=0)
            shape = tuple(shape)
            trainable = False
            # if tid == 0:
            #     init_value = np.zeros(shape, dtype=np.float32)
            # else:
            #     tensor = torch.empty(shape).view(1,-1)
            #     init_value = self.initializer(tensor).detach().numpy()
            # init_value = np.zeros(shape, dtype=np.float32)
            # tensor = torch.tensor(init_value)
            # init_value = self.initializer(init_value).detach().numpy()
            init_value = np.zeros(shape, dtype=np.float32)
        elif var_type == 'bias':
            init_value = torch.randn(self.shapes[lid][1])
        elif var_type == 'mask' or var_type == 'atten':
            if len(self.shapes[lid]) > 2:
                shape = (self.shapes[lid][-1])
                tensor = torch.empty(shape).view(1,-1)
                init_value = self.initializer(tensor).detach().numpy()
            else:
                tensor = torch.empty((self.shapes[lid][1],self.shapes[lid][0]))
                init_value = self.initializer(tensor).detach().numpy()
        else:
            shape = (self.shapes[lid][-1])
            tensor = torch.empty(shape).view(1,-1)
            init_value = self.initializer(tensor).detach().numpy()
            # init_value = self.initializer((self.shapes[lid][-1],)).detach().numpy()
        
        var = torch.tensor(init_value, requires_grad=trainable, dtype=torch.float32)
        self.decomposed_variables[var_type][tid][lid] = var

    def get_variable(self, var_type, lid, tid=None):
        if var_type == 'shared':
            return self.decomposed_variables[var_type][lid]
        else:
            return self.decomposed_variables[var_type][tid][lid]

    def generate_mask(self, mask):
        mask = torch.tensor(mask)
        return torch.sigmoid(mask)
    
    def get_body_weights(self, task_id=None):
        prev_weights = {}
        for lid in range(len(self.shapes)):
            prev_weights[lid] = {}
            sw = self.get_variable(var_type='shared', lid=lid).detach().numpy()
            for tid in range(task_id+1):
                prev_aw = self.get_variable(var_type='adaptive', lid=lid, tid=tid).detach().numpy()
                prev_mask = self.get_variable(var_type='mask', lid=lid, tid=tid).detach().numpy()
                prev_mask_sig = self.generate_mask(prev_mask).detach().numpy()
                #################################################
                prev_weights[lid][tid] = sw * prev_mask_sig + prev_aw
                #################################################
        return prev_weights
        
    

class DecomposedDense(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 use_bias=False,
                 lambda_l1=None,
                 lambda_mask=None,
                 shared=None,
                 adaptive=None,
                 from_kb=None,
                 atten=None,
                 mask=None,
                 bias=None,
                 args=None,
                 **kwargs):
        super(DecomposedDense, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.args = args
        self.use_bias = use_bias
        
        self.sw = nn.Parameter(shared) if shared is not None else None
        self.aw = nn.Parameter(adaptive) if adaptive is not None else None
        self.mask = nn.Parameter(mask) if mask is not None else None
        self.bias = nn.Parameter(bias) if bias is not None else None
        self.aw_kb = nn.ParameterList([nn.Parameter(kb) for kb in from_kb]) if from_kb is not None else None
        self.atten = nn.Parameter(atten) if atten is not None else None
        
        self.lambda_l1 = lambda_l1
        self.lambda_mask = lambda_mask

    def l1_pruning(self, weights, hyp):
        hard_threshold = torch.gt(torch.abs(weights), hyp).float()
        return weights * hard_threshold
    
    def forward(self, inputs):
        aw = self.aw if self.training else self.l1_pruning(self.aw, self.lambda_l1)
        mask = self.mask if self.training else self.l1_pruning(self.mask, self.lambda_mask)
        atten = self.atten
        aw_kbs = self.aw_kb

        # Decomposed Kernel
        self.my_theta = self.sw * mask + aw + torch.sum(torch.stack([kb * atten for kb in aw_kbs]), dim=0)
        
        layer = nn.Linear(in_features=self.in_features, out_features=self.out_features)
       
        layer.weight = nn.Parameter(self.my_theta)
        layer.bias = nn.Parameter(self.bias)
        layer.to(self.args.device)
        outputs = layer(inputs)
        
        return outputs


class DecomposedConv(nn.Module):
    """ Custom conv layer that decomposes parameters into shared and specific parameters.
    
    Base code is referenced from official tensorflow code (https://github.com/tensorflow/tensorflow/)

    Created by:
        Wonyong Jeong (wyjeong@kaist.ac.kr)
    """
    def __init__(self, 
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 activation=None,
                 use_bias=False,                 
                 lambda_l1=None,
                 lambda_mask=None,
                 shared=None,
                 adaptive=None,
                 from_kb=None,
                 atten=None,
                 mask=None,
                 bias=None,
                 args=None,
                 **kwargs):
        super(DecomposedConv, self).__init__()
        
        self.args = args
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = [1]
        self.padding = [0]
        self.dilation = [1]
        self.groups = [1]
        self.activation = activation
        self.use_bias = use_bias
        self.stride = stride
        self.padding = padding
        
        self.sw = nn.Parameter(shared) if shared is not None else None
        self.aw = nn.Parameter(adaptive) if adaptive is not None else None
        self.mask = nn.Parameter(mask) if mask is not None else None
        self.bias = nn.Parameter(bias) if bias is not False else None
        self.aw_kb = nn.ParameterList([nn.Parameter(kb) for kb in from_kb]) if from_kb is not None else None
        self.atten = nn.Parameter(atten) if atten is not None else None
        
        self.lambda_l1 = lambda_l1
        self.lambda_mask = lambda_mask

    def l1_pruning(self, weights, hyp):
        hard_threshold = torch.gt(torch.abs(weights), hyp).float()
        return weights * hard_threshold
    
    def forward(self, inputs):
        aw = self.aw if self.training else self.l1_pruning(self.aw, self.lambda_l1)
        mask = self.mask if self.training else self.l1_pruning(self.mask, self.lambda_mask)
        atten = self.atten
        aw_kbs = self.aw_kb

        # Decomposed Kernel
        self.my_theta = self.sw * mask + aw + torch.sum(torch.stack([kb * atten for kb in aw_kbs]), dim=0)
      
        
        layer = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, stride=self.stride, padding=self.padding, kernel_size=self.kernel_size)
       
        layer.weight = nn.Parameter(self.my_theta)
        layer.bias = nn.Parameter(self.bias)
        layer.to(self.args.device)
        outputs = layer(inputs)
        
        
        return outputs
