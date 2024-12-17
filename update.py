#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
from loss import DistillationLoss
from elastic_weight_consolidation import ElasticWeightConsolidation
import copy
from importlib import import_module
import pickle as pkl
from tqdm import tqdm
import math
import os

from utils import init_network, draw_model_heatmap, add_trigger


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        # # for image
        # return torch.tensor(image, dtype=torch.float), torch.tensor(label, dtype=torch.long)
        # # for text
        return torch.tensor(image), torch.tensor(label)


class MyDataSet(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, label = self.dataset[item]
        # # for image
        # return torch.tensor(image, dtype=torch.float), torch.tensor(label, dtype=torch.long)
        # # for text
        return torch.tensor(image), torch.tensor(label)


class GlobalDataSetSplit(Dataset):
    def __init__(self, dataset, idxs, local_models, device):
        self.device = device
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        self.local_models = local_models
        for i in self.local_models:
            i.eval()
        self.model_idx = 0

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        temp_image, label = torch.tensor(image).to(self.device), torch.tensor(label).to(self.device)
        temp_image = temp_image.view(1, temp_image.shape[0])
        soft_label = self.local_models[self.model_idx](temp_image)
        self.model_idx = (self.model_idx + 1) % len(self.local_models)
        return torch.tensor(image), soft_label.view(soft_label.shape[1])


class LocalUpdate(object):
    def __init__(self, args, idx,train_data, test_data, user_task, user_data, logger, current_task, lr):
        self.train_data = train_data
        self.test_data = test_data
        self.user_task = user_task
        self.user_data = user_data
        self.idx = idx
        self.args = args
        self.logger = logger
        self.current_task = current_task
        self.device = args.device
        self.criterion = F.cross_entropy
        self.lr = lr
        self.trainloader, self.validloader, self.testloader, self.user_data_test_task = self.train_val_test()
        self.gradients = []
        
    def train_val_test(self):
        task_length = int(self.args.num_classes / self.args.task_number)
        user_train_data = []
        user_data_test_task = []
        test_data = []
        if self.args.scenario == 'class':
            for i, j in enumerate(self.user_data[0]):
                if (self.args.attack_type == 'label_flipping') and (self.idx in self.args.atttack_client_id) and (self.current_task == self.args.attack_task_id):
                    for k in self.user_task[self.args.attacked_task_id]:
                        train_data = list(self.train_data[k][self.user_data[k][i]])
                        # train_data[1] = self.args.num_classes-1-train_data[1] #+ task_length*(self.args.attack_task_id-self.args.attacked_task_id)
                        train_data[1] = random.randint(0, self.args.num_classes-1)
                        user_train_data.append(tuple(train_data))
                else:
                    for k in self.user_task[self.current_task]:
                        user_train_data.append(self.train_data[k][self.user_data[k][i]])
            random.shuffle(user_train_data)
            for j in range(self.current_task+1):
                user_data_test_task.append(self.user_task[j])
            for tid, i in enumerate(user_data_test_task):
                # if (self.args.attack_type == 'label_flipping') and (self.idx in self.args.atttack_client_id) and (self.current_task == self.args.attack_task_id):
                #     if tid != self.args.attack_task_id:
                #         for cls in i:
                #             test_data.extend(self.test_data[cls])
                #     elif tid == self.args.attacked_task_id:
                #         i = self.user_task[self.args.attacked_task_id]
                #         for cls in i:
                #             data = self.test_data[cls]
                #             #对data进行标签翻转处理
                #             for j in range(len(data)):
                #                 data[j][1] = self.args.num_classes-data[j][1]#+task_length*(self.args.attack_task_id-self.args.attacked_task_id)
                #             test_data.extend(self.test_data[cls])
                # else:
                for cls in i:
                    test_data.extend(self.test_data[cls])

        elif self.args.scenario == 'domain':
            for i, j in enumerate(self.user_data[0]):
                for k in self.user_task[self.current_task]:
                    user_train_data.append(self.train_data[self.current_task*10+k][self.user_data[k][i]])
            random.shuffle(user_train_data)
            for j in range(self.current_task+1):
                for i in range(self.args.num_classes):
                    test_data.extend(self.test_data[j*10+i])

        trainloader = DataLoader(user_train_data, batch_size=self.args.local_bs, shuffle=False, drop_last=True)
        # idxs_test -> idxs_val
        if self.args.model_name == 'TextCNN':
            batch_size = 1000
        elif self.args.model_name == 'CNN_Cifar10':
            batch_size = 1000
        elif self.args.model_name == 'CNN_Cifar100':
            batch_size = 100
        elif self.args.model_name == 'LeNet_FashionMNIST':
            batch_size = 1000
        validloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)
        testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)
        return trainloader, validloader, testloader, user_data_test_task

    def detect_shift(self, model, datasets, idx, global_round, change_dict, last_loss, task_shift):
        # datasets_loader = DataLoader(datasets, batch_size=16, shuffle=False)
        datasets_loader = datasets
        last_task_global_model = copy.deepcopy(model)
        current_dict = model.state_dict()
        backbone_dict = torch.load('./save_model/global_model_{}.pth'.format(self.args.model))
        pretrained_dict = torch.load('./save_model/Client_{}_{}.pth'.format(idx, self.args.model))

        if task_shift == True:
            for index, (name, param) in enumerate(pretrained_dict.items()):
                if index == len(pretrained_dict)-2:
                    weight_shape = pretrained_dict[name].shape
                    current_dict[name][:weight_shape[0], :] = pretrained_dict[name]
                elif index == len(pretrained_dict)-1:
                    bias_shape = pretrained_dict[name].shape
                    current_dict[name][:bias_shape[0]] = pretrained_dict[name]
                else:
                    current_dict[name] = backbone_dict[name]
        last_task_global_model.load_state_dict(current_dict)

        loss = 0
        for batch_idx, (x, target) in enumerate(datasets_loader):
            if batch_idx <= 9:
                x, target = x.to(self.device), target.to(self.device)

                x.requires_grad = False
                target.requires_grad = False

                if 'cifar' in self.args.model_name.lower():
                    x = x.to(torch.float32)
                    sub_model = torch.nn.Sequential(*list(model.children())[:-1])
                    pro1 = sub_model(x)
                    sub_last_task_global_model = torch.nn.Sequential(*list(last_task_global_model.children())[:-1])
                    pro2 = sub_last_task_global_model(x)
                else:
                    _, pro1 = model(x)
                    _, pro2 = last_task_global_model(x)
                pdist = nn.PairwiseDistance(p=1)
                diff = pdist(pro1, pro2)
                gap = torch.norm(diff)
                loss += gap
        # print('loss. vs. thresh:', "{:.2e}".format(loss))
        # thresh = maxval
        if 'MNIST' in self.args.model:
            thresh = 500
        elif self.args.model == 'CNN_Cifar10':
            thresh = 100
        elif self.args.model == 'CNN_Cifar100':
            thresh = 100
        elif 'Text' in self.args.model:
            thresh = 1000
        print('idx_{}_epoch_{}_thresh_{}'.format(idx, global_round,loss))
        if loss > thresh:
            shift_or_not = True
            torch.save(backbone_dict, './save_model/global_{}_history_model_{}.pth'.format(global_round-1, self.args.model))
            file_path = './save_model/Epoch_{}_Client_{}_head_{}.pth'.format(global_round - 1, idx, self.args.model)
            torch.save(dict(list(pretrained_dict.items())[-2:]), file_path)
            change_dict[idx].append(global_round)
        else:
            shift_or_not = False
        return loss, shift_or_not, change_dict

    def regulization_loss(self, model, datasets, idx, global_round, change_dict, last_loss, task_shift):
        # datasets_loader = DataLoader(datasets, batch_size=16, shuffle=False)
        datasets_loader = datasets
        last_task_global_model = copy.deepcopy(model)
        current_dict = model.state_dict()
        backbone_dict = torch.load('./save_model/global_{}_history_model_{}.pth'.format(self.current_task*self.args.num_epochs-1, self.args.model), map_location='cuda')
        pretrained_dict = torch.load('./save_model/Client_{}_{}.pth'.format(idx, self.args.model))

        if task_shift == True:
            for index, (name, param) in enumerate(pretrained_dict.items()):
                if index == len(pretrained_dict)-2:
                    weight_shape = pretrained_dict[name].shape
                    current_dict[name][:weight_shape[0], :] = pretrained_dict[name]
                elif index == len(pretrained_dict)-1:
                    bias_shape = pretrained_dict[name].shape
                    current_dict[name][:bias_shape[0]] = pretrained_dict[name]
                else:
                    current_dict[name] = backbone_dict[name]
        last_task_global_model.load_state_dict(current_dict)

        loss = 0
        for batch_idx, (x, target) in enumerate(datasets_loader):
            if batch_idx <= 9:
                x, target = x.to(self.device), target.to(self.device)

                x.requires_grad = False
                target.requires_grad = False

                if 'cifar' in self.args.model_name.lower():
                    x = x.to(torch.float32)
                    sub_model = torch.nn.Sequential(*list(model.children())[:-1])
                    pro1 = sub_model(x)
                    sub_last_task_global_model = torch.nn.Sequential(*list(last_task_global_model.children())[:-1])
                    pro2 = sub_last_task_global_model(x)
                else:
                    _, pro1 = model(x)
                    _, pro2 = last_task_global_model(x)
                # pdist = nn.PairwiseDistance(p=1)
                pro1 = F.log_softmax(pro1, dim=1)  # 对 pro1 应用 log_softmax
                pro2 = F.softmax(pro2, dim=1)      # 对 pro2 应用 softmax
                loss += F.kl_div(pro1, pro2, reduction='batchmean')  # 计算 KL 散度

                # diff = pdist(pro1, pro2)
                # gap = torch.norm(diff)
                # loss += gap
        
        return loss

    def update_weights_sacfl(self, model, global_round, idx, change_dict, mu, shift_or_not, two_epoch_loss=None, task_shift=False):
        # Set mode to train model
        model.train()

        if self.args.paradigm.lower() == 'ccfed':
            if self.current_task > 0:
                length = len(list(model.parameters()))
                for index, p in enumerate(model.modules()):
                    if index < length - 2:
                        p.requires_grad_ = False
        epoch_loss = []
        epoch_acc = []

        sum_time = 0

        print('shift_or_not_{}'.format(shift_or_not))

        for iter in range(self.args.local_ep):
            total = 0
            correct = 0
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                if (self.args.attack_type == 'backdoor') and (self.current_task == self.args.attack_task_id) and (idx in self.args.atttack_client_id):
                    if np.random.rand() < 0.5:
                        images = add_trigger(images)
                        labels[:] = 8  # 攻击者指定目标标签      

                model.zero_grad()
                if 'cifar' in self.args.model_name.lower():
                    images = images.to(torch.float32)
                out, pro1 = model(images)
                
                optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
               
                loss = self.criterion(out, labels)
                
                
                if (self.args.defense == 'sacfl') and (idx in self.args.atttack_client_id) and (self.current_task == self.args.attack_task_id) and (iter>0):
                    loss_gap= self.regulization_loss(model, self.trainloader, idx, global_round, change_dict, two_epoch_loss[idx], task_shift)
                    loss =+ loss_gap* 0.
                # with torch.autograd.set_detect_anomaly(True): 
                #     loss.backward()
                #     optimizer.step()
                loss.backward()               
                optimizer.step()
                # Prediction
                _, pred_labels = torch.max(out, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)

                if self.args.verbose and (batch_idx % 100 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{}]\tLoss: {:.6f}'.format(global_round, iter, batch_idx * len(images), len(self.trainloader.dataset), loss.item()))
                self.logger.add_scalar('loss', loss.item())

                batch_loss.append(loss.item())

            epoch_acc.append(correct/total)
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
           
            std_time = time.time()
            if iter == self.args.local_ep-1:
                torch.save(model.state_dict(), './save_model/Client_{}_{}.pth'.format(idx, self.args.model))
                torch.save(dict(list(model.state_dict().items())[-2:]), './save_model/Client_{}_head_{}.pth'.format(idx, self.args.model))
            if iter == 0:
                if global_round > 0:
                    gap, shift_or_not, change_dict = self.detect_shift(model, self.trainloader, idx, global_round, change_dict, two_epoch_loss[idx], task_shift)
                    gap_ = gap.item()
                    two_epoch_loss[idx] = abs(gap_)
                else:
                    gap = torch.tensor(0)
                    gap = gap.to(self.device)
                    shift_or_not = False
            end_time = time.time()
            sum_time += end_time - std_time

        print('global_round_{}_user_id_{}_train_acc_{}'.format(global_round, idx, epoch_acc[-1]))

        return model.state_dict(), sum(epoch_loss)/len(epoch_loss), change_dict, gap, shift_or_not, two_epoch_loss, sum_time

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        model.train()
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                if 'cifar' in self.args.model_name.lower():
                    images = images.to(torch.float32)
                    log_prob,_ = model(images)
                else:
                    log_prob, _ = model(images)
                loss = self.criterion(log_prob, labels)

                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 100 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(global_round, iter, batch_idx * len(images), len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model, idx):
        x = import_module('models.' + self.args.model_name)
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        his_task = []
        his_task_dict = {}
        test_model = copy.deepcopy(model)
        test_model_state_dict = model.state_dict()
        if self.args.paradigm == 'ccfed':
            head = torch.load('./save_model/Client_{}_head_{}.pth'.format(idx, self.args.model))
            for index, (key, v) in enumerate(test_model_state_dict.items()):
                if index >= len(test_model_state_dict) - 2:
                    test_model_state_dict[key] = head[key]
        test_model.load_state_dict(test_model_state_dict)
        diff_task_acc = {}
        for task in range(self.current_task+1):
            diff_task_acc[task] = []

        for i in range(self.current_task+1):
            his_task.extend(self.user_task[i])
            his_task_dict[i] = self.user_task[i]
        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)
            if (self.args.attack_type == 'backdoor') and (idx in self.args.atttack_client_id) and (self.current_task == self.args.attack_task_id):
                images = add_trigger(images)
                # labels[:] = 8
            if self.args.paradigm == 'ccfed':#加载预测头模型
                if 'class' in self.args.scenario:
                    for k, cls in enumerate(his_task):
                        if cls in labels:
                            task_id = [key for key, value in his_task_dict.items() if cls in value]
                else:
                    task_id = [batch_idx//10]
                if self.current_task > 0 and task_id[0] < self.current_task:
                    test_model = copy.deepcopy(model)
                    new_dict = test_model.state_dict()
                    head = torch.load('./save_model/Epoch_{}_Client_{}_head_{}.pth'.format((task_id[0]+1)*self.args.num_epochs-1, idx, self.args.model))
                    for index, (key, v) in enumerate(new_dict.items()):
                        if index < len(new_dict) - 2:
                            new_dict[key] = test_model_state_dict[key]
                        if index >= len(new_dict) - 2:
                            new_dict[key] = head[key]
                    test_model.load_state_dict(new_dict)
                else:
                    test_model = copy.deepcopy(model)
            if 'cifar' in self.args.model_name.lower():
                images = images.to(torch.float32)
                log_prob, _ = test_model(images)
            else:
                log_prob, _ = test_model(images)
            if isinstance(log_prob, list):
                log_prob = log_prob[self.current_task]
                labels -= int(self.args.task_list[self.current_task][0])
            batch_loss = self.criterion(log_prob, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(log_prob, 1)
            pred_labels = pred_labels.view(-1)

            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
            if self.args.paradigm == 'ccfed':
                diff_task_acc[task_id[0]].append(torch.sum(torch.eq(pred_labels, labels)).item()/len(labels))
            
        accuracy = correct/total
        return accuracy, loss


class GlobalUpdate(object):
    def __init__(self, args, train_data, idxs, logger, current_task, local_models):
        self.args = args
        self.logger = logger
        self.local_models = local_models
        self.device = args.device
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            train_data, list(idxs))
        self.current_task = current_task
        self.criterion = F.cross_entropy

    def train_val_test(self, dataset, idxs):
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8 * len(idxs))]
        idxs_val = idxs[int(0.8 * len(idxs)):]

        trainloader = DataLoader(GlobalDataSetSplit(dataset, idxs_train, local_models=self.local_models, device=self.device),
                                 batch_size=128, shuffle=False)
        validloader = DataLoader(GlobalDataSetSplit(dataset, idxs_val, local_models=self.local_models, device=self.device),
                                 batch_size=int(len(idxs_val) / 10), shuffle=False)
        # idxs_test -> idxs_val
        testloader = DataLoader(GlobalDataSetSplit(dataset, idxs_val, local_models=self.local_models, device=self.device),
                                batch_size=int(len(idxs_val) / 10), shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        model.train()
        dis_loss = DistillationLoss()
        for local_model in self.local_models:
            local_model.eval()
        epoch = self.args.local_ep
        for iter in range(int(epoch)):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                if 'cifar' in self.args.model_name.lower():
                    images = images.to(torch.float32)
                    outputs, _ = model(images)
                else:
                    outputs, _ = model(images)

                loss = dis_loss(outputs, labels, 2.0, 0.1)

                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 100 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                                            100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)


def test_inference(args, model, test_dataset, current_task):
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = args.device
    criterion = F.cross_entropy
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        # if 'class' in args.scenario and args.paradigm == 'ccfed':
        #     for k, val in enumerate(args.subtract_val):
        #         if val in labels:
        #             index = torch.where(labels == val)
        #             for ele in index:
        #                 labels[ele] = labels[ele] - val + k

        # Inference
        if 'cifar' in args.model_name.lower():
            images = images.to(torch.float32)
            outputs, _ = model(images)
        else:
            outputs, _ = model(images)

        if isinstance(outputs, list):
            outputs = outputs[current_task]
            labels -= int(args.task_list[current_task][0])
        batch_loss = criterion(outputs, labels)

        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss