#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utils import build_dataset, build_iterator, get_time_dif, build_dataset_from_csv_fed, init_network
import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import random

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from utils import build_usergroup, build_usergroup_non_iid, calculate_cosine_similarity
from update import LocalUpdate, test_inference, GlobalUpdate
from utils import average_weights, exp_details, draw_model_heatmap, weighted_weights, attack_sample_detection, krum_aggregation, median_aggregation, trimmed_mean_aggregation
import pandas as pd
from importlib import import_module
from models import LeNet_FashionMNIST, TextCNN
from algo_utils.fcil import participant_exemplar_storing, GLFC_model, proxyServer
from algo_utils.fedweit import *
from models.MyModel import network, NetModule


def train_sacfl(config, model, train_dataset, dev_datasets, mu):
    x = import_module('models.' + config.model_name)
    start_time = time.time()
    logger = SummaryWriter('../logs')

    # Training
    train_loss, train_accuracy = [], []
    print_every = 1
    change_dict = {}
    for i in range(config.num_users):
        change_dict[i] = []
    train_acc_info_list = []
    diff_epoch_list = []
    shift_or_not_dict = {}
    two_epoch_loss = {}
    for i in range(config.num_users):
        two_epoch_loss[i] = 0
        shift_or_not_dict[i] = False
    current_task = -1
    cal_time = []
    attack_degradation = []
    for epoch in range(config.num_epochs * config.task_number):
        print('epoch_{}_two_epoch_loss_{}'.format(epoch, two_epoch_loss))
        current_task_copy = copy.deepcopy(current_task)
        current_task = epoch // config.num_epochs
        if current_task != current_task_copy:
            lr = config.learning_rate
            task_shift = True
            if current_task != 0:
                pretrained_dict = current_dict
                global_dict = model.state_dict()
                for index, (name, param) in enumerate(pretrained_dict.items()):
                    if index < len(pretrained_dict) - 2:
                        global_dict[name] = pretrained_dict[name]
                model.load_state_dict(global_dict)
        else:
            model.load_state_dict(torch.load('./save_model/global_model_{}.pth'.format(config.model), map_location='cuda'))
            task_shift = False

        local_weights, local_losses = [], [], []

        print('\n | Global Training Round : {}|\n'.format(epoch))

        m = max(int(config.frac * config.num_users), 1)
        idxs_users = np.random.choice(range(config.num_users), m, replace=False)
        std_time = time.time()
        diff_user_list = []
        sum_user_time = 0
        for idx in idxs_users:
            subtract_val = config.user_task[idx][current_task]
            config.subtract_val = subtract_val
            user_task = config.user_task[idx]
            user_data = config.user_data[idx]
            local_model = LocalUpdate(args=config, idx=idx, train_data=train_dataset, test_data=dev_datasets, user_task=user_task, user_data=user_data, logger=logger, current_task=current_task, lr=lr)
            w, loss, change_dict, diff, shift_or_not, two_epoch_loss, sum_time = local_model.update_weights_sacfl(model=copy.deepcopy(model), global_round=epoch, idx=idx, change_dict=change_dict, mu=mu,
                                                                                                        shift_or_not=shift_or_not_dict[idx], two_epoch_loss=two_epoch_loss, task_shift=task_shift)
            sum_user_time += sum_time
            diff = diff.cpu()
            diff = diff.detach().numpy()
            if epoch > 0:
                diff_user_list.append(diff)
            shift_or_not_dict[idx] = shift_or_not
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
        
        # test wether the attack is harmful
        if config.attack_type != 'none':
            user_pf_degration = attack_sample_detection(config, model, shift_or_not_dict, current_task, epoch, local_weights, idxs_users)
            print('user_pf_degration:', user_pf_degration)
            attack_degradation.extend(user_pf_degration)
                    
        diff_user_avg = np.mean(diff_user_list)
        diff_epoch_list.append(diff_user_avg)
        
        # update global weights
        if config.defense == 'krum' and (current_task == config.attack_task_id):
            global_weights = krum_aggregation(local_weights, num_to_select=len(local_weights) - 2)
        elif config.defense == 'sacfl' and (current_task == config.attack_task_id):
            global_weights = krum_aggregation(local_weights, num_to_select=len(local_weights) - 2)
        elif config.defense == 'median' and (current_task == config.attack_task_id):
            global_weights = median_aggregation(local_weights)
        elif config.defense == 'trimmed_mean' and (current_task == config.attack_task_id):
            global_weights = trimmed_mean_aggregation(local_weights, trim_ratio=0.2)
        else:
            global_weights = average_weights(local_weights)

        end_time = time.time()
        cal_time.append(end_time - std_time - sum_user_time)
        avg_time = sum(cal_time) / len(cal_time)

        if epoch == config.num_epochs * config.task_number - 1:
            print('method-{}-epoch-{}-cal_ls-{}-avg-{}'.format(config.paradigm.lower(), epoch, cal_time, avg_time))

        if current_task > 0: # when current_task > 0, mix the history global Encoder weights with the current global Encoder weights
            current_dict = global_weights
            global_weights_ = copy.deepcopy(global_weights)
            pre_weights_ls = []
            keys_to_remove = list(global_weights_.keys())[-2:]
            for key in keys_to_remove:
                global_weights_.pop(key)
            pre_weights_ls.append(global_weights_)
            for i in range(current_task):
                pretrained_dict = torch.load('./save_model/global_{}_history_model_{}.pth'.format((i+1)*config.num_epochs-1, config.model), map_location='cuda')
                keys_to_remove = list(pretrained_dict.keys())[-2:]
                for key in keys_to_remove:
                    pretrained_dict.pop(key)
                pre_weights_ls.append(pretrained_dict)
            
            avg_dict = average_weights(pre_weights_ls)
            
            for index, (name, param) in enumerate(avg_dict.items()):
                if index < len(avg_dict) - 2:
                    current_dict[name] = copy.deepcopy(avg_dict[name])
        else:
            current_dict = global_weights
        # update global weights
        model.load_state_dict(current_dict)
        torch.save(current_dict, './save_model/global_model_{}.pth'.format(config.model))

        # return
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []

        model.eval()
        for c in range(config.num_users):
            acc_info = []
            subtract_val = config.user_task[c][current_task]
            config.subtract_val = subtract_val
            user_task = config.user_task[c]
            user_data = config.user_data[c]
            local_model = LocalUpdate(args=config, idx=c, train_data=train_dataset, test_data=dev_datasets, user_task=user_task, user_data=user_data, logger=logger, current_task=current_task, lr=lr)
            # acc, loss, weighted = local_model.inference(model=model, idx=c)
            acc, loss = local_model.inference(model=model, idx=c)

            list_acc.append(acc)
            list_loss.append(loss)
            acc_info.append(current_task)
            acc_info.append(epoch)
            acc_info.append(c)
            acc_info.append(acc)
            train_acc_info_list.append(acc_info)

        train_accuracy.append(sum(list_acc) / len(list_acc)) # avg_acc of clients

        if (epoch + 1) % print_every == 0:
            print(' \nAvg Training Stats after {} global rounds:'.format(epoch + 1))
            print('Current Task {} Training Loss : {}'.format(current_task, np.mean(np.array(train_loss))))
            print('Current Task {} Train Accuracy: {:.2f}% \n'.format(current_task, 100 * train_accuracy[-1]))

    df1 = pd.DataFrame(train_acc_info_list, columns=['Current_task', 'Epoch', 'Client_id', 'Train_acc'])
    # df2 = pd.DataFrame(test_acc_info_list, columns=['Current_task', 'Epoch', 'Test_task', 'Test_acc'])
    df3 = pd.DataFrame(diff_epoch_list, columns=['Diff'])
    if config.attack_type != 'none':
        df1.to_csv('Results/attack_{}_defense_{}_train_acc_ccfed_tasknum_{}_{}.csv'.format(config.attack_type, config.defense, config.task_number, config.model_name))
        if config.defense == 'none':
            df4 = pd.DataFrame(attack_degradation, columns=['Client_id', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'Epoch', 'Task'])
            df4.to_csv('Results/attack_{}_degradation_ccfed_tasknum_{}_{}_attacktaskid_{}.csv'.format(config.attack_type, config.task_number, config.model_name, config.attack_task_id))
            df3.to_csv('Results/attack_{}_diff_ccfed_tasknum_{}_{}.csv'.format(config.attack_type, config.task_number, config.model_name))
   
    elif config.attack_type == 'none':
        df1.to_csv('Results/train_acc_ccfed_tasknum_{}_{}.csv'.format(config.task_number, config.model_name))
        # df2.to_csv('Results/test_acc_ccfed_tasknum_{}_{}.csv'.format(config.task_number, config.model_name))
        df3.to_csv('Results/diff_ccfed_tasknum_{}_{}.csv'.format(config.task_number, config.model_name))
    
    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))


def train(config, global_model, train_dataset, dev_datasets):
    start_time = time.time()
    logger = SummaryWriter('../logs')

    # Training
    train_loss, train_accuracy = [], []
    print_every = 1
    train_acc_info_list = []
    test_acc_info_list = []
    current_task = -1
    for epoch in tqdm(range(config.num_epochs * config.task_number)):
        current_task_copy = copy.deepcopy(current_task)
        current_task = epoch // config.num_epochs
        if current_task != current_task_copy:
            lr = config.learning_rate
        else:
            if epoch % 10 == 0 and epoch > 0:
                lr = lr * 0.9

        local_weights, local_losses = [], []
        print('\n | Global Training Round : {}|\n'.format(epoch+1))

        global_model.train()
        m = max(int(config.frac * config.num_users), 1)
        idxs_users = np.random.choice(range(config.num_users), m, replace=False)

        for idx in idxs_users:
            user_task = config.user_task[idx]
            user_data = config.user_data[idx]
            local_model = LocalUpdate(args=config, idx=idx, train_data=train_dataset, test_data=dev_datasets,
                                      user_task=user_task, user_data=user_data, logger=logger,
                                      current_task=current_task, lr=lr)
            if config.paradigm.lower() == 'fedavg':
                w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
            elif config.paradigm.lower() == 'fedprox':
                w, loss = local_model.update_fedprox_weights(0.3, model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(config.num_users):
            acc_info = []
            user_task = config.user_task[c]
            user_data = config.user_data[c]
            local_model = LocalUpdate(args=config, idx=c, train_data=train_dataset, test_data=dev_datasets,
                                      user_task=user_task, user_data=user_data, logger=logger,
                                      current_task=current_task, lr=lr)
            acc, loss = local_model.inference(model=global_model, idx=c)#验证集准确率

            list_acc.append(acc)
            list_loss.append(loss)
            acc_info.append(current_task)
            acc_info.append(epoch)
            acc_info.append(c)
            acc_info.append(acc)
            train_acc_info_list.append(acc_info)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        if (epoch+1) % print_every == 0:
            print(' \nAvg Training Stats after {} global rounds:'.format(epoch+1))
            print('Training Loss : {}'.format(np.mean(np.array(train_loss))))
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

        for idx in range(config.num_users):
            for i in range(config.task_number):
                if i <= current_task:
                    test_acc_info = []
                    user_task = config.user_task[idx]
                    user_data = config.user_data[idx]
                    local_model = LocalUpdate(args=config, idx=idx, train_data=train_dataset, test_data=dev_datasets,
                                              user_task=user_task, user_data=user_data, logger=logger,
                                              current_task=i, lr=lr)
                    test_acc, test_loss = local_model.inference(model=global_model, idx=idx)
                    print("Current Task {} Test Task {} Accuracy: {:.2f}%".format(current_task, i, 100 * test_acc))
                    test_acc_info.append(current_task)
                    test_acc_info.append(epoch)
                    test_acc_info.append(i)
                    test_acc_info.append(test_acc)
                    test_acc_info_list.append(test_acc_info)

    df1 = pd.DataFrame(train_acc_info_list, columns=['Current_task', 'Epoch', 'Client_id', 'Train_acc'])
    df2 = pd.DataFrame(test_acc_info_list, columns=['Current_task', 'Epoch', 'Test_task', 'Test_acc'])
    df1.to_csv('Results/train_acc_{}_tasknum_{}_{}.csv'.format(config.paradigm, config.task_number, config.model_name))
    df2.to_csv('Results/test_acc_{}_tasknum_{}_{}.csv'.format(config.paradigm, config.task_number, config.model_name))
    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))



def test(config, model, test_iter, current_task):
    # test
    model.eval()
    start_time = time.time()
    subtract_val = 0
    for k in range(current_task):
        subtract_val += config.task_class_length[k]
    config.subtract_val = subtract_val
    test_acc, test_loss = evaluate(config, model, test_iter, current_task, test=True)
    time_dif = get_time_dif(start_time)
  
    return test_acc, test_loss

def evaluate(config, model, data_iter, current_task, test=False):
    acc, loss_total = test_inference(config, model, data_iter, current_task)
    return acc, loss_total / len(data_iter)