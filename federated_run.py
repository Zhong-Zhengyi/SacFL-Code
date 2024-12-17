# coding: UTF-8
import time
import torch
import torch.nn as nn
import numpy as np
from train_eval_fed import train, test, train_CFeD, train_ewc, train_multihead, train_lwf, train_DMC, train_sacfl, train_sacfl_nodetection, train_fedweit, train_fcil
from importlib import import_module
from utils import build_usergroup, get_parameter_number, init_network, init_network_resnet
import argparse
import copy
from utils import build_dataset, build_iterator, get_time_dif, build_dataset_from_csv_fed, build_dataset_cifar10, \
    build_dataset_mnist, build_cifar_iterator, build_dataset_cifar100
from models.CNN_Cifar10 import ResNet18Feature

def get_args():
    parser = argparse.ArgumentParser(description='Chinese Text Classification')
    parser.add_argument('--model', type=str, required=False, default='LeNet_FashionMNIST', help='choose a model: TextCNN, LeNet_FashionMNIST, CNN_Cifar10, CNN_Cifar100')#LeNet_FashionMNIST,CNN_Cifar10,CNN_Cifar100,TextCNN
    parser.add_argument('--num_epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
    parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
    parser.add_argument('--paradigm', default='sacfl', type=str, help='choose the training paradigm: lwf,sacfl,ewc,multihead,cfed,fedweit, fcil, fedavg, fedprox')#dmc,
    parser.add_argument('--scenario', default='class', type=str, help=':Class-IL or Domain-IL') # class or domain
    parser.add_argument('--distribution', default=True, type=bool, help='True means iid, while False means non-iid')
    parser.add_argument('--num_channels', default=1, type=int, help='num_channels')
    parser.add_argument('--num_filters', default=256, type=int, help='num_filters')
    # fedweit setting
    parser.add_argument('--wd', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--lambda_l1', default=1e-3, type=float, help='L1 regularization parameter')
    parser.add_argument('--lambda_l2', default=100., type=float, help='L2 regularization parameter')
    parser.add_argument('--lambda_mask', default=0, type=float, help='L2 regularization parameter')
    #fcil setting
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--memory_size', default=200, type=int, help='memory size')
    parser.add_argument('--epochs_local', default=10, type=int, help='epochs local')
    #adverserial
    parser.add_argument('--attack_type', default='none', type=str, help='none, label_flipping, backdoor')
    parser.add_argument('--atttack_client_id', default=[0])
    parser.add_argument('--attack_task_id', default=1)
    parser.add_argument('--attacked_task_id', default=0, type=int)
    parser.add_argument('--defense', default='none', type=str, help='sacfl, none, krum, median, trimmed_mean')
    args = parser.parse_args()
    return args

args = get_args()


if __name__ == '__main__':
    print(args.paradigm)
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model  # TextCNN, TextCNN_multihead
    scenario = args.scenario.lower()

    x = import_module('models.' + model_name)
    config = x.Config(scenario, embedding, args.scenario)
    for key, value in vars(args).items():
        setattr(config, key, value)

    torch.backends.cudnn.deterministic = True

    start_time = time.time()
    print(start_time)
    print("Loading data...")
    if args.model == 'CNN_Cifar100':
        vocab, train_datas, dev_datas, test_datas = build_dataset_cifar100(config)
    elif args.model == 'CNN_Cifar10':
        vocab, train_datas, dev_datas, test_datas = build_dataset_cifar10(config)
    elif 'MNIST' in args.model:
        vocab, train_datas, dev_datas, test_datas = build_dataset_mnist(config)
    else:
        vocab, train_datas, dev_datas, test_datas = build_dataset_from_csv_fed(config, args.word)
    # train
    config.n_vocab = len(vocab)
    config.model_name = model_name
    print('config.task_number', config.task_number)
    if 'cifar' in config.model_name.lower():
        from torchvision import models
        original_model = models.resnet18(pretrained=True)
        original_model.fc = torch.nn.Linear(512, config.num_classes)
        model = ResNet18Feature(original_model)
        model = model.to(config.device)
        config.num_channels = 3
        config.num_filters = 400
    else:
        model = x.Model(config).to(config.device)
        init_network(model)

    if args.paradigm.lower() == 'sacfl':
        train_sacfl(config, model, train_datas, dev_datas, 1)
   