import torch
import torchvision
import torchvision.transforms as transforms
import pickle
import csv
import os
import random
import numpy as np
from scipy.ndimage import gaussian_filter


def process_fashionmnist(num_user, task_number):
    train_dataset = torchvision.datasets.FashionMNIST(root='dataset/data/FashionMNIST', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = torchvision.datasets.FashionMNIST(root='dataset/data/FashionMNIST', train=False, download=True, transform=transforms.ToTensor())
    num_class = 10
    task_length = int(num_class / task_number)
    task_class_length = []
    train_tasks = []
    dev_tasks = []
    for item in range(task_number):
        if item < task_number-1:
            task_class_length.append(task_length)
        elif item == task_number-1:
            task_class_length.append(num_class-task_length*(task_number-1))

    numdata_onecls = 6000//num_user
    user_data = {}
    user_task = {}
    for i in range(num_user):
        user_data[i] = {}
        user_task[i] = {}

    for k in range(num_class):
        id_onecls = [i for i in range(6000)]
        for i in range(num_user):
            if len(id_onecls) >= numdata_onecls:
                cls_user_id = random.sample(id_onecls, numdata_onecls)
                user_data[i][k] = cls_user_id
                id_onecls = list(set(id_onecls)-set(cls_user_id))

    for i in range(num_user):
        class_id_ls = [i for i in range(10)]
        for idx, j in enumerate(task_class_length):
            if len(class_id_ls) >= j:
                class_id = random.sample(class_id_ls, j)
                class_id = sorted(class_id)
                class_id_ls = list(set(class_id_ls) - set(class_id))
                user_task[i][idx] = class_id

    doc_dict = {}
    for i in range(num_class):
        key1 = 'FashionMNIST_{}_train'.format(i)
        train_tasks.append('dataset/data/FashionMNIST/{}'.format(key1))
        key2 = 'FashionMNIST_{}_eval'.format(i)
        dev_tasks.append('dataset/data/FashionMNIST/{}'.format(key2))
        doc_dict[key1] = []
        doc_dict[key2] = []

    for i in range(len(train_dataset)):
        image, label = train_dataset[i]
        for j in range(num_class):
            if label == j:
                key = 'FashionMNIST_{}_train'.format(j)
                doc_dict[key].append(train_dataset[i])

    for i in range(len(test_dataset)):
        image, label = test_dataset[i]
        for j in range(num_class):
            if label == j:
                key = 'FashionMNIST_{}_eval'.format(j)
                doc_dict[key].append(test_dataset[i])

    for k, v in doc_dict.items():
        with open('dataset/data/FashionMNIST/{}'.format(k), 'wb') as f:
            pickle.dump(v, f)
    return task_class_length, train_tasks, dev_tasks, user_data, user_task


def process_cifar10(num_user, task_number):
    train_dataset = torchvision.datasets.CIFAR10(root='dataset/data/cifar10', train=True, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='dataset/data/cifar10', train=False, download=True)
    num_class = 10
    task_length = int(num_class / task_number)
    task_class_length = []
    train_tasks = []
    dev_tasks = []
    for item in range(task_number):
        if item < task_number - 1:
            task_class_length.append(task_length)
        elif item == task_number - 1:
            task_class_length.append(num_class - task_length * (task_number - 1))

    numdata_onecls = 5000 // num_user
    user_data = {}
    user_task = {}
    for i in range(num_user):
        user_data[i] = {}
        user_task[i] = {}

    for k in range(num_class):
        id_onecls = [i for i in range(5000)]
        for i in range(num_user):
            if len(id_onecls) >= numdata_onecls:
                cls_user_id = random.sample(id_onecls, numdata_onecls)
                user_data[i][k] = cls_user_id
                id_onecls = list(set(id_onecls) - set(cls_user_id))

    for i in range(num_user):
        class_id_ls = [i for i in range(10)]
        for idx, j in enumerate(task_class_length):
            if len(class_id_ls) >= j:
                class_id = random.sample(class_id_ls, j)
                class_id = sorted(class_id)
                class_id_ls = list(set(class_id_ls) - set(class_id))
                user_task[i][idx] = class_id

    doc_dict = {}
    for i in range(num_class):
        key1 = 'CIFAR10_{}_train'.format(i)
        train_tasks.append('dataset/data/cifar10/{}'.format(key1))
        key2 = 'CIFAR10_{}_eval'.format(i)
        dev_tasks.append('dataset/data/cifar10/{}'.format(key2))
        doc_dict[key1] = []
        doc_dict[key2] = []

    for i in range(len(train_dataset)):
        image, label = train_dataset[i]
        for j in range(num_class):
            if label == j:
                key = 'CIFAR10_{}_train'.format(j)
                doc_dict[key].append(train_dataset[i])

    for i in range(len(test_dataset)):
        image, label = test_dataset[i]
        for j in range(num_class):
            if label == j:
                key = 'CIFAR10_{}_eval'.format(j)
                doc_dict[key].append(test_dataset[i])

    for k, v in doc_dict.items():
        with open('dataset/data/cifar10/{}'.format(k), 'wb') as f:
            pickle.dump(v, f)
    return task_class_length, train_tasks, dev_tasks, user_data, user_task


def process_cifar100(num_user, task_number):
    train_dataset = torchvision.datasets.CIFAR100(root='dataset/data/cifar100', train=True, download=True)
    test_dataset = torchvision.datasets.CIFAR100(root='dataset/data/cifar100', train=False, download=True)
    num_class = 100
    task_length = int(num_class / task_number)
    task_class_length = []
    train_tasks = []
    dev_tasks = []
    for item in range(task_number):
        if item < task_number - 1:
            task_class_length.append(task_length)
        elif item == task_number - 1:
            task_class_length.append(num_class - task_length * (task_number - 1))

    numdata_onecls = 500 // num_user
    user_data = {}
    user_task = {}
    for i in range(num_user):
        user_data[i] = {}
        user_task[i] = {}

    for k in range(num_class):
        id_onecls = [i for i in range(500)]
        for i in range(num_user):
            if len(id_onecls) >= numdata_onecls:
                cls_user_id = random.sample(id_onecls, numdata_onecls)
                user_data[i][k] = cls_user_id
                id_onecls = list(set(id_onecls) - set(cls_user_id))

    for i in range(num_user):
        class_id_ls = [i for i in range(100)]
        for idx, j in enumerate(task_class_length):
            if len(class_id_ls) >= j:
                class_id = random.sample(class_id_ls, j)
                class_id = sorted(class_id)
                class_id_ls = list(set(class_id_ls) - set(class_id))
                user_task[i][idx] = class_id

    doc_dict = {}
    for i in range(num_class):
        key1 = 'CIFAR100_{}_train'.format(i)
        train_tasks.append('dataset/data/cifar100/{}'.format(key1))
        key2 = 'CIFAR100_{}_eval'.format(i)
        dev_tasks.append('dataset/data/cifar100/{}'.format(key2))
        doc_dict[key1] = []
        doc_dict[key2] = []

    for i in range(len(train_dataset)):
        image, label = train_dataset[i]
        for j in range(num_class):
            if label == j:
                key = 'CIFAR100_{}_train'.format(j)
                doc_dict[key].append(train_dataset[i])

    for i in range(len(test_dataset)):
        image, label = test_dataset[i]
        for j in range(num_class):
            if label == j:
                key = 'CIFAR100_{}_eval'.format(j)
                doc_dict[key].append(test_dataset[i])

    for k, v in doc_dict.items():
        with open('dataset/data/cifar100/{}'.format(k), 'wb') as f:
            pickle.dump(v, f)
    return task_class_length, train_tasks, dev_tasks, user_data, user_task


def process_text_class(num_user, task_number):
    num_class = 10
    task_length = int(num_class / task_number)
    task_class_length = []
    train_tasks = []
    dev_tasks = []
    for item in range(task_number):
        if item < task_number - 1:
            task_class_length.append(task_length)
        elif item == task_number - 1:
            task_class_length.append(num_class - task_length * (task_number - 1))
    numdata_onecls = 4000 // num_user
    user_data = {}
    user_task = {}
    for i in range(num_user):
        user_data[i] = {}
        user_task[i] = {}

    for k in range(num_class):
        id_onecls = [i for i in range(4000)]
        for i in range(num_user):
            if len(id_onecls) >= numdata_onecls:
                cls_user_id = random.sample(id_onecls, numdata_onecls)
                user_data[i][k] = cls_user_id
                id_onecls = list(set(id_onecls) - set(cls_user_id))

    for i in range(num_user):
        class_id_ls = [i for i in range(10)]
        for idx, j in enumerate(task_class_length):
            if len(class_id_ls) >= j:
                class_id = random.sample(class_id_ls, j)
                class_id = sorted(class_id)
                class_id_ls = list(set(class_id_ls) - set(class_id))
                user_task[i][idx] = class_id

    for i in range(num_class):
        train_tasks.append('dataset/text_data/text_train_{}_class.csv'.format(i))
        dev_tasks.append('dataset/text_data/text_eval_{}_class.csv'.format(i))

    return task_class_length, train_tasks, dev_tasks, user_data, user_task


def add_multiplicative_noise(image, scale):
    noisy_image = np.copy(image).astype(np.float32)  
    noisy_image *= np.random.uniform(1 - scale, 1 + scale, size=image.shape)
    noisy_image = np.clip(noisy_image, 0, 255)
    noisy_image = torch.tensor(noisy_image)
    return noisy_image


def add_gaussian_noise(image, sigma):
    image_np = np.array(image)
    noisy_image = gaussian_filter(image_np, sigma=sigma)
    noisy_image = torch.tensor(noisy_image)
    return noisy_image


def process_fashionmnist_domain(num_user, task_number):
    train_dataset = torchvision.datasets.FashionMNIST(root='dataset/data/FashionMNIST', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = torchvision.datasets.FashionMNIST(root='dataset/data/FashionMNIST', train=False, download=True, transform=transforms.ToTensor())
    num_class = 10
    task_class_length = []
    train_tasks = []
    dev_tasks = []
    for item in range(task_number):
        task_class_length.append(10)
    numdata_onecls = 6000//num_user
    user_data = {}
    user_task = {}
    for i in range(num_user):
        user_data[i] = {}
        user_task[i] = {}

    for k in range(num_class):
        id_onecls = [i for i in range(6000)]
        for i in range(num_user):
            if len(id_onecls) >= numdata_onecls:
                cls_user_id = random.sample(id_onecls, numdata_onecls)
                user_data[i][k] = cls_user_id
                id_onecls = list(set(id_onecls)-set(cls_user_id))

    for i in range(num_user):
        for idx, j in enumerate(task_class_length):
            user_task[i][idx] = [k for k in range(num_class)]

    doc_dict = {}
    for i in range(task_number):
        for j in range(num_class):
            key1 = 'FashionMNIST_{}_train_domain'.format(j)
            train_tasks.append('dataset/noise_data/FashionMNIST/task{}/{}'.format(i, key1))
            key2 = 'FashionMNIST_{}_eval_domain'.format(j)
            dev_tasks.append('dataset/noise_data/FashionMNIST/task{}/{}'.format(i, key2))
            doc_dict[key1] = []
            doc_dict[key2] = []

    return task_class_length, train_tasks, dev_tasks, user_data, user_task

def process_cifar10_domain(num_user, task_number):
    train_dataset = torchvision.datasets.CIFAR10(root='dataset/data/cifar10', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = torchvision.datasets.CIFAR10(root='dataset/data/cifar10', train=False, download=True, transform=transforms.ToTensor())
    num_class = 10
    task_class_length = []
    train_tasks = []
    dev_tasks = []
    for item in range(task_number):
        task_class_length.append(10)
    numdata_onecls = 5000//num_user
    user_data = {}
    user_task = {}
    for i in range(num_user):
        user_data[i] = {}
        user_task[i] = {}

    for k in range(num_class):
        id_onecls = [i for i in range(5000)]
        for i in range(num_user):
            if len(id_onecls) >= numdata_onecls:
                cls_user_id = random.sample(id_onecls, numdata_onecls)
                user_data[i][k] = cls_user_id
                id_onecls = list(set(id_onecls)-set(cls_user_id))

    for i in range(num_user):
        for idx, j in enumerate(task_class_length):
            user_task[i][idx] = [k for k in range(num_class)]

    doc_dict = {}
    for i in range(task_number):
        for j in range(num_class):
            key1 = 'CIFAR10_{}_train_domain'.format(j)
            train_tasks.append('dataset/noise_data/cifar10/task{}/{}'.format(i, key1))
            key2 = 'CIFAR10_{}_eval_domain'.format(j)
            dev_tasks.append('dataset/noise_data/cifar10/task{}/{}'.format(i, key2))
            doc_dict[key1] = []
            doc_dict[key2] = []

    return task_class_length, train_tasks, dev_tasks, user_data, user_task
