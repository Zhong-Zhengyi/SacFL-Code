# coding: UTF-8
import os
import torch
import torch.nn as nn
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
import pandas as pd
import copy
import math
import matplotlib.pyplot as plt
# import cv2 as cv
import matplotlib.pyplot as plt


MAX_VOCAB_SIZE = 10000  
UNK, PAD = '<UNK>', '<PAD>'
import numpy as np
from scipy import stats


def euclidean_distance(A, B):
    # 计算差值
    diff = A - B
    # 计算差值的平方
    squared_diff = diff * diff
    # 对每一行求和
    sum_squared_diff = torch.sum(squared_diff, dim=1)
    # 计算平方差的平均值
    mean_squared_diff = torch.mean(sum_squared_diff)
    # 取平均值的平方根
    distance = torch.sqrt(mean_squared_diff)
    return distance


def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        # print('Name: ', name)
        if exclude in name:
            continue
        if 'bias' in name:
            nn.init.constant_(w, 0)
        else:
            if 'batch' in name:
                nn.init.normal_(w)
                continue
            if method == 'xavier':
                if len(w.shape) < 2:
                    nn.init.kaiming_normal_(w.unsqueeze(0))
                else:
                    nn.init.kaiming_normal_(w)
                # nn.init.xavier_normal_(w)
            elif method == 'kaiming':
                nn.init.kaiming_normal_(w)
            else:
                nn.init.normal_(w)

def init_network_resnet(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            fan_in = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
            nn.init.normal_(m.weight.data, mean=0., std=math.sqrt(2. / fan_in))
            if m.bias is not None: nn.init.zeros_(m.bias.data)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.uniform_(m.weight.data)
            if m.bias is not None: nn.init.zeros_(m.bias.data)
        elif isinstance(m, nn.Linear):
            fan_in = m.in_features
            nn.init.normal_(m.weight.data, mean=0., std=math.sqrt(2. / fan_in))
            if m.bias is not None: nn.init.zeros_(m.bias.data)
            
            
def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[0]
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def build_dataset(config, ues_word):
    if ues_word:
        tokenizer = lambda x: x.split(' ')  # word-level, split the word with space
    else:
        tokenizer = lambda x: [y for y in x]  # char-level
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print('Vocab size: {}'.format(len(vocab)))

    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')
                words_line = []
                token = tokenizer(content)
                seq_len = len(token)
                if pad_size:
                    if len(token) < pad_size:
                        token.extend([vocab.get(PAD)] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size
                # word to id
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                contents.append((words_line, int(label)))
        return contents  # [([...], 0), ([...], 1), ...]
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return vocab, train, dev, test


def build_dataset_mnist(config):
    def load_dataset_from_pkl(path):
        with open(path, 'rb') as f:
            dataset = pkl.load(f, encoding='bytes')

        contents = []
        for line in tqdm(dataset):
            img = np.array(line[0])
            # if (img.shape != (1, 28, 28)):
            #     img = img.reshape(1, 28, 28)
            # img = img.transpose((1, 2, 0))
            # img = Image.fromarray(img)
            # print(img.shape)
            contents.append((img, int(line[1])))

            # print(line[1])
        np.random.shuffle(contents)
        return contents  # [([...], 0), ([...], 1), ...]

    vocab = {}
    # train = load_dataset_from_csv(config.train_path, config.pad_size)
    trains = [load_dataset_from_pkl(train_path) for train_path in config.train_tasks]
    devs = [load_dataset_from_pkl(dev_path) for dev_path in config.dev_tasks]
    tests = [load_dataset_from_pkl(test_path) for test_path in config.test_tasks]
    return vocab, trains, devs, tests

def build_dataset_cifar10(config):
    def load_dataset_from_pkl(path):
        with open(path, 'rb') as f:
            dataset = pkl.load(f, encoding='bytes')
        contents = []
        for line in tqdm(dataset):
            img = np.array(line[0])
            if(img.shape != (3, 32, 32)):
                img = img.reshape(3, 32, 32)
            contents.append((img, int(line[1])))
            
            # print(line[1])
        np.random.shuffle(contents)
        return contents  # [([...], 0), ([...], 1), ...]
    vocab = {}
    # train = load_dataset_from_csv(config.train_path, config.pad_size)
    trains = [load_dataset_from_pkl(train_path) for train_path in config.train_tasks]
    devs = [load_dataset_from_pkl(dev_path) for dev_path in config.dev_tasks]
    tests = [load_dataset_from_pkl(test_path) for test_path in config.test_tasks]
    return vocab, trains, devs, tests

def build_dataset_cifar100(config):
    def load_dataset_from_pkl(path):
        with open(path, 'rb') as f:
            dataset = pkl.load(f, encoding='bytes')
        
        contents = []
        for line in tqdm(dataset):
            img = np.array(line[0])
            if(img.shape != (3, 32, 32)):
                img = img.reshape(3, 32, 32)
            contents.append((img, int(line[1])))
            
            # print(line[1])
        np.random.shuffle(contents)
        return contents  # [([...], 0), ([...], 1), ...]
    
    vocab = {}
    trains = [load_dataset_from_pkl(train_path) for train_path in config.train_tasks]
    devs = [load_dataset_from_pkl(dev_path) for dev_path in config.dev_tasks]
    tests = [load_dataset_from_pkl(test_path) for test_path in config.test_tasks]
    return vocab, trains, devs, tests

def build_dataset_from_csv(config, ues_word):
    if ues_word:
        tokenizer = lambda x: x.split(' ')  # word-level, split the word with space
    else:
        tokenizer = lambda x: [y for y in x]  # char-level
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab_from_csv(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print('Vocab size: {}'.format(len(vocab)))

    def load_dataset_from_csv(path, pad_size=32):
        dataset = pd.read_csv(path)
        dataset = dataset.values
        contents = []
        for line in tqdm(dataset):
            try:
                content, label = str(line[0]).strip() + str(line[1]).strip(), int(line[2])
            except AttributeError:
                print(content, label)
                content, label = line[0].strip(), int(line[2])

            words_line = []
            token = tokenizer(content)
            seq_len = len(token)
            if pad_size:
                if len(token) < pad_size:
                    token.extend([vocab.get(PAD)] * (pad_size - len(token)))
                else:
                    token = token[:pad_size]
                    seq_len = pad_size
            # word to id
            for word in token:
                words_line.append(vocab.get(word, vocab.get(UNK)))
            contents.append((words_line, int(label)))
        np.random.shuffle(contents)
        return contents  # [([...], 0), ([...], 1), ...]
    trains = [load_dataset_from_csv(train_path, config.pad_size) for train_path in config.train_tasks]
    devs = [load_dataset_from_csv(dev_path, config.pad_size) for dev_path in config.dev_tasks]
    tests = [load_dataset_from_csv(test_path, config.pad_size) for test_path in config.test_tasks]
    return vocab, trains, devs, tests

def build_dataset_from_csv_fed(config, ues_word):
    if ues_word:
        tokenizer = lambda x: x.split(' ')  # word-level, split the word with space
    else:
        tokenizer = lambda x: [y for y in x]  # char-level
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab_from_csv(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print('Vocab size: {}'.format(len(vocab)))

    def load_dataset_from_csv(path, pad_size=100):
        print(path)
        dataset = pd.read_csv(path)
        dataset = dataset.values
        contents = []
        for line in tqdm(dataset):
            try:
                # content, label = str(line[-3]).strip() + str(line[-2]).strip(), int(line[-1])
                content, label = str(line[-2]).strip(), int(line[-1])
            except:
                content, label = line[1].strip(), int(line[2])

            words_line = []
            token = tokenizer(content)

            # if pad_size:
            if len(token) < pad_size:
                token.extend([vocab.get(PAD)] * (pad_size - len(token)))
            else:
                token = token[:pad_size]

            # word to id
            for word in token:
                words_line.append(vocab.get(word, vocab.get(UNK)))
            contents.append((np.array(words_line), int(label)))
        np.random.shuffle(contents)
        return contents  # [([...], 0), ([...], 1), ...]

    trains = [load_dataset_from_csv(train_path, config.pad_size) for train_path in config.train_tasks]
    devs = [load_dataset_from_csv(dev_path, config.pad_size) for dev_path in config.dev_tasks]
    tests = [load_dataset_from_csv(test_path, config.pad_size) for test_path in config.test_tasks]
    return vocab, trains, devs, tests


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        return x, y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches

class CifarIterater(DatasetIterater):
    def __init__(self, batches, batch_size, device):
        super(CifarIterater, self).__init__(batches, batch_size, device)

    def _to_tensor(self, datas):
        x = torch.FloatTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        return x, y

def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter

def build_cifar_iterator(dataset, config):
    iter = CifarIterater(dataset, config.batch_size, config.device)
    return iter

def build_usergroup(dataset, config):
    num_shards, num_texts = 60, len(dataset) // 60
    num_assign = num_shards // config.num_users

    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(config.num_users)}
    idxs = np.arange((num_shards) * num_texts)

    for i in range(config.num_users):
        rand_set = set(np.random.choice(idx_shard, num_assign, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_texts:(rand + 1) * num_texts]), axis=0)
    return dict_users

def build_usergroup_non_iid(dataset, config):
    num_shards, num_texts = 120, len(dataset) // 120
    num_assign = num_shards // config.num_users
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(config.num_users)}
    idxs = np.arange(num_shards * num_texts)
    # labels = dataset.train_labels.numpy()

    labels = np.asarray([content[1] for content in dataset], dtype=np.float64)

    # sort labels
    idxs_labels = np.vstack((idxs, labels[:num_shards * num_texts]))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(config.num_users):
        rand_set = set(np.random.choice(idx_shard, num_assign, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_texts:(rand + 1) * num_texts]), axis=0)
    return dict_users


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def _update_mean_params(model):
    for param_name, param in model.named_parameters():
        _buff_param_name = param_name.replace('.', '_')
        model.register_buffer(_buff_param_name, param.data.clone())

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = w_avg[key].float()
        for i in range(1, len(w)):
            w_avg[key] += w[i][key].float()
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def krum_aggregation(local_weights, num_to_select):
    # 计算每个客户端更新与其他客户端更新之间的欧几里得距离
    distances = []
    for i, w_i in enumerate(local_weights):
        dist = []
        for j, w_j in enumerate(local_weights):
            if i != j:
                for k in w_i.keys():
                    if not isinstance(w_i[k], torch.LongTensor):
                        dist.append((w_i[k] - w_j[k]).float().norm())
                dist.append(sum((w_i[k] - w_j[k]).float().norm() for k in w_i.keys()))
        distances.append((i, sum(sorted(dist)[:num_to_select])))
    # 选择距离最小的更新
    distances.sort(key=lambda x: x[1])
    return local_weights[distances[0][0]]

def median_aggregation(local_weights):
    # 对每个参数的更新值取中位数
    median_weights = {}
    for key in local_weights[0].keys():
        stacked_weights = torch.stack([w[key] for w in local_weights])
        median_weights[key] = torch.median(stacked_weights, dim=0)[0]
    return median_weights

def trimmed_mean_aggregation(local_weights, trim_ratio):
    # 对每个参数的更新值进行排序，去掉一定比例的最大值和最小值，然后对剩余的值进行平均
    trimmed_mean_weights = {}
    for key in local_weights[0].keys():
        stacked_weights = torch.stack([w[key] for w in local_weights])
        sorted_weights, _ = torch.sort(stacked_weights, dim=0)
        trim_count = int(trim_ratio * len(local_weights))
        trimmed_weights = sorted_weights[trim_count:-trim_count]
        trimmed_mean_weights[key] = torch.mean(trimmed_weights.float(), dim=0)
    return trimmed_mean_weights


def weighted_weights(w, weighted):
    weighted_w = copy.deepcopy(w[0])
    for key in weighted_w.keys():
        for i in range(1, len(w)):
            if i == 1:
                weighted_w[key] = w[i][key]*weighted[i]
            else:
                weighted_w[key] += w[i][key]*weighted[i]
    return weighted_w

def exp_details(args):
    print('\nExperimental details:')
    print('    Learning  : {}'.format(args.learning_rate))
    print('    Global Rounds   : {}\n'.format(args.num_epochs))

    print('    Federated parameters:')
    print('    Fraction of users  : {}'.format(args.frac))
    print('    Local Batch size   : {}'.format(args.local_bs))
    print('    Local Epochs       : {}\n'.format(args.local_ep))

    
def calculate_cosine_similarity(grad1, grad2):
    # 确保两个梯度都是张量并展平
    grad1_flat = grad1.view(-1)
    grad2_flat = grad2.view(-1)

    # 计算点积
    dot_product = torch.dot(grad1_flat, grad2_flat)

    # 计算范数
    norm_grad1 = torch.norm(grad1_flat)
    norm_grad2 = torch.norm(grad2_flat)

    # 计算余弦相似度
    if norm_grad1 == 0 or norm_grad2 == 0:
        return None  # 避免除以零的情况
    cosine_similarity = dot_product / (norm_grad1 * norm_grad2)
    
    return cosine_similarity.item()

def attack_sample_detection(config, model, shift_or_not_dict, current_task, epoch, local_weights, idxs_users):
    from train_eval_fed import test
    all_hist_cls = []
    for idx in range(config.num_users):
        user_task_ls = []
        for t in range(current_task):
            user_task_ls.append(config.user_task[idx][t])
        all_hist_cls.append(user_task_ls)
    user_pf_degration = []
    
    for k, v in shift_or_not_dict.items():
        user_cls_pf_degration_ls = []
        user_cls_pf_degration = {}
        for dataclass in range(config.num_classes):
            user_cls_pf_degration[dataclass] = []
        # if v == True:#如果当前用户数据发生变化，需要在所有历史类别中测试效果是否发生下降
        index = np.where(idxs_users == k)[0]
        encoder_weight = copy.deepcopy(local_weights[index[0]])
        keys_to_remove = list(encoder_weight.keys())[-2:]
        for key in keys_to_remove:
            encoder_weight.pop(key)
        new_test_model = copy.deepcopy(model)
        old_test_model = copy.deepcopy(model)
        if current_task>0:
            old_test_model.load_state_dict(torch.load('./save_model/global_{}_history_model_{}.pth'.format(current_task*config.num_epochs-1, config.model), map_location='cuda'))
        new_dict = new_test_model.state_dict()
        old_dict = old_test_model.state_dict()
        for key, value in encoder_weight.items():#更新encoder参数
            new_dict[key] = value
        for user_id, user_task_ls in enumerate(all_hist_cls):#一旦监测出任务发生变化后，将最新的encoder参数与所有历史知识对应的decoder进行结合测试准确率
            for task_id, cls_ls in enumerate(user_task_ls):
                head_dict = torch.load('./save_model/Epoch_{}_Client_{}_head_{}.pth'.format((task_id+1)*config.num_epochs-1, user_id, config.model))
                for key, value in head_dict.items():
                    new_dict[key] = value
                    old_dict[key] = value
                new_test_model.load_state_dict(new_dict)
                old_test_model.load_state_dict(old_dict)
                # 运用proxy data对cls_ls中涉及的类进行测试
                for cls in cls_ls:     
                    cls_test_data = config.proxy_dict[cls]
                    # print('cls_test_data_idx:', type(cls_test_data_idx))

                    # prox_cls_data = dev_datasets[cls]
                    # cls_test_data = prox_cls_data[cls_test_data_idx]
                    acc_new, loss_new = test(config, new_test_model, cls_test_data, task_id)
                    acc_old, loss_old = test(config, old_test_model, cls_test_data, task_id)
                    if acc_old - acc_new > 0:
                        performance_degration = (acc_old - acc_new)/acc_old
                    else:
                        performance_degration = acc_old - acc_new
                    user_cls_pf_degration[cls].append(performance_degration)
        user_cls_pf_degration_ls.append(k)
        #对user_cls_pf_degration[cls]求平均值
        for c, v in user_cls_pf_degration.items():
            user_cls_pf_degration[c] = np.mean(v)
            user_cls_pf_degration_ls.append(user_cls_pf_degration[c])
        user_cls_pf_degration_ls.append(epoch)
        user_cls_pf_degration_ls.append(config.user_task[k][current_task])
        user_pf_degration.append(user_cls_pf_degration_ls)
    return user_pf_degration

def add_trigger(img, trigger_size=5):
    # 在右下角添加一个trigger_size x trigger_size的白色方块
    img[:, -trigger_size:, -trigger_size:] = 1
    return img