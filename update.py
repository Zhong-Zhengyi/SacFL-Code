import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.MyModel import NetModule
from torch.utils.data import DataLoader, Dataset
import random
import copy

def aggregate(updates):
    client_weights = [u[0][0] for u in updates]
    client_masks = [u[0][1] for u in updates]
    new_weights = [torch.zeros_like(w) for w in client_weights[0]]
    epsi = 1e-15
    total_sizes = epsi
    client_masks = torch.stack([torch.tensor(m) for m in client_masks])
    for _cs in client_masks:
        total_sizes += _cs
    for c_idx, c_weights in enumerate(client_weights): # by client
        for lidx, l_weights in enumerate(c_weights): # by layer
            ratio = 1/total_sizes[lidx]
            new_weights[lidx] += torch.mul(l_weights, ratio)
    
    return new_weights

def get_adapts(curr_round, nets, args, client_adapts):
    if curr_round%args.num_epochs==1 and not curr_round==1:
        from_kb = []
        for lid, shape in enumerate(nets.shapes):
            if len(nets.shapes[lid]) > 2:
                shape = np.concatenate([[int(round(nets.args.num_users * nets.args.frac))], [nets.shapes[lid][1],nets.shapes[lid][0],nets.shapes[lid][2],nets.shapes[lid][2]]], axis=0)
            else:
                shape = np.concatenate([[int(round(nets.args.num_users * nets.args.frac))], [nets.shapes[lid][1],nets.shapes[lid][0]]], axis=0)
            # shape = np.concatenate([nets.shapes[lid],[int(round(args.num_users*args.frac))]], axis=0)
            from_kb_l = np.zeros(shape)
            for cid, ca in enumerate(client_adapts):
                
                if len(shape)==5:
                    from_kb_l[cid,:,:,:,:] = ca[lid]
                else:
                    from_kb_l[cid,:,:] = ca[lid]         
            from_kb.append(from_kb_l)
        return from_kb
    else:
        return None

class FedWeIT:
    def __init__(self, args, train_data, test_data, user_task, user_data, logger, current_task, lr):
        self.train_data = train_data
        self.test_data = test_data
        self.user_task = user_task
        self.user_data = user_data
        self.args = args
        self.logger = logger
        self.current_task = current_task
        self.device = args.device
        self.criterion = F.cross_entropy
        self.lr = lr
        self.nets = NetModule(self.args)
        self.init_model()
        

    def init_model(self):
        decomposed = True
        self.nets.build_model(decomposed=decomposed, initial_weights=self.nets.init_global_weights())
        

    def train_val_test(self):
        user_train_data = []
        user_data_test_task = []
        test_data = []
        if self.args.scenario == 'class':
            for i, j in enumerate(self.user_data[0]):
                for k in self.user_task[self.current_task]:
                    user_train_data.append(self.train_data[k][self.user_data[k][i]])
            random.shuffle(user_train_data)
            for j in range(self.current_task+1):
                user_data_test_task.extend(self.user_task[j])
            for i in user_data_test_task:
                test_data.extend(self.test_data[i])
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
    
    def update_weight_fedweit(self, client_idx, curr_round, global_weights=None, from_kb=None):#客户端训练函数
        if from_kb is not None:
            for lid, weights in enumerate(from_kb):
                self.nets.decomposed_variables['from_kb'][self.current_task][lid].data.copy_(torch.from_numpy(weights))
        
        if self.current_task < 0:
            # self.init_new_task()
            self.set_weights(global_weights) 
        else:
            is_last_task = (self.current_task == self.args.task_number-1)
            is_last_round = (curr_round % self.args.num_epochs == 0 and curr_round != 0)
            is_last = is_last_task and is_last_round
            if is_last_round:
                self.prev_body_weights = self.nets.get_body_weights(self.current_task)
        
        self.set_weights(global_weights)

        
        local_weight, loss = self.train_one_round(self.current_task)
        
        return local_weight, loss
    
    def train_one_round(self, curr_task):
        self.curr_model = self.nets.get_model_by_tid(curr_task)
        self.curr_model.to(self.args.device)
        self.curr_model.train()
        optimizer = torch.optim.Adam(self.curr_model.parameters(), lr=self.lr)
        epoch_loss = []
        self.trainloader, self.validloader, self.testloader, self.user_data_test_task = self.train_val_test()
        for epoch in range(self.args.num_epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                self.curr_model.zero_grad()
                images, labels = images.to(self.device), labels.to(self.device)
                labels = labels.long()
                if 'cifar' in self.args.model_name.lower():
                    images = images.to(torch.float32)
                    log_prob, _ = self.curr_model(images)
                else:
                    log_prob, _ = self.curr_model(images)
                
                optimizer.zero_grad()
                loss = self.loss(labels, log_prob)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        global_weights = {}
        for k, v in self.curr_model.state_dict().items():
            if 'sw' in k:
                global_weights[k] = v
        return global_weights, sum(epoch_loss) / len(epoch_loss)
            
        # self.adaptive_lr_decay()
    
    
    def set_weights(self, weights):
        if weights is None:
            return None
        if isinstance(weights, dict):
            weights = list(weights.values())
        for i, w in enumerate(weights):
            w = torch.tensor(w)
            sw = self.nets.get_variable('shared', i)
            sw = sw.to(self.args.device)
            w = w.to(self.args.device)
            residuals = torch.eq(w, torch.zeros_like(w)).float()

            param = sw*residuals+w
            param = param.to(self.args.device)
            sw.data.copy_(param)
        

    def get_adaptives(self):
        adapts = []
        for lid in range(len(self.nets.shapes)):
            aw = self.nets.get_variable(var_type='adaptive', lid=lid, tid=self.current_task).detach().cpu().numpy()
            hard_threshold = np.greater(np.abs(aw), self.args.lambda_l1).astype(np.float32)
            adapts.append(aw*hard_threshold)
        return adapts
    
    def loss(self, y_true, y_pred):
        weight_decay, sparseness, approx_loss = 0, 0, 0
        loss = F.cross_entropy(y_pred, y_true)
        for lid in range(len(self.nets.shapes)):
            sw = self.nets.get_variable(var_type='shared', lid=lid)
            aw = self.nets.get_variable(var_type='adaptive', lid=lid, tid=self.current_task)
            mask = self.nets.get_variable(var_type='mask', lid=lid, tid=self.current_task)
            g_mask = self.nets.generate_mask(mask)
            weight_decay += self.args.wd * (aw.pow(2).sum() / 2)
            weight_decay += self.args.wd * (mask.pow(2).sum() / 2)
            sparseness += self.args.lambda_l1 * torch.abs(aw).sum()
            sparseness += self.args.lambda_mask * torch.abs(mask).sum()
            if self.current_task == 0:
                weight_decay += self.args.wd * (sw.pow(2).sum() / 2)
            else:
                for tid in range(self.current_task):
                    prev_aw = self.nets.get_variable(var_type='adaptive', lid=lid, tid=tid)
                    prev_mask = self.nets.get_variable(var_type='mask', lid=lid, tid=tid)
                    g_prev_mask = self.nets.generate_mask(prev_mask)
                    #################################################
                    restored = sw * g_prev_mask + prev_aw
                    # a_l2 = (restored - self.prev_body_weights[lid][tid]).pow(2).sum() / 2
                    a_l2 = (restored.detach() - self.prev_body_weights[lid][tid]).pow(2).sum() / 2
                    approx_loss += self.args.lambda_l2 * a_l2
                    #################################################
                    sparseness += self.args.lambda_l1 * torch.abs(prev_aw).sum()
        
        loss += weight_decay + sparseness + approx_loss 
        return loss
    
    def inference(self, weight, idx, test_task):
        self.trainloader, self.validloader, self.testloader, self.user_data_test_task = self.train_val_test()
        self.set_weights(weight)
        model = self.nets.get_model_by_tid(test_task)
        model.to(self.args.device)
        model.eval()

        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)
            if 'cifar' in self.args.model_name.lower():
                images = images.to(torch.float32)
                log_prob, _ = model(images)
            else:
                log_prob, _ = model(images)
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
            
        accuracy = correct/total
        return accuracy, loss
