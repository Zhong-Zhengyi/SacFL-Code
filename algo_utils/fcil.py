import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import random
from PIL import Image
from torch.nn import functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.MyModel import network

def participant_exemplar_storing(clients, num, model_g, old_client, task_id, clients_index):
    for index in range(num):
        clients[index].model = copy.deepcopy(model_g)
        if index not in clients_index:
            if index in old_client:
                clients[index].beforeTrain(task_id, 0)
            else:
                clients[index].beforeTrain(task_id, 1)
            clients[index].update_new_set()



def entropy(input_):
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy

class GLFC_model:

    def __init__(self, config, numclass, feature_extractor, train_data, test_data, user_task, user_data, encode_model, current_task,task_shift):

        super(GLFC_model, self).__init__()
        self.epochs = config.local_ep
        self.learning_rate = config.learning_rate
        self.model = network(config.num_classes, feature_extractor)
        self.encode_model = encode_model
        self.train_data = train_data
        self.test_data = test_data
        self.user_task = user_task
        self.user_data = user_data
        self.args = config
        self.feature_extractor = feature_extractor
        self.current_task = current_task

        self.exemplar_set = []
        self.class_mean_set = []
        self.numclass = numclass
        self.learned_numclass = numclass
        self.learned_classes, self.last_class = self.cal_learned_classes(current_task)
        self.transform = self.transform()
        self.old_model = None
        self.start = True
        self.task_shift = task_shift

        self.batchsize = config.batch_size
        self.memory_size = config.memory_size
        # self.task_size = task_size

        self.current_class = user_task[current_task]
        self.device = config.device
        self.last_entropy = 0
        self.train_loader, self.validloader, self.testloader, self.user_data_test_task = self.train_val_test(current_task)

    def transform(self):
        if self.args.model_name == 'LeNet_FashionMNIST':
            transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.5,), (0.5,))])
        elif 'cifar' in self.args.model_name.lower():
            transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        else:
            transform = transforms.Compose([transforms.ToTensor()])
        return transform
    def get_one_hot(self,target, device):
        one_hot=torch.zeros(target.shape[0], self.args.num_classes).cuda(device)
        one_hot=one_hot.scatter(dim=1,index=target.long().view(-1,1),value=1.)
        return one_hot
    
    def cal_learned_classes(self, current_task):
        learned_classes = []
        for i in range(current_task):
            learned_classes.extend(self.user_task[i])
        if current_task == 0:
            last_class = None
        else:   
            last_class = self.user_task[current_task-1]
        return learned_classes, last_class

    def train_val_test(self, current_task):
        user_train_data = []
        user_data_test_task = []
        test_data = []
        if self.args.scenario == 'class':
            for i, j in enumerate(self.user_data[0]):
                for k in self.user_task[current_task]:
                    user_train_data.append(self.train_data[k][self.user_data[k][i]])
            random.shuffle(user_train_data)
            for j in range(current_task+1):
                user_data_test_task.extend(self.user_task[j])
            for i in user_data_test_task:
                test_data.extend(self.test_data[i])
        elif self.args.scenario == 'domain':
            for i, j in enumerate(self.user_data[0]):
                for k in self.user_task[current_task]:
                    user_train_data.append(self.train_data[current_task*10+k][self.user_data[k][i]])
            random.shuffle(user_train_data)
            for j in range(current_task+1):
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
    

    def local_train(self, model_g,  model_old, ep_g):
        self.model = network(self.args.num_classes, self.feature_extractor)
        self.model.load_state_dict(model_g.state_dict())
        # self.model = copy.deepcopy(model_g)
        # self.beforeTrain(current_task)
        self.update_new_set()
        self.train(ep_g, model_old)
        local_model = self.model.state_dict()
        proto_grad = self.proto_grad_sharing()

        return local_model, proto_grad
        
    def update_new_set(self):
        if self.task_shift == True:        
            m = int(self.memory_size / self.learned_numclass)
            self._reduce_exemplar_sets(m)
            if self.last_class != None:
                for i in self.last_class: 
                    images = self.get_class_data(i)
                if images != []:
                    self._construct_exemplar_set(images, m)
                else:
                    print('no images')
        self.model.train()

    # train model
    def train(self, ep_g, model_old):
        self.model.to(self.device)

        if model_old[1] != None:
            if self.task_shift==True:
                self.old_model = model_old[1]
            else:
                self.old_model = model_old[0]
        else:
            if self.task_shift==True:
                self.old_model = model_old[0]

        if self.old_model != None:
            print('load old model')
            self.old_model.to(self.device)
            self.old_model.eval()
        
        for epoch in range(self.epochs):
            loss_cur_sum, loss_mmd_sum = [], []
            opt = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=0.00001)
            for step, (images, target) in enumerate(self.train_loader):
                images, target = images.to(self.device), target.to(self.device)
                loss_value = self._compute_loss(images, target)
                opt.zero_grad()
                loss_value.backward()
                opt.step()

    def entropy_signal(self, loader):
        self.model.eval()
        start_ent = True
        res = False

        for step, (indexs, imgs, labels) in enumerate(loader):
            imgs, labels = imgs.cuda(self.device), labels.cuda(self.device)
            with torch.no_grad():
                outputs, _ = self.model(imgs)
            softmax_out = nn.Softmax(dim=1)(outputs)
            ent = entropy(softmax_out)

            if start_ent:
                all_ent = ent.float().cpu()
                all_label = labels.long().cpu()
                start_ent = False
            else:
                all_ent = torch.cat((all_ent, ent.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.long().cpu()), 0)

        overall_avg = torch.mean(all_ent).item()
        print(overall_avg)
        if overall_avg - self.last_entropy > 1.2:
            res = True
        
        self.last_entropy = overall_avg

        self.model.train()

        return res

    def _compute_loss(self, imgs, label):
        if self.args.model_name != 'TextCNN':
            imgs = imgs.float()
        output, _ = self.model(imgs)

        target = self.get_one_hot(label, self.device)
     
        output, target = output.to(self.device), target.to(self.device)
        if self.old_model == None:
            w = self.efficient_old_class_weight(output, label)
            loss_cur = torch.mean(w * F.binary_cross_entropy_with_logits(output, target, reduction='none'))

            return loss_cur
        else:
            w = self.efficient_old_class_weight(output, label)
            loss_cur = torch.mean(w * F.binary_cross_entropy_with_logits(output, target, reduction='none'))

            distill_target = target.clone()
            outputs, _ = self.old_model(imgs)
            old_target = torch.sigmoid(outputs)
            old_task_size = old_target.shape[1]
            distill_target[..., :old_task_size] = old_target
            loss_old = F.binary_cross_entropy_with_logits(output, distill_target)

            return 0.5 * loss_cur + 0.5 * loss_old

    def efficient_old_class_weight(self, output, label):
        pred = torch.sigmoid(output)
        
        N, C = pred.size(0), pred.size(1)

        class_mask = pred.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = label.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        target = self.get_one_hot(label, self.device)
        g = torch.abs(pred.detach() - target)
        g = (g * class_mask).sum(1).view(-1, 1)

        if len(self.learned_classes) != 0:
            for i in self.learned_classes:
                ids = torch.where(ids != i, ids, ids.clone().fill_(-1))

            index1 = torch.eq(ids, -1).float()
            index2 = torch.ne(ids, -1).float()
            if index1.sum() != 0:
                w1 = torch.div(g * index1, (g * index1).sum() / index1.sum())
            else:
                w1 = g.clone().fill_(0.)
            if index2.sum() != 0:
                w2 = torch.div(g * index2, (g * index2).sum() / index2.sum())
            else:
                w2 = g.clone().fill_(0.)

            w = w1 + w2
        
        else:
            w = g.clone().fill_(1.)

        return w

    def _construct_exemplar_set(self, images, m):
        
        class_mean, feature_extractor_output = self.compute_class_mean(images, self.transform)
        exemplar = []
        now_class_mean = np.zeros((1, 512))
     
        for i in range(m):
            x = class_mean - (now_class_mean + feature_extractor_output) / (i + 1)
            x = np.linalg.norm(x, axis=1)
            index = np.argmin(x)
            now_class_mean += feature_extractor_output[index]
            exemplar.append(images[index])

        self.exemplar_set.append(exemplar)

    def _reduce_exemplar_sets(self, m):
        for index in range(len(self.exemplar_set)):
            self.exemplar_set[index] = self.exemplar_set[index][:m]

    def Image_transform(self, images, transform):
        if self.args.model_name == 'LeNet_FashionMNIST':
            for idx,image in enumerate(images):
                images[idx] = image.squeeze(0)
            data = transform(Image.fromarray(images[0])).unsqueeze(0)
            for index in range(1, len(images)): 
                img = transform(Image.fromarray(images[index])).unsqueeze(0)
                data = torch.cat((data, img), dim=0)
        elif 'cifar' in self.args.model_name.lower():
            data = transform(Image.fromarray((images[0].transpose(1, 2, 0) * 255).astype(np.uint8))).unsqueeze(0)
            for index in range(1, len(images)): 
                img = transform(Image.fromarray((images[index].transpose(1, 2, 0) * 255).astype(np.uint8))).unsqueeze(0)
                data = torch.cat((data, img), dim=0)
        elif self.args.model_name == 'TextCNN':
            data = torch.tensor(images)
        return data

    def compute_class_mean(self, images, transform):
        x = self.Image_transform(images, transform).to(self.device)
        if self.args.model_name == 'TextCNN':
            x = x.long()
        feature = self.model.feature_extractor(x)
        feature_extractor_output = F.normalize(feature.detach()).cpu().numpy()
        class_mean = np.mean(feature_extractor_output, axis=0)
        return class_mean, feature_extractor_output

    def compute_exemplar_class_mean(self):
        self.class_mean_set = []
        for index in range(len(self.exemplar_set)):
            exemplar = self.exemplar_set[index]
            class_mean, _ = self.compute_class_mean(exemplar, self.transform)
            class_mean_,_ = self.compute_class_mean(exemplar,self.classify_transform)
            class_mean = (class_mean/np.linalg.norm(class_mean)+class_mean_/np.linalg.norm(class_mean_))/2
            self.class_mean_set.append(class_mean)

    def proto_grad_sharing(self):
        if self.task_shift:
            proto_grad = self.prototype_mask()
        else:
            proto_grad = None

        return proto_grad
    
    def get_class_data(self, target_class):
        class_data = []
        class_labels = []

        for images, labels in self.train_loader.dataset:
            if labels == target_class:
                class_data.append(images)
                class_labels.append(labels)

        return class_data
        

    def prototype_mask(self):
        tt = transforms.Compose([transforms.ToTensor()])
        tp = transforms.Compose([transforms.ToPILImage()])
        iters = 50
        criterion = nn.CrossEntropyLoss().to(self.device)
        proto = []
        proto_grad = []

        for i in self.current_class:
            images = self.get_class_data(i)
            class_mean, feature_extractor_output = self.compute_class_mean(images, self.transform)
            dis = class_mean - feature_extractor_output
            dis = np.linalg.norm(dis, axis=1)
            pro_index = np.argmin(dis)
            proto.append(images[pro_index])

        for i in range(len(proto)):
            self.model.eval()
            data = proto[i]
            label = self.current_class[i]
            if self.args.model_name == 'LeNet_FashionMNIST':
                data = Image.fromarray(data)
            elif 'cifar' in self.args.model_name.lower():
                data = (data.transpose(1, 2, 0) * 255).astype(np.uint8)
                data = Image.fromarray(data)
            elif self.args.model_name == 'TextCNN':
                data = torch.tensor(data, dtype=torch.float32)
            label_np = label
            
            if self.args.model_name != 'TextCNN':
                data, label = tt(data), torch.Tensor([label]).long()
            else:
                data, label = data, torch.Tensor([label]).long()
            data, label = data.cuda(self.device), label.cuda(self.device)
            
            target = self.get_one_hot(label, self.device)

            opt = optim.SGD([data, ], lr=self.learning_rate / 10, weight_decay=0.00001)
            proto_model = network(self.args.num_classes, self.feature_extractor)
            proto_model.load_state_dict(self.model.state_dict())
            proto_model.to(self.device)

            for ep in range(iters):
                proto_model.eval()
                if self.args.model_name == 'TextCNN':
                    data = data.long()
                if (self.args.model_name == 'TextCNN' and data.ndim != 2) or (self.args.model_name != 'TextCNN' and data.ndim != 4):
                    data = data.unsqueeze(0)
                
                outputs, _ = proto_model(data)
                proto_model.train()
                loss_cls = F.binary_cross_entropy_with_logits(outputs, target)
                opt.zero_grad()
                loss_cls.backward()
                opt.step()

            self.encode_model.to(self.device)
            data = data.detach().clone().to(self.device).requires_grad_(False)
            outputs, _ = self.encode_model(data)
            loss_cls = criterion(outputs, label)
            dy_dx = torch.autograd.grad(loss_cls, self.encode_model.parameters())
            original_dy_dx = list((_.detach().clone() for _ in dy_dx))
            proto_grad.append(original_dy_dx)

        return proto_grad
    
    def inference(self, model, idx):
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        his_task = []
        his_task_dict = {}
        test_model = copy.deepcopy(model)
        test_model_state_dict = model.state_dict()
        criterion = nn.CrossEntropyLoss().to(self.device)
        if self.args.paradigm == 'sacfl':
            head = torch.load('./save_model/Client_{}_head.pth'.format(idx))
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
            if self.args.paradigm == 'sacfl':
                if 'class' in self.args.scenario:
                    for k, cls in enumerate(his_task):
                        if cls in labels:
                            task_id = [key for key, value in his_task_dict.items() if cls in value]

                else:
                    task_id = [batch_idx//10]
                if self.current_task > 0 and task_id[0] < self.current_task:
                    test_model = copy.deepcopy(model)
                    new_dict = test_model.state_dict()
                    head = torch.load('./save_model/Epoch_{}_Client_{}_head.pth'.format((task_id[0]+1)*self.args.num_epochs-1, idx))
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
            batch_loss = criterion(log_prob, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(log_prob, 1)
            pred_labels = pred_labels.view(-1)

            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
            if self.args.paradigm == 'sacfl':
                diff_task_acc[task_id[0]].append(torch.sum(torch.eq(pred_labels, labels)).item()/len(labels))
            
        accuracy = correct/total
        return accuracy, loss


class proxyServer:
    def __init__(self, config, numclass, feature_extractor, encode_model):
        super(proxyServer, self).__init__()
        self.Iteration = 50
        self.args = config
        self.learning_rate = config.learning_rate
        self.model = network(config.num_classes, feature_extractor)
        self.encode_model = encode_model
        self.monitor_dataset = Proxy_Data()
        self.new_set = []
        self.new_set_label = []
        self.numclass = numclass
        self.device = config.device
        self.num_image = 20
        self.pool_grad = None
        self.best_model_1 = None
        self.best_model_2 = None
        self.best_perf = 0

    def dataloader(self, pool_grad):
        self.pool_grad = pool_grad
        if len(pool_grad) != 0:
            self.reconstruction()
            data=self.monitor_dataset.getTestData(self.new_set, self.new_set_label)
            print(self.monitor_dataset)
            self.monitor_loader = DataLoader(dataset=data, shuffle=True, batch_size=8, drop_last=True)
            self.last_perf = 0
            self.best_model_1 = self.best_model_2

        cur_perf = self.monitor()
        print(cur_perf)
        if cur_perf >= self.best_perf:
            self.best_perf = cur_perf
            self.best_model_2 = copy.deepcopy(self.model)

    def model_back(self):
        return [self.best_model_1, self.best_model_2]

    def monitor(self):
        self.model.eval()
        self.model.to(self.device)
        correct, total = 0, 0
        for step, (imgs, labels) in enumerate(self.monitor_loader):
            if self.args.model_name == 'LeNet_FashionMNIST':
                imgs = imgs.unsqueeze(1)
                imgs = imgs.to(torch.float32)
            elif 'cifar' in self.args.model_name.lower():
                imgs=imgs.permute(0, 3, 1, 2)
                imgs = imgs.to(torch.float32)
            elif self.args.model_name == 'TextCNN':
                imgs = imgs.long()
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            with torch.no_grad():
                outputs, _ = self.model(imgs)
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracy = 100 * correct / total
        
        return accuracy

    def gradient2label(self):
        pool_label = []
        for w_single in self.pool_grad:
            pred = torch.argmin(torch.sum(w_single[-2], dim=-1), dim=-1).detach().reshape((1,)).requires_grad_(False)
            pool_label.append(pred.item())

        return pool_label

    def reconstruction(self):
        self.new_set, self.new_set_label = [], []

        tt = transforms.Compose([transforms.ToTensor()])
        tp = transforms.Compose([transforms.ToPILImage()])

        pool_label = self.gradient2label()
        pool_label = np.array(pool_label)

        class_ratio = np.zeros((1, self.numclass))

        for i in pool_label:
            class_ratio[0, i] += 1

        for label_i in range(self.numclass):
            if class_ratio[0, label_i] > 0:
                num_augmentation = self.num_image
                augmentation = []

                grad_index = np.where(pool_label == label_i)
                for j in range(len(grad_index[0])):

                    grad_truth_temp = self.pool_grad[grad_index[0][j]]

                    if self.args.model_name == 'LeNet_FashionMNIST':
                        dummy_data = torch.randn((1, 1, 28, 28)).to(self.device).requires_grad_(True)
                    elif 'cifar' in self.args.model_name.lower():
                        dummy_data = torch.randn((1, 3, 32, 32)).to(self.device).requires_grad_(True)
                    elif self.args.model_name == 'TextCNN':
                        vocab_size = 6615 
                        dummy_data = torch.randint(0, vocab_size, (1, 24)).to(self.device)
                        dummy_data = dummy_data.long()

                    label_pred = torch.Tensor([label_i]).long().to(self.device).requires_grad_(False)

                    optimizer = torch.optim.LBFGS([dummy_data, ], lr=0.1)

                    criterion = nn.CrossEntropyLoss().to(self.device)

                    recon_model = copy.deepcopy(self.encode_model)
                    recon_model.to(self.device)

                    for iters in range(self.Iteration):
                        def closure():
                            optimizer.zero_grad()
                            pred, _ = recon_model(dummy_data)
                            dummy_loss = criterion(pred, label_pred)

                            dummy_dy_dx = torch.autograd.grad(dummy_loss, recon_model.parameters(), create_graph=True)

                            grad_diff = 0
                            for gx, gy in zip(dummy_dy_dx, grad_truth_temp):
                                grad_diff += ((gx - gy) ** 2).sum()
                            grad_diff.backward()
                            return grad_diff

                        optimizer.step(closure)
                        current_loss = closure().item()

                        if iters == self.Iteration - 1:
                            print(current_loss)

                        if iters >= self.Iteration - self.num_image:
                            if self.args.model_name == 'TextCNN':
                                dummy_data_temp = np.asarray(dummy_data.squeeze(0).cpu())
                            else:
                                dummy_data_temp = np.asarray(tp(dummy_data.clone().squeeze(0).cpu()))
                            augmentation.append(dummy_data_temp)

                self.new_set.append(augmentation)
                self.new_set_label.append(label_i)


class Proxy_Data():
    def __init__(self, test_transform=None):
        super(Proxy_Data, self).__init__()
        self.test_transform = test_transform
        self.TestData = []
        self.TestLabels = []

    def concatenate(self, datas, labels):
        con_data = datas[0]
        con_label = labels[0]
        for i in range(1, len(datas)):
            con_data = np.concatenate((con_data, datas[i]), axis=0)
            con_label = np.concatenate((con_label,labels[i]), axis=0)
        return con_data, con_label

    def getTestData(self, new_set, new_set_label):
        datas, labels = [], []
        self.TestData, self.TestLabels = [], []
        if len(new_set) != 0 and len(new_set_label) != 0:
            datas = [exemplar for exemplar in new_set]
            for i in range(len(new_set)):
                length = len(datas[i])
                labels.append(np.full((length), new_set_label[i]))
        self.TestData, self.TestLabels = self.concatenate(datas, labels)
        data_zip = list(zip(self.TestData, self.TestLabels))

        return data_zip
    

    def getTestItem(self, index):
        img, target = Image.fromarray(self.TestData[index]), self.TestLabels[index]
        return img, target

    def __getitem__(self, index):
        return self.getTestItem(index)

    def __len__(self):
        if isinstance(self.TestData, list):
            return len(self.TestData)
        else:
            return self.TestData.shape[0]

