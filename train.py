from Load_dataset import load_split_MUTAG_data, accuracy
from Model import GCN
import time

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

model_path = 'model/gcn_first.pth'

epochs = 1000
seed = 200
lr = 0.001
dropout = 0.1
weight_decay = 5e-4

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


class EarlyStopping():
    def __init__(self, patience=10, min_loss=0.5, hit_min_before_stopping=False):
        self.patience = patience
        self.counter = 0
        self.hit_min_before_stopping = hit_min_before_stopping
        if hit_min_before_stopping:
            self.min_loss = min_loss
        self.best_loss = None
        self.early_stop = False

    def __call__(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
        elif loss > self.best_loss:
            self.counter += 1
            if self.counter > self.patience:
                if self.hit_min_before_stopping == True and loss > self.min_loss:
                    print("Cannot hit mean loss, will continue")
                    self.counter -= self.patience
                else:
                    self.early_stop = True
        else:
            self.best_loss = loss
            counter = 0


if __name__ == '__main__':
    # adj_list: [188, 29, 29]
    # features_list: [188, 29, 7]
    # graph_labels: [188]
    adj_list, features_list, graph_labels, idx_map, idx_train, idx_val, idx_test = load_split_MUTAG_data()
    idx_train = torch.cat([idx_train, idx_val, idx_test])

    model = GCN(nfeat=features_list[0].shape[1], # nfeat = 7
                nclass=graph_labels.max().item() + 1, # nclass = 2
                dropout=dropout)
    optimizer = optim.Adam(model.parameters(),
                           lr=lr, weight_decay=weight_decay)

    model.cuda()
    features_list = features_list.cuda()
    adj_list = adj_list.cuda()
    graph_labels = graph_labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

    # 训练模型
    early_stopping = EarlyStopping(10, hit_min_before_stopping=True)
    t_total = time.time()

    for epoch in range(epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()

        # # Split
        outputs = []
        for i in idx_train:
            output = model(features_list[i], adj_list[i])
            output = output.unsqueeze(0)
            outputs.append(output)
        output = torch.cat(outputs, dim=0)


        loss_train = F.cross_entropy(output, graph_labels[idx_train])
        acc_train = accuracy(output, graph_labels[idx_train])
        loss_train.backward()
        optimizer.step()

        model.eval()
        outputs = []
        for i in idx_val:
            output = model(features_list[i], adj_list[i])
            output = output.unsqueeze(0)
            outputs.append(output)
        output = torch.cat(outputs, dim=0)
        loss_val = F.cross_entropy(output, graph_labels[idx_val])
        acc_val = accuracy(output, graph_labels[idx_val])

        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))

        print(loss_val)
        early_stopping(loss_val)
        if early_stopping.early_stop == True:
            break

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    torch.save(model.state_dict(), model_path)
