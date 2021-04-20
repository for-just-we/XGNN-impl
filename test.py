from Load_dataset import load_split_MUTAG_data, accuracy
from Model import GCN
import time

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

model_path = 'model/gcn_first.pth'

if __name__ == '__main__':
    adj_list, features_list, graph_labels, idx_map, idx_train, idx_val, idx_test = load_split_MUTAG_data()
    model = GCN(nfeat=features_list[0].shape[1],  # nfeat = 7
                nclass=graph_labels.max().item() + 1,  # nclass = 2
                dropout=0.1)

    model.eval()
    outputs = []
    for i in idx_test:
        output = model(features_list[i], adj_list[i])
        output = output.unsqueeze(0)
        outputs.append(output)
    output = torch.cat(outputs, dim=0)

    loss_test = F.cross_entropy(output, graph_labels[idx_test])
    acc_test = accuracy(output, graph_labels[idx_test])
    print(loss_test)
    print(acc_test)