import numpy as np
import scipy.sparse as sp
from Load_dataset import encode_onehot, normalize

import torch

src_path = 'datas/MUTAG/MUTAG_{}'
split_train=0.7
split_val=0.15

if __name__ == '__main__':
    nodeidx_features = np.genfromtxt(src_path.format('node_labels.txt'), delimiter=",",
                                     dtype=np.dtype(int))
    node_features = np.zeros((nodeidx_features.shape[0], max(nodeidx_features) + 1))
    node_features[np.arange(nodeidx_features.shape[0]), nodeidx_features] = 1


    graph_labels = np.genfromtxt(src_path.format('graph_labels.txt'), dtype=np.dtype(int))
    graph_labels = encode_onehot(graph_labels)
    graph_labels = torch.LongTensor(np.where(graph_labels)[1])


    graph_idx = np.genfromtxt(src_path.format('graph_indicator.txt'),dtype=np.dtype(int))
    graph_idx = np.array(graph_idx, dtype=np.int32)


    edges_unordered = np.genfromtxt(src_path.format('A.txt'), delimiter=",",
                                    dtype=np.int32)  # (7442,2)

    edges_label = np.genfromtxt(src_path.format('edge_labels.txt'), delimiter=",",
                                dtype=np.int32)

    # 邻接矩阵
    adj = sp.coo_matrix((edges_label, (edges_unordered[:, 0] - 1, edges_unordered[:, 1] - 1)))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) # (3371, 3371)

    idx_map = {j: i for i, j in enumerate(graph_idx)} # key, value表示第key个图的起始结点索引号为value
    length = len(idx_map.keys())  # 总共有多少个图 , 188
    num_nodes = [idx_map[n] - idx_map[n - 1] if n - 1 > 1 else idx_map[n] for n in range(1, length + 1)]  # 一个长度188的list，表示没个图有多少个结点
    max_num_nodes = max(num_nodes) # 最大的一个图有多少个结点 实际29
    features_list = []
    adj_list = []
    prev = 0

    node_features = normalize(node_features) # (3371, 7)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = adj.todense()

    for n in range(1, length + 1):
        # entry为图的特征矩阵X
        entry = np.zeros((max_num_nodes, max(nodeidx_features) + 1))
        entry[:idx_map[n] - prev] = node_features[prev:idx_map[n]]
        entry = torch.FloatTensor(entry)
        features_list.append(entry)

        # entry为图的邻接矩阵A
        entry = np.zeros((max_num_nodes, max_num_nodes))
        entry[:idx_map[n] - prev, :idx_map[n] - prev] = adj[prev:idx_map[n], prev:idx_map[n]]
        entry = torch.FloatTensor(entry)
        adj_list.append(entry)

        prev = idx_map[n] # prev为下个图起始结点的索引号

    num_total = max(graph_idx)
    num_train = int(split_train * num_total)
    num_val = int((split_train + split_val) * num_total)

    idx_train = range(num_train)
    idx_val = range(num_train, num_val)
    idx_test = range(num_val, num_total)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)


    print(graph_labels[idx_train])
    print(graph_labels[idx_val])

