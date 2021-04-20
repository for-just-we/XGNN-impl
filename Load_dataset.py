import numpy as np
import scipy.sparse as sp
import torch


# MUTAG数据集特征，188个图，总共3371个结点，7442条边，为无向图

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)



def load_split_MUTAG_data(path="datas/MUTAG/", dataset="MUTAG_", split_train=0.7, split_val=0.15):
    """Load MUTAG data """
    print('Loading {} dataset...'.format(dataset))

    # 加载图的标签
    graph_labels = np.genfromtxt("{}{}graph_labels.txt".format(path, dataset),
                           dtype=np.dtype(int))
    graph_labels = encode_onehot(graph_labels)  # (188, 2)
    graph_labels = torch.LongTensor(np.where(graph_labels)[1]) # (188, 1)


    # 图结点的索引号
    graph_idx = np.genfromtxt("{}{}graph_indicator.txt".format(path, dataset),
                              dtype=np.dtype(int))

    graph_idx = np.array(graph_idx, dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(graph_idx)} # key, value表示第key个图的起始结点索引号为value
    length = len(idx_map.keys()) # 总共有多少个图
    num_nodes = [idx_map[n] - idx_map[n - 1] if n - 1 > 1 else idx_map[n] for n in range(1, length + 1)] # 一个长度188的list，表示没个图有多少个结点
    max_num_nodes = max(num_nodes) # 最大的一个图有多少个结点
    features_list = []
    adj_list = []
    prev = 0

    # 结点的标签
    nodeidx_features = np.genfromtxt("{}{}node_labels.txt".format(path, dataset), delimiter=",",
                                     dtype=np.dtype(int))
    node_features = np.zeros((nodeidx_features.shape[0], max(nodeidx_features) + 1))
    node_features[np.arange(nodeidx_features.shape[0]), nodeidx_features] = 1

    # 边信息
    edges_unordered = np.genfromtxt("{}{}A.txt".format(path, dataset), delimiter=",",
                                    dtype=np.int32)

    # 边的标签
    edges_label = np.genfromtxt("{}{}edge_labels.txt".format(path, dataset), delimiter=",",
                                dtype=np.int32)  # shape = (7442,)

    # 生成邻接矩阵A，该邻接矩阵包括了数据集中所有的边
    adj = sp.coo_matrix((edges_label, (edges_unordered[:, 0] - 1, edges_unordered[:, 1] - 1)))

    # 论文里A^=(D~)^0.5 A~ (D~)^0.5这个公式
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    node_features = normalize(node_features)
    adj = normalize(adj + sp.eye(adj.shape[0])) # 对应公式A~=A+IN
    adj = adj.todense()

    for n in range(1, length + 1):
        # entry为第n个图的特征矩阵X
        entry = np.zeros((max_num_nodes, max(nodeidx_features) + 1))
        entry[:idx_map[n] - prev] = node_features[prev:idx_map[n]]
        entry = torch.FloatTensor(entry)
        features_list.append(entry.tolist())

        # entry为第n个图的邻接矩阵A
        entry = np.zeros((max_num_nodes, max_num_nodes))
        entry[:idx_map[n] - prev, :idx_map[n] - prev] = adj[prev:idx_map[n], prev:idx_map[n]]
        entry = torch.FloatTensor(entry)
        adj_list.append(entry.tolist())

        prev = idx_map[n] # prev为下个图起始结点的索引号

    num_total = max(graph_idx)
    num_train = int(split_train * num_total)
    num_val = int((split_train + split_val) * num_total)

    if (num_train == num_val or num_val == num_total):
        return

    features_list = torch.FloatTensor(features_list)
    adj_list = torch.FloatTensor(adj_list)

    idx_train = range(num_train)
    idx_val = range(num_train, num_val)
    idx_test = range(num_val, num_total)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # 返回值一次为 188个图的邻接矩阵列表  188个图的特征矩阵列表  188个图的label， 每个图的起始结点索引号， 训练集索引号，
    # 验证集索引号， 测试集索引号
    return adj_list, features_list, graph_labels, idx_map, idx_train, idx_val, idx_test