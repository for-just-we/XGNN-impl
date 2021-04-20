import random
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from Model import GraphConvolution, GCN

rollout = 10
MAX_NUM_NODES = 28 # for mutag
random.seed(200)

class Generator(nn.Module):
    def __init__(self, model_path: str, C: list, node_feature_dim: int ,num_class = 2, c=0, hyp1=1, hyp2=2, start=None, nfeat=7, dropout=0.1):
        """
        :param C: Candidate set of nodes (list)
        :param start: Starting node (defaults to randomised node)
        """
        super(Generator, self).__init__()
        self.nfeat = nfeat
        self.dropout = dropout
        self.c = c # c为要指定生成类别c的图

        self.fc = nn.Linear(nfeat, 8)
        self.gc1 = GraphConvolution(8, 16)
        self.gc2 = GraphConvolution(16, 24)
        self.gc3 = GraphConvolution(24, 32)

        # MLP1
        # 2 FC layers with hidden dimension 16
        self.mlp1 = nn.Sequential(nn.Linear(32, 16), nn.Linear(16, 1))

        # MLP2
        # 2 FC layers with hidden dimension 24
        self.mlp2 = nn.Sequential(nn.Linear(64, 24), nn.Linear(24, 1))

        # Hyperparameters
        self.hyp1 = hyp1
        self.hyp2 = hyp2
        self.candidate_set = C

        # Default starting node (if any)
        if start is not None:
            self.start = start
            self.random_start = False
        else:
            self.start = random.choice(np.arange(0, len(self.candidate_set)))
            self.random_start = True

        # Load GCN for calculating reward
        self.model = GCN(nfeat=node_feature_dim,
                         nclass=num_class,
                         dropout=dropout)

        self.model.load_state_dict(torch.load(model_path))
        for param in self.model.parameters():
            param.requires_grad = False

        self.reset_graph()

    def reset_graph(self):
        """
        Reset g.G to default graph with only start node， 生成一个只有1个结点的图
        """
        if self.random_start == True:
            self.start = random.choice(np.arange(0, len(self.candidate_set)))

        # 初始图除了第1个结点全被mask，这里由于邻接矩阵的边长为MAX_NUM_NODES + len(self.candidate_set)，所以mask的不仅为候选集结点，还有图中的所以虚结点
        mask_start = torch.BoolTensor(
            [False if i == 0 else True for i in range(MAX_NUM_NODES + len(self.candidate_set))])

        adj = torch.zeros((MAX_NUM_NODES + len(self.candidate_set), MAX_NUM_NODES + len(self.candidate_set)),
                          dtype=torch.float32)   # 这里adj shape为 [MAX_NUM_NODES + len(self.candidate_set), MAX_NUM_NODES + len(self.candidate_set)] 中间可能有空结点

        feat = torch.zeros((MAX_NUM_NODES + len(self.candidate_set), len(self.candidate_set)), dtype=torch.float32)
        feat[0, self.start] = 1
        feat[np.arange(-len(self.candidate_set), 0), np.arange(0, len(self.candidate_set))] = 1

        degrees = torch.zeros(MAX_NUM_NODES)

        self.G = {'adj': adj, 'feat': feat, 'degrees': degrees, 'num_nodes': 1, 'mask_start': mask_start}

    ## 计算Gt->Gt+1
    def forward(self, G_in):
        ## G_in为 Gt
        G = copy.deepcopy(G_in)

        x = G['feat'].detach().clone() # Gt的特征矩阵
        adj = G['adj'].detach().clone() # Gt的邻接矩阵

        ## 对应 X = GCNs(Gt​,C)
        x = F.relu6(self.fc(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu6(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu6(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu6(self.gc3(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)

        ## pt,start​=Softmax(MLPs(X))
        p_start = self.mlp1(x)
        p_start = p_start.masked_fill(G['mask_start'].unsqueeze(1), 0)
        p_start = F.softmax(p_start, dim=0)
        a_start_idx = torch.argmax(p_start.masked_fill(G['mask_start'].unsqueeze(1), -1))

        ## pt,end​=Softmax(MLPs([X,x^start​))
        # broadcast
        x1, x2 = torch.broadcast_tensors(x, x[a_start_idx])
        x = torch.cat((x1, x2), 1)  # cat increases dim from 32 to 64

        # 计算maskt,end，除了候选集和Gt结点中未被选为初始结点的结点之外，其它均被mask
        mask_end = torch.BoolTensor([True for i in range(MAX_NUM_NODES + len(self.candidate_set))])
        mask_end[MAX_NUM_NODES:] = False
        mask_end[:G['num_nodes']] = False
        mask_end[a_start_idx] = True

        p_end = self.mlp2(x)
        p_end = p_end.masked_fill(mask_end.unsqueeze(1), 0)
        p_end = F.softmax(p_end, dim=0)
        a_end_idx = torch.argmax(p_end.masked_fill(mask_end.unsqueeze(1), -1))

        # Return new G
        # If a_end_idx is not masked, node exists in graph, no new node added
        if G['mask_start'][a_end_idx] == False:
            G['adj'][a_end_idx][a_start_idx] += 1
            G['adj'][a_start_idx][a_end_idx] += 1

            # Update degrees
            G['degrees'][a_start_idx] += 1
            G['degrees'][G['num_nodes']] += 1
        else:
            # Add node
            G['feat'][G['num_nodes']] = G['feat'][a_end_idx]
            # Add edge
            G['adj'][G['num_nodes']][a_start_idx] += 1
            G['adj'][a_start_idx][G['num_nodes']] += 1
            # Update degrees
            G['degrees'][a_start_idx] += 1
            G['degrees'][G['num_nodes']] += 1

            # Update start mask
            G_mask_start_copy = G['mask_start'].detach().clone()
            G_mask_start_copy[G['num_nodes']] = False
            G['mask_start'] = G_mask_start_copy

            G['num_nodes'] += 1

        return p_start, a_start_idx, p_end, a_end_idx, G


    ### reward函数
    def calculate_reward(self, G_t_1):
        """
        Rtr     Calculated from graph rules to encourage generated graphs to be valid
                1. Only one edge to be added between any two nodes
                2. Generated graph cannot contain more nodes than predefined maximum node number
                3. (For chemical) Degree cannot exceed valency
                If generated graph violates graph rule, Rtr = -1

        Rtf     Feedback from trained model
        """

        rtr = self.check_graph_rules(G_t_1)

        rtf = self.calculate_reward_feedback(G_t_1)
        rtf_sum = 0
        for m in range(rollout):
            p_start, a_start, p_end, a_end, G_t_1 = self.forward(G_t_1)
            rtf_sum += self.calculate_reward_feedback(G_t_1)
        rtf = rtf + rtf_sum * self.hyp1 / rollout

        return rtf + self.hyp2 * rtr

    def calculate_reward_feedback(self, G_t_1):
        """
        p(f(G_t_1) = c) - 1/l
        where l denotes number of possible classes for f
        """
        f = self.model(G_t_1['feat'], G_t_1['adj'])
        return f[self.c] - 1 / len(f)


    ## graph rules
    def check_graph_rules(self, G_t_1):
        """
        For mutag, node degrees cannot exceed valency
        """
        idx = 0

        for d in G_t_1['degrees']:
            if d is not 0:
                node_id = torch.argmax(G_t_1['feat'][idx])  # Eg. [0, 1, 0, 0] -> 1
                node = self.candidate_set[node_id]  # Eg ['C.4', 'F.2', 'Br.7'][1] = 'F.2'
                max_valency = int(node.split('.')[1])  # Eg. C.4 -> ['C', '4'] -> 4

                # If any node degree exceeds its valency, return -1
                if max_valency < d:
                    return -1

        return 0


    ## 计算loss
    def calculate_loss(self, Rt, p_start, a_start, p_end, a_end, G_t_1):
        """
        Calculated from cross entropy loss (Lce) and reward function (Rt)
        where loss = -Rt*(Lce_start + Lce_end)
        """

        Lce_start = F.cross_entropy(torch.reshape(p_start, (1, 35)), a_start.unsqueeze(0))
        Lce_end = F.cross_entropy(torch.reshape(p_end, (1, 35)), a_end.unsqueeze(0))

        return -Rt * (Lce_start + Lce_end)