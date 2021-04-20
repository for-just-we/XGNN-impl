from GraphGenerator import Generator
import copy

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import torch
import torch.optim as optim

lr = 0.01
b1 = 0.9
b2 = 0.99
hyp1 = 1
hyp2 = 2
max_gen_step = 10  # T = 10

candidate_set = ['C.4', 'N.5', 'O.2', 'F.1', 'I.7', 'Cl.7', 'Br.5']  # C.4表明碳原子的度不超过4
model_path = 'model/gcn_first.pth'

## 训练generator
def train_generator(c=0, max_nodes=5):
    g.c = c
    for i in range(max_gen_step):
        optimizer.zero_grad()
        G = copy.deepcopy(g.G)
        p_start, a_start, p_end, a_end, G = g.forward(G)

        Rt = g.calculate_reward(G)
        loss = g.calculate_loss(Rt, p_start, a_start, p_end, a_end, G)
        loss.backward()
        optimizer.step()

        if G['num_nodes'] > max_nodes:
            g.reset_graph()
        elif Rt > 0:
            g.G = G


## 生成图
def generate_graph(c=0, max_nodes=5):
    g.c = c
    g.reset_graph()

    for i in range(max_gen_step):
        G = copy.deepcopy(g.G)
        p_start, a_start, p_end, a_end, G = g.forward(G)
        Rt = g.calculate_reward(G)

        if G['num_nodes'] > max_nodes:
            return g.G
        elif Rt > 0:
            g.G = G

    return g.G

## 画图
def display_graph(G):
    G_nx = nx.from_numpy_matrix(np.asmatrix(G['adj'][:G['num_nodes'], :G['num_nodes']].numpy()))
    # nx.draw_networkx(G_nx)

    layout=nx.spring_layout(G_nx)
    nx.draw(G_nx, layout)

    coloring=torch.argmax(G['feat'],1)
    colors=['b','g','r','c','m','y','k']

    for i in range(7):
        nx.draw_networkx_nodes(G_nx,pos=layout,nodelist=[x for x in G_nx.nodes() if coloring[x]==i],node_color=colors[i])
        nx.draw_networkx_labels(G_nx,pos=layout,labels={x:candidate_set[i].split('.')[0] for x in G_nx.nodes() if coloring[x]==i})
    nx.draw_networkx_edges(G_nx,pos=layout,width=list(nx.get_edge_attributes(G_nx,'weight').values()))
    nx.draw_networkx_edge_labels(G_nx,pos=layout,edge_labels=nx.get_edge_attributes(G_nx, "weight"))

    plt.show()

if __name__ == '__main__':
    g = Generator(model_path = model_path, C = candidate_set, node_feature_dim=7 ,c=0, start=0)
    optimizer = optim.Adam(g.parameters(), lr=lr, betas=(b1, b2))

    for i in range(1, 10):
        ## 生成最多分别包括i个结点的图结构
        g.reset_graph()
        train_generator(c=1, max_nodes=i)
        to_display = generate_graph(c=1, max_nodes=i)
        display_graph(to_display)
        print(g.model(to_display['feat'], to_display['adj']))