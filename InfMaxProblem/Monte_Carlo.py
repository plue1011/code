# 隣接リストを入力とするモンテカルロシュミレーションに関する関数

import networkx as nx
import numpy as np
from tqdm import tqdm_notebook as tqdm
from collections import deque

# ------------------------------------ #

# モンテカルロシュミレーション
# 入力:隣接リスト(numpy)[[from_node, to_node, edge_prob],...], 隣接リストの長さ
# 出力:エッジ確率に従って、シュミレーション後に残った隣接リスト(numpy)[[from_node, to_node],...]
def live_edge_graph_edges(p, p_len):
    rand = np.random.uniform(0, 1, p_len)
    prob = p.T[2]
    l = np.where(rand < prob)[0]
    return np.array([[p[i][0], p[i][1]] for i in l])

# live_edge_graph_edges(network_np, len(network_np))
# array([[0.0000e+00, 4.0000e+00],
#        [0.0000e+00, 5.0000e+00],
#        [0.0000e+00, 7.0000e+00],
#        ...,
#        [7.5885e+04, 7.9000e+03],
#        [7.5885e+04, 1.6086e+04],
#        [7.5886e+04, 5.1414e+04]])

# ------------------------------------ #

# ------------------------------------ #

# シミュレーション(幅優先探索)
# アルゴリズム by https://todo314.hatenadiary.org/entry/20151013/1444720166
def IC_simulation(G, S):
    visited = {s:s for s in S}
    queue = deque(S)
    while queue:
        v = queue.popleft()
        weighted_edges = np.array([[i, G[v][i]["weight"]] for i in G[v]])
        if len(weighted_edges) != 0:
            prob = weighted_edges.T[1]
            rand = np.random.uniform(0, 1, len(prob))
            l = np.where(rand < prob)[0]

            out_node = weighted_edges[l].T[0]
            for u in out_node:
                if not (u in visited):
                    queue.append(u)
                    visited[u] = v
    return len(visited)

# 空の有向グラフを作成
# G = nx.DiGraph()
# 重み付きの枝を加える
# G.add_weighted_edges_from([[0,1,0.2],
#                            [0,2,0.3],
#                            [0,1,0.4]
#                           ])
# IC_simulation(G, [0])
# {訪問できる頂点:pre頂点,...}


# ------------------------------------ #

# ------------------------------------ #

# 試行回数を算出して、ICモデルでの期待影響サイズを返す関数
def approx_inf_size_IC(G, seed, epsi, delta):
    # 試行回数を算出する
    n = G.number_of_nodes()
    T = int(((n**2) / (epsi**2)) * np.log(1/delta)) + 1
    inf_size = sum([len(IC_simulation(G, seed)) for i in tqdm(range(T))])
    return inf_size / T

# ------------------------------------ #

# ------------------------------------ #

# 試行回数を指定して、ICモデルでの期待影響サイズを返す関数
def approx_inf_size_IC_T(G, seed, T):
    inf_size = sum([len(IC_simulation(G, seed)) for i in range(T)])
    return inf_size / T

# ------------------------------------ #

# ------------------------------------ #

def experiment_IC(G, seed, T=10000):
    inf_sum = 0
    inf_size_list = []
    for i in range(T):
        simulation = mc.IC_simulation(G, seed)
        
        # 影響数
        inf_sum += simulation
        inf_size_list.append(inf_sum / (i+1))
    return inf_size_list, inf_sum/T

# ------------------------------------ #