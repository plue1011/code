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
    return np.array([[p[i][0], p[i][1]] for i in range(p_len) if rand[i] < p[i][2]])

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
        out_node = G.successors(v)
        for u in out_node:
            if not (u in visited):
                coin = np.random.uniform(0,1)
                if G[v][u]["weight"] > coin:
                    queue.append(u)
                    visited[u] = v
    return visited

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

