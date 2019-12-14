# 隣接リストを入力とするモンテカルロシュミレーションに関する関数

import networkx as nx
import numpy as np
from tqdm import tqdm_notebook as tqdm

# ------------------------------------ #

# モンテカルロシュミレーション
# 入力:隣接リスト(numpy)[[from_node, to_node, edge_prob],...], 隣接リストの長さ
# 出力:エッジ確率に従って、シュミレーション後に残った隣接リスト(numpy)[[from_node, to_node],...]
def live_edge_graph_edges(p, p_len):
    rand = np.random.uniform(0, 1, p_len)
    return np.array([[p[i][0], p[i][1]] for i in range(p_len) if rand[i] > p[i][2]])

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

# シュミレーション後のグラフについて、影響数を計算
# 入力:重みなし隣接リスト, シード集合
# 出力:(int)影響数
def reach_node_size(live_edge, seed):
    # live_edgeグラフ作成
    H = nx.DiGraph()
    H.add_edges_from(live_edge)
    
    # 到達可能な頂点集合をシードごとに和集合して求める
    reach_set = set([])
    for s in seed:
        # たどり着いているノードがシードでない場合だけ計算する
        if s not in reach_set:
            reach_set |= set(nx.dfs_preorder_nodes(H,source=s))
    return len(reach_set)

# simulation = live_edge_graph_edges(network_np, len(network_np))
# reach_node_size(simulation, [i for i in range(50)])
# 23440

# ------------------------------------ #

# ------------------------------------ #

# 影響数の期待値(近似)
# ε近似を確率1-δで達成する
# 入力:枝確率, シード, 近似率ε, δ)
# 出力:(int)近似値
def approx_expect_inf_size(p, seed, epsi, delta):
    # グラフの作成
    G = nx.DiGraph()
    G.add_weighted_edges_from(p)
    
    # 頂点数
    n = G.number_of_nodes()
    
    # 試行回数を算出する
    T = int(((n**2) / (epsi**2)) * np.log(1/delta)) + 1
        
    # 各回のシュミレーションの結果の和が格納される
    X = 0
    len_p = len(p)
    # T回シュミレーションしていく
    for i in tqdm(range(T)):
        # live_edgeグラフを作る
        live_edge = live_edge_graph_edges(p, len_p)
        X += reach_node_size(live_edge, seed)
    expected_num = X / T
    return expected_num

# ------------------------------------ #

# ------------------------------------ #

# シュミレーションの結果的に変化していない場合は、回数を減らす関数
# 影響数の期待値(近似)
# 入力:枝確率, シード, ε, δ, 変化数)
# 出力:ヒューリスティック的な値
def approx_expect_inf_size_heuris(p, seed, epsi, delta, change):
    # グラフの作成
    G = nx.DiGraph()
    G.add_weighted_edges_from(p)
    
    # 頂点数
    n = G.number_of_nodes()
    
    # 試行回数を算出する
    T = int(((n**2) / (epsi**2)) * np.log(1/delta)) + 1
        
    # 各回のシュミレーションの結果の和が格納される
    X = 0
    len_p = len(p)
    # T回シュミレーションしていく
    for i in tqdm(range(T)):
        # live_edgeグラフを作る
        live_edge = live_edge_graph_edges(p, len_p)
        X += reach_node_size(live_edge, seed)
        
        # 前回と比較して変化なし
        # if (i != 0) and (abs(((X / (i+1)) - expected_num) / expected_num) < rate_change):
        if (i != 0) and (abs(((X / (i+1)) - expected_num) < change)):    
            return expected_num
        
        expected_num = X / (i+1)
    return expected_num

# ------------------------------------ #