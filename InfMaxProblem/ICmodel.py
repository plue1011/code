import itertools
import random
import numpy as np
import networkx as nx
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra #フィボナッチヒープを使用している
import math
from tqdm import tqdm_notebook as tqdm


#--------------------------------#

def make_random_graph(V_size, E_size):
    E = np.zeros(V_size**2)
    p = np.zeros(V_size**2)
    
    index_list = [i for i in range(V_size**2) if i%(V_size+1) != 0]
    
    for i in random.sample(index_list, k=E_size):
        E[i] = 1
        p[i] = random.randint(1,100)/100
    E = E.reshape(V_size, V_size)
    p = p.reshape(V_size, V_size)
    
    return [E, p]

#--------------------------------#

#--------------------------------#

# 1であるところだけ分岐して順列を全列挙
def permutation_01(l):
    l_result = []
    # パターン数:n
    n = int(sum(l))
    zero = np.array([0 for i in range(2**n)])
    l_01 = np.array(list(itertools.product([0,1], repeat=n)))
    l_01_T = l_01.T
    j = 0
    for i in l:
        if i == 0:
            l_result += [zero]
        else:
            l_result += [l_01_T[j]]
            j += 1
    return np.array(l_result).T

# permutation_01([0,1,0,1,1])
# array([[0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 1],
#        [0, 0, 0, 1, 0],
#        [0, 0, 0, 1, 1],
#        [0, 1, 0, 0, 0],
#        [0, 1, 0, 0, 1],
#        [0, 1, 0, 1, 0],
#        [0, 1, 0, 1, 1]])

#--------------------------------#

#--------------------------------#

# ネットワークの全パターンlive_edgeグラフの作成
# live_edge_set(枝集合)
def live_edge_set(E_i):
    l_list = []
    for l in E_i:
        l_list.append(permutation_01(l))
    return np.array(list(itertools.product(*l_list))) 

# live_edge_set(E[0])
# array([[[0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0]],
#
#        [[0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0],
#         [0, 0, 0, 1, 0]],
#
#        ...,

#--------------------------------#

#--------------------------------#

# そのグラフになる確率
# live_edge_prob(元ネットワーク, live_edgeグラフ, 枝確率)
def live_edge_prob(original, live_edge, p):
    n = len(live_edge)
    # 枝がある確率
    prob_T = live_edge * p
    prob_T = np.where(prob_T == 0, 1, prob_T)
    # 総積を求める
    prob_T = np.prod(prob_T)
    
    # 枝がない確率
    prob_F = (original - live_edge) * (np.ones((n,n))-p)
    prob_F = np.where(prob_F == 0, 1, prob_F)
    # 総積を求める
    prob_F = np.prod(prob_F)
    
    return prob_T * prob_F

# live_edge_prob(E[0], live_edge_set(E[0])[10000], p[0])


# 確率計算のため、数が小さくなっていくためlogで計算する必要があるかもしれない

#--------------------------------#

#--------------------------------#

# live_edgeグラフ上でシードから到達可能な頂点数
def reachable_num(live_edge, seed):
    # live_edgeグラフの各頂点までの最短距離を求める
    short_path = dijkstra(csr_matrix(live_edge), indices=seed)
    
    # inf(到達できない)を0に変換している
    short_path = np.where(np.float('inf') == short_path, 0, short_path)
    
    # seedの到達可能な頂点をor演算することで重複を消している
    reachable_node_num = np.logical_or.reduce(short_path)
    
    # seedの頂点をFalseにしている(重複しないように)
    reachable_node_num[seed] = False
    
    # seedの頂点数を足して出力
    return len(set(seed)) + sum(reachable_node_num)

# reachable_num(live_edge_set(E[0])[10002], [0,1])
# 5

#--------------------------------#

#--------------------------------#

# live_edgeグラフ上でシードから到達可能な頂点数
def reachable_num_nx(live_edge, seed):
    # グラフ作成
    G = nx.DiGraph(live_edge)
    
    # 到達可能な頂点集合をシードごとに和集合して求める
    reach_set = set([])
    for s in seed:
        reach_set |= set(nx.dfs_preorder_nodes(G,source=s))
    return len(reach_set)

# reachable_num_nx(live_edge_set(E[0])[10002], [0,1])
# 5

#--------------------------------#

#--------------------------------#

# 影響数の期待値
# expected_influence_num(ネットワーク, 枝確率, シード)
def expected_influence_num(E, p, seed):
    # live_edgeグラフを全列挙
    live_edge_sets = live_edge_set(E)
    
    expected_num = 0
    for l_e_s in tqdm(live_edge_sets):
        # そのlive_edgeグラフになる確率 * そのグラフ上でシードからの到達頂点数の総和
        expected_num += live_edge_prob(E, l_e_s, p) * reachable_num(l_e_s, seed)
    return expected_num

# for i in range(5):
#     print(expected_influence_num(E[0], p[0], [i]))
# 3.710841757347267
# 2.7803227401623367
# 2.9310440468303938
# 3.8615958011324687
# 4.544981497080014

#--------------------------------#

#--------------------------------#

def expected_influence_num_nx(E, p, seed):
    # live_edgeグラフを全列挙
    live_edge_sets = live_edge_set(E)
    
    expected_num = 0
    for l_e_s in tqdm(live_edge_sets):
        # そのlive_edgeグラフになる確率 * そのグラフ上でシードからの到達頂点数の総和
        expected_num += live_edge_prob(E, l_e_s, p) * reachable_num_nx(l_e_s, seed)
    return expected_num

#--------------------------------#

#--------------------------------#

# Independet Cascade Modelにおいて、考えられるlive_edgeグラフの枝集合を返す
# sub_gragh_edge(枝確率集合)
def live_edge_gragh_edges(p):
    prop_edge = []
    for e_prop_set in p:
        prop_edge_temp = []
        for e_prop in e_prop_set:
            if e_prop > random.random():
                prop_edge_temp.append(1)
            else:
                prop_edge_temp.append(0)
        prop_edge.append(prop_edge_temp)
    return prop_edge

# live_edge_gragh_edges(p[0])
# [[0, 0, 0, 1, 0],
#  [0, 0, 0, 0, 0],
#  [0, 0, 0, 0, 0],
#  [1, 0, 1, 0, 0],
#  [1, 0, 1, 0, 0]]

#--------------------------------#

#--------------------------------#

# 影響数の期待値(近似)
# approxim_expected_influence_num(枝確率, シード, ε, δ)
def approxim_expected_influence_num(p, seed, epsi, delta):
    # 頂点数
    n = len(p)
    # 試行回数を算出する
    T = int(((n**2) / (epsi**2)) * np.log(1/delta)) + 1
    
    # 各回のシュミレーションの結果の和が格納される
    X = 0
    # T回シュミレーションしていく
    for i in tqdm(range(T)):
        # live_edgeグラフを作る
        live_edge_sets = np.array(live_edge_gragh_edges(p))
        X += reachable_num(live_edge_sets, seed)
    expected_num = X / T
    return expected_num

#--------------------------------#

#--------------------------------#



#--------------------------------#

#--------------------------------#


#--------------------------------#

#--------------------------------#


#--------------------------------#

#--------------------------------#


#--------------------------------#

#--------------------------------#


#--------------------------------#

#--------------------------------#


#--------------------------------#


#--------------------------------#

#--------------------------------#

#--------------------------------#


#--------------------------------#

#--------------------------------#


#--------------------------------#

#--------------------------------#


#--------------------------------#



#--------------------------------#



#--------------------------------#



#--------------------------------#



#--------------------------------#



#--------------------------------#



#--------------------------------#



#--------------------------------#



#--------------------------------#

