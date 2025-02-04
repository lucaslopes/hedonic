import pickle
import numpy as np
import os
import pandas as pd
import utils
from tqdm import tqdm
from hedonic import Game
from collections import Counter


def count_unique_fractions(base_dir):
    unique_counts = {}
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                if 'fractions' in df.columns:
                    unique_counts[file_path] = df['fractions'].nunique()
    return unique_counts

base_dir = '/Users/lucas/Databases/Hedonic/PHYSA_2000/networks/2C_2000N'
paths = []
for root, _, files in os.walk(base_dir):
    for file in files:
        if file.endswith('.pkl'):
            file_path = os.path.join(root, file)
            paths.append(file_path)

V = 2000
membership = [0 if i < int(V/2) else 1 for i in range(V)]

for seed in range(100):
    n_coms = 10
    c_size = 500
    V = int(n_coms * c_size)
    membership = [i // c_size for i in range(V)]
    g = utils.generate_graph(n_coms, c_size, .01, .8, seed)
    nodes_info = g.get_nodes_info(membership)
    count = Counter([Game.classify_node_satisfaction(info, membership[node]) for node, info in nodes_info.items()])
    print(count)

for file_path in tqdm(paths):        
    with open(file_path, 'rb') as f:
        g = pickle.load(f)
    nodes_info = g.get_nodes_info(membership)
    for node, info in nodes_info.items():
        node_satisfaction = Game.classify_node_satisfaction(info, membership[node])
        if node_satisfaction == 'relatively_satisfied':
            print(node, info)

    # resolutions, fractions, robustness = g.resolution_spectrum(membership, np.linspace(0, 1, 1001))


    # nodes_info = g.get_nodes_info(membership)
    # for res in resolutions:
    #     satisfied = 0
    #     for node, community in enumerate(membership):
    #         c0 = nodes_info[node][0]['friends'] - res * nodes_info[node][0]['strangers']
    #         c1 = nodes_info[node][1]['friends'] - res * nodes_info[node][1]['strangers']
    #         if (community == 0 and c0 >= c1) or (community == 1 and c1 >= c0):
    #             satisfied += 1
    #         else:
    #             print(res, node, community, c0, c1, nodes_info[node])
    #     print(f'{res:.2f} {satisfied/V:.2f}')