import time
import gzip
import numpy as np
import igraph as ig
import pandas as pd
from hedonic import Game
from tqdm import tqdm


def read_txt_gz_to_igraph(file_path):
    edges = []
    with gzip.open(file_path, 'rt') as file:  # 'rt' mode for text mode reading
        for line in file:
            if line.startswith('#'):  # Skip comment lines
                continue
            # Split line into source and target node IDs and convert to integers
            nodes = list(map(int, line.strip().split()))
            if len(nodes) == 2:
                source, target = nodes
                edges.append((source, target))
            else:
                print(line, nodes)
    # Assuming the file contains an edge list with each line as 'source target'
    graph = ig.Graph(edges=edges, directed=False)
    return graph


def read_communities(file_path, mode='list_of_communities'):
    communities = []
    with gzip.open(file_path, 'rt') as file:  # 'rt' mode for text mode reading
        for line in file:
            if mode == 'list_of_communities':
                nodes = list(map(int, line.strip().split()))
                communities.append(nodes)
            elif mode == 'node_labels':
                node, community = map(int, line.strip().split())
                communities.append((node, community))
    if mode == 'node_labels':
        communities = dict()
        for node, community in communities:
            try:
                communities[community].add(node)
            except:
                communities[community] = set({node})
    return communities



def get_accuracy(graph, communities, noise=0.):
    accuracy_hedonic, accuracy_leiden = list(), list()
    times_hedonic, times_leiden = list(), list()
    for comm in tqdm(communities):
        initial_membership = np.zeros(graph.vcount(), dtype=int)
        for n in comm:
            if np.random.random() > noise:
                initial_membership[n] = 1
        start = time.time()
        res_hedonic = graph.community_leiden(hedonic=True, initial_membership=initial_membership, n_iterations=-1, resolution=graph.density())
        end = time.time()
        times_hedonic.append(end - start)
        start = time.time()
        res_leiden = graph.community_leiden(initial_membership=initial_membership, n_iterations=-1, resolution=graph.density())
        end = time.time()
        times_leiden.append(end - start)
        accuracy_hedonic.append(ig.compare_communities(res_hedonic, initial_membership, method="rand"))
        accuracy_leiden.append(ig.compare_communities(res_leiden, initial_membership, method="rand"))
    return accuracy_hedonic, accuracy_leiden, times_hedonic, times_leiden


def main():
    file_path = '/Users/lucas/Databases/Hedonic/Networks/DBLP/com-dblp.ungraph.txt.gz'
    graph = read_txt_gz_to_igraph(file_path)
    g = Game(graph)
    print(g.summary())
    # communities_path = '/Users/lucas/Databases/Hedonic/Networks/DBLP/com-dblp.top5000.cmty.txt.gz'
    communities_path = '/Users/lucas/Databases/Hedonic/Networks/DBLP/com-dblp.all.cmty.txt.gz'
    communities = read_communities(communities_path)
    robustness = [g.evaluate_community_stability(communities[i]) for i in tqdm(range(len(communities)))]
    df = pd.DataFrame(robustness)
    df.index.name = 'community_index'
    output_path = communities_path.replace(communities_path.split('/')[-1], 'fraction_stability.csv')
    df.to_csv(output_path)
    
    
    # accuracy_hedonic, accuracy_leiden, times_hedonic, times_leiden = get_accuracy(graph, communities[:10], noise=0.)
    # print(accuracy_hedonic)
    # print(accuracy_leiden)
    # print(times_hedonic)
    # print(times_leiden)




if __name__ == "__main__":
    main()

