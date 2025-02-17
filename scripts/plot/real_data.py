import gzip
import numpy as np
import igraph as ig
import seaborn as sns
import matplotlib.pyplot as plt
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
    graph = Game(edges=edges, directed=False)
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

def get_robustness(graph, communities):
    robustness = list()
    for comm in tqdm(communities):
        initial_membership = np.zeros(graph.vcount(), dtype=int)
        for n in comm:
            initial_membership[n] = 1
        # robustness.append(graph.robustness(initial_membership))
        robustness.append(graph.robustness_per_community(initial_membership, only_community_of_index=1))
    return robustness

def get_accuracy(graph: Game, communities, noise=0.):
    accuracy_hedonic, accuracy_leiden = list(), list()
    for comm in tqdm(communities):
        initial_membership = np.zeros(graph.vcount(), dtype=int)
        for n in comm:
            if np.random.random() > noise:
                initial_membership[n] = 1
        print('running for hedonic')
        res_hedonic = graph.community_hedonic(initial_membership=initial_membership)
        print('running for leiden')
        res_leiden = graph.community_leiden(initial_membership=initial_membership, n_iterations=-1, resolution=graph.density())
        accuracy_hedonic.append(ig.compare_communities(res_hedonic, comm, method="rand"))
        accuracy_leiden.append(ig.compare_communities(res_leiden, initial_membership, method="rand"))
    return accuracy_hedonic, accuracy_leiden

file_path = '/Users/lucas/Databases/Hedonic/Networks/DBLP/com-dblp.ungraph.txt.gz'
graph = read_txt_gz_to_igraph(file_path)
print(graph.summary())

communities_path = '/Users/lucas/Databases/Hedonic/Networks/DBLP/com-dblp.top5000.cmty.txt.gz'
communities = read_communities(communities_path)[:10]

accuracy_hedonic, accuracy_leiden = get_accuracy(graph, communities)
# robustness_values = get_robustness(graph, communities)
# Creating subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plotting the violin plots
sns.violinplot(data=accuracy_hedonic, ax=axes[0])
sns.violinplot(data=accuracy_leiden, ax=axes[1])

# Adding labels and titles
axes[0].set_title('Accuracy Hedonic')
axes[1].set_title('Accuracy Leiden')
for ax in axes:
    ax.set_xlabel('Community')
    ax.set_ylabel('Accuracy')

# Display the plot
plt.tight_layout()
plt.show()
