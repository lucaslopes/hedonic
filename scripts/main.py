import igraph as ig
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from networkx.generators.community import stochastic_block_model as SBM
from hedonic import community_hedonic

#################################################

def networkx_to_igraph(g):
    return ig.Graph(g.number_of_nodes(), g.edges())

def probs_matrix(n_communities, p, q):
    probs = np.full((n_communities, n_communities), q) # fill with q
    np.fill_diagonal(probs, p) # fill diagonal with p
    return probs # return probability matrix

def ppm_igraph(n_communities, community_size, p_in, multiplier):
    block_sizes = np.full(n_communities, community_size) # all blocks are same size
    p_out = p_in * multiplier # probability of edge between communities
    p = probs_matrix(n_communities, p_in, p_out) # probability matrix
    h = SBM(block_sizes, p) # generate networkx graph
    g = networkx_to_igraph(h) # convert to igraph
    return g # return igraph graph

def cap_n_communities(g, partition, max_n_comm):
    new_partition = None
    if len(partition.sizes()) > max_n_comm:
        new_membership = [m if (
            m < max_n_comm
        ) else (
            max_n_comm - 1
        ) for m in partition.membership]
        new_partition = ig.clustering.VertexClustering(g, new_membership)
    return new_partition if new_partition else partition

def accuracy(partition, ground_truth):
    n_correct = 0
    for i in range(partition.n):
        for j in range(partition.n):
            pair_ij = partition.membership[i] == partition.membership[j]
            pair_gt = ground_truth.membership[i] == ground_truth.membership[j]
            if pair_ij == pair_gt:
                n_correct += 1
    acc = n_correct / partition.n ** 2
    return (acc - .5) / .5

def generate_probability_array(length):
    powers = np.arange(length-1, -1, -1)
    result = 1 / np.power(2, powers)
    return result

#################################################

def run_experiment(
        length_of_probability_array = 3, # Define length of probability array
        community_size = 100, # Define block sizes (all blocks are same size)
        number_of_communities = 2, # Define number of communities
    ):
    results = []
    ps_in = generate_probability_array(length_of_probability_array) # Define probability of edge within community
    multipliers = generate_probability_array(length_of_probability_array) # `multiplier = p_out / p_in` and p_out is the probability of edge between communities
    gt_membership = np.concatenate((np.zeros(community_size), np.ones(community_size)))
    ground_truth = ig.clustering.VertexClustering(g, gt_membership)
    for p_in in ps_in:
        for mult in multipliers:
            g = ppm_igraph(number_of_communities, community_size, p_in, mult)
            # Get the communities
            comms_ml = g.community_multilevel(resolution=g.density())
            comms_ld = g.community_leiden(resolution=g.density())
            comms_hd = community_hedonic(g, resolution=g.density())

            cap_comms_ml = cap_n_communities(g, comms_ml, number_of_communities)
            cap_comms_ld = cap_n_communities(g, comms_ld, number_of_communities)
            acc = accuracy(method['Partition'], ground_truth)
            
            # Define the methods
            methods = [
                {
                    'Method': 'community multilevel',
                    'Partition': cap_comms_ml
                },
                {
                    'Method': 'community leiden',
                    'Partition': cap_comms_ld
                },
                {
                    'Method': 'community hedonic',
                    'Partition': comms_hd
                }
            ]

            # Calculate accuracy for each method
            for method in methods:
                results.append({
                    'Method': method['Method'],
                    'p_in': p_in,
                    'p_out': p_in * mult,
                    'Multiplier': mult,
                    'Accuracy': accuracy(method['Partition'], ground_truth)
                })

    # Create a pandas dataframe
    df = pd.DataFrame(results)

    return df

#################################################

def plot_heatmaps(df):
    methods = df["Method"].unique()
    fig, axes = plt.subplots(1, len(methods), figsize=(15, 5))
    for i, method in enumerate(methods):
        method_df = df[df["Method"] == method]
        method_data = method_df.pivot_table(index="p_in", columns="Multiplier", values="Accuracy")
        
        ax = sns.heatmap(method_data, ax=axes[i], cmap="Greens", vmin=0, vmax=1) # , annot=True, fmt=".2f",
        ax.set_title(method)
        ax.set_xlabel("multiplier")
        ax.set_ylabel("p_in")
    
    return fig, axes

#################################################

df = run_experiment(10)
fig, axes = plot_heatmaps(df)
plt.tight_layout()
plt.show()