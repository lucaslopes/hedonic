import igraph as ig
import numpy as np
from networkx.generators.community import stochastic_block_model as SBM
from hedonic import community_hedonic

#################################################

# Parameters
community_size = 500 # Define block sizes
p_in = .1
multiplier = .25

#################################################

def probs_matrix(n_communities, p, q):
    probs = np.full((n_communities, n_communities), q)
    np.fill_diagonal(probs, p)
    return probs

def ppm_igraph(community_size, p_in, multiplier):
    # Define edge probabilities between blocks
    block_sizes = [community_size] * 2
    p_out = p_in * multiplier
    p = probs_matrix(len(block_sizes), p_in, p_out)
    h = SBM(block_sizes, p)
    g = ig.Graph.TupleList(h.edges()) # convert to igraph
    return g

def cap_n_communities(g, partition, max_n_comm=2):
    new_partition = None
    if len(partition.sizes()) > max_n_comm:
        new_membership = [m if (
            m < max_n_comm
        ) else (max_n_comm - 1) for m in partition.membership]
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

#################################################

g = ppm_igraph(community_size, p_in, multiplier)

# Get the communities
comms_ml = g.community_multilevel()
comms_ld = g.community_leiden(resolution=g.density())
comms_hd = community_hedonic(g)

cap_comms_ml = cap_n_communities(g, comms_ml)
cap_comms_ld = cap_n_communities(g, comms_ld)

gt_membership = np.concatenate(
    (np.zeros(community_size), np.ones(community_size)))
ground_truth = ig.clustering.VertexClustering(g, gt_membership)

# Print the accuracy of the methods
print('community multilevel acc: ', accuracy(cap_comms_ml, ground_truth))
print('community leiden acc: ', accuracy(cap_comms_ld, ground_truth))
print('community hedonic acc: ', accuracy(comms_hd, ground_truth))