import igraph as ig
import numpy as np
from networkx.generators.community import stochastic_block_model as SBM
from hedonic import community_hedonic

#################################################

# Parameters
community_size = 10 # Define block sizes
p_in = .5
multiplier = .5

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
    # convert to igraph
    g = ig.Graph.TupleList(h.edges())
    return g

#################################################

g = ppm_igraph(community_size, p_in, multiplier)
# print(g.summary()) # Print the graph summary

# Get the communities
comms_ml = g.community_multilevel()
comms_ld = g.community_leiden()

# Print the communities
print('community multilevel: \n', comms_ml)
print('community leiden: \n', comms_ld)
print('community hedonic: \n', community_hedonic(g))