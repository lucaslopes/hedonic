import numpy as np
import igraph as ig
from collections import defaultdict
from hedonic import Game
from networkx.generators.community import stochastic_block_model as SBM

def probs_matrix(n_communities, p, q):
  probs = np.full((n_communities, n_communities), q) # fill with q
  np.fill_diagonal(probs, p) # fill diagonal with p
  return probs # return probability matrix

def generate_graph(n_communities, community_size, p_in, multiplier, seed):
  block_sizes = np.full(n_communities, community_size) # all blocks are same size
  p_out = p_in * multiplier # probability of edge between communities
  p = probs_matrix(n_communities, p_in, p_out) # probability matrix
  g = SBM(sizes=block_sizes, p=p, seed=seed) # generate networkx graph
  h = ig.Graph()
  h.add_vertices(g.number_of_nodes())
  h.add_edges(g.edges())
  h = Game(h)  # convert to Game (igraph subclass)
  
  return h # return Hedonic Game

def get_ground_truth(number_of_communities: int, community_size: int, g: Game = None):
  gt_membership = np.concatenate([
    np.full(community_size, i) for i in range(number_of_communities)]).tolist() # ground truth membership
  if g is not None:
    gt_membership = ig.clustering.VertexClustering(g, gt_membership) # return ground truth
  return gt_membership

def shuffle_with_noise(membership, noise=1.0, seed=None):
  if seed is not None:
    np.random.seed(seed)
  
  # Group nodes by community
  community_dict = defaultdict(list)
  for node, community in enumerate(membership):
    community_dict[community].append(node)
  
  # Shuffle nodes within each community
  for community_nodes in community_dict.values():
    np.random.shuffle(community_nodes)
  
  # Flatten the community_dict to get the shuffled membership
  shuffled_membership = [None] * len(membership)
  for community, nodes in community_dict.items():
    for node in nodes:
      shuffled_membership[node] = community
  
  # Calculate the number of nodes to shuffle between communities
  n = len(membership)
  num_to_shuffle = int(noise * n)
  
  # Select nodes to shuffle between communities
  indices_to_shuffle = np.random.choice(range(n), size=num_to_shuffle, replace=False)
  
  # Shuffle the selected nodes between communities
  shuffled_indices = np.random.permutation(indices_to_shuffle)
  for i, j in zip(indices_to_shuffle, shuffled_indices):
    shuffled_membership[i], shuffled_membership[j] = shuffled_membership[j], shuffled_membership[i]
  
  return shuffled_membership

def get_initial_membership(ground_truth, noise=1, seed=None):
  if seed is not None:
    np.random.seed(seed)
  membership = ground_truth if type(ground_truth) == list else ground_truth.membership
  if noise > 1:
    membership = [node if type(node) == int else node.index for node in range(len(membership))] # Singleton partition
  else:
    membership = shuffle_with_noise(membership, noise=noise, seed=seed)
  return membership

def limit_community_count(g: Game, partition, max_n_communities): # since we known in advance the number of communities we can limit it
  new_partition = None
  if len(partition.sizes()) > max_n_communities:
    new_membership = [m if (
      m < max_n_communities
    ) else (
      max_n_communities - 1
    ) for m in partition.membership]
    new_partition = ig.clustering.VertexClustering(g, new_membership)
  return new_partition if new_partition else partition

def generate_sequence(num: float, n: int) -> list:
  if n < 3:
    raise ValueError("n must be at least 3")
  sequence = [num, 0.0, 1.0]
  while len(sequence) < n:
    last_two = sequence[-2:]
    mid1 = (last_two[0] + num) / 2
    mid2 = (last_two[1] + num) / 2
    sequence.append(mid1)
    sequence.append(mid2)
  return sorted(sequence[:n])

