import os
import json
import config
import numpy as np
import igraph as ig
from time import time
from tqdm import tqdm
from random import choice
from stopwatch import Stopwatch
from hedonic import HedonicGame
from methods import CommunityMethods  # CommunityMethods is a subclass of HedonicGame
from networkx.generators.community import stochastic_block_model as SBM

#################################################

def probs_matrix(n_communities, p, q):
  probs = np.full((n_communities, n_communities), q) # fill with q
  np.fill_diagonal(probs, p) # fill diagonal with p
  return probs # return probability matrix

def generate_graph(n_communities, community_size, p_in, multiplier, seed):
  block_sizes = np.full(n_communities, community_size) # all blocks are same size
  p_out = p_in * multiplier # probability of edge between communities
  p = probs_matrix(n_communities, p_in, p_out) # probability matrix
  g = SBM(sizes=block_sizes, p=p, seed=seed) # generate networkx graph
  h = CommunityMethods(g.number_of_nodes(), g.edges()) # convert to HedonicGame (igraph subclass)
  return h # return Hedonic Game

def get_ground_truth(g: HedonicGame, number_of_communities, community_size):
  gt_membership = np.concatenate([
    np.full(community_size, i) for i in range(number_of_communities)
  ]) # ground truth membership
  return ig.clustering.VertexClustering(g, gt_membership) # return ground truth

def get_initial_membership(g: HedonicGame, ground_truth, number_of_communities, know_n_clusters=False, noise=1):
  know_n_clusters = False if noise > 1 else know_n_clusters
  if not know_n_clusters:
    membership = [node.index for node in g.vs] # Singleton partition
  else:
    membership = [choice(range(number_of_communities)) if np.random.rand() < noise else c for c in ground_truth.membership]
  return membership # ig.clustering.VertexClustering(g, membership) # return initial membership

def limit_community_count(g: HedonicGame, partition, max_n_communities): # since we known in advance the number of communities we can limit it
  new_partition = None
  if len(partition.sizes()) > max_n_communities:
    new_membership = [m if (
      m < max_n_communities
    ) else (
      max_n_communities - 1
    ) for m in partition.membership]
    new_partition = ig.clustering.VertexClustering(g, new_membership)
  return new_partition if new_partition else partition

#################################################

def get_method_result(
    g: HedonicGame,
    method_name,
    method_params,
    p_in,
    multiplier,
    community_size,
    number_of_communities,
    ground_truth,
  ):
  method = getattr(g, method_name) # get method
  stopwatch = Stopwatch() # create Stopwatch instance
  stopwatch.start()
  try:
    partition = method(**method_params) # run method
  except Exception as e:
    partition = ig.clustering.VertexClustering(g, [0] * g.vcount())
    print(f"\nPARTITIONING ERROR:\n{e}\n{method_name=}\n{p_in=}\n{multiplier=}\n{community_size=}\n{number_of_communities=}")
    return None
  stopwatch.stop()
  accuracy = g.accuracy(partition, ground_truth) # calculate accuracy wrt the ground truth
  robustness = g.robustness(partition) # calculate robustness
  result = {
    'method': method_name.split("_")[1],
    'number_of_communities': number_of_communities,
    'community_size': community_size,
    'p_in': p_in,
    'p_out': p_in * multiplier,
    'multiplier': multiplier,
    'resolution': method_params['resolution'] if 'resolution' in method_params else None,
    'duration': stopwatch.duration,
    'accuracy': accuracy,
    'robustness': robustness,
    'partition': partition.membership,}
  return result

def run_experiment(
    folder_name,
    number_of_communities,
    community_size,
    probabilities,
    difficulties,
    methods,
    noise=1,
    seed=42,
  ):
  for p_in in tqdm(probabilities, desc='p_in', leave=False):
    for multiplier in tqdm(difficulties, desc='multiplier', leave=False):
      g = generate_graph(number_of_communities, community_size, p_in, multiplier, seed)
      gt = get_ground_truth(g, number_of_communities, community_size) # get ground truth
      initial_membership = get_initial_membership(g, gt, number_of_communities, True, noise) # get initial membership
      edge_density = g.density()
      for method, params in tqdm(methods.items(), desc='method', leave=False, total=len(methods)):
        if method in {'community_leiden', 'community_hedonic'}:
          params['resolution'] = edge_density
        if 'initial_membership' in params:
          params['initial_membership'] = initial_membership
        result = get_method_result(
          g,
          method,
          params,
          p_in,
          multiplier,
          community_size,
          number_of_communities,
          gt)
        result['noise'] = noise
        result['seed'] = seed
        output_results_path = f"~/Databases/Hedonic/Experiments/{folder_name}/Clusters = {number_of_communities}/Size = {community_size}/Noise = {noise}/P_in = {p_in}/Multiplier = {multiplier}/Method = {method}"
        file_path = os.path.join(
          os.path.expanduser(output_results_path),
          f'{method.split("_")[1]}_{time()}.json')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
          file.write(json.dumps(result))
  return True

#################################################

def main():
  folder_name = 'comparison'
  noises = [0, .25, .5, 1]
  max_n_nodes = 60 * 10
  n_communities = [5]#[2, 3, 4, 5]
  probabilities = [.10, .09, .08, .07, .06, .05, .04, .03, .02, .01]
  difficulties = [.75, .7, .65, .6, .55, .5, .4, .3, .2, .1]
  samples = 1
  seed = 0
  for noise in tqdm(noises, desc='noise', leave=False, total=len(noises)):
    for n_comm in tqdm(n_communities, desc='n_communities', leave=False, total=len(n_communities)):
      for _ in tqdm(range(samples), desc='samples', leave=False, total=len(range(samples))):
        community_size = int(max_n_nodes / n_comm)
        run_experiment(
          folder_name,
          n_comm,
          community_size,
          probabilities,
          difficulties,
          config.methods,
          noise,
          seed)
        seed += 1
  return True


__name__ == '__main__' and main()