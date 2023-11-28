import os
import json
import config
import numpy as np
import igraph as ig
from time import time
from tqdm import tqdm
from random import choice
from hedonic import HedonicGame
from networkx.generators.community import stochastic_block_model as SBM

#################################################

def probs_matrix(n_communities, p, q):
  probs = np.full((n_communities, n_communities), q) # fill with q
  np.fill_diagonal(probs, p) # fill diagonal with p
  return probs # return probability matrix

def generate_graph(n_communities, community_size, p_in, multiplier):
  block_sizes = np.full(n_communities, community_size) # all blocks are same size
  p_out = p_in * multiplier # probability of edge between communities
  p = probs_matrix(n_communities, p_in, p_out) # probability matrix
  g = SBM(block_sizes, p) # generate networkx graph
  h = HedonicGame(g.number_of_nodes(), g.edges()) # convert to HedonicGame (igraph subclass)
  return h # return Hedonic Game

def get_ground_truth(g, community_size):
  c0 = np.zeros(community_size) # community 0
  c1 = np.ones(community_size) # community 1
  gt_membership = np.concatenate((c0, c1)) # ground truth membership
  return ig.clustering.VertexClustering(g, gt_membership) # return ground truth

def limit_community_count(g, partition, max_n_communities): # since we known in advance the number of communities we can limit it
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
    g,
    method_name,
    method_params,
    p_in,
    multiplier,
    community_size,
    number_of_communities,
  ):
  method = getattr(g, method_name) # get method
  start_time = time() # start timer
  partition = method(**method_params) # run method
  duration = time() - start_time # end timer
  partition = limit_community_count(g, partition, number_of_communities) # limit community count
  gt = get_ground_truth(g, community_size) # get ground truth
  accuracy = g.accuracy(partition, gt) # calculate accuracy
  result = {
    'method': method_name.split("_")[1],
    'number_of_communities': number_of_communities,
    'community_size': community_size,
    'p_in': p_in,
    'p_out': p_in * multiplier,
    'multiplier': multiplier,
    'resolution': method_params['resolution'],
    'duration': duration,
    'accuracy': accuracy,}
  return result

def run_experiment(
    number_of_communities,
    community_size,
    probabilities,
    output_results_path,
    methods,
  ):
  for p_in in tqdm(probabilities, desc='p_in', leave=False):
    for multiplier in tqdm(probabilities, desc='multiplier', leave=False):
      g = generate_graph(number_of_communities, community_size, p_in, multiplier)
      initial_membership = [choice(range(number_of_communities)) for _ in g.vs]
      edge_density = g.density()
      for method, params in tqdm(methods.items(), desc='method', leave=False):
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
          number_of_communities,)
        file_path = os.path.join(
          os.path.expanduser(output_results_path),
          f'{method.split("_")[1]}_{time()}.json')
        with open(file_path, 'w') as file:
          file.write(json.dumps(result))
  return True

#################################################

def main():
  for _ in tqdm(range(config.experiment_params['samples'])):
    run_experiment(
      config.experiment_params['number_of_communities'],
      config.experiment_params['community_size'],
      config.experiment_params['probabilities'],
      config.experiment_params['output_results_path'],
      config.methods,)
  return True


__name__ == '__main__' and main()