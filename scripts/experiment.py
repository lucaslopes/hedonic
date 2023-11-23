import os
import time
import json
import random
import igraph as ig
import numpy as np
from networkx.generators.community import stochastic_block_model as SBM
from hedonic import HedonicGame
from config import methods

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
  return h # return igraph graph

def get_ground_truth(g, community_size):
  gt_membership = np.concatenate((np.zeros(community_size), np.ones(community_size)))
  return ig.clustering.VertexClustering(g, gt_membership)

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

def generate_probability_array(length):
  powers = np.arange(length-1, -1, -1)
  result = 1 / np.power(2, powers)
  return result


def sequential_probability_ratio_test(func, error=0.01, z=1.96):
    samples = []
    while True:
        sample = func()
        samples.append(sample)
        n = len(samples)
        mean = np.mean(samples)
        se = np.sqrt((mean * (1 - mean)) / n)
        ci = z * se
        if ci < error:
            break
    return mean
    # z = 1.96 # 95% confidence interval
    # z = 1.645 # 90% confidence interval
    # error = 5/100 # 5% error
    # se = np.sqrt((mean * (1 - mean)) / n)
    # ci = z * se
    # if ci < error:
    #     break

#################################################

def experiment(
    method_name,
    method_params,
    p_in,
    multiplier,
    community_size,
    number_of_communities,
  ):
  g = generate_graph(number_of_communities, community_size, p_in, multiplier)
  ground_truth = get_ground_truth(g, community_size)
  method = getattr(g, method_name)
  start_time = time.time()
  partition = method(**method_params)
  duration = time.time() - start_time
  partition = limit_community_count(g, partition, number_of_communities)
  accuracy = g.accuracy(partition, ground_truth)
  g.community_leiden()
  result = {
    'method': method_name,
    'p_in': p_in,
    'p_out': p_in * multiplier,
    'multiplier': multiplier,
    'community_size': community_size,
    'number_of_communities': number_of_communities,
    'accuracy': accuracy,
    'duration': duration,
  }
  return result

#################################################

def run_exp(community_size, number_of_communities, length_of_probability_array):
  print('Running experiment')
  ps_in = generate_probability_array(length_of_probability_array) # Define probability of edge within community
  multipliers = generate_probability_array(length_of_probability_array) # `multiplier = p_out / p_in` and p_out is the probability of edge between communities
  for p_in in ps_in:
    for multiplier in multipliers:
      g = generate_graph(number_of_communities, community_size, p_in, multiplier)
      for method, params in methods.items():
        if 'initial_membership' in params:
          params['initial_membership'] = [
            random.choice(range(number_of_communities)) for _ in g.vs]
        res = experiment(
          method,
          params,
          p_in,
          multiplier,
          community_size,
          number_of_communities,
        )
        directory = os.path.expanduser('~/Databases/hedonic')
        file_path = os.path.join(directory, f'{method}_{time.time()}.json')
        with open(file_path, 'w') as file:
            file.write(json.dumps(res))
            print(f'Wrote to {file_path}')