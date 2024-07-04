import os
import json
import config
import random
import argparse
import numpy as np
import igraph as ig
from tqdm import tqdm
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
    np.full(community_size, i) for i in range(number_of_communities)]).tolist() # ground truth membership
  return ig.clustering.VertexClustering(g, gt_membership) # return ground truth

def get_initial_membership(g: HedonicGame, ground_truth, number_of_communities, know_n_clusters=False, noise=1):
  know_n_clusters = False if noise > 1 else know_n_clusters
  if not know_n_clusters:
    membership = [node.index for node in g.vs] # Singleton partition
  else:
    membership = [np.random.choice(range(number_of_communities)) if np.random.rand() < noise else c for c in ground_truth.membership]
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
  stopwatch.stop()
  accuracy = g.accuracy(partition, ground_truth) # calculate accuracy wrt the ground truth
  # accuracy_edges = g.accuracy_classify_edges(partition, ground_truth) # calculate accuracy wrt the ground truth
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
    # 'accuracy_edges': accuracy_edges,
    'robustness': robustness,
    'partition': partition.membership,}
  return result

def run_experiment(
    folder_name,
    number_of_communities,
    community_size,
    p_in,
    difficulty,
    methods,
    noises=[0],
    seed=42,
  ):
  np.random.seed(seed)
  # for p_in in tqdm(probabilities, desc='p_in', leave=False):
  #   for multiplier in tqdm(difficulties, desc='multiplier', leave=False):
  g = generate_graph(number_of_communities, community_size, p_in, difficulty, seed)
  gt = get_ground_truth(g, number_of_communities, community_size) # get ground truth
  edge_density = g.density()
  for method, method_params in tqdm(methods.items(), desc='method', leave=False, total=len(methods)):
    result = None
    params = {k: v for k, v in method_params.items()}
    if method == 'community_groundtruth':
      params['groundtruth'] = gt
    if method in {'community_leiden', 'community_hedonic'}:
      params['resolution'] = edge_density
    for noise in tqdm(noises, desc='noise', leave=False, total=len(noises)):
      if 'initial_membership' in params:
        params['initial_membership'] = get_initial_membership(g, gt, number_of_communities, True, noise) # get initial membership
        result = None
      if result is None:
        result = get_method_result(
          g,
          method,
          params,
          p_in,
          difficulty,
          community_size,
          number_of_communities,
          gt)
      result['noise'] = noise
      result['seed'] = seed
      output_results_path = f"~/Databases/hedonic/{folder_name}/{number_of_communities} Communities of {community_size} nodes/Noise = {noise:.2f}/P_in = {p_in:.2f}/Difficulty = {difficulty:.2f}/Method = {method}"
      file_path = os.path.join(
        os.path.expanduser(output_results_path),
        f'{method.split("_")[1]}_{seed}.json')
      os.makedirs(os.path.dirname(file_path), exist_ok=True)
      with open(file_path, 'w') as file:
        file.write(json.dumps(result))
  return True

#################################################

def main():
  # Parse command line arguments
  # "python scripts/experiment.py --folder_name sample_exp --max_n_nodes 120 --n_communities $n_community --seed $seed --p_in $p_in --difficulty $difficulty"
  parser = argparse.ArgumentParser(description='Run hedonic game experiments.')
  parser.add_argument('--folder_name', type=str, required=False, help='Name of the folder to store results', default='test')
  parser.add_argument('--max_n_nodes', type=int, required=False, help='Maximum number of nodes', default=60)
  parser.add_argument('--n_communities', type=int, nargs='+', required=False, help='Number of clusters', default=[2])
  parser.add_argument('--seeds', type=int, nargs='+', required=False, help='Seeds', default=[42])
  parser.add_argument('--p_in', type=float, nargs='+', required=False, help='Probability of edge within communities', default=[0.1])
  parser.add_argument('--difficulty', type=float, nargs='+', required=False, help='Difficulty of the problem', default=[0.5])
  args = parser.parse_args()

  folder_name = args.folder_name # MainResultExperiment
  max_n_nodes = args.max_n_nodes
  n_communities = args.n_communities # [2, 3, 4, 5, 6]
  seeds = args.seeds # [1, 2, 3, 4, 5]
  probabilities = args.p_in # [.10, .09, .08, .07, .06, .05, .04, .03, .02, .01]
  difficulties = args.difficulty # [.75, .7, .65, .6, .55, .5, .4, .3, .2, .1]
  noises = [0.01, 0.25, 0.5, 0.6, 0.7, .75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1]
  for n_community in tqdm(n_communities, desc='n_community', leave=False):
    community_size = int(max_n_nodes / n_community)
    for seed in tqdm(seeds, desc='seed', leave=False):
      for p_in in tqdm(probabilities, desc='p_in', leave=False):
        for difficulty in tqdm(difficulties, desc='difficulty', leave=False):
          run_experiment(
            folder_name,
            n_community,
            community_size,
            p_in,
            difficulty,
            config.methods,
            noises,
            seed)
  print(f'Experiments completed successfully.')
  return True


__name__ == '__main__' and main()

# caffeinate -s python scripts/experiment.py --folder_name exp600 --max_n_nodes 600 --n_communities 2 3 4 5 6 --seeds 1 2 3 4 5