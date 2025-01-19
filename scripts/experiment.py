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
from utils import generate_graph, get_ground_truth, get_initial_membership

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
    n_partitions=1,
    seed=42,
  ):
  # for p_in in tqdm(probabilities, desc='p_in', leave=False):
  #   for multiplier in tqdm(difficulties, desc='multiplier', leave=False):
  g = generate_graph(number_of_communities, community_size, p_in, difficulty, seed)
  gt = get_ground_truth(g, number_of_communities, community_size) # get ground truth
  edge_density = g.density()
  for method_name, method_info in tqdm(methods.items(), desc='method', leave=False, total=len(methods)):
    result = None
    method_call_name = method_info['method_call_name']
    parameters = method_info['parameters']
    params = {k: v for k, v in parameters.items()}
    if method_call_name == 'community_groundtruth':
      params['groundtruth'] = gt
    if method_call_name in {'community_leiden', 'community_hedonic'}:
      params['resolution'] = edge_density
    for noise in tqdm(noises, desc='noise', leave=False, total=len(noises)):
      for partition_seed in range(n_partitions):
        if 'initial_membership' in params:
          params['initial_membership'] = get_initial_membership(g, gt, number_of_communities, True, noise, seed=partition_seed) # get initial membership
          result = None
        if result is None:
          result = get_method_result(
            g,
            method_call_name,
            params,
            p_in,
            difficulty,
            community_size,
            number_of_communities,
            gt)
        result['method'] = method_name
        result['noise'] = noise
        result['network_seed'] = seed
        result['partition_seed'] = partition_seed
        output_results_path = f"~/Databases/hedonic/{folder_name}/{number_of_communities} Communities of {community_size} nodes/Noise = {noise:.2f}/P_in = {p_in:.2f}/Difficulty = {difficulty:.2f}/Network ({seed:03d})/Partition ({partition_seed:03d})"
        file_path = os.path.join(
          os.path.expanduser(output_results_path),
          f'{method_name}.json')
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
  parser.add_argument('--noises', type=float, nargs='+', required=False, help='Noise levels', default=[0.01, 0.25, 0.5, 0.6, 0.7, .75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1])
  parser.add_argument('--n_partitions', type=int, default=1, help='Number of partitions')
  args = parser.parse_args()

  folder_name = args.folder_name # MainResultExperiment
  max_n_nodes = args.max_n_nodes
  n_communities = args.n_communities # [2, 3, 4, 5, 6]
  seeds = args.seeds # [1, 2, 3, 4, 5]
  probabilities = args.p_in # [.10, .09, .08, .07, .06, .05, .04, .03, .02, .01]
  difficulties = args.difficulty # [.75, .7, .65, .6, .55, .5, .4, .3, .2, .1]
  noises = args.noises # [0.01, 0.25, 0.5, 0.6, 0.7, .75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1]
  n_partitions = args.n_partitions
  for n_community in tqdm(n_communities, desc='n_community', leave=False):
    community_size = int(max_n_nodes / n_community)
    for p_in in tqdm(probabilities, desc='p_in', leave=False):
      for difficulty in tqdm(difficulties, desc='difficulty', leave=False):
        for seed in tqdm(seeds, desc='seed', leave=False):
          run_experiment(
            folder_name,
            n_community,
            community_size,
            p_in,
            difficulty,
            config.methods,
            noises,
            n_partitions,
            seed)
  print(f'Experiments completed successfully.')
  return True


__name__ == '__main__' and main()

# caffeinate -s python scripts/experiment.py --folder_name exp600 --max_n_nodes 600 --n_communities 2 3 4 5 6 --seeds 1 2 3 4 5