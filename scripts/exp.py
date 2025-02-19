import re
import os
import json
import config
import argparse
import numpy as np
import igraph as ig
from tqdm import tqdm
from collections import defaultdict
from stopwatch import Stopwatch
from hedonic import Game
from utils import read_pickle, get_ground_truth, get_all_subpaths, network_path_to_memberships_path, read_csv_partition

#################################################

def get_method_result(
    g: Game,
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
  robustness = g.partition_robustness(partition.membership) # calculate robustness
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
    'partition': partition.membership}
  return result

def run_experiment(network_path: str, methods: dict) -> bool:
  g = read_pickle(network_path)
  edge_density = g.density()
  number_of_communities, community_size = re.findall(r'(\d+)C_(\d+)N', network_path)[0]
  number_of_communities = int(number_of_communities)
  community_size = int(community_size)  # int(int(total_nodes)/number_of_communities)
  network_seed = int(re.findall(r'network_0*(\d+)\.pkl', network_path)[0])
  p_in = float(re.findall(r'P_in = (\d+\.\d+)/', network_path)[0])
  multiplier = float(re.findall(r'Difficulty = (\d+\.\d+)/', network_path)[0])
  gt = get_ground_truth(number_of_communities, community_size, g) # get ground truth
  partitions_path = get_all_subpaths(network_path_to_memberships_path(network_path))
  results_dict = defaultdict(lambda: [])
  for method_name, method_info in tqdm(methods.items(), desc='method', leave=False, total=len(methods)):
    result = None
    runs = 1
    method_call_name = method_info['method_call_name']
    parameters = method_info['parameters']
    params = {k: v for k, v in parameters.items()}
    if method_call_name == 'community_groundtruth':
      params['groundtruth'] = gt
    if method_call_name == 'community_leading_eigenvector':
      params['clusters'] = number_of_communities
    if method_call_name in {'community_leiden', 'community_hedonic'}:
      params['resolution'] = edge_density
      runs = 10
    for partition_path in tqdm(partitions_path, desc='partitions', leave=False, total=len(partitions_path)):
      if 'initial_membership' in params:
        initial_membership = read_csv_partition(partition_path)
        params['initial_membership'] = initial_membership
        result = None
      if result is None:
        saved_partitions = set()  # Set to keep track of saved partitions
        for _ in range(runs):
          result = get_method_result(
            g = g,
            method_name = method_call_name,
            method_params = params,
            p_in = p_in,
            multiplier = multiplier,
            community_size = community_size,
            number_of_communities = number_of_communities,
            ground_truth = gt)
          result['method'] = method_name
          result['noise'] = float(re.findall(r'Noise = (\d+\.\d+)/', partition_path)[0])
          result['network_seed'] = network_seed
          result['partition_seed'] = int(re.findall(r'partition_0*(\d+)\.csv', partition_path)[0])
          if (part := tuple(result['partition'])) not in saved_partitions or runs == 1:  # Convert list to tuple for set membership
            saved_partitions.add(part)  # Add as tuple to the set
            results_dict[(result['noise'], result['partition_seed'])].append(result)
  for ((noise, partition_seed), results) in results_dict.items():
    output_results_path = f"{'/'.join(partitions_path[0].split('/')[:-2])}/Noise = {noise:.2f}/P_in = {p_in:.2f}/Difficulty = {multiplier:.2f}/Network ({network_seed:03d})".replace('/memberships/', '/resultados/')
    file_path = os.path.join(
      os.path.expanduser(output_results_path),
      f'partition_{partition_seed:03d}.json'
    )
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
      json.dump(results, file)
  return True

#################################################

def main():
  # Parse command line arguments: python scripts/experiment.py DIR
  parser = argparse.ArgumentParser(description='Run hedonic game experiments.')
  parser.add_argument('dir', type=str, help='Path of the directory containing the networks')
  args = parser.parse_args()

  if run_experiment(network_path=args.dir, methods=config.methods):
    # Mark the experiment as completed
    with open(args.dir.replace('.pkl', '.completed'), 'w') as file:
      file.write('')
  
  return True


__name__ == '__main__' and main()

# caffeinate -s python scripts/experiment.py --folder_name exp600 --max_n_nodes 600 --n_communities 2 3 4 5 6 --seeds 1 2 3 4 5