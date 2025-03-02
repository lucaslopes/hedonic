import os
import re
import gzip
import json
import pickle
import numpy as np
import igraph as ig
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from networkx.generators.community import stochastic_block_model as SBM
from hedonic import Game


def network_path_to_memberships_path(pth: str) -> str:
  l = pth.split('/')
  index = l.index('networks')
  l[index] = 'memberships'
  return'/'.join(l[:index+2])


def get_all_subpaths(path: str, endswith:str = '.csv') -> list[str]:
  paths = []
  for root, _, files in os.walk(path):
    for file in files:
      if file.endswith(endswith):
        paths.append(os.path.join(root, file))
  paths.sort()
  return paths


def read_csv_partition(partition_path: str) -> list[int]:
  with open(partition_path, 'r') as f:
    return [int(x) for x in f.read().strip().split(',')]

def load_json_files(file_paths, ignore_partition_key=True, verbose=False):
  """
  Reads all JSON files from a list of file paths and returns a combined list of result dictionaries.
  """
  # Load JSON data
  json_files = []
  for root, dirs, files in tqdm(os.walk(file_paths), desc="Walking directory tree"):
    for file in files:
      if file.endswith(".json"):
        json_files.append(os.path.join(root, file))
  json_files = sort_files(json_files)
  with open('json_files.txt', 'w') as f:
    for line in json_files:
      f.write(line + '\n')
  print(f"Saved list of JSON files to: `json_files.txt`")
  data = []
  last_noise = None
  for fp in tqdm(json_files, desc="Loading JSON files"):
    network_seed = int(re.findall(r'Network \(0*(\d+)\)', fp)[0])
    noise = float(re.findall(r'Noise = (\d+\.\d+)', fp)[0])
    csv_path = fp.replace('/resultados/', '/csv_results/')
    csv_path = csv_path.split('/')[:-1]
    csv_path[-1] = f'network_{network_seed:03d}.csv.gzip'
    csv_path = '/'.join(csv_path)
    if os.path.exists(os.path.dirname(csv_path)):
      continue
    Path(os.path.dirname(csv_path)).mkdir(parents=True, exist_ok=True)
    if last_noise is None:
      last_noise = noise
    if noise != last_noise:
      last_noise = noise
      df = pd.DataFrame(data)
      data = []
      if verbose:
        print(f"Saving dataframe to: `{csv_path}`")
      df.to_csv(csv_path, index=False, compression="gzip")
    with open(fp, 'r') as f:
      file_data = json.load(f)  # Each file is assumed to be a JSON list
      for d in file_data:
        if ignore_partition_key and 'partition' in d:
          del d['partition']
        data.append(d)
  return True

def read_pickle(graph_path: str, verbose: bool = False) -> Game:
  with open(graph_path, 'rb') as f:
    if verbose:
      print(f"Loading graph from: `{graph_path}`")
    g = pickle.load(f)
  return g

def read_txt_gz_to_igraph(file_path):
  edges = []
  with gzip.open(file_path, 'rt') as file:  # 'rt' mode for text mode reading
    for line in file:
      if line.startswith('#'):  # Skip comment lines
        continue
      # Split line into source and target node IDs and convert to integers
      nodes = list(map(int, line.strip().split()))
      if len(nodes) == 2:
        source, target = nodes
        edges.append((source, target))
      else:
        print(line, nodes)
  # Assuming the file contains an edge list with each line as 'source target'
  graph = ig.Graph(edges=edges, directed=False)
  return graph

def read_communities(file_path, mode='list_of_communities'):
  communities = []
  with gzip.open(file_path, 'rt') as file:  # 'rt' mode for text mode reading
    for line in file:
      if mode == 'list_of_communities':
        nodes = list(map(int, line.strip().split()))
        communities.append(nodes)
      elif mode == 'node_labels':
        node, community = map(int, line.strip().split())
        communities.append((node, community))
  if mode == 'node_labels':
    communities = dict()
    for node, community in communities:
      try:
        communities[community].add(node)
      except:
        communities[community] = set({node})
  return communities

def delete_non_format_files(path: str, format: str):
  '''
  Delete all files in the given path that do not have the given format.
  Useful for cleaning up temporary files like the .completed files.
  '''
  for root, _, files in os.walk(path):
    for file in files:
      if not file.endswith(format):
        os.remove(os.path.join(root, file))

def extract_sorting_keys(filename):
  """Extract sorting keys from the file name"""
  match = re.search(r'network_(\d+)\.', filename)
  network_index = int(match.group(1)) if match else float('inf')
  if network_index == float('inf'):
    match = re.findall(r'Network \(0*(\d+)\)', filename)
    network_index = int(match[0]) if match else float('inf')
  if network_index == float('inf'):
    raise ValueError(f"Could not find network index in filename: `{filename}`")

  difficulty_match = re.search(r'Difficulty = (\d+\.\d+)', filename)
  difficulty = float(difficulty_match.group(1)) if difficulty_match else float('inf')
  
  p_in_match = re.search(r'P_in = (\d+\.\d+)', filename)
  p_in = float(p_in_match.group(1)) if p_in_match else float('inf')
  
  n_communities_match = re.search(r'(\d+)C_', filename)
  n_communities = int(n_communities_match.group(1)) if n_communities_match else float('inf')

  noise_match = re.search(r'Noise = (\d+\.\d+)', filename)
  noise = float(noise_match.group(1)) if noise_match else float('inf')
  
  return (network_index, n_communities, p_in, difficulty, -noise)  

def sort_files(file_list):
  """Sort the list of files based on the extracted keys"""
  return sorted(file_list, key=extract_sorting_keys)

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

