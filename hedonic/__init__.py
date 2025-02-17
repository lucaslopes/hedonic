import numpy as np
from tqdm import tqdm
from collections import Counter
from igraph import Graph, compare_communities
from igraph.clustering import VertexClustering

#################################################

class Game(Graph):
  def __init__(self, graph=None, *args, **kwargs):
    # Initialize the base class with empty parameters
    super().__init__(*args, **kwargs)
    if type(graph) == Graph: # if graph is an igraph
      self.add_vertices(graph.vcount())
      self.add_edges(graph.get_edgelist())
      # Copy vertex attributes
      for attr in graph.vertex_attributes():
        self.vs[attr] = graph.vs[attr]
      # Copy edge attributes
      for attr in graph.edge_attributes():
        self.es[attr] = graph.es[attr]
      # Copy graph attributes
      for attr in graph.attributes():
        self[attr] = graph[attr]

  def hedonic_value(self, neighbors, strangers, resolution):
    pros = neighbors * (1-resolution) # prosocial value
    cons = strangers *    resolution # antisocial value
    return pros - cons # hedonic value

  def initialize_game(self, initial_membership):
    self['communities_nodes'] = dict() # communities are sets of nodes
    self['communities_edges'] = dict() # communities are count of internal edges
    initial_membership = initial_membership if type(initial_membership) == list else [node.index for node in self.vs]
    for node, community in zip(self.vs, initial_membership):
      self.vs[node.index]['community'] = community
      try:
        self['communities_nodes'][community].add(node.index)
      except:
        self['communities_nodes'][community] = set({node.index})
    for community in self['communities_nodes']:
      self['communities_edges'][community] = 0
    for node in self.vs:
      node['neighbors_in_community'] = {}
      for community in self['communities_nodes']:
        node['neighbors_in_community'][community] = 0
      neighbors_communities = [
        self.vs[neighbor]['community'] for neighbor in self.neighbors(node)]
      for community in neighbors_communities:
        node['neighbors_in_community'][community] += 1
        if community == node['community']:
          self['communities_edges'][community] += 1
    for community, edges in self['communities_edges'].items():
      self['communities_edges'][community] = int(edges/2)

  def value_of_node_in_community(self, node, community, resolution):
    members = len(self['communities_nodes'][community]) # number of members in community
    neighbors = node['neighbors_in_community'][community] # number of neighbors in community
    strangers = members - neighbors # number of strangers in community
    if node['community'] == community: # if node is a member of the community
      strangers -= 1 # node is not a stranger to itself
    return self.hedonic_value(neighbors, strangers, resolution) # return hedonic value of node in community

  def get_preferable_community(self, node, resolution=None):
    resolution = resolution if resolution else self.density()
    pref_community = node['community']
    highest_hedonic_value = self.value_of_node_in_community(node, node['community'], resolution)
    communities = [c for c, amount in node['neighbors_in_community'].items() if amount > 0]
    for community in communities:
      hedonic = self.value_of_node_in_community(node, community, resolution)
      if hedonic > highest_hedonic_value:
        pref_community = community
        highest_hedonic_value = hedonic
    return pref_community
  
  def in_equibrium(self, resolution):
    for node in self.vs:
      pref_comm = self.get_preferable_community(node, resolution)
      if pref_comm != node['community']:
        return False
    return True

  def move_node_to_community(self, node, community):
    derparture, arrival = node['community'], community # departure and arrival communities
    self['communities_nodes'][derparture].remove(node.index) # remove node from derparture community
    self['communities_nodes'][arrival].add(node.index) # add node to arrival community
    self['communities_edges'][derparture] -= node['neighbors_in_community'][derparture] # update derparture community edges
    self['communities_edges'][arrival] += node['neighbors_in_community'][arrival] # update arrival community edges
    node['community'] = community # update node's community
    for neighbor in self.neighbors(node):
      self.vs[neighbor]['neighbors_in_community'][derparture] -= 1 # neighbor no longer community neighbor
      self.vs[neighbor]['neighbors_in_community'][arrival]    += 1 # neighbor is now community neighbor
    if len(self['communities_nodes'][derparture]) == 0: # if community is empty
      del self['communities_nodes'][derparture] # this community no longer exist

  def membership(self):
    return [int(node['community']) for node in self.vs]

  ## statistics #################################

  def accuracy(self, partition, ground_truth):
    # Rand index of Rand (1971)
    partition = VertexClustering(self, partition) if type(partition) == list else partition
    ground_truth = VertexClustering(self, ground_truth) if type(ground_truth) == list else ground_truth
    return compare_communities(partition, ground_truth, method="rand")
    n_communities = len({partition.membership[i] for i in range(partition.n)})
    if n_communities > 0:
      n_correct = 0
      for i in range(partition.n):
        for j in range(partition.n):
          pair_ij = partition.membership[i] == partition.membership[j]
          pair_gt = ground_truth.membership[i] == ground_truth.membership[j]
          if pair_ij == pair_gt:
            n_correct += 1
      acc = n_correct / partition.n ** 2
      acc = (acc - .5) / .5  # normalize
    else:
      acc = 0
    return acc

  def robustness_per_community(self, partition, only_community_of_index=None):
    """Calculate fraction of nodes that are robust wrt the resolution parameter (i.e. they do not change community when the resolution parameter is changed)
    """
    partition = partition if type(partition) == VertexClustering else VertexClustering(self, partition)
    self.initialize_game(partition.membership)
    communities = list()
    for idx, community in enumerate(partition):
      if only_community_of_index is not None and idx != only_community_of_index:
        continue
      full_robust_nodes = 0
      for n in community:
        node = self.vs[n]
        pref_comm_res0 = self.get_preferable_community(node, 0)
        pref_comm_res1 = self.get_preferable_community(node, 1)
        on_best_community = pref_comm_res0 == pref_comm_res1 == idx
        full_robust_nodes += 1 if on_best_community else 0
      communities.append((full_robust_nodes, len(community)))
    return communities

  def robustness(self, partition):
    """Calculate fraction of nodes that are robust wrt the resolution parameter (i.e. they do not change community when the resolution parameter is changed)
    """
    robust = 0
    communities = self.robustness_per_community(partition)
    if len(self['communities_nodes']) > 0:
      robust = sum([robust for robust, total in communities]) / self.vcount()
    return robust

  ## need to verify #############################

  def potential(self): # need to verify
    global_potential = 0
    for community in list(self['communities_nodes']):
      global_potential += self.potential_of_community(community)
    return global_potential

  def potential_of_community(self, community, alpha):
    connections = self['communities_edges'][community]
    missed_connections = self.total_possible_edges(len(self['communities_nodes'][community])) - connections
    return self.hedonic_value(connections, missed_connections, alpha)

  def edges_between(self, community_A, community_B):
    bridges = 0
    for node in self['communities_nodes'][community_A]:
      bridges += self.vs[node]['friends_in_community'][community_B]
    return bridges

  def potential_merged(self, community_A, community_B, alpha):
    total_verts = len(self['communities_nodes'][community_A]) + len(self['communities_nodes'][community_B])
    connections = self['communities_edges'][community_A] + self['communities_edges'][community_B] + self.edges_between(community_A, community_B)
    missed_connections = self.total_possible_edges(total_verts) - connections
    return self.hedonic_value(connections, missed_connections, alpha)

  def worthy_merge(self, community_A, community_B, alpha=None):
    potential_A = self.potential_of_community(community_A)
    potential_B = self.potential_of_community(community_B)
    potential_together = self.potential_merged(community_A, community_B, alpha)
    return potential_together > potential_A + potential_B

  def find_a_pair_of_communities_to_merge(self):
    communities_list = list(self['communities_nodes'])
    for community in communities_list:
      # random.shuffle(communities_list)
      for other_community in communities_list:
        if other_community != community and self.worthy_merge(community, other_community):
          return community, other_community
    return None, None

  def merge_two_communities(self, C1, C2):
    smaller_community = C1 if len(self['communities_nodes'][C1]) < len(self['communities_nodes'][C2]) else C2
    greater_community = C2 if smaller_community == C1 else C1
    nodes_to_move = list(self['communities_nodes'][smaller_community])
    for node in nodes_to_move:
      self.move_node_to_community(self.vs[node], greater_community)

  ## Other Clustering Methods #############################

  def community_groundtruth(self, groundtruth):
    return VertexClustering(self, groundtruth) if type(groundtruth) == list else groundtruth

  def community_hedonic_traversal(self, resolution=None, initial_membership=None, log_memberships=False):
    resolution = resolution if resolution else self.density()
    self['log_memberships'] = []
    self.initialize_game(initial_membership)
    a_node_has_moved = True
    while a_node_has_moved:
      a_node_has_moved = False
      for node in self.vs:
        pref_comm = self.get_preferable_community(node, resolution)
        if pref_comm != node['community']:
          self.move_node_to_community(node, pref_comm)
          a_node_has_moved = True
          if log_memberships:
            self['log_memberships'].append(self.membership())
    return VertexClustering(self, self.membership())
  
  def community_hedonic_queue(self, resolution=None, initial_membership=None, log_memberships=False):
    resolution = resolution if resolution else self.density()
    self['log_memberships'] = []
    self.initialize_game(initial_membership)
    while not self.in_equibrium(resolution):
      # Initialize the queue with all nodes
      queue = [node.index for node in self.vs]
      in_queue = set(queue)  # Track nodes in the queue to avoid duplicates
      
      while queue:
        node_index = queue.pop(0)
        in_queue.remove(node_index)

        node = self.vs[node_index]
        pref_comm = self.get_preferable_community(node, resolution)
        if pref_comm != node['community']:
          self.move_node_to_community(node, pref_comm)

          # Add neighbors to the queue if not already in it or the new community
          for neighbor_index in self.neighbors(node):
            neighbor = self.vs[neighbor_index]
            if neighbor_index not in in_queue and neighbor['community'] != pref_comm:
              queue.append(neighbor_index)
              in_queue.add(neighbor_index)

            if log_memberships:
              self['log_memberships'].append(self.membership())
    
    return VertexClustering(self, self.membership())

  def community_mirror(self, initial_membership=None):
    if initial_membership is None:
      initial_membership = [0] * self.vcount()
    return VertexClustering(self, initial_membership)

  def community_onepass_improvement(self, initial_membership=None):
    for node, community in zip(self.vs, initial_membership):
      self.vs[node.index]['community'] = int(community)
    nodes_to_move = list()
    for node in self.vs:
      neighbors_comms = [self.vs[n]['community'] for n in self.neighbors(node)]
      if len(neighbors_comms) > 0:
        pref_comm = max(set(neighbors_comms), key=neighbors_comms.count)
        if pref_comm != node['community']:
          nodes_to_move.append((node.index, pref_comm))
    new_membership = [int(i) for i in initial_membership]
    while nodes_to_move:
      node, community = nodes_to_move.pop()
      new_membership[node] = community
    return VertexClustering(self, new_membership)

  def community_onepass_improvement_hedonic(self, initial_membership=None):
    self.initialize_game(initial_membership)
    nodes_to_move = list()
    for node in self.vs:
      pref_comm = self.get_preferable_community(node, 0)  # resolution = 0 means that the algorithm will try to move nodes to the community with the highest number of neighbors in the community
      if pref_comm != node['community']:
        nodes_to_move.append((node.index, pref_comm))
    while nodes_to_move:
      node, community = nodes_to_move.pop()
      self.move_node_to_community(self.vs[node], community)
    return VertexClustering(self, self.membership())

  def get_nodes_info(self, membership_list, nodes_subset=None):
    """
    Computes for each node (in nodes_subset or all nodes) a dictionary of community-based friend
    and stranger counts by iterating once over all edges of the graph.

    For each node, for every community 'c', the number of friends is the number of neighbors
    in community 'c'. The number of strangers is computed as:
    
        strangers = (total nodes in community c) - (friend count in c) - (1 if the node itself is in c else 0)
    
    Parameters:
        graph (ig.Graph): The input graph.
        membership_list (list): A list where the i-th element is the community membership of node i.
        nodes_subset (iterable, optional): An iterable of node indices for which to compute info.
            If None, info is computed for all nodes.
    
    Returns:
        dict: A dictionary mapping each node (from nodes_subset) to a dictionary that maps each
              community to a dict with keys 'friends' and 'strangers'.
    """
    # Total count of nodes per community.
    community_counter = Counter(membership_list)
    
    # Determine which nodes to process.
    if nodes_subset is None:
        nodes_subset = set(self.vs.indices)
    else:
        nodes_subset = set(nodes_subset)
    
    # Initialize friend counts for each node in nodes_subset.
    # Each value is a Counter mapping community -> friend count.
    friends_counts = {node: Counter() for node in nodes_subset}
    
    # Iterate over each edge once.
    for u, v in self.get_edgelist():
      # If u is in the subset, increment its friend count for v's community.
      if u in nodes_subset:
        friends_counts[u][membership_list[v]] += 1
      # Similarly, if v is in the subset, increment its friend count for u's community.
      if v in nodes_subset:
        friends_counts[v][membership_list[u]] += 1
    
    # Build the nodes_info dictionary.
    nodes_info = {}
    for node in friends_counts:
      node_membership = membership_list[node]
      node_info = {}
      # Ensure every community is represented, even if the friend count is 0.
      for community, total in community_counter.items():
        friend_count = friends_counts[node].get(community, 0)
        # If the node is in community 'community', subtract one (to exclude the node itself).
        stranger_count = total - friend_count - (1 if community == node_membership else 0)
        node_info[community] = {
          'friends': friend_count,
          'strangers': stranger_count
        }
      nodes_info[node] = node_info
    
    return nodes_info

  @staticmethod
  def classify_node_satisfaction(node_info, node_membership):
    max_friends_in_community = max([info['friends'] for info in node_info.values()])
    min_strangers_in_community = min([info['strangers'] for info in node_info.values()])
    robust_communities = set()
    for community in node_info:
      satisfy_max = node_info[community]['friends'] == max_friends_in_community
      satisfy_min = node_info[community]['strangers'] == min_strangers_in_community
      if satisfy_max and satisfy_min:
        robust_communities.add(community)
    if len(robust_communities) > 0:
      satisfaction = 'always_satisfied'
      if node_membership not in robust_communities:
        satisfaction = 'never_satisfied' # because there exists a robust community that is not the node's community, so no matter the resolution, the node will never be satisfied
    else:
      satisfaction = 'relatively_satisfied' # it depends on the resolution
    return satisfaction

  @staticmethod
  def is_node_robust(node_info, membership, node):
    satisfaction = Game.classify_node_satisfaction(node_info, membership[node])
    return satisfaction == 'always_satisfied'

  @staticmethod
  def get_nodes_robustness(nodes_info, membership):
    return {node: Game.is_node_robust(info, membership, node) for node, info in nodes_info.items()}

  @staticmethod
  def get_robustness(nodes_robustness):
    return sum(nodes_robustness.values()) / len(nodes_robustness)

  @staticmethod
  def get_partition_robustness(nodes_info, membership):
    nodes_robustness = Game.get_nodes_robustness(nodes_info, membership)
    robustness = Game.get_robustness(nodes_robustness)
    return robustness

  def partition_robustness(self, partition):
    nodes_info = self.get_nodes_info(partition)
    robustness = Game.get_partition_robustness(nodes_info, partition)
    return robustness

  @staticmethod
  def count_nodes_wanting_to_move(nodes_info, target_community):
    count = 0
    for info in nodes_info.values():
      prefer_community = max(info, key=lambda c: info[c]['friends'])  # TODO: consider resolution
      if prefer_community == target_community:
        count += 1
    return count
  
  def community_to_partition(self, community):
    partition = np.zeros(self.vcount(), dtype=int)
    for n in set(community):
      partition[n] = 1
    return partition
  
  def evaluate_community_stability(self, community):
    membership_list = np.zeros(self.vcount(), dtype=int)
    nodes_in_community = set(community)
    outer_neighbors = set()
    for n in nodes_in_community:
      membership_list[n] = 1
      for neighbor in self.neighbors(n):
        if neighbor not in nodes_in_community:
            outer_neighbors.add(neighbor)
    inside_nodes_info = self.get_nodes_info(membership_list, nodes_in_community)
    outside_nodes_info = self.get_nodes_info(membership_list, outer_neighbors)
    # TODO: consider nodes_info as input
    want_to_leave = Game.count_nodes_wanting_to_move(inside_nodes_info, 0)
    want_to_join = Game.count_nodes_wanting_to_move(outside_nodes_info, 1)
    fraction_want_to_leave = want_to_leave / len(nodes_in_community)
    fraction_want_to_join = want_to_join / len(outer_neighbors)
    return {
      'fraction_want_to_leave': fraction_want_to_leave,
      'fraction_want_to_join': fraction_want_to_join
    }

  def resolution_spectrum(self, membership, resolutions=None, return_robustness=True):
    resolutions = np.linspace(0, 1, 11) if resolutions is None else resolutions 
    nodes_info = self.get_nodes_info(membership)
    nodes_satisfaction = [Game.classify_node_satisfaction(info, membership[node]) for node, info in nodes_info.items()]
    satisfaction_count = Counter(nodes_satisfaction)
    always_satisfied = satisfaction_count['always_satisfied']
    robustness = always_satisfied / len(nodes_satisfaction)
    if satisfaction_count['relatively_satisfied'] == 0:
      fractions = [robustness] * len(resolutions)
    else:
      nodes_in_doubt = [node for node, satisfaction in enumerate(nodes_satisfaction) if satisfaction == 'relatively_satisfied']
      nodes_info_subset = {node: nodes_info[node] for node in nodes_in_doubt}
      potentials, nodes, communities = Game.get_nodes_potential(nodes_info_subset, resolutions)
      nodes_eq = Game.is_in_equilibrium(membership, potentials, nodes, communities, return_dict=True)
      fractions = (sum(nodes_eq.values()) + always_satisfied) / len(membership)
    if return_robustness:
      return resolutions, fractions, robustness
    return resolutions, fractions

  def equilibrium_fraction(self, resolution, membership):
    nodes_info = self.get_nodes_info(membership)
    potentials, nodes, communities = Game.get_nodes_potential(nodes_info, resolution)
    nodes_equilibrium = Game.is_in_equilibrium(membership, potentials, nodes, communities, return_dict=True)
    fraction = sum(nodes_equilibrium.values()) / len(nodes_equilibrium)
    return fraction

  @staticmethod
  def fraction_equilibrium_nodes(resolution, nodes_info, membership):
    nodes_potential = Game.get_nodes_potential(nodes_info, resolution)
    nodes_equilibrium = Game.are_nodes_in_equilibrium(nodes_potential, membership)
    fraction = sum(nodes_equilibrium.values()) / len(nodes_equilibrium)
    return fraction

  @staticmethod
  def get_node_potential(node_info, resolution):
    node_potential = dict()
    for community, counts in node_info.items():
      pros = counts['friends'] * (1-resolution)
      cons = counts['strangers'] * resolution
      node_potential[community] = pros - cons
    return node_potential

  @staticmethod
  def get_nodes_potential(nodes_info, resolutions=1):
    """
    Compute the potential of each node for one or more resolution values in a vectorized manner.

    For each node and each community, the potential is defined as:
      potential = friends * (1 - resolution) - strangers * resolution
    where "friends" and "strangers" are the counts provided in nodes_info.

    Parameters:
      nodes_info (dict):
        A dictionary mapping node IDs to dictionaries that map community IDs to 
        a dict with keys 'friends' and 'strangers'. For example:
          {
            0: {0: {'friends': 3, 'strangers': 7}, 1: {'friends': 2, 'strangers': 5}},
            1: {0: {'friends': 4, 'strangers': 6}, 1: {'friends': 1, 'strangers': 8}},
            ...
          }
      resolutions (float or array-like):
        A scalar or an array of resolution values in the interval [0, 1].

    Returns:
      potentials: If multiple resolutions are provided, an array of shape 
        (num_resolutions, num_nodes, num_communities) with the computed potentials.
        If a single resolution is provided, a 2D array of shape (num_nodes, num_communities).
      nodes (list):
        Sorted list of node IDs (rows in the resulting arrays).
      communities (list):
        Sorted list of community IDs (columns in the resulting arrays).
    """
    # Extract sorted lists of nodes and communities.
    nodes = sorted(nodes_info.keys())
    # Assume each node_info has the same set of communities.
    communities = sorted(next(iter(nodes_info.values())).keys())
    
    num_nodes = len(nodes)
    num_communities = len(communities)
    
    # Build arrays for friends and strangers counts.
    friends = np.empty((num_nodes, num_communities), dtype=float)
    strangers = np.empty((num_nodes, num_communities), dtype=float)
    
    for i, node in enumerate(nodes):
      for j, comm in enumerate(communities):
        counts = nodes_info[node][comm]
        friends[i, j] = counts['friends']
        strangers[i, j] = counts['strangers']
    
    # Ensure resolutions is a NumPy array.
    resolutions = np.atleast_1d(resolutions).astype(float)
    # Reshape resolutions to allow broadcasting: shape (num_res, 1, 1)
    res = resolutions[:, None, None]
    
    # Compute potentials for each resolution, node, and community:
    # potential = friends*(1 - resolution) - strangers*(resolution)
    potentials = friends[None, :, :] * (1 - res) - strangers[None, :, :] * res
    
    # If a single resolution was provided, return a 2D array instead of 3D.
    if potentials.shape[0] == 1:
      potentials = potentials[0]
    
    return potentials, nodes, communities

  @staticmethod
  def is_node_in_equilibrium(node_comm, potentials):
    best = max(potentials.values())
    return best == potentials[node_comm]

  @staticmethod
  def are_nodes_in_equilibrium(nodes_potential, membership_list):
    nodes_equilibrium = dict()
    for node, potentials in nodes_potential.items():
      nodes_equilibrium[node] = Game.is_node_in_equilibrium(membership_list[node], potentials)
    return nodes_equilibrium

  @staticmethod
  def resolution_effect_on_equilibrium(nodes_info, membership_list):
    fractions = []
    resolutions = np.linspace(0, 1, 11)
    for res in resolutions:
      nodes_potential = Game.get_nodes_potential(nodes_info, res)
      nodes_equilibrium = Game.are_nodes_in_equilibrium(nodes_potential, membership_list)
      fraction = sum(nodes_equilibrium.values()) / len(nodes_equilibrium)
      fractions.append(fraction)
    return resolutions, fractions

  @staticmethod
  def is_in_equilibrium(membership_list, potentials, nodes, communities,
                                 node_ids=None, resolution_ids=None, return_dict=False):
    """
    Check the equilibrium status for multiple nodes and resolutions in a fully vectorized manner.
    
    A node is in equilibrium at a given resolution if the potential for its own community
    is (approximately) equal to the maximum potential over all communities for that node.
    
    Parameters:
      membership_list (list): Global list (indexed by node id) of each node's community.
      potentials (np.ndarray): Either a 3D array of shape 
        (num_resolutions, num_nodes, num_communities) or a 2D array 
        (num_nodes, num_communities) for a single resolution.
      nodes (list): Sorted list of node IDs corresponding to the second axis of potentials.
      communities (list): Sorted list of community IDs corresponding to the third axis of potentials.
      node_ids (int, list, or None): Node id(s) to check. If None, all nodes in `nodes` are considered.
      resolution_ids (int, list, or None): Resolution index/indices to check.
        If None, all resolutions are used.
      return_dict (bool): If True, returns a dict mapping each node id to its equilibrium status.
        Otherwise, returns a tuple (eq_status, node_ids_array, resolution_ids_array), where:
          - eq_status is a NumPy boolean array of shape (num_resolutions, num_selected_nodes) (or 1D if only one resolution)
          - node_ids_array and resolution_ids_array are the used indices.
    
    Returns:
      Either a dictionary mapping node ids to equilibrium status (boolean or boolean vector)
      or a tuple (eq_status, node_ids_array, resolution_ids_array).
      
    Raises:
      ValueError: If any specified resolution index is out of bounds.
    """
    # Ensure potentials is a NumPy array.
    potentials = np.asarray(potentials)
    if potentials.ndim == 2:
      # Convert to 3D with one resolution.
      potentials = potentials[None, :, :]
    num_res, num_nodes, num_comms = potentials.shape

    # Process resolution_ids.
    if resolution_ids is None:
      resolution_ids = np.arange(num_res)
    else:
      if isinstance(resolution_ids, int):
        resolution_ids = np.array([resolution_ids])
      else:
        resolution_ids = np.array(resolution_ids)
      if np.any(resolution_ids < 0) or np.any(resolution_ids >= num_res):
        raise ValueError(f"resolution_ids must be between 0 and {num_res - 1}.")

    # Process node_ids.
    if node_ids is None:
      # Use all nodes as given by the ordering in 'nodes'.
      node_ids = np.array(nodes)
      node_indices = np.arange(len(nodes))
    else:
      if isinstance(node_ids, int):
        node_ids = np.array([node_ids])
      else:
        node_ids = np.array(node_ids)
      # Create a mapping from node id to its index in the sorted list.
      node_to_index = {n: i for i, n in enumerate(nodes)}
      node_indices = np.array([node_to_index[n] for n in node_ids])
    
    # Create a mapping for communities (from community value to its column index).
    comm_to_index = {c: i for i, c in enumerate(communities)}
    # For each selected node, map its community (from membership_list) to a community index.
    own_comm_indices = np.array([comm_to_index[membership_list[n]] for n in node_ids])
    
    # Slice potentials for the selected resolutions and nodes.
    # This gives an array of shape (len(resolution_ids), len(node_ids), num_comms).
    subpotentials = potentials[resolution_ids][:, node_indices, :]
    
    # Compute the maximum potential across communities for each (resolution, node).
    # Shape: (len(resolution_ids), len(node_ids))
    max_potentials = subpotentials.max(axis=2)
    
    # Build index arrays for the resolutions and nodes.
    n_res = len(resolution_ids)
    n_sel = len(node_ids)
    res_idx = np.arange(n_res)[:, None]   # shape (n_res, 1)
    node_idx = np.arange(n_sel)[None, :]    # shape (1, n_sel)
    # Use fancy indexing to select the potential corresponding to each node's own community.
    # own_comm_indices is (n_sel,) and we want to broadcast it to shape (n_res, n_sel).
    own_potentials = subpotentials[res_idx, node_idx, own_comm_indices[None, :]]
    
    # Check equilibrium: node is in equilibrium if its own potential equals the maximum (within tolerance).
    eq_status = np.isclose(own_potentials, max_potentials)
    
    # If only one resolution is selected, squeeze the resolution axis.
    if eq_status.shape[0] == 1:
      eq_status = eq_status[0]
    
    if return_dict:
      # Create a dictionary mapping each node id to its equilibrium status.
      # If multiple resolutions are used, each value is a boolean array.
      result = {node: (eq_status[:, i] if eq_status.ndim == 2 else bool(eq_status[i]))
                for i, node in enumerate(node_ids)}
      return result
    else:
      return eq_status, node_ids, resolution_ids

