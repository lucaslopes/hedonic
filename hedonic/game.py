import igraph as ig

#################################################

class HedonicGame(ig.Graph):
  def __init__(self, graph=None, *args, **kwargs):
    # Initialize the base class with empty parameters
    super().__init__(*args, **kwargs)
    if type(graph) == ig.Graph: # if graph is an igraph
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

  def community_hedonic_old(self, resolution=None, initial_membership=None, log_memberships=False):
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
    return ig.clustering.VertexClustering(self, self.membership())
  
  def community_hedonic(self, resolution=None, initial_membership=None, log_memberships=False):
    resolution = resolution if resolution else self.density()
    self['log_memberships'] = []
    print("Initializing game...")
    self.initialize_game(initial_membership)
    print("Running hedonic game...")
    while not self.in_equibrium(resolution):
      # Initialize the queue with all nodes
      print("Initializing queue...")
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
    
    return ig.clustering.VertexClustering(self, self.membership())

  ## statistics #################################

  def accuracy(self, partition, ground_truth):
    # Rand index of Rand (1971)
    partition = ig.clustering.VertexClustering(self, partition) if type(partition) == list else partition
    ground_truth = ig.clustering.VertexClustering(self, ground_truth) if type(ground_truth) == list else ground_truth
    return ig.compare_communities(partition, ground_truth, method="rand")
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
    partition = partition if type(partition) == ig.clustering.VertexClustering else ig.clustering.VertexClustering(self, partition)
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