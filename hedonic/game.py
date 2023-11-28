import random
import igraph as ig

#################################################

class HedonicGame(ig.Graph):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def hedonic_value(self, neighbors, strangers, resolution):
    pros = neighbors * (1-resolution) # prosocial value
    cons = strangers *    resolution # antisocial value
    return pros - cons # hedonic value

  def initialize_game(self, initial_membership):
    self['communities_nodes'] = dict() # communities are sets of nodes
    self['communities_edges'] = dict() # communities are count of internal edges
    initial_membership = initial_membership if initial_membership else [node.index for node in self.vs]
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

  def get_preferable_community(self, node, resolution):
    pref_community = node['community']
    highest_hedonic_value = self.value_of_node_in_community(node, node['community'], resolution)
    communities = [c for c, amount in node['neighbors_in_community'].items() if amount > 0]
    for community in communities:
      hedonic = self.value_of_node_in_community(node, community, resolution)
      if hedonic > highest_hedonic_value:
        pref_community = community
        highest_hedonic_value = hedonic
    return pref_community

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

  def community_hedonic(self, resolution=1, initial_membership=None):
    self.initialize_game(initial_membership)
    a_node_has_moved = True
    while a_node_has_moved:
      a_node_has_moved = False
      for node in self.vs:
        pref_comm = self.get_preferable_community(node, resolution)
        if pref_comm != node['community']:
          self.move_node_to_community(node, pref_comm)
          a_node_has_moved = True
    membership = [int(node['community']) for node in self.vs]
    return ig.clustering.VertexClustering(self, membership)
  
  ## statistics #################################

  def accuracy(self, partition, ground_truth):
    n_correct = 0
    for i in range(partition.n):
      for j in range(partition.n):
        pair_ij = partition.membership[i] == partition.membership[j]
        pair_gt = ground_truth.membership[i] == ground_truth.membership[j]
        if pair_ij == pair_gt:
          n_correct += 1
    acc = n_correct / partition.n ** 2
    return (acc - .5) / .5
  
  def robustness(self, partition):
    self.initialize_game(partition.membership)
    robust_nodes = []
    for node in self.vs:
      pref_comm_0 = self.get_preferable_community(node, 0)
      pref_comm_1 = self.get_preferable_community(node, 1)
      robust_nodes.append(
        pref_comm_0 == pref_comm_1 == node['community'])
    return robust_nodes.count(True) / len(robust_nodes)

  ## need to verify #############################

  def potential(self): # need to verify
    global_potential = 0
    for community in list(self['communities_nodes']):
      global_potential += self.potential_of_community(community)
    return global_potential

  def potential_of_community(self, community, alpha=None):
    connections = self['communities_edges'][community]
    missed_connections = self.total_possible_edges(len(self['communities_nodes'][community])) - connections
    return self.hedonic_value(connections, missed_connections, alpha)

  def edges_between(self, community_A, community_B):
    bridges = 0
    for node in self['communities_nodes'][community_A]:
      bridges += self.vs[node]['friends_in_community'][community_B]
    return bridges

  def potential_merged(self, community_A, community_B, alpha=None):
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
      random.shuffle(communities_list)
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