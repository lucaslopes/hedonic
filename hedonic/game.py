import random
import igraph as ig

class HedonicGame(ig.Graph):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def hedonic_value(self, neighbors, strangers, resolution):
    pros = neighbors * (1-resolution) # prosocial value
    cons = strangers *    resolution # antisocial value
    return pros - cons # hedonic value
  
  def initial_membership(self, max_communitites=0):
    if max_communitites < 1:
      membership = [node.index for node in self.vs] # all nodes are isolated
    else:
      membership = [random.choice(range(max_communitites)) for _ in self.vs]
    return membership

  def initialize_game(self, max_communities):
    self['communities_nodes'] = dict() # communities are sets of nodes
    self['communities_edges'] = dict() # communities are count of internal edges
    initial_state = self.initial_membership(max_communities)
    for node, community in zip(self.vs, initial_state):
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

  def community_hedonic(self, resolution=1, max_communities=2):
    self.initialize_game(max_communities)
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