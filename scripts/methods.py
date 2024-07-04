import igraph as ig
from hedonic import HedonicGame

#################################################

class CommunityMethods(HedonicGame):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def community_groundtruth(self, groundtruth):
    return ig.clustering.VertexClustering(self, groundtruth) if type(groundtruth) == list else groundtruth

  def community_local_improvement(self, initial_membership=None):
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
    return ig.clustering.VertexClustering(self, new_membership)

  def community_local_improvement_hedonic(self, initial_membership=None):
    self.initialize_game(initial_membership)
    nodes_to_move = list()
    for node in self.vs:
      pref_comm = self.get_preferable_community(node, 0)  # resolution = 0 means that the algorithm will try to move nodes to the community with the highest number of neighbors in the community
      if pref_comm != node['community']:
        nodes_to_move.append((node.index, pref_comm))
    while nodes_to_move:
      node, community = nodes_to_move.pop()
      self.move_node_to_community(self.vs[node], community)
    return ig.clustering.VertexClustering(self, self.membership())

