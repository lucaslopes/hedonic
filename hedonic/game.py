import igraph as ig

class HedonicGame(ig.Graph):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def community_hedonic(self, resolution=1):
    membership = [0 for _ in self.vs]
    return ig.clustering.VertexClustering(self, membership)