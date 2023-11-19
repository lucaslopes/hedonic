import igraph as ig

class HedonicGame(ig.Graph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add any additional initialization code here

    def community_hedonic(self, resolution=1):
        membership = [0] * self.vcount()
        return ig.clustering.VertexClustering(self, membership)