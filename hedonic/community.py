import igraph as ig

def community_hedonic(g):
    membership = [0] * g.vcount()
    return ig.clustering.VertexClustering(g, membership)