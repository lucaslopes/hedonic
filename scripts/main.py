import igraph as ig
from hedonic import community_hedonic


g = ig.Graph.Erdos_Renyi(30,0.3)

comms_ml = g.community_multilevel()
comms_ld = g.community_leiden()

print('community multilevel: \n', comms_ml)
print('community leiden: \n', comms_ld)
print('community hedonic: \n', community_hedonic(g))