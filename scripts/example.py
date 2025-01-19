import igraph as ig
import hedonic as hd
from stopwatch import Stopwatch
from utils import generate_graph, get_ground_truth, get_initial_membership

stopwatch = Stopwatch() # create Stopwatch instance

# for seed in range(100):
seed=42
n_communities = 2
community_size = 500
g = generate_graph(n_communities, community_size, 0.05, 0.4, seed)
gt = get_ground_truth(g, n_communities, community_size)
density = g.density()

stopwatch.reset()
stopwatch.start()
p1 = g.community_hedonic()
stopwatch.stop()
print('community_hedonic:', stopwatch.duration, g.in_equibrium(density), ig.compare_communities(p1, gt, method="rand"))

stopwatch.reset()
stopwatch.start()
p2 = g.community_hedonic_old()
stopwatch.stop()
print('community_hedonic_old:', stopwatch.duration, g.in_equibrium(density), ig.compare_communities(p2, gt, method="rand"))


stopwatch.reset()
stopwatch.start()
p3 = g.community_leiden(resolution=density, n_iterations=-1)
stopwatch.stop()
g.initialize_game(p3.membership)
print('community_leiden:', stopwatch.duration, g.in_equibrium(density), ig.compare_communities(p3, gt, method="rand"))

stopwatch = Stopwatch() # create Stopwatch instance
stopwatch.start()
p4 = g.community_leiden(resolution=density, n_iterations=-1, hedonic=True)
g.initialize_game(p4.membership)
stopwatch.stop()
print('community_leiden_hedonic:', stopwatch.duration, g.in_equibrium(density), ig.compare_communities(p4, gt, method="rand"))

print('\n'*3)
