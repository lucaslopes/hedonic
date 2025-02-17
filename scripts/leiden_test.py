import numpy as np
import igraph as ig
from tqdm import tqdm
from stopwatch import Stopwatch
from hedonic import Game
from collections import Counter
from utils import read_pickle, read_csv_partition, get_ground_truth, get_initial_membership, read_txt_gz_to_igraph


partition_pth = '/Users/lucas/Databases/Hedonic/PHYSA/Synthetic_Networks/V2040/memberships/4C_2040N/Noise = 0.90/partition_009.csv'

graph_pth = '/Users/lucas/Databases/Hedonic/PHYSA/Synthetic_Networks/V2040/networks/4C_2040N/P_in = 0.03/Difficulty = 0.30/network_003.pkl'

g = read_pickle(graph_pth)
membership = read_csv_partition(partition_pth)

p1a = g.community_leiden(
    resolution=g.density(), 
    n_iterations=-1, 
    initial_membership=membership
)
p2a = g.community_leiden(
    resolution=g.density(), 
    n_iterations=-1, 
    initial_membership=membership,
    can_create_new_clusters=False
)
p3a = g.community_leiden(
    resolution=g.density(), 
    n_iterations=-1, 
    initial_membership=membership,
    only_first_phase=True,
    can_create_new_clusters=False
)
p4a = g.community_hedonic_queue(
    resolution=g.density(), 
    initial_membership=membership
)


p1b = g.community_leiden(
    resolution=g.density(), 
    n_iterations=-1, 
    initial_membership=membership
)
p2b = g.community_leiden(
    resolution=g.density(), 
    n_iterations=-1, 
    initial_membership=membership,
    can_create_new_clusters=False
)
p3b = g.community_leiden(
    resolution=g.density(), 
    n_iterations=-1, 
    initial_membership=membership,
    only_first_phase=True,
    can_create_new_clusters=False
)
p4b = g.community_hedonic_queue(
    resolution=g.density(), 
    initial_membership=membership
)


print(ig.compare_communities(p1a, p1b, method="rand"))
print(ig.compare_communities(p2a, p2b, method="rand"))
print(ig.compare_communities(p3a, p3b, method="rand"))
print(ig.compare_communities(p4a, p4b, method="rand"))

print("Partition 1a:", g.equilibrium_fraction(g.density(), p1a.membership), Counter(p1a.membership))
print("Partition 1b:", g.equilibrium_fraction(g.density(), p1b.membership), Counter(p1b.membership))
print("Partition 2a:", g.equilibrium_fraction(g.density(), p2a.membership), Counter(p2a.membership))
print("Partition 2b:", g.equilibrium_fraction(g.density(), p2b.membership), Counter(p2b.membership))
print("Partition 3a:", g.equilibrium_fraction(g.density(), p3a.membership), Counter(p3a.membership))
print("Partition 3b:", g.equilibrium_fraction(g.density(), p3b.membership), Counter(p3b.membership))
print("Partition 4a:", g.equilibrium_fraction(g.density(), p4a.membership), Counter(p4a.membership))
print("Partition 4b:", g.equilibrium_fraction(g.density(), p4b.membership), Counter(p4b.membership))
