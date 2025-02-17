import numpy as np
import igraph as ig
from tqdm import tqdm
from stopwatch import Stopwatch
from collections import Counter
from hedonic import Game
from utils import generate_graph, get_ground_truth, get_initial_membership, read_txt_gz_to_igraph, read_pickle

stopwatch = Stopwatch() # create Stopwatch instance

# Initialize counters and duration sums
counts = {'hedonic_queue': 0, 'hedonic_traversal': 0, 'leiden': 0, 'leiden_hedonic': 0}
durations = {'hedonic_queue': 0, 'hedonic_traversal': 0, 'leiden': 0, 'leiden_hedonic': 0}

if (real_data := True):

    # file_path = '/Users/lucas/Databases/Hedonic/Networks/DBLP/raw/com-dblp.ungraph.txt.gz'
    # g = Game(read_txt_gz_to_igraph(file_path))
    # g.vs["label"] = list(range(g.vcount()))
    # g.delete_vertices(g.vs.select(_degree_eq=0))
    
    file_path = '/Users/lucas/Databases/Hedonic/Networks/DBLP/pkl/com-dblp.ungraph.pkl'
    communities_path = '/Users/lucas/Databases/Hedonic/Networks/DBLP/pkl/com-dblp.top5000.cmty.pkl'
    g = read_pickle(file_path)
    ground_truth = read_pickle(communities_path)

    density = g.density()
    print(g.summary())

    for community in ground_truth[:11]:
        # initial_membership = [0 if np.random.random() < 0.5 else 1 for _ in range(g.vcount())]
        print(g.vcount(), len(community))
        subset = {j for i in community for j in g.neighbors(i)}.union(set(community))
        for i in subset:
            g.vs[i]["label"] = i
        new_g = g.subgraph(subset)
        initial_membership = np.zeros(len(subset), dtype=int)
        for i in community:
            index = new_g.vs.find(label=i).index
            initial_membership[index] = 1
        print(initial_membership)
                
        stopwatch.reset()
        stopwatch.start()
        p3 = new_g.community_leiden(resolution=density, n_iterations=-1, initial_membership=initial_membership, can_create_new_clusters=False)
        stopwatch.stop()
        new_g.initialize_game(p3.membership)
        counts['leiden'] += new_g.in_equibrium(density)
        durations['leiden'] += stopwatch.duration
        print('community_leiden:', stopwatch.duration, new_g.in_equibrium(density), ig.compare_communities(p3, initial_membership, method="rand"))

        stopwatch.reset()
        stopwatch.start()
        p4 = new_g.community_leiden(resolution=density, n_iterations=-1, only_first_phase=True, initial_membership=initial_membership, can_create_new_clusters=False)
        new_g.initialize_game(p4.membership)
        stopwatch.stop()
        counts['leiden_hedonic'] += new_g.in_equibrium(density)
        durations['leiden_hedonic'] += stopwatch.duration
        print('community_leiden_hedonic:', stopwatch.duration, new_g.in_equibrium(density), ig.compare_communities(p4, initial_membership, method="rand"))


if (synthetic := False):

    for seed in tqdm(range(100)):
        # seed=42
        n_communities = 5
        community_size = 1000
        g = generate_graph(n_communities, community_size, 0.05, 0.4, seed)
        gt = get_ground_truth(n_communities, community_size, g)
        density = g.density()

        # stopwatch.reset()
        # stopwatch.start()
        # p1 = g.community_hedonic_queue()
        # stopwatch.stop()
        # counts['hedonic_queue'] += g.in_equibrium(density)
        # durations['hedonic_queue'] += stopwatch.duration
        # # print('community_hedonic:', stopwatch.duration, g.in_equibrium(density), ig.compare_communities(p1, gt, method="rand"))

        # stopwatch.reset()
        # stopwatch.start()
        # p2 = g.community_hedonic_traversal()
        # stopwatch.stop()
        # counts['hedonic_traversal'] += g.in_equibrium(density)
        # durations['hedonic_traversal'] += stopwatch.duration
        # print('community_hedonic_traversal:', stopwatch.duration, g.in_equibrium(density), ig.compare_communities(p2, gt, method="rand"))

        stopwatch.reset()
        stopwatch.start()
        p3 = g.community_leiden(resolution=density, n_iterations=-1)
        stopwatch.stop()
        g.initialize_game(p3.membership)
        counts['leiden'] += g.in_equibrium(density)
        durations['leiden'] += stopwatch.duration
        # print('community_leiden:', stopwatch.duration, g.in_equibrium(density), ig.compare_communities(p3, gt, method="rand"))

        stopwatch = Stopwatch() # create Stopwatch instance
        stopwatch.start()
        p4 = g.community_leiden(resolution=density, n_iterations=-1, hedonic=True)
        g.initialize_game(p4.membership)
        stopwatch.stop()
        counts['leiden_hedonic'] += g.in_equibrium(density)
        durations['leiden_hedonic'] += stopwatch.duration
        # print('community_leiden_hedonic:', stopwatch.duration, g.in_equibrium(density), ig.compare_communities(p4, gt, method="rand"))

        # print('\n'*3)

# Print final counts and mean durations
print('Final Counts:', counts)
print('Mean Durations:', {key: durations[key] / 100 for key in durations})

print('\n'*3)
