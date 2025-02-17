import os
import pickle
import csv
from tqdm import tqdm
from config import experiment_params
from hedonic import Game
from utils import (
    generate_graph, get_ground_truth, get_initial_membership, read_txt_gz_to_igraph, read_communities)


def save_real_graph(file_path, communities_path, noises, partition_seeds):

    def get_new_index(old_index):
        return g.vs.find(label=old_index).index

    g = Game(read_txt_gz_to_igraph(file_path))
    g.vs["label"] = [v.index for v in g.vs]
    g.delete_vertices(g.vs.select(_degree_eq=0))
    
    communities = read_communities(communities_path)
    communities = [[get_new_index(c.index) for c in comm] for comm in communities]
    ground_truth = [g.community_to_partition(c) for c in communities]
    for gt in ground_truth:
        for noise in tqdm(noises, desc="Noises"):
            for partition_seed in tqdm(seeds, desc="Partition Seeds", leave=False):    
                initial_membership = get_initial_membership(gt, noise, partition_seed)
    
    with open(pickle_path, 'wb') as f:
        # TODO: this should be done in other process
        pickle.dump(g, f)


def save_graphs(base_path, number_of_communities, community_size, probabilities, difficulties, network_seeds):
    os.makedirs(base_path, exist_ok=True)
    
    for net_seed in tqdm(network_seeds, desc="Network Seeds", leave=False):
        for p_in in tqdm(probabilities, desc="Probabilities"):
            for multiplier in tqdm(difficulties, desc="Difficulties", leave=False):
                g = generate_graph(number_of_communities, community_size, p_in, multiplier, net_seed)
                # Create full path for graph
                graph_path = os.path.join(
                    base_path,
                    f"{number_of_communities}C_{community_size}N",
                    f"P_in = {p_in:.2f}",
                    f"Difficulty = {multiplier:.2f}"
                )
                os.makedirs(graph_path, exist_ok=True)
                
                # Save graph
                graph_filename = os.path.join(graph_path, f"network_{net_seed:03d}.pkl")
                with open(graph_filename, 'wb') as file:
                    pickle.dump(g, file)


def save_memberships(base_path, number_of_communities, community_size, noises, partition_seeds):
    os.makedirs(base_path, exist_ok=True)
    
    ground_truth = get_ground_truth(number_of_communities, community_size)
    for noise in tqdm(noises, desc="Noises"):
        seeds = [0] if noise > 1 else partition_seeds
        for partition_seed in tqdm(seeds, desc="Partition Seeds", leave=False):    
            initial_membership = get_initial_membership(ground_truth, noise, partition_seed)
            save_membership(
                base_path,
                number_of_communities,
                community_size,
                noise,
                partition_seed,
                initial_membership
            )


def save_membership(base_path, number_of_communities, community_size, noise, partition_seed, membership):
    """
    Saves a membership list as a CSV file.
    """
    # Create full path for membership
    membership_path = os.path.join(
        base_path,
        f"{number_of_communities}C_{community_size}N",
        f"Noise = {noise:.2f}"
    )
    os.makedirs(membership_path, exist_ok=True)
    
    # Save membership as CSV
    membership_filename = os.path.join(membership_path, f"partition_{partition_seed:03d}.csv")
    with open(membership_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(membership)

def main():

    # communities_path = '/Users/lucas/Databases/Hedonic/Networks/DBLP/com-dblp.all.cmty.txt.gz'
    # communities_path = '/Users/lucas/Databases/Hedonic/Networks/DBLP/com-dblp.top5000.cmty.txt.gz'

    experiment_params = {
        'output_results_path': '/Users/lucas/Databases/Hedonic/PHYSA/Synthetic_Networks/V1020',
        'number_of_communities': 6,
        'community_size': int(1020/6),
        'network_seeds': [i for i in range(100)],
        'partition_seeds': [i for i in range(10)],
        'noises': [.1, .25, .5, .75, 1],
        'probabilities': [.10, .09, .08, .07, .06, .05, .04, .03, .02, .01],
        'difficulties': [.75, .7, .65, .6, .55, .5, .4, .3, .2, .1],
        }


    # Call the functions with the correct paths
    base_path = os.path.expanduser(experiment_params['output_results_path'])
    graphs_path = os.path.join(base_path, 'networks')
    memberships_path = os.path.join(base_path, 'memberships')

    for n_communities in [2, 3, 4, 5, 6]:
        save_memberships(
            memberships_path,
            n_communities,
            int(1020/n_communities),
            experiment_params['noises'],
            experiment_params['partition_seeds']
        )
        save_graphs(
            graphs_path,
            n_communities,
            int(1020/n_communities),
            experiment_params['probabilities'],
            experiment_params['difficulties'],
            experiment_params['network_seeds']
        )

    return True


__name__ == '__main__' and main()