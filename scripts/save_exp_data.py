import os
import pickle
import csv
from tqdm import tqdm
from config import experiment_params
from utils import generate_graph, get_ground_truth, get_initial_membership


def save_graphs(base_path, number_of_communities, community_size, probabilities, difficulties, network_seeds):
    os.makedirs(base_path, exist_ok=True)
    
    for net_seed in tqdm(network_seeds, desc="Network Seeds", leave=False):
        for p_in in tqdm(probabilities, desc="Probabilities"):
            for multiplier in tqdm(difficulties, desc="Difficulties", leave=False):
                g = generate_graph(number_of_communities, community_size, p_in, multiplier, net_seed)
                # Create full path for graph
                graph_path = os.path.join(
                    base_path,
                    f"{number_of_communities}C_{community_size*number_of_communities}N",
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
            # Create full path for membership
            membership_path = os.path.join(
                base_path,
                f"{number_of_communities}C_{community_size*number_of_communities}N",
                f"Noise = {noise:.2f}"
            )
            os.makedirs(membership_path, exist_ok=True)
            
            # Save membership as CSV
            membership_filename = os.path.join(membership_path, f"partition_{partition_seed:03d}.csv")
            with open(membership_filename, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(initial_membership)

def main():

    # Call the functions with the correct paths
    base_path = os.path.expanduser(experiment_params['output_results_path'])
    graphs_path = os.path.join(base_path, 'networks')
    memberships_path = os.path.join(base_path, 'memberships')


    save_memberships(
        memberships_path,
        experiment_params['number_of_communities'],
        experiment_params['community_size'],
        experiment_params['noises'],
        experiment_params['partition_seeds']
    )


    save_graphs(
        graphs_path,
        experiment_params['number_of_communities'],
        experiment_params['community_size'],
        experiment_params['probabilities'],
        experiment_params['difficulties'],
        experiment_params['network_seeds']
    )

    return True


__name__ == '__main__' and main()