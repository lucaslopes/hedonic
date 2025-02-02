import argparse
import os
from robust import get_edges_and_partitions
from plot import plot_network
import matplotlib.pyplot as plt

def get_unique_filename(directory, base_filename):
    """
    Generates a unique filename by appending a number if the file already exists.
    
    Args:
        directory (str): The directory where the file will be saved.
        base_filename (str): The base filename without extension.
    
    Returns:
        str: A unique filename with extension.
    """
    filename = f"{base_filename}.png"
    counter = 1
    while os.path.exists(os.path.join(directory, filename)):
        filename = f"{base_filename}_{counter}.png"
        counter += 1
    return filename

def main():
    parser = argparse.ArgumentParser(description='Generate edges and partitions for a given size n.')
    parser.add_argument('n', type=int, help='The size of the set to partition and the number of nodes in the network')
    args = parser.parse_args()
    
    result = get_edges_and_partitions(args.n, filter_robust=True)
    for idx, (network, partitions) in enumerate(result.items()):
        for partition_idx, (partition, fraction) in enumerate(partitions.items()):
            fig, ax = plot_network(network, partition)
            base_filename = f"network_{idx}_partition_{partition_idx}"
            
            # Create directory for the number of nodes if it doesn't exist
            directory = os.path.join("figs", f"nodes_{ args.n}")
            os.makedirs(directory, exist_ok=True)
            
            unique_filename = get_unique_filename(directory, base_filename)
            fig.savefig(os.path.join(directory, unique_filename))
            plt.close(fig)
            print(f"Saved plot as {unique_filename}")

if __name__ == "__main__":
    main()