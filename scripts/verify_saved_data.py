import os
import pickle
import csv
import hedonic as hd
from config import experiment_params


def load_and_verify_graph(graph_path):
    """Load a graph and verify its basic properties."""
    try:
        with open(graph_path, 'rb') as file:
            print(file)
            g = pickle.load(file)
            
        # Basic verification
        assert isinstance(g, hd.Game), "Loaded object is not a NetworkX graph"
        assert len(g.vs()) > 0, "Graph has no nodes"
        assert len(g.es()) > 0, "Graph has no edges"
        
        print(f"Successfully loaded graph from {graph_path}")
        print(f"Nodes: {len(g.vs())}")
        print(f"Edges: {len(g.es())}")
        return g
    
    except Exception as e:
        print(f"Error loading graph from {graph_path}: {str(e)}")
        return None


def load_and_verify_membership(membership_path):
    """Load a membership list and verify its basic properties."""
    try:
        with open(membership_path, 'r', newline='') as file:
            reader = csv.reader(file)
            membership = [int(x) for x in next(reader)]  # Read first row
            
        # Basic verification
        assert len(membership) > 0, "Membership list is empty"
        assert all(isinstance(x, int) for x in membership), "Not all elements are integers"
        
        print(f"Successfully loaded membership from {membership_path}")
        print(f"Number of nodes: {len(membership)}")
        print(f"Number of communities: {len(set(membership))}")
        return membership
    
    except Exception as e:
        print(f"Error loading membership from {membership_path}: {str(e)}")
        return None


def verify_saved_data():
    """Verify a sample of saved graphs and memberships."""
    base_path = os.path.expanduser(experiment_params['output_results_path'])
    
    # Check one graph from each configuration
    graphs_path = os.path.join(base_path, 'graphs')
    for root, dirs, files in os.walk(graphs_path):
        for file in files:
            if file.endswith('.pkl'):
                graph_path = os.path.join(root, file)
                print("\nVerifying graph:", graph_path)
                g = load_and_verify_graph(graph_path)
                break  # Only check first graph in each directory
    
    # Check one membership from each noise level
    memberships_path = os.path.join(base_path, 'memberships')
    for root, dirs, files in os.walk(memberships_path):
        for file in files:
            if file.endswith('.csv'):
                membership_path = os.path.join(root, file)
                print("\nVerifying membership:", membership_path)
                membership = load_and_verify_membership(membership_path)
                break  # Only check first membership in each directory


def get_graph_and_partition_paths(base_path):
    """Get lists of graph and partition paths from the given base path."""
    graphs_path = os.path.join(base_path, 'graphs')
    partitions_path = os.path.join(base_path, 'partitions')
    
    graph_files = []
    partition_files = []
    
    for root, dirs, files in os.walk(graphs_path):
        for file in files:
            if file.endswith('.pkl'):
                graph_files.append(os.path.join(root, file))
    
    for root, dirs, files in os.walk(partitions_path):
        for file in files:
            if file.endswith('.csv'):
                partition_files.append(os.path.join(root, file))
    
    return graph_files, partition_files


if __name__ == "__main__":
    verify_saved_data()
