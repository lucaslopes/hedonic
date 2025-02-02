import argparse
from partitions import partition_set
from edges import generate_valid_graphs
from collections import Counter
from tqdm import tqdm

def is_robust(node: int, membership: list, neighbors: set, cluster_counts: Counter):
    """
    Placeholder function to determine if a node is robust.
    
    Args:
        node (int): The node to check.
        edges (list of tuple): The list of edges in the network.
        membership (list of int): The membership list where index is the node and value is the cluster index.
    
    Returns:
        bool: True if the node is robust, False otherwise.
    """
    neighbors_in_cluster = Counter(membership[neighbor] for neighbor in neighbors)
    strangers_in_cluster = {cluster: n - neighbors_in_cluster[cluster] for cluster, n in cluster_counts.items()}
    strangers_in_cluster[membership[node]] -= 1
    
    max_neighbors_cluster = max(neighbors_in_cluster.values())
    min_strangers_cluster = min(strangers_in_cluster.values())
    
    is_max_neighbors_cluster = neighbors_in_cluster[membership[node]] == max_neighbors_cluster
    is_min_strangers_cluster = strangers_in_cluster[membership[node]] == min_strangers_cluster

    return is_max_neighbors_cluster and is_min_strangers_cluster


def fraction_of_robust_nodes(edges, membership, neighbors, cluster_counts: Counter):
    """
    Calculates the fraction of robust nodes in the network.
    
    Args:
        edges (list of tuple): The list of edges in the network.
        membership (list of int): The membership list where index is the node and value is the cluster index.
    
    Returns:
        float: The fraction of robust nodes.
    """
    total_nodes = len(membership)
    robust_nodes = sum(is_robust(node, membership, neighbors[node], cluster_counts) for node in range(total_nodes))
    return robust_nodes / total_nodes if total_nodes > 0 else 0.0


def verify_cluster_degree_distributions(edges, partition):
    """
    Verifies the degree distributions within the clusters defined by the partition.
    
    Args:
        edges (list of tuple): The list of edges in the network.
        partition (list): The partition to verify.
    
    Returns:
        list: A list where each element is a list of tuples. Each tuple contains the degree and the number of nodes with that degree in the cluster.
    """
    cluster_degrees = {i: [] for i in range(len(set(partition)))}
    node_degrees = {i: 0 for i in range(len(partition))}
    
    for edge in edges:
        u, v = edge
        if partition[u] == partition[v]:
            node_degrees[u] += 1
            node_degrees[v] += 1
    
    for node, degree in node_degrees.items():
        cluster_degrees[partition[node]].append(degree)
    
    degree_distribution = []
    for cluster in sorted(cluster_degrees.keys()):
        degree_count = Counter(cluster_degrees[cluster])
        degree_distribution.append(sorted(degree_count.items()))
    
    return sorted(degree_distribution)

def get_edges_and_partitions(n, filter_robust=True):
    """
    Returns the edges and partitions for a given size n.
    
    Args:
        n (int): The size of the set to partition and the number of nodes in the network.
    
    Returns:
        tuple: A tuple containing the list of edges and the list of partitions.
    """
    result = {}
    networks = generate_valid_graphs(n)
    partitions = partition_set(n, format='membership')
    for network in tqdm(networks, desc="Processing networks"):
        neighbors = {node: set() for node in range(n)}
        for edge in network:
            neighbors[edge[0]].add(edge[1])
            neighbors[edge[1]].add(edge[0])
        network_key = tuple(sorted(network))
        result[network_key] = {}
        seen_distributions = set()
        for partition in partitions:
            degree_distributions = verify_cluster_degree_distributions(network, partition)
            distribution_key = tuple(tuple(tuple(x) for x in dist) for dist in degree_distributions)
            if distribution_key not in seen_distributions:
                seen_distributions.add(distribution_key)
                cluster_counts = Counter(partition)
                f = fraction_of_robust_nodes(network, partition, neighbors, cluster_counts)
                if not filter_robust or (f == 1.0 and len(set(partition)) > 1):
                    partition_key = tuple(partition)
                    result[network_key][partition_key] = f
    return result

def main():
    parser = argparse.ArgumentParser(description='Generate edges and partitions for a given size n.')
    parser.add_argument('n', type=int, help='The size of the set to partition and the number of nodes in the network')
    args = parser.parse_args()
    
    result = get_edges_and_partitions(args.n)
    for network, partitions in result.items():
        for partition, fraction in partitions.items():
            if fraction == 1.0 and len(set(partition)) > 1:
                print(f"Network: {network}, Partition: {partition}, Fraction of robust nodes: {fraction:.4f}")

if __name__ == "__main__":
    main()

