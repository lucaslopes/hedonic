#!/usr/bin/env python3

import argparse
from itertools import combinations_with_replacement
from collections import defaultdict, deque


def is_connected(n, edges):
    """
    Check if 'edges' form a connected graph of 'n' nodes (labeled 0..n-1).
    """
    if not edges:
        return False
    
    adj_list = defaultdict(list)
    for u, v in edges:
        adj_list[u].append(v)
        adj_list[v].append(u)

    visited = set()
    queue = deque([0])  # start BFS from node 0
    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            queue.extend(adj_list[node])
    
    return (len(visited) == n)


def degree_distributions(n):
    """
    Returns a list of all unique ways to distribute degrees among n nodes,
    satisfying:
      - Each degree d_i is between 1 and n.
      - The sum of degrees <= (n*(n-1))//2.
      - Order does not matter (distributions are treated as multisets).
    """
    max_sum = n * (n - 1)  # maximum sum of degrees for a simple graph
    possible_degrees = range(1, n + 1)
    
    valid_distributions = []
    # combinations_with_replacement generates sorted n-tuples where order doesn't matter
    for combo in combinations_with_replacement(possible_degrees, n):
        if sum(combo) <= max_sum and sum(combo) % 2 == 0:
            valid_distributions.append(combo)
    
    return valid_distributions

def distribution_to_edges(deg_sequence):
    """
    Given a tuple or list of non-negative integer degrees deg_sequence,
    attempts to construct a simple, undirected graph with those degrees.
    Returns a list of edges (u, v) with u < v if possible, or False if not.
    """
    n = len(deg_sequence)

    # Make a local copy (list of (degree, node_id)) so we can mutate
    nodes = [(deg_sequence[i], i) for i in range(n)]
    # Sort descending by degree
    nodes.sort(key=lambda x: x[0], reverse=True)

    edges = []
    
    while True:
        # If all degrees are 0, we are done
        if nodes[0][0] == 0:
            # All degrees are zero at this point => valid construction
            return edges
        
        # Take the node with the highest degree
        d, node_id = nodes[0]
        nodes = nodes[1:]  # remove it from the list

        # If the node's degree is larger than the remaining nodes, not possible
        if d > len(nodes):
            return False

        # Connect this node with the next d highest-degree nodes
        for i in range(d):
            adj_deg, adj_id = nodes[i]
            
            # If any "target" node has degree 0, we can't reduce it => not valid
            if adj_deg == 0:
                return False

            # Add edge between node_id and adj_id
            u, v = sorted([node_id, adj_id])
            edges.append((u, v))

            # Decrement that node's degree
            nodes[i] = (adj_deg - 1, adj_id)

        # Re-sort the list after decrementing
        nodes.sort(key=lambda x: x[0], reverse=True)

        # If after sorting, we get negative degrees => not valid
        if any(nd[0] < 0 for nd in nodes):
            return False

def generate_valid_graphs(n):
    distributions = degree_distributions(n)
    valid_graphs = []
    for d in distributions:
        edges = distribution_to_edges(d)
        if edges and is_connected(n, edges):
            valid_graphs.append(edges)
    return sorted(valid_graphs)

def main():
    parser = argparse.ArgumentParser(
        description="Generate all connected networks on n nodes, treating two networks "
                    "as identical if they share the same degree distribution."
    )
    parser.add_argument('n', type=int, help='Number of nodes')
    args = parser.parse_args()

    unique_edge_lists = generate_valid_graphs(args.n)
    print(f"Number of valid degree distributions for n = {args.n}: {len(unique_edge_lists)}")
    for g in unique_edge_lists:
        print(g)

if __name__ == "__main__":
    main()