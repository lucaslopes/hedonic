from os import name
import numpy as np
import igraph as ig
from collections import Counter


def get_neighbor_memberships(graph, membership, node):
    """
    Returns a list of memberships for all neighbors of the given node.
    
    Parameters:
    graph (igraph.Graph): The input graph
    membership (list): List of community memberships for all nodes
    node (int): The node ID to get neighbors' memberships for
    
    Returns:
    list: List of memberships of the node's neighbors
    """
    neighbors = graph.neighbors(node)
    return Counter([membership[neighbor] for neighbor in neighbors])


def get_strangers_in_community(community_counter, neighbors_counter, node_membership):
    """
    Returns a Counter with the number of strangers in each community.
    
    Parameters:
    community_counter (Counter): Counter of memberships for all nodes
    neighbors_counter (Counter): Counter of memberships for all neighbors of the given node
    node_membership (int): Community membership of the node
    
    Returns:
    Counter: Counter of the number of strangers in each community
    """
    strangers_counter = Counter()

    for c, f in neighbors_counter.items():
        strangers_in_community = community_counter[c] - f
        if node_membership == c:
            strangers_in_community -= 1
        strangers_counter[c] = strangers_in_community

    return strangers_counter


def get_node_info(graph, membership_list, community_counter, node):
    """
    """
    node_membership = membership_list[node]
    neighbors_counter = get_neighbor_memberships(graph, membership_list, node)
    neighbors_counter = {c: neighbors_counter[c] if c in neighbors_counter else 0 for c in community_counter}
    strangers_counter = get_strangers_in_community(community_counter, neighbors_counter, node_membership)
    node_info = dict()
    for c in strangers_counter:
        node_info[c] = {
            'friends': neighbors_counter[c],
            'strangers': strangers_counter[c]
        }
    return node_info


def get_nodes_info(graph, membership_list):
    community_counter = Counter(membership_list)
    nodes_info = dict()
    for node in range(graph.vcount()):
        nodes_info[node] = get_node_info(graph, membership_list, community_counter, node)
    return nodes_info


def is_node_robust(node_info, membership_list, node):
    max_friends_in_community = max([info['friends'] for info in node_info.values()])
    min_strangers_in_community = min([info['strangers'] for info in node_info.values()])
    is_maximal_friends = node_info[membership_list[node]]['friends'] == max_friends_in_community
    is_minimal_strangers = node_info[membership_list[node]]['strangers'] == min_strangers_in_community
    return is_maximal_friends and is_minimal_strangers


def get_nodes_robustness(nodes_info, membership_list):
    return {node: is_node_robust(nodes_info[node], membership_list, node) for node in nodes_info}


def get_robustness(nodes_robustness):
    return sum(nodes_robustness.values()) / len(nodes_robustness)


def get_community_robustness(graph, community, intra=False):
    membership_list = np.zeros(graph.vcount(), dtype=int)
    nodes_in_community = set(community)
    for n in nodes_in_community:
        membership_list[n] = 1
    nodes_info = get_nodes_info(graph, membership_list)
    if intra:
        nodes_info = {node: nodes_info[node] for node in nodes_in_community}
    nodes_robustness = get_nodes_robustness(nodes_info, membership_list)
    robustness = get_robustness(nodes_robustness)
    # resolutions, fractions = resolution_effect_on_equilibrium(nodes_info, membership_list)
    return robustness


def get_node_potential(node_info, resolution):
    node_potential = dict()
    for community, counts in node_info.items():
        pros = counts['friends'] * (1-resolution)
        cons = counts['strangers'] * -resolution
        node_potential[community] = pros - cons
    return node_potential

def get_nodes_potential(nodes_info, resolution=1):
    nodes_potential = dict()
    for node, info in nodes_info.items():
        nodes_potential[node] = get_node_potential(info, resolution)
    return nodes_potential


def is_node_in_equilibrium(nodes_potential, membership_list):
    nodes_equilibrium = dict()
    for node, potentials in nodes_potential.items():
        node_comm = membership_list[node]
        best = max(potentials.values())
        nodes_equilibrium[node] = best == potentials[node_comm]
    return nodes_equilibrium


def resolution_effect_on_equilibrium(nodes_info, membership_list):
    fractions = []
    resolutions = np.linspace(0, 1, 11)
    for res in resolutions:
        nodes_potential = get_nodes_potential(nodes_info, res)
        nodes_equilibrium = is_node_in_equilibrium(nodes_potential, membership_list)
        fraction = sum(nodes_equilibrium.values()) / len(nodes_equilibrium)
        fractions.append(fraction)
    return resolutions, fractions


def main():
    # Create example graph
    g = ig.Graph.Famous("zachary")
    membership_list = np.random.randint(0, 2, size=g.vcount()).tolist()
    nodes_info = get_nodes_info(g, membership_list)
    nodes_robustness = get_nodes_robustness(nodes_info, membership_list)
    robustness = get_robustness(nodes_robustness)
    resolutions, fractions = resolution_effect_on_equilibrium(nodes_info, membership_list)
    print("Robustness:", robustness)


if __name__ == "__main__":
    main()

