import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
from collections import defaultdict


def degree_histogram(graph):
    hist, bins = np.histogram(graph.degree(), bins=range(1, max(graph.degree())+2))
    plt.bar(bins[:-1], hist, align='center', alpha=0.5)
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Degree histogram')
    plt.show()


def resolution_threshold(friends_a, friends_b, strangers_a, strangers_b):
    delta_friends = len(friends_a) - len(friends_b)
    delta_size = len(strangers_a) + len(strangers_b)
    try:
        resolution = delta_friends / delta_size
    except ZeroDivisionError:
        resolution = None
    return resolution


def update_matrix(matrix, idx_a, idx_b, resolution, comparison_friends, comparison_strangers):
    if '=' in comparison_friends:
        if '<' in comparison_strangers:
            matrix[idx_b, idx_a] = 1 # '0<=r<=1'
        elif '>' in comparison_strangers:
            matrix[idx_a, idx_b] = 1 # '0<=r<=1'
    elif '>' in comparison_friends:
        if '=' or '<' in comparison_strangers:
            matrix[idx_b, idx_a] = 1 # '0<=r<=1'
        elif '>' in comparison_strangers:
            matrix[idx_a, idx_b] = 1-resolution # f'{resolution}<=r<=1'
            matrix[idx_b, idx_a] = resolution
    elif '<' in comparison_friends:
        if '=' or '>' in comparison_strangers:
            matrix[idx_a, idx_b] = 1 # '0<=r<=1'
        elif '<' in comparison_strangers:
            matrix[idx_a, idx_b] = resolution
            matrix[idx_b, idx_a] = 1-resolution # f'{resolution}<=r<=1'


def matrix_count_nan(matrix):
    return [np.count_nonzero(np.isnan(matrix[c, :])) for c in range(matrix.shape[0])]


def matrix_resolution_internval(matrix):
    return [np.nansum(matrix[c,:]).item() for c in range(matrix.shape[0])]


def compare_communities(community_a, community_b):
    comparison = ''
    if len(community_a) == len(community_b):
        comparison = 'a=b'
    elif len(community_a) < len(community_b):
        comparison = 'a<b'
    else:
        comparison = 'a>b'
    return comparison


def get_friendship_density(friends_in_common, friends_of_neighbor):
    return len(friends_in_common) / len(friends_of_neighbor) if friends_of_neighbor else 0


def get_friendship_info(community, friend_of_node, g):
    friends_of_neighbor = {friend for neighbor in community for friend in g.neighbors(neighbor)}
    friends_in_common = friend_of_node.intersection(friends_of_neighbor)
    strangers = friends_of_neighbor - friend_of_node
    return friends_in_common, strangers


def reciprocity_matrix(g, node, membership):
    friends_of_node = set(g.neighbors(node))
    neighbors_community = [membership[n] for n in friends_of_node]
    neighbors_community.append(membership[node])
    neighbor_communities = [set(x) for x in set(frozenset(x) for x in neighbors_community)]
    matrix = np.full((len(neighbor_communities), len(neighbor_communities)), np.nan, dtype=np.float64)
    for idx_a, community_a in enumerate(neighbor_communities):
        friends_in_common_a, strangers_of_community_a = get_friendship_info(community_a - {node}, friends_of_node, g)
        for idx_b, community_b in enumerate(neighbor_communities[idx_a+1:]):
            friends_in_common_b, strangers_of_community_b = get_friendship_info(community_b - {node}, friends_of_node, g)
            resolution = resolution_threshold(friends_in_common_a, friends_in_common_b, strangers_of_community_a, strangers_of_community_b)
            comparison_friends = compare_communities(friends_in_common_a, friends_in_common_b)
            comparison_strangers = compare_communities(strangers_of_community_a, strangers_of_community_b)
            update_matrix(matrix, idx_a, (idx_a+1+idx_b), resolution, comparison_friends, comparison_strangers)
            
    return matrix, neighbor_communities


def find_reciprocal_pairs(graph, membership=None):
    invitations = dict()
    membership = membership or singleton(graph)
    for node in range(graph.vcount()):
        matrix, neighbor_communities = reciprocity_matrix(graph, node, membership)
        communities_count = matrix_count_nan(matrix)
        max_count = np.max(communities_count)
        best_communities = [community for community, count in zip(neighbor_communities, communities_count) if count == max_count]
        invitations[node] = {friend for community in best_communities for friend in community} - {node}
    
    common_desire = defaultdict(set)
    for node, invited_nodes in invitations.items():
        if len(invited_nodes) > 1:
            group = tuple(sorted(invited_nodes))
            common_desire[group].add(node)

    reciprocal_pairs = set()
    for node, invited_nodes in invitations.items():
        for target in invited_nodes:
            if node in invitations.get(target, set()):
                reciprocal_pairs.add(tuple(sorted([node, target])))

    return reciprocal_pairs, common_desire


def singleton(graph):
    return {node.index: {node.index} for node in graph.vs}


def get_membership(graph, reciprocal_pairs):
    membership = singleton(graph)
    for v, u in reciprocal_pairs:
        membership[v].add(u)
        membership[u].add(v)
    return membership

def merge_common_desires(common_desire, membership):
    for _, nodes in common_desire.items():
        for node in nodes:
            membership[node] = nodes
    return membership

def network_partitioning(graph):
    old_membership = singleton(graph)
    old_reciprocal = None
    while True:
        reciprocal_pairs, common_desire = find_reciprocal_pairs(graph, old_membership)
        if reciprocal_pairs != old_reciprocal:
            new_membership = get_membership(graph, reciprocal_pairs)
        else:
            new_membership = merge_common_desires(common_desire, old_membership)
        print('pairs:', reciprocal_pairs)
        print('desire:',common_desire)
        print('membership:', new_membership, '\n')
        if new_membership == old_membership:
            break
        old_membership = new_membership
        old_reciprocal = reciprocal_pairs
    
    community_labels = [next((label for label, nodes in new_membership.items() if node in nodes), node) for node in range(len(new_membership))]
    return community_labels


g = ig.Graph.Famous('Krackhardt_Kite')  # Krackhardt_Kite | Zachary https://igraph.org/c/doc/igraph-Generators.html#igraph_famous
partition = network_partitioning(g)
print(partition)

