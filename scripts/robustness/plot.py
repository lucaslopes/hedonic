import networkx as nx
import matplotlib.pyplot as plt


def plot_network(network, partition):
    """
    Plots the network with nodes colored according to their community assignments.
    
    Args:
        network (list of tuple): The list of edges in the network.
        partition (list of int): The membership list where index is the node and value is the cluster index.
    
    Returns:
        matplotlib.figure.Figure: The figure object of the plot.
    """
    G = nx.Graph()
    G.add_edges_from(network)
    
    # Create a color map based on the partition
    color_map = []
    for node in G:
        color_map.append(partition[node])
    
    pos = nx.spring_layout(G)
    fig, ax = plt.subplots()
    nx.draw(G, pos, node_color=color_map, with_labels=True, cmap=plt.cm.jet, node_size=500, font_color='white', ax=ax)
    return fig, ax


def main():
    # Example usage
    # network = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 4), (1, 5), (2, 3), (2, 5), (3, 4), (4, 5)]
    # partition = [0, 1, 0, 0, 1, 1]

    network = [(0, 4), (0, 5), (1, 2), (1, 3), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)]
    partition = [0, 1, 1, 1, 0, 0]

    fig, = plot_network(network, partition)
    fig.show()

    return True


__name__ == '__main__' and main()