#!/usr/bin/env python3
import matplotlib.pyplot as plt
import networkx as nx

def draw_subfigure_a(ax):
    """
    (a) T-shaped network with edges:
       (1,2), (2,3), (2,4), (4,5), (5,6)
    """
    G = nx.Graph()
    # Add the 6 nodes
    G.add_nodes_from([1,2,3,4,5,6])
    # Add T-shape edges
    G.add_edges_from([(1,2), (2,3), (2,4), (4,5), (5,6)])
    
    # Positions to form a T shape:
    # 1
    # 2 -- 4 -- 5 -- 6
    # 3
    pos = {
        1: (0,2),
        2: (0,1),
        3: (0,0),
        4: (1,1),
        5: (2,1),
        6: (3,1)
    }
    
    # Example color pattern (adjust to your preference)
    node_colors = []
    for n in [1,3,5]:
        node_colors.append("lightgray")
    for n in [2,4,6]:
        node_colors.append("white")
    
    # We must match colors to the correct node index order
    # E.g., node_colors[i] is color for node i+1 if we rely on sorted(G.nodes())
    # Instead, let's build a dictionary: {node: color}
    color_map = {
        1: "lightgray",
        2: "white",
        3: "lightgray",
        4: "white",
        5: "lightgray",
        6: "white"
    }
    
    nx.draw(
        G, pos, ax=ax,
        node_color=[color_map[n] for n in G.nodes()],
        edgecolors="black",
        with_labels=True,
        node_size=800
    )
    ax.set_title(r"(a) equilibrium for $\alpha \in [0,1]$")
    ax.axis("off")


def draw_subfigure_b(ax):
    """
    (b) Same T-shaped network, different color pattern.
    """
    G = nx.Graph()
    G.add_nodes_from([1,2,3,4,5,6])
    G.add_edges_from([(1,2), (2,3), (2,4), (4,5), (5,6)])
    
    pos = {
        1: (0,2),
        2: (0,1),
        3: (0,0),
        4: (1,1),
        5: (2,1),
        6: (3,1)
    }
    
    # Different color pattern than (a)
    color_map = {
        1: "white",
        2: "lightgray",
        3: "white",
        4: "lightgray",
        5: "white",
        6: "lightgray"
    }
    
    nx.draw(
        G, pos, ax=ax,
        node_color=[color_map[n] for n in G.nodes()],
        edgecolors="black",
        with_labels=True,
        node_size=800
    )
    ax.set_title(r"(b) equilibrium for $\alpha \in [0,0.4]$")
    ax.axis("off")


def draw_subfigure_c(ax):
    """
    (c) Path of 5 nodes: 1-2-3-4-5
    """
    G = nx.Graph()
    G.add_nodes_from([1,2,3,4,5])
    G.add_edges_from([(1,2), (2,3), (3,4), (4,5)])
    
    # Arrange them in a horizontal row
    pos = {
        1: (0,0),
        2: (1,0),
        3: (2,0),
        4: (3,0),
        5: (4,0)
    }
    
    # Example color pattern
    color_map = {
        1: "lightgray",
        2: "white",
        3: "lightgray",
        4: "white",
        5: "lightgray"
    }
    
    nx.draw(
        G, pos, ax=ax,
        node_color=[color_map[n] for n in G.nodes()],
        edgecolors="black",
        with_labels=True,
        node_size=800
    )
    ax.set_title(r"(c) equilibrium for $\alpha \in [0,1]$")
    ax.axis("off")


def draw_subfigure_d(ax):
    """
    (d) Same 5-node path, different coloring.
    """
    G = nx.Graph()
    G.add_nodes_from([1,2,3,4,5])
    G.add_edges_from([(1,2), (2,3), (3,4), (4,5)])
    
    pos = {
        1: (0,0),
        2: (1,0),
        3: (2,0),
        4: (3,0),
        5: (4,0)
    }
    
    # Another color pattern
    color_map = {
        1: "white",
        2: "lightgray",
        3: "white",
        4: "lightgray",
        5: "white"
    }
    
    nx.draw(
        G, pos, ax=ax,
        node_color=[color_map[n] for n in G.nodes()],
        edgecolors="black",
        with_labels=True,
        node_size=800
    )
    ax.set_title(r"(d) equilibrium for $\alpha \in [0,\frac{1}{7}]$")
    ax.axis("off")


def draw_subfigure_e(ax):
    """
    (e) Cycle of 8 nodes, arranged in a circle.
    """
    G = nx.cycle_graph(8)  # nodes labeled 0..7 by default
    # Relabel to 1..8
    mapping = {i: i+1 for i in range(8)}
    G = nx.relabel_nodes(G, mapping)
    
    pos = nx.circular_layout(G, scale=1.5)
    
    # Example: alternate gray/white
    color_map = {}
    for n in G.nodes():
        if n % 2 == 1:
            color_map[n] = "lightgray"
        else:
            color_map[n] = "white"
    
    nx.draw(
        G, pos, ax=ax,
        node_color=[color_map[n] for n in G.nodes()],
        edgecolors="black",
        with_labels=True,
        node_size=800
    )
    ax.set_title(r"(e) equilibrium for $\alpha = 1$")
    ax.axis("off")


def main():
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12,6))
    
    # Top row: (a), (c), (e)
    draw_subfigure_a(axes[0, 0])
    draw_subfigure_c(axes[0, 1])
    draw_subfigure_e(axes[0, 2])
    
    # Bottom row: (b), (d), and empty
    draw_subfigure_b(axes[1, 0])
    draw_subfigure_d(axes[1, 1])
    axes[1, 2].axis("off")
    
    plt.tight_layout()
    plt.savefig("reproduced_figure.pdf", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
