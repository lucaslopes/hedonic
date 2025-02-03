from os import name
import numpy as np
import igraph as ig
from hedonic import Game


def main():
    # Create example graph
    g = ig.Graph.Famous("zachary")
    membership_list = np.random.randint(0, 2, size=g.vcount()).tolist()
    nodes_info = g.get_nodes_info(membership_list)
    nodes_robustness = Game.get_nodes_robustness(nodes_info, membership_list)
    robustness = Game.get_robustness(nodes_robustness)
    resolutions, fractions = Game.resolution_effect_on_equilibrium(nodes_info, membership_list)
    print("Robustness:", robustness)


if __name__ == "__main__":
    main()

