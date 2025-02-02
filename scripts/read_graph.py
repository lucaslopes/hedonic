import os
import pickle
import hedonic as hd
from pathlib import Path

graph_path = Path('/Users/lucas/Databases/Hedonic/PHYSA_100/graphs/2C_100N/P_in = 0.09/Difficulty = 0.30/network_013.pkl')

with open(graph_path, 'rb') as f:
    print(f"Loading graph from {graph_path}")
    print(f)
    g: hd.Game = pickle.load(f)

print("Successfully loaded graph")
print(f"Number of nodes: {len(g.vs())}")
print(f"Number of edges: {len(g.es())}")