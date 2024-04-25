import gzip
from hedonic import HedonicGame
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def read_txt_gz_to_igraph(file_path):
    edges = []
    with gzip.open(file_path, 'rt') as file:  # 'rt' mode for text mode reading
        for line in file:
            if line.startswith('#'):  # Skip comment lines
                continue
            # Split line into source and target node IDs and convert to integers
            source, target = map(int, line.strip().split())
            edges.append((source, target))
    # Assuming the file contains an edge list with each line as 'source target'
    graph = HedonicGame(edges=edges, directed=False)
    return graph

def read_communities(file_path):
    communities = []
    with gzip.open(file_path, 'rt') as file:  # 'rt' mode for text mode reading
        for line in file:
            nodes = list(map(int, line.strip().split()))
            communities.append(nodes)
    return communities

def get_robustness(graph, communities):
    robustness = list()
    for comm in tqdm(communities):
        initial_membership = [0] * graph.vcount()
        for n in comm:
            initial_membership[n] = 1
        robustness.append(graph.robustness(initial_membership))
    return robustness

file_path = '/Users/lucas/Desktop/com-dblp.ungraph.txt.gz'
graph = read_txt_gz_to_igraph(file_path)
print(graph.summary())

communities_path = '/Users/lucas/Desktop/com-dblp.top5000.cmty.txt.gz'
communities = read_communities(communities_path)[:10]



# Assuming you have a list of robustness values
# calculated in the get_robustness function
robustness_values = get_robustness(graph, communities)

# Plotting the violin plot
sns.violinplot(data=robustness_values)

# Adding labels and title
plt.xlabel('Community')
plt.ylabel('Robustness')
plt.title('Robustness Distribution')

# Display the plot
plt.show()