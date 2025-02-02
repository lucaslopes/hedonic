import matplotlib.pyplot as plt
import pickle
from utils import generate_graph
from stopwatch import Stopwatch
from hedonic import Game
from tqdm import tqdm

stopwatch = Stopwatch() # create Stopwatch instance

n_communities = 2
community_sizes = [100, 500, 1000, 5000, 10000]  # Varying community sizes
p_in = .05
multiplier = .5
seed = 42

generate_times = []
read_times_gml = []
read_times_pickle = []

for community_size in tqdm(community_sizes, desc="Processing community sizes"):
    # Measure generate_graph time
    stopwatch.reset()
    stopwatch.start()
    g = generate_graph(n_communities, community_size, p_in, multiplier, seed)
    stopwatch.stop()
    generate_times.append(stopwatch.duration)
    
    # Write to GML
    g.write_gml('test.gml')
    
    # Measure read time for GML
    stopwatch.reset()
    stopwatch.start()
    h_gml = Game.Read_GML('test.gml')
    stopwatch.stop()
    read_times_gml.append(stopwatch.duration)
    
    # Write to Pickle
    with open('test.pkl', 'wb') as f:
        pickle.dump(g, f)
    
    # Measure read time for Pickle
    stopwatch.reset()
    stopwatch.start()
    with open('test.pkl', 'rb') as f:
        h_pickle = pickle.load(f)
    stopwatch.stop()
    read_times_pickle.append(stopwatch.duration)

# Plot the results
plt.plot(community_sizes, generate_times, label='Generate Graph', linestyle='-')
plt.plot(community_sizes, read_times_gml, label='Read Graph (GML)', linestyle='--')
plt.plot(community_sizes, read_times_pickle, label='Read Graph (Pickle)', linestyle='-.')
plt.xlabel('Community Size')
plt.ylabel('Time (seconds)')
plt.legend()
plt.title('Generate vs Read Time for Different Community Sizes and Formats')
plt.show()
