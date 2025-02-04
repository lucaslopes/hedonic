import time
import numpy as np
import igraph as ig
import pandas as pd
import utils
from hedonic import Game
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# Global variable to hold the Game instance accessible in worker processes
global_g = None

def get_accuracy(graph, communities, noise=0.):
    accuracy_hedonic, accuracy_leiden = list(), list()
    times_hedonic, times_leiden = list(), list()
    for comm in tqdm(communities):
        initial_membership = np.zeros(graph.vcount(), dtype=int)
        for n in comm:
            if np.random.random() > noise:
                initial_membership[n] = 1
        start = time.time()
        res_hedonic = graph.community_leiden(hedonic=True, initial_membership=initial_membership, n_iterations=-1, resolution=graph.density())
        end = time.time()
        times_hedonic.append(end - start)
        start = time.time()
        res_leiden = graph.community_leiden(initial_membership=initial_membership, n_iterations=-1, resolution=graph.density())
        end = time.time()
        times_leiden.append(end - start)
        accuracy_hedonic.append(ig.compare_communities(res_hedonic, initial_membership, method="rand"))
        accuracy_leiden.append(ig.compare_communities(res_leiden, initial_membership, method="rand"))
    return accuracy_hedonic, accuracy_leiden, times_hedonic, times_leiden


def run_frac_leave_and_join(file_path):
    g = Game(utils.read_txt_gz_to_igraph(file_path))
    print(g.summary())
    # communities_path = '/Users/lucas/Databases/Hedonic/Networks/DBLP/com-dblp.top5000.cmty.txt.gz'
    communities_path = '/Users/lucas/Databases/Hedonic/Networks/DBLP/com-dblp.all.cmty.txt.gz'
    communities = utils.read_communities(communities_path)
    robustness = [g.evaluate_community_stability(communities[i]) for i in tqdm(range(len(communities)))]
    df = pd.DataFrame(robustness)
    df.index.name = 'community_index'
    output_path = communities_path.replace(communities_path.split('/')[-1], 'fraction_stability.csv')
    df.to_csv(output_path)    
    # accuracy_hedonic, accuracy_leiden, times_hedonic, times_leiden = get_accuracy(graph, communities[:10], noise=0.)
    # print(accuracy_hedonic)
    # print(accuracy_leiden)
    # print(times_hedonic)
    # print(times_leiden)

def compute_spectrum(partition):
    return global_g.resolution_spectrum(partition)

def compute_spectrum_wrapper(comm):
    # Convert community to partition within the worker process
    partition = global_g.community_to_partition(comm)
    return compute_spectrum(partition)

def init_worker(g):
    # Initializer function to set up the global_g in each worker
    global global_g
    global_g = g

def run_resolution_spectrum(file_path):
    global global_g  # Reference to the global variable in the main process
    # Load the graph and assign to global_g
    global_g = Game(utils.read_txt_gz_to_igraph(file_path))
    print(global_g.summary())
    
    communities_path = '/Users/lucas/Databases/Hedonic/Networks/DBLP/com-dblp.all.cmty.txt.gz'
    communities = utils.read_communities(communities_path)

    # Use a generator to stream communities instead of loading all at once if possible
    # For this example, assuming communities is a list that fits in memory
    
    # Using ProcessPoolExecutor with initializer to pass the global_g to workers
    with ProcessPoolExecutor(max_workers=1, initializer=init_worker, initargs=(global_g,)) as executor:
        spectrum = list(tqdm(executor.map(compute_spectrum_wrapper, communities, chunksize=10), total=len(communities)))

    # Process results and save as before
    dfs = []
    for idx, spectra in enumerate(spectrum):
        resolutions, fractions, robustness = spectra
        df = pd.DataFrame({
            'resolutions': resolutions,
            'fractions': fractions
        }).round(5)
        df['community_index'] = idx
        df['robustness'] = robustness
        dfs.append(df)
    df = pd.concat(dfs)
    output_path = communities_path.replace(communities_path.split('/')[-1], 'resolution_spectra.csv')
    df.to_csv(output_path)

def main():
    file_path = '/Users/lucas/Databases/Hedonic/Networks/DBLP/com-dblp.ungraph.txt.gz'
    # run_frac_leave_and_join(file_path)
    run_resolution_spectrum(file_path)

if __name__ == "__main__":
    main()