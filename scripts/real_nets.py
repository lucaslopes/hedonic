import time
import argparse
import numpy as np
import igraph as ig
import pandas as pd
import utils
from tqdm import tqdm
from hedonic import Game
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

def run_resolution_spectrum_parallel(file_path, use_parallel=True, ignore_first=None, ignore_last=None, first=None, last=None):
    global global_g  # Reference to the global variable in the main process
    # Load the graph and assign to global_g
    global_g = Game(utils.read_txt_gz_to_igraph(file_path))
    print(global_g.summary())
    
    communities_path = '/Users/lucas/Databases/Hedonic/Networks/DBLP/com-dblp.all.cmty.txt.gz'
    communities = utils.read_communities(communities_path)

    if ignore_first:
        n = int(len(communities) * (ignore_first / 100))
        communities = communities[n:]
    if ignore_last:
        n = int(len(communities) * (ignore_last / 100))
        communities = communities[:-n]

    if first:
        n = int(len(communities) * (first / 100))
        communities = communities[:n]
    elif last:
        n = int(len(communities) * (last / 100))
        communities = communities[-n:]

    spectrum = list()
    if use_parallel:
        # Using ProcessPoolExecutor with initializer to pass the global_g to workers
        with ProcessPoolExecutor(max_workers=1, initializer=init_worker, initargs=(global_g,)) as executor:
            spectrum = list(tqdm(executor.map(compute_spectrum_wrapper, communities, chunksize=10), total=len(communities)))
    else:
        # Non-parallelized approach
        pbar = tqdm(total=len(communities), desc="Communities", position=0)
        for comm in communities:
            partition = global_g.community_to_partition(comm)
            spectra = global_g.resolution_spectrum(partition)
            spectrum.append(spectra)  # TODO: spectrum not initialized and using global_g is making the tqdm not update
            pbar.update(1)
        pbar.close()

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
    df = df.round(5)
    df = df[['community_index', 'resolutions', 'fractions', 'robustness']]
    output_path = communities_path.replace(communities_path.split('/')[-1], 'resolution_spectra.csv')
    df.to_csv(output_path)


def run_resolution_spectrum(file_path, ignore_first=None, ignore_last=None, first=None, last=None):
    g = Game(utils.read_txt_gz_to_igraph(file_path))
    n_vertices_before = len(g.vs)
    g.vs.select(_degree_eq=0).delete()
    n_vertices_after = len(g.vs)
    print(f"Deleted {n_vertices_before - n_vertices_after} isolated vertices")

    communities_path = '/Users/lucas/Databases/Hedonic/Networks/DBLP/com-dblp.all.cmty.txt.gz'
    communities = utils.read_communities(communities_path)

    if ignore_first:
        n = int(len(communities) * (ignore_first / 100))
        communities = communities[n:]
    if ignore_last:
        n = int(len(communities) * (ignore_last / 100))
        communities = communities[:-n]

    if first:
        n = int(len(communities) * (first / 100))
        communities = communities[:n]
    elif last:
        n = int(len(communities) * (last / 100))
        communities = communities[-n:]

    pbar = tqdm(total=len(communities), desc="Communities", position=0)
    spectrum = []
    for comm in communities:
        partition = g.community_to_partition(comm)
        spectra = g.resolution_spectrum(partition)
        spectrum.append(spectra)
        pbar.update(1)
    pbar.close()

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
    df = df.round(5)
    df = df[['community_index', 'resolutions', 'fractions', 'robustness']]
    output_path = communities_path.replace(communities_path.split('/')[-1], 'resolution_spectra.csv')
    df.to_csv(output_path)


def main():
    parser = argparse.ArgumentParser(description="Process resolution spectrum files.")
    parser.add_argument('--use_parallel', type=bool, default=True, help="Use parallel processing")
    parser.add_argument('--ignore_first', type=float, help="Percentage of communities to ignore from the start")
    parser.add_argument('--ignore_last', type=float, help="Percentage of communities to ignore from the end")
    parser.add_argument('--first', type=float, help="Percentage of first N communities to process")
    parser.add_argument('--last', type=float, help="Percentage of last N communities to process")
    args = parser.parse_args()

    file_path = '/Users/lucas/Databases/Hedonic/Networks/DBLP/com-dblp.ungraph.txt.gz'
    # run_frac_leave_and_join(file_path)
    run_resolution_spectrum(file_path, ignore_first=args.ignore_first, ignore_last=args.ignore_last, first=args.first, last=args.last)
    # run_resolution_spectrum_parallel(file_path, use_parallel=args.use_parallel, ignore_first=args.ignore_first, ignore_last=args.ignore_last, first=args.first, last=args.last)

if __name__ == "__main__":
    main()