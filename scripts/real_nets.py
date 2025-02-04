import time
import numpy as np
import igraph as ig
import pandas as pd
import utils
from hedonic import Game
from tqdm import tqdm


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
    graph = utils.read_txt_gz_to_igraph(file_path)
    g = Game(graph)
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


def run_resolution_spectrum():
    file_path = '/Users/lucas/Databases/Hedonic/Networks/DBLP/com-dblp.ungraph.txt.gz'
    graph = utils.read_txt_gz_to_igraph(file_path)
    g = Game(graph)
    print(g.summary())
    communities_path = '/Users/lucas/Databases/Hedonic/Networks/DBLP/com-dblp.top5000.cmty.txt.gz'
    # communities_path = '/Users/lucas/Databases/Hedonic/Networks/DBLP/com-dblp.all.cmty.txt.gz'
    communities = utils.read_communities(communities_path)
    partitions = [g.community_to_partition(comm) for comm in communities]
    spectrum = [g.resolution_spectrum(p, [0,1]) for p in tqdm(partitions)]
    dfs = list()
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
    run_resolution_spectrum()


if __name__ == "__main__":
    main()

