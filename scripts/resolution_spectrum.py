import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import argparse
from utils import generate_sequence


def output_files_exist(output, robustness_output):
    return os.path.exists(output) and os.path.exists(robustness_output)

def get_output_paths(pth):
    output = pth.replace('/networks/', '/robustness/').replace('network_', 'spectrum_').replace('.pkl', '.csv')
    robustness_output = output.replace('spectrum_', 'robustness_').replace('.csv', '.txt')
    return output, robustness_output

def process_file(pth, membership, n_resolutions=2):
    with open(pth, 'rb') as f:
        g = pickle.load(f)

    spectra_output, robustness_output = get_output_paths(pth)

    # if output_files_exist(spectra_output, robustness_output):
    #     print(f"Skipping {pth}, output files already exist.")
    #     return
    
    resolutions, fractions, robustness = g.resolution_spectrum(membership, np.linspace(0, 1, n_resolutions))
    df = pd.DataFrame({
        'resolutions': resolutions,
        'fractions': fractions
    }).round(5)

    os.makedirs(os.path.dirname(spectra_output), exist_ok=True)
    df.to_csv(spectra_output, index=False)
    robustness_output = spectra_output.replace('spectrum_', 'robustness_').replace('.csv', '.txt')
    with open(robustness_output, 'w') as f:
        f.write(f'{robustness:.5f}')


def main():
    parser = argparse.ArgumentParser(description="Process resolution spectrum files.")
    parser.add_argument('--first', type=float, help="Percentage of first N paths to process")
    parser.add_argument('--last', type=float, help="Percentage of last N paths to process")
    parser.add_argument('--ignore_first', type=float, help="Percentage of paths to ignore from the start")
    parser.add_argument('--ignore_last', type=float, help="Percentage of paths to ignore from the end")
    parser.add_argument('--n_resolutions', type=int, default=2, help="Number of resolutions to use")
    parser.add_argument('--V', type=int, default=2040, help="Number of nodes in the network")
    args = parser.parse_args()

    V = args.V
    membership = [0 if i < int(V/2) else 1 for i in range(V)]

    paths = []
    base = f'/Users/lucas/Databases/Hedonic/PHYSA_{V}/networks/'
    for root, _, files in os.walk(base):
        for file in files:
            if file.endswith('.pkl'):
                paths.append(os.path.join(root, file))
    paths.sort()

    paths = [p for p in paths if not output_files_exist(*get_output_paths(p))]

    if args.ignore_first:
        n = int(len(paths) * (args.ignore_first / 100))
        paths = paths[n:]
    if args.ignore_last:
        n = int(len(paths) * (args.ignore_last / 100))
        paths = paths[:-n]

    if args.first:
        n = int(len(paths) * (args.first / 100))
        paths = paths[:n]
    elif args.last:
        n = int(len(paths) * (args.last / 100))
        paths = paths[-n:]

    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(lambda p: process_file(p, membership, args.n_resolutions), paths), total=len(paths), desc="Processing files"))


__name__ == '__main__' and main()


# import pickle
# import numpy as np
# pth = '/Users/lucas/Databases/Hedonic/PHYSA_2000/networks/2C_2000N/P_in = 0.03/Difficulty = 0.65/network_088.pkl'
# with open(pth, 'rb') as f:
#     g = pickle.load(f)
# V = 2000
# membership = [0 if i < int(V/2) else 1 for i in range(V)]
# resolutions, fractions, robustness = g.resolution_spectrum(membership, np.linspace(0, 1, 1001))
# nodes_info = g.get_nodes_info(membership)
# for res in resolutions:
#     satisfied = 0
#     for node, community in enumerate(membership):
#         c0 = nodes_info[node][0]['friends'] - res * nodes_info[node][0]['strangers']
#         c1 = nodes_info[node][1]['friends'] - res * nodes_info[node][1]['strangers']
#         if (community == 0 and c0 >= c1) or (community == 1 and c1 >= c0):
#             satisfied += 1
#         else:
#             print(res, node, community, c0, c1, nodes_info[node])
#     print(f'{res:.2f} {satisfied/V:.2f}')