import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import generate_sequence


def main():
    V = 2000
    membership = [0 if i < int(V/2) else 1 for i in range(V)]

    paths = []
    base = f'/Users/lucas/Databases/Hedonic/PHYSA_{V}/networks/2C_{V}N/'
    for root, _, files in os.walk(base):
        for file in files:
            if file.endswith('.pkl'):
                paths.append(os.path.join(root, file))

    for pth in tqdm(paths, desc="Processing files"):
        with open(pth, 'rb') as f:
            g = pickle.load(f)
        resolutions, fractions, robustness = g.resolution_spectrum(membership, np.linspace(0, 1, 1001), return_robustness=True)
        df = pd.DataFrame({
            'resolutions': resolutions,
            'fractions': fractions
        }).round(5)
        output = pth.replace('/networks/', '/resolution_spectra/').replace('network_', 'spectrum_').replace('.pkl', '.csv')
        os.makedirs(os.path.dirname(output), exist_ok=True)
        df.to_csv(output, index=False)
        robustness_output = output.replace('spectrum_', 'robustness_').replace('.csv', '.txt')
        with open(robustness_output, 'w') as f:
            f.write(f'{robustness:.5f}')


__name__ == '__main__' and main()