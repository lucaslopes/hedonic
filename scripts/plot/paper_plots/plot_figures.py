#!/usr/bin/env python3
"""
Optimized & Refactored Plotting Script with Persistent Data

This script computes plot-ready data once, saves it as pickle files, and reuses it
on subsequent runs. It also leverages parallel processing for expensive KDE calculations
and removes redundant computations (e.g. for the noisy subsets in Figure 4).

Usage:
  python persist_plots_refactored.py [PATH_TO_DATA]

Requirements:
  - numpy, pandas, matplotlib, seaborn, scipy, joblib, tqdm, pickle
"""

import os
import gc
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from scipy.stats import gaussian_kde
from joblib import Parallel, delayed

# =============================================================================
# Global Constants
# =============================================================================

# Methods labels
METHODS_MAPPING = {
    'Leiden': 'Leiden (full-fledged)',
    'Hedonic': 'Leiden (phase 1)',
    'Spectral': 'Spectral Clustering',
    'OnePass': 'OnePass',
    'Mirror': 'Mirror'
}

# Colormap for contour plots (used with reversed color scales)
COLORMAP_DICT = {
    'Leiden': 'Blues_r',
    'Hedonic': 'Greens_r',
    'Spectral': 'Oranges_r',
    'OnePass': 'Reds_r',
    'Mirror': 'Purples_r'
}

# Bar plot color dictionary (from tab20b, chosen once)
_TAB20B = plt.get_cmap('tab20b').colors
BAR_COLOR_DICT = {
    'Leiden': _TAB20B[2],
    'Hedonic': _TAB20B[6],
    'Spectral': _TAB20B[10],
    'OnePass': _TAB20B[14],
    'Mirror': _TAB20B[18],
}

# Persistence directory
PERSIST_DIR = None

# =============================================================================
# Utility Functions
# =============================================================================

def load_data(path: str) -> pd.DataFrame:
    """Load data from a parquet file or a gzipped CSV."""
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    else:
        return pd.read_csv(path, compression="gzip")

def include_spectral(df: pd.DataFrame) -> pd.DataFrame:
    """
    Augment data by repeating the 'Spectral' rows (noise=0.1) with new noise values.
    """
    unique_noise = [n for n in df['noise'].unique() if n != 0.1]
    spectral_rows = df[(df['method'] == 'Spectral') & (df['noise'] == 0.1)]
    new_rows = []
    for noise in unique_noise:
        temp = spectral_rows.copy()
        temp['noise'] = noise
        new_rows.append(temp)
    return pd.concat([df] + new_rows, ignore_index=True)

def load_or_compute(filepath, compute_func, *args, **kwargs):
    """
    Load persisted data from 'filepath' if available; otherwise compute it,
    save it, and then return the computed object.
    """
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        print(f"Loaded persisted data from {filepath}")
        return data
    else:
        print(f"Computing and persisting data to {filepath}")
        data = compute_func(*args, **kwargs)
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        print(f"Computed and persisted data to {filepath}")
        return data

# =============================================================================
# Figure 1: Ground Truth Histograms & Heatmaps
# =============================================================================

def compute_figure1_data(df: pd.DataFrame, cmap: str = "BuPu"):
    """
    Compute histogram and heatmap data for ground truth.
    """
    gt_df = df[df['method'] == 'GroundTruth'].drop_duplicates()
    communities = sorted(gt_df['number_of_communities'].unique())
    global_max = 0
    hist_data = []
    
    for nc in communities:
        subset = gt_df[gt_df['number_of_communities'] == nc]['robustness']
        counts, _ = np.histogram(subset, bins=100, range=(0, 1))
        global_max = max(global_max, counts.max())
        hist_data.append((nc, subset, counts))
    
    heatmaps = {}
    for nc in communities:
        sub = gt_df[gt_df['number_of_communities'] == nc]
        pivot = sub.pivot_table(values='robustness', index='p_in', columns='multiplier', aggfunc='mean')
        pivot = pivot.sort_index().reindex(sorted(pivot.columns), axis=1)
        heatmaps[nc] = pivot

    return {
        "communities": communities,
        "global_max": global_max,
        "hist_data": hist_data,
        "heatmaps": heatmaps,
        "cmap": cmap
    }

def plot_figure1(data: dict, filename: str = "figure1", fig_dir: str = None):
    """
    Plot histograms (upper row) and heatmaps (bottom row) for ground truth.
    """
    communities = data["communities"]
    global_max = data["global_max"]
    hist_data = data["hist_data"]
    heatmaps = data["heatmaps"]
    cmap = "BuPu"  # Alternatively, use data["cmap"]

    fig, axs = plt.subplots(2, len(communities), figsize=(15, 6))
    
    hist_color = plt.get_cmap(cmap+'_r')(0)
    for i, (nc, subset, counts) in enumerate(hist_data):
        ax = axs[0, i]
        ax.hist(subset, bins=100, range=(0, 1), color=hist_color)
        ax.set_title(f"{nc} Communities")
        ax.set_xlabel("Fraction of robust nodes")
        ax.set_ylim(0, global_max)
        ax.set_ylabel("Ground Truth Count" if i == 0 else "")
    
    for i, nc in enumerate(communities):
        ax = axs[1, i]
        pivot = heatmaps[nc]
        im = ax.imshow(pivot.values, aspect='auto', vmin=0, vmax=1, cmap=cmap)
        ax.set_xlabel(r"Difficulty Factor ($\lambda$)")
        ax.set_ylabel(r"Intra Edge Probability ($p$)" if i == 0 else "")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([f"{val:.2f}" for val in pivot.columns], rotation=45, fontsize=8)
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([f"{val:.2f}" for val in pivot.index], fontsize=8)
    
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Robustness")
    fig.subplots_adjust(right=0.9, hspace=0.3, wspace=0.3)
    fig.savefig(os.path.join(fig_dir, f"{filename}.pdf"), dpi=300)
    plt.close(fig)

# =============================================================================
# Figure 2: Bar Plots with Means & Confidence Intervals
# =============================================================================

def compute_figure2_data(df: pd.DataFrame):
    """
    Aggregate means and confidence intervals for each metric (duration, robustness, accuracy)
    per noise level and method.
    """
    methods = list(METHODS_MAPPING.keys())
    df_methods = df[df['method'].isin(methods)]
    noise_levels = sorted(df_methods['noise'].unique())
    metrics = ['duration', 'robustness', 'accuracy']
    results = {metric: {} for metric in metrics}
    
    for metric in metrics:
        for meth in methods:
            rows = []
            for nl in noise_levels:
                sub = df_methods[(df_methods['noise'] == nl) & (df_methods['method'] == meth)]
                if sub.empty:
                    mean_val, ci = np.nan, 0
                else:
                    mean_val = sub[metric].mean()
                    ci = 1.96 * sub[metric].std() / np.sqrt(len(sub))
                rows.append((nl, mean_val, ci))
            results[metric][meth] = rows

    return {
        "methods": methods,
        "noise_levels": noise_levels,
        "metrics": metrics,
        "results": results
    }

def plot_figure2(fig2_data: dict, filename: str = "figure2", fig_dir: str = None):
    """
    Plot bar charts (one subplot per metric) with means and confidence intervals.
    """
    methods = fig2_data["methods"]
    noise_levels = fig2_data["noise_levels"]
    metrics = fig2_data["metrics"]
    results = fig2_data["results"]
    
    fig, axs = plt.subplots(1, len(metrics), figsize=(15, 3))
    bar_width = 0.15
    indices = np.arange(len(noise_levels))
    handles, labels = [], []
    
    metric_names = {'duration': 'Efficiency', 'robustness': 'Robustness', 'accuracy': 'Accuracy'}
    
    for col, metric in enumerate(metrics):
        ax = axs[col]
        for i, m in enumerate(methods):
            rows = results[metric][m]
            means = [r[1] for r in rows]
            cis = [r[2] for r in rows]
            bars = ax.bar(indices + i * bar_width, means, width=bar_width, yerr=cis,
                          color=BAR_COLOR_DICT[m], capsize=3,
                          label=METHODS_MAPPING[m] if col == 0 else "_nolegend_")
            if col == 0:
                handles.append(bars[0])
                labels.append(METHODS_MAPPING[m])
        ax.set_xticks(indices + bar_width * (len(methods)-1) / 2)
        ax.set_xticklabels([f"{n:.2f}" for n in noise_levels])
        ax.set_xlabel("Noise")
        ax.set_ylabel(metric_names[metric])
        ax.set_title(metric_names[metric])
    
    fig.legend(handles, labels, loc='upper center', ncol=len(methods), bbox_to_anchor=(0.5, -0.1))
    fig.subplots_adjust(bottom=0.1)
    fig.savefig(os.path.join(fig_dir, f"{filename}.pdf"), dpi=300, bbox_inches='tight')
    plt.close(fig)

# =============================================================================
# Figure 3: Bar Plots Split by Communities and Noise
# =============================================================================

def compute_figure3_data(df: pd.DataFrame, precomputed_all=None, precomputed_noisy=None):
    """
    Aggregate data for metrics split by number of communities and noise.
    """
    methods = list(METHODS_MAPPING.keys())
    if precomputed_all is None:
        df_methods_all = df[df['method'].isin(methods)]
    else:
        df_methods_all = precomputed_all
    if precomputed_noisy is None:
        df_methods_noisy = df_methods_all[df_methods_all['noise'] == 1]
    else:
        df_methods_noisy = precomputed_noisy
    communities = sorted(df_methods_all['number_of_communities'].unique())
    metrics = ['duration', 'robustness', 'accuracy']
    results = {0: {metric: {} for metric in metrics}, 1: {metric: {} for metric in metrics}}
    
    for noise_val in [0, 1]:
        subset = df_methods_noisy if noise_val == 1 else df_methods_all
        for metric in metrics:
            for meth in methods:
                rows = []
                for nc in communities:
                    sub = subset[(subset['number_of_communities'] == nc) & (subset['method'] == meth)]
                    if sub.empty:
                        mean_val, ci = np.nan, 0
                    else:
                        mean_val = sub[metric].mean()
                        ci = 1.96 * sub[metric].std() / np.sqrt(len(sub))
                    rows.append((nc, mean_val, ci))
                results[noise_val][metric][meth] = rows

    return {
        "methods": methods,
        "communities": communities,
        "metrics": metrics,
        "results": results
    }

def plot_figure3(fig3_data: dict, filename: str = "figure3", fig_dir: str = None):
    """
    Plot bar charts for metrics split by communities for noise=0 (upper row) and noise=1 (lower row).
    """
    methods = fig3_data["methods"]
    communities = fig3_data["communities"]
    metrics = fig3_data["metrics"]
    results = fig3_data["results"]
    
    fig, axs = plt.subplots(2, len(metrics), figsize=(15, 6))
    bar_width = 0.15
    indices = np.arange(len(communities))
    handles, labels = [], []
    
    metric_names = {'duration': 'Efficiency', 'robustness': 'Robustness', 'accuracy': 'Accuracy'}
    
    for row, noise_val in enumerate([0, 1]):
        for col, metric in enumerate(metrics):
            ax = axs[row, col]
            for i, m in enumerate(methods):
                rows = results[noise_val][metric][m]
                means = [r[1] for r in rows]
                cis = [r[2] for r in rows]
                bars = ax.bar(indices + i * bar_width, means, width=bar_width, yerr=cis,
                              color=BAR_COLOR_DICT[m], capsize=3,
                              label=METHODS_MAPPING[m] if (row == 0 and col == 0) else "_nolegend_")
                if row == 0 and col == 0:
                    handles.append(bars[0])
                    labels.append(METHODS_MAPPING[m])
            ax.set_xticks(indices + bar_width * (len(methods)-1) / 2)
            ax.set_xticklabels([str(nc) for nc in communities])
            if row == 0:
                ax.set_title(metric_names[metric])
            if row == 1:
                ax.set_xlabel("Number of Communities")
            ax.set_ylabel(metric_names[metric])
    
    fig.legend(handles, labels, loc='upper center', ncol=len(methods), bbox_to_anchor=(0.5, -0.1))
    fig.subplots_adjust(bottom=0)
    fig.savefig(os.path.join(fig_dir, f"{filename}.pdf"), dpi=300, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# Common KDE Computation for Figure 4
# =============================================================================

def compute_kde2d_generic(xvals, yvals, X, Y):
    """Compute 2D KDE for given x and y values over grid (X, Y)."""
    if len(xvals) < 2:
        return np.zeros_like(X)
    kde = gaussian_kde(np.vstack([xvals, yvals]))
    coords = np.vstack([X.ravel(), Y.ravel()])
    return kde(coords).reshape(X.shape)

def compute_kde_for_method(m_key, df_all, df_noisy, x_col, y_col, X, Y):
    """
    Compute KDEs for a given method using both the full dataset and the precomputed noisy subset.
    """
    sub_all = df_all[df_all['method'] == m_key]
    sub_noisy = df_noisy[df_noisy['method'] == m_key]
    Z_clean = compute_kde2d_generic(sub_all[x_col].values, sub_all[y_col].values, X, Y)
    Z_noisy = compute_kde2d_generic(sub_noisy[x_col].values, sub_noisy[y_col].values, X, Y)
    return m_key, Z_clean, Z_noisy


def compute_figure4_data_common(df: pd.DataFrame, x_col: str, y_col: str, x_range: tuple, y_range: tuple, grid_size: int = 200, x_axis_log: bool = False,
                                precomputed_all=None, precomputed_noisy=None):
    """
    Common function for computing KDE results for Figure 4.
    Precomputes 'all' and 'noisy' subsets once, then computes KDEs for each method.
    """
    if precomputed_all is None:
        df_methods_all = df[df['method'].isin(METHODS_MAPPING.keys())]
    else:
        df_methods_all = precomputed_all
    if precomputed_noisy is None:
        df_methods_noisy = df_methods_all[df_methods_all['noise'] == 1]
    else:
        df_methods_noisy = precomputed_noisy
    xgrid = np.linspace(x_range[0], x_range[1], grid_size)
    ygrid = np.linspace(y_range[0], y_range[1], grid_size)
    X, Y = np.meshgrid(xgrid, ygrid)

    results = Parallel(n_jobs=-1)(
        delayed(compute_kde_for_method)(m_key, df_methods_all, df_methods_noisy, x_col, y_col, X, Y)
        for m_key in METHODS_MAPPING
    )
    kde_results = {m_key: {"clean": Z_clean, "noisy": Z_noisy}
                   for m_key, Z_clean, Z_noisy in results}
    return X, Y, kde_results

def precompute_fig4_subsets(df: pd.DataFrame):
    """Precomputes 'all' and 'noisy' subsets for Figure 4."""
    df_methods_all = df[df['method'].isin(METHODS_MAPPING.keys())]
    df_methods_noisy = df_methods_all[df_methods_all['noise'] == 1]
    return df_methods_all, df_methods_noisy

def compute_figure4a_robustness_data(df: pd.DataFrame, precomputed_all, precomputed_noisy):
    """Wrapper for computing Figure 4 (Robustness): x-axis is 'robustness'."""
    X, Y, kde_results = compute_figure4_data_common(
        df, x_col='robustness', y_col='accuracy', 
        x_range=(-.1,1.1), y_range=(0,1.1),
        precomputed_all=precomputed_all, precomputed_noisy=precomputed_noisy
    )
    return {"X": X, "Y": Y, "kde_results": kde_results}

def compute_figure4b_efficiency_data(df: pd.DataFrame, precomputed_all, precomputed_noisy):
    """Wrapper for computing Figure 4 (Efficiency): x-axis is 'duration'."""
    # max_dur = df_methods_all['duration'].max() if not df_methods_all.empty else 1.0
    X, Y, kde_results = compute_figure4_data_common(
        df, x_col='duration', y_col='accuracy', 
        x_range=(-0.05, 0.3), y_range=(0,1.1),
        precomputed_all=precomputed_all, precomputed_noisy=precomputed_noisy
    )
    return {"X": X, "Y": Y, "kde_results": kde_results}

def plot_figure4(fig4_data: dict, xlabel: str, fig_dir: str = None):
    """
    General plotting routine for Figure 4.
    The upper row plots KDE for all data (clean) and the bottom row for the noisy subset.
    'xlabel' should be "Robustness" or "Efficiency" accordingly.
    """
    X = fig4_data["X"]
    Y = fig4_data["Y"]
    kde_results = fig4_data["kde_results"]
    n_methods = len(METHODS_MAPPING)
    
    fig, axs = plt.subplots(2, n_methods, figsize=(15, 6))
    for j, (m_key, m_label) in enumerate(METHODS_MAPPING.items()):
        # Upper row: all data (clean)
        ax = axs[0, j]
        Z = kde_results[m_key]["clean"]
        if Z is not None and np.any(Z):
            ax.contourf(X, Y, Z, levels=14, cmap=COLORMAP_DICT[m_key])
        ax.set_title(m_label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Accuracy" if j == 0 else "")
        # Bottom row: noisy subset
        ax2 = axs[1, j]
        Z = kde_results[m_key]["noisy"]
        if Z is not None and np.any(Z):
            ax2.contourf(X, Y, Z, levels=14, cmap=COLORMAP_DICT[m_key])
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel("Accuracy" if j == 0 else "")
    
    fig.tight_layout()
    fname = "acc_robustness.pdf" if xlabel == "Robustness" else "acc_efficiency.pdf"
    fig.savefig(os.path.join(fig_dir, fname), dpi=300)
    plt.close(fig)

# =============================================================================
# Main Routine
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate persistent figures from a dataset.')
    parser.add_argument('path', type=str, nargs='?',
                        default='/Users/lucas/Databases/Hedonic/PHYSA/Synthetic_Networks/V1020/resultados.csv.gzip',
                        help='Path to the parquet or CSV.GZ dataset.')
    args = parser.parse_args()
    data_path = args.path
    fig_dir = os.path.join(os.path.dirname(data_path), 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    PERSIST_DIR = os.path.join(os.path.dirname(data_path), 'persist')
    os.makedirs(PERSIST_DIR, exist_ok=True)
    
    print(f"Loading data from {data_path} ...")
    df = load_data(data_path)
    df = include_spectral(df)
    df_methods_all, df_methods_noisy = precompute_fig4_subsets(df)
    print(f"Data loaded. Rows = {len(df)}")

    # Figure 1
    print("Plotting figure 1...")
    fig1_file = os.path.join(PERSIST_DIR, "fig1_data.pkl")
    fig1_data = load_or_compute(fig1_file, compute_figure1_data, df)
    plot_figure1(fig1_data, "gt_robustness", fig_dir)
    del fig1_data
    gc.collect()
    print("Done: figure1.pdf")
    
    # Figure 2
    print("Plotting figure 2...")
    fig2_file = os.path.join(PERSIST_DIR, "fig2_data.pkl")
    fig2_data = load_or_compute(fig2_file, compute_figure2_data, df)
    plot_figure2(fig2_data, "noise", fig_dir)
    del fig2_data
    gc.collect()
    print("Done: figure2.pdf")
    
    # Figure 3
    print("Plotting figure 3...")
    fig3_file = os.path.join(PERSIST_DIR, "fig3_data.pkl")
    fig3_data = load_or_compute(fig3_file, compute_figure3_data, df, precomputed_all=df_methods_all, precomputed_noisy=df_methods_noisy)
    plot_figure3(fig3_data, "n_communities", fig_dir)
    del fig3_data
    gc.collect()
    print("Done: figure3.pdf")
   
    # Figure 4a (Robustness)
    print("Plotting figure 4a...")
    fig4a_rob_file = os.path.join(PERSIST_DIR, "fig4a_robustness_data.pkl")
    fig4a_rob_data = load_or_compute(
        fig4a_rob_file, compute_figure4a_robustness_data, 
        df, precomputed_all=df_methods_all, precomputed_noisy=df_methods_noisy
    )
    plot_figure4(fig4a_rob_data, xlabel="Robustness", fig_dir=fig_dir)
    del fig4a_rob_data
    gc.collect()
    print("Done: figure4a_robustness.pdf")

    # Figure 4b (Efficiency)
    print("Plotting figure 4b...")
    fig4b_eff_file = os.path.join(PERSIST_DIR, "fig4b_efficiency_data.pkl")
    fig4b_eff_data = load_or_compute(
        fig4b_eff_file, compute_figure4b_efficiency_data, 
        df, precomputed_all=df_methods_all, precomputed_noisy=df_methods_noisy
    )
    plot_figure4(fig4b_eff_data, xlabel="Efficiency", fig_dir=fig_dir)
    del fig4b_eff_data
    gc.collect()
    print("Done: figure4b_efficiency.pdf")
    
    print("All figures generated successfully.")

if __name__ == "__main__":
    main()
