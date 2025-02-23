#!/usr/bin/env python3
"""
Rewrite of the original plotting code with caching of expensive steps (notably KDE).
If the data or the code changes, recomputation occurs. If only plot style changes,
cached data will be reused, making repeated runs much faster.

Usage:
  python cache_plots.py [PATH_TO_DATA]

Requirements:
  - seaborn
  - matplotlib
  - joblib
  - tqdm
  - numpy, pandas, etc.

Author: Marxist AGI, rewriting for maximum efficiency & clarity
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from scipy.stats import gaussian_kde
from scipy.interpolate import griddata

from joblib import Memory

# Create a local directory "cache" to store all intermediate results
memory = Memory("./cache", verbose=0)

# =============================================================================
# 1) Utility to load data
# =============================================================================

def load_data(path: str) -> pd.DataFrame:
    """
    Loads the data from parquet or CSV (gzip). 
    Adjust as needed for your real data location and format.
    """
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    else:
        # e.g., 'somefile.csv.gzip'
        return pd.read_csv(path, compression="gzip")

def include_spectral(df: pd.DataFrame) -> pd.DataFrame:
    """
    Repeats Spectral rows (original script).
    """
    unique_noise_values = [n for n in df['noise'].unique() if n != 0.1]
    spectral_rows = df[(df['method'] == 'Spectral') & (df['noise'] == 0.1)]
    new_rows = []
    for noise in unique_noise_values:
        temp = spectral_rows.copy()
        temp['noise'] = noise
        new_rows.append(temp)
    return pd.concat([df] + new_rows, ignore_index=True)

# =============================================================================
# 2) Figure 1: Ground Truth Histograms & Heatmaps
# =============================================================================

@memory.cache
def compute_figure1_data(df: pd.DataFrame, cmap: str = "BuPu"):
    """
    Returns all data needed to plot figure 1 without redoing the
    heavy computations on subsequent runs.
    """
    # Focus on ground truth
    gt_df = df[df['method'] == 'GroundTruth'].drop_duplicates()
    communities = sorted(gt_df['number_of_communities'].unique())

    # -- Hist data (2 passes)
    # 1st pass: global max histogram count
    global_max = 0
    hist_data = []
    
    for nc in communities:
        subset = gt_df[gt_df['number_of_communities'] == nc]['robustness']
        counts, bin_edges = np.histogram(subset, bins=100, range=(0, 1))
        if counts.max() > global_max:
            global_max = counts.max()
        hist_data.append((nc, subset, counts, bin_edges))
    
    # -- Heatmap data (pivot tables)
    # For each community, we create the pivot table for p_in vs multiplier
    heatmaps = {}
    for nc in communities:
        subset = gt_df[gt_df['number_of_communities'] == nc]
        pivot = subset.pivot_table(
            values='robustness',
            index='p_in',
            columns='multiplier',
            aggfunc='mean'
        )
        # sort index & columns
        pivot = pivot.sort_index(ascending=True)
        pivot = pivot.reindex(sorted(pivot.columns), axis=1)
        heatmaps[nc] = pivot
    
    return {
        "communities": communities,
        "global_max": global_max,
        "hist_data": hist_data,
        "heatmaps": heatmaps,
        "cmap": cmap
    }

def plot_figure1(data: dict):
    """
    Plots the histogram (top row) & heatmaps (bottom row) for the ground truth
    from precomputed data (in `data`).
    """
    communities = data["communities"]
    global_max = data["global_max"]
    hist_data = data["hist_data"]
    heatmaps = data["heatmaps"]
    cmap = "cubehelix"  # data["cmap"]

    figsize = (15, 6)
    fig, axs = plt.subplots(2, 5, figsize=figsize)
    
    # === First row: Histograms ===
    hist_color = plt.get_cmap(cmap + '_r')(0)
    for i, (nc, subset, counts, bin_edges) in enumerate(hist_data):
        ax = axs[0, i]
        ax.hist(subset, bins=100, range=(0, 1), color=hist_color)
        ax.set_title(f"{nc} Communities")
        ax.set_xlabel("Fraction of robust nodes")
        ax.set_ylim(0, global_max)
        if i == 0:
            ax.set_ylabel("Ground Truth Partitions Count")
        else:
            ax.set_ylabel("")
    
    # === Second row: Heatmaps ===
    for i, nc in enumerate(communities):
        ax = axs[1, i]
        pivot = heatmaps[nc]
        im = ax.imshow(pivot.values, aspect='auto', vmin=0, vmax=1, cmap=cmap)
        ax.set_xlabel(r"Difficulty Factor ($\lambda$)")
        if i == 0:
            ax.set_ylabel(r"Intra Edge Probability ($p$)")
        else:
            ax.set_ylabel("")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([f"{val:.2f}" for val in pivot.columns], rotation=45, fontsize=8)
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([f"{val:.2f}" for val in pivot.index], fontsize=8)
    
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Robustness")
    fig.subplots_adjust(right=0.9, hspace=0.3, wspace=0.3)
    fig.savefig("figure1.pdf", dpi=300)
    plt.close(fig)

# =============================================================================
# 3) Figure 2 (Robustness): Contour Plots for Methods (Excluding GT)
# =============================================================================
@memory.cache
def compute_figure2_robustness_data(df: pd.DataFrame):
    """
    Precompute the data needed for Figure 2 (robustness version).
    The most time-consuming part is the 2D KDE estimate for each method.
    We'll manually compute and store it so we don't have to do it again.
    """
    methods_mapping = {
        'Leiden': 'Leiden (full-fledged)',
        'Hedonic': 'Leiden (phase 1)',
        'Spectral': 'Spectral Clustering',
        'OnePass': 'OnePass',
        'Mirror': 'Mirror'
    }
    colormap_dict = {
        'Leiden': 'Blues',
        'Hedonic': 'Greens',
        'Spectral': 'Oranges',
        'OnePass': 'Reds',
        'Mirror': 'Purples'
    }
    
    df_methods = df[df['method'].isin(methods_mapping.keys())]
    
    # We'll collect a dict of {method_key -> { "clean":(X, Y, Z), "noisy":(X, Y, Z) }}
    # where (X, Y, Z) are the meshgrid and KDE values for the contour.
    # This way we compute them once.
    kde_results = {}
    
    # Some grid definition for all. Adjust resolution as needed.
    xgrid = np.linspace(0, 1, 200)  # robust can be in [0,1]
    ygrid = np.linspace(0, 1, 200)  # accuracy in [0,1], if that’s your real range
    X, Y = np.meshgrid(xgrid, ygrid)
    
    for m_key in methods_mapping:
        subset_clean = df_methods[(df_methods['method'] == m_key)]
        subset_noisy = subset_clean[(subset_clean['noise'] == 1)]
        
        # compute KDE only if subset has enough points
        def compute_kde2d(xvals, yvals):
            if len(xvals) < 2:
                return np.zeros_like(X)
            kde = gaussian_kde([xvals, yvals])
            coords = np.vstack([X.ravel(), Y.ravel()])
            Z_ = kde(coords).reshape(X.shape)
            return Z_
        
        Z_clean = compute_kde2d(subset_clean['robustness'].values, subset_clean['accuracy'].values)
        Z_noisy = compute_kde2d(subset_noisy['robustness'].values, subset_noisy['accuracy'].values)
        
        kde_results[m_key] = {
            "clean": Z_clean,
            "noisy": Z_noisy
        }
    
    return {
        "methods_mapping": methods_mapping,
        "colormap_dict": colormap_dict,
        "X": X,
        "Y": Y,
        "kde_results": kde_results
    }

def plot_figure2_robustness(fig2_data: dict):
    """
    Using the precomputed meshgrid & 2D KDE, generate the 2x5 subplots:
    - 1st row: (robustness vs accuracy) for noise=0
    - 2nd row: (robustness vs accuracy) for noise=1
    """
    methods_mapping = fig2_data["methods_mapping"]
    colormap_dict = fig2_data["colormap_dict"]
    X = fig2_data["X"]
    Y = fig2_data["Y"]
    kde_results = fig2_data["kde_results"]
    
    figsize = (15, 6)
    fig, axs = plt.subplots(2, 5, figsize=figsize)
    
    for j, (m_key, m_label) in enumerate(methods_mapping.items()):
        ax = axs[0, j]
        Z = kde_results[m_key]["clean"]
        if Z is not None and np.any(Z):
            ax.contourf(X, Y, Z, levels=14, cmap=colormap_dict[m_key])
        ax.set_title(m_label)
        ax.set_xlabel("Robustness")
        if j == 0:
            ax.set_ylabel("Accuracy")
        
        # second row: noisy
        ax2 = axs[1, j]
        Z = kde_results[m_key]["noisy"]
        if Z is not None and np.any(Z):
            ax2.contourf(X, Y, Z, levels=14, cmap=colormap_dict[m_key])
        ax2.set_xlabel("Robustness")
        if j == 0:
            ax2.set_ylabel("Accuracy")
    
    fig.tight_layout()
    fig.savefig("figure2_robustness.pdf", dpi=300)
    plt.close(fig)

# =============================================================================
# 4) Figure 2 (efficiency): Contour Plots for Methods (Excluding GT)
# =============================================================================
@memory.cache
def compute_figure2_efficiency_data(df: pd.DataFrame):
    """
    Similar approach to the above, but now the x-axis is 'duration' and y-axis is 'accuracy'.
    We store a 2D KDE for (duration vs accuracy) for each method, noise=0 and noise=1.
    """
    methods_mapping = {
        'Leiden': 'Leiden (full-fledged)',
        'Hedonic': 'Leiden (phase 1)',
        'Spectral': 'Spectral Clustering',
        'OnePass': 'OnePass',
        'Mirror': 'Mirror'
    }
    colormap_dict = {
        'Leiden': 'Blues',
        'Hedonic': 'Greens',
        'Spectral': 'Oranges',
        'OnePass': 'Reds',
        'Mirror': 'Purples'
    }
    
    df_methods = df[df['method'].isin(methods_mapping.keys())]
    
    # Decide on a grid range for 'duration' if needed
    # Suppose durations are in [0, maxD] for example
    if len(df_methods['duration']) > 0:
        max_dur = df_methods['duration'].max()
    else:
        max_dur = 1.0
    
    xgrid = np.linspace(0, max_dur, 200)
    ygrid = np.linspace(0, 1, 200)  # accuracy in [0,1]
    X, Y = np.meshgrid(xgrid, ygrid)
    
    kde_results = {}
    
    for m_key in methods_mapping:
        subset_clean = df_methods[(df_methods['method'] == m_key)]
        subset_noisy = subset_clean[(subset_clean['noise'] == 1)]
        
        def compute_kde2d(xvals, yvals):
            if len(xvals) < 2:
                return np.zeros_like(X)
            kde = gaussian_kde([xvals, yvals])
            coords = np.vstack([X.ravel(), Y.ravel()])
            Z_ = kde(coords).reshape(X.shape)
            return Z_
        
        Z_clean = compute_kde2d(subset_clean['duration'].values, subset_clean['accuracy'].values)
        Z_noisy = compute_kde2d(subset_noisy['duration'].values, subset_noisy['accuracy'].values)
        
        kde_results[m_key] = {
            "clean": Z_clean,
            "noisy": Z_noisy
        }
    
    return {
        "methods_mapping": methods_mapping,
        "colormap_dict": colormap_dict,
        "X": X,
        "Y": Y,
        "kde_results": kde_results
    }

def plot_figure2_efficiency(fig2_data: dict):
    """
    2 rows x 5 columns: first row = noise=0, second = noise=1
    x-axis = duration, y-axis = accuracy
    """
    methods_mapping = fig2_data["methods_mapping"]
    colormap_dict = fig2_data["colormap_dict"]
    X = fig2_data["X"]
    Y = fig2_data["Y"]
    kde_results = fig2_data["kde_results"]
    
    figsize = (15, 6)
    fig, axs = plt.subplots(2, 5, figsize=figsize)
    
    for j, (m_key, m_label) in enumerate(methods_mapping.items()):
        ax = axs[0, j]
        Z = kde_results[m_key]["clean"]
        if Z is not None and np.any(Z):
            ax.contourf(X, Y, Z, levels=14, cmap=colormap_dict[m_key])
        ax.set_title(m_label)
        ax.set_xlabel("Efficiency")
        if j == 0:
            ax.set_ylabel("Accuracy")
        
        ax2 = axs[1, j]
        Z = kde_results[m_key]["noisy"]
        if Z is not None and np.any(Z):
            ax2.contourf(X, Y, Z, levels=14, cmap=colormap_dict[m_key])
        ax2.set_xlabel("Efficiency")
        if j == 0:
            ax2.set_ylabel("Accuracy")
    
    fig.tight_layout()
    fig.savefig("figure2_efficiency.pdf", dpi=300)
    plt.close(fig)

# =============================================================================
# 5) Figure 3: Bar Plots with Means & Confidence Intervals
# =============================================================================
@memory.cache
def compute_figure3_data(df: pd.DataFrame):
    """
    Return the aggregated means & confidence intervals for the 3 metrics
    at each noise level, for each method. This is relatively quick, but we
    still cache for consistency.
    """
    methods = ['Leiden', 'Hedonic', 'Spectral', 'OnePass', 'Mirror']
    df_methods = df[df['method'].isin(methods)]
    noise_levels = sorted(df_methods['noise'].unique())
    
    # We'll store a dict keyed by metric -> method -> [ (noise, mean, ci), ... ]
    metrics = ['duration', 'robustness', 'accuracy']
    results = {m: {} for m in metrics}
    
    for metric in metrics:
        for meth in methods:
            row_list = []
            for nl in noise_levels:
                sub = df_methods[(df_methods['noise'] == nl) & (df_methods['method'] == meth)]
                if len(sub) == 0:
                    mean_val, ci = (np.nan, 0)
                else:
                    mean_val = sub[metric].mean()
                    ci = 1.96 * sub[metric].std() / np.sqrt(len(sub))
                row_list.append((nl, mean_val, ci))
            results[metric][meth] = row_list

    return {
        "methods": methods,
        "noise_levels": noise_levels,
        "metrics": metrics,
        "results": results
    }

def plot_figure3(fig3_data: dict):
    """
    Plot bar charts from the precomputed means & confidence intervals.
    Each of the 3 metrics is a subplot.
    """
    methods = fig3_data["methods"]
    noise_levels = fig3_data["noise_levels"]
    metrics = fig3_data["metrics"]
    results = fig3_data["results"]
    
    # Some color / label settings
    cmap = plt.get_cmap('tab20b').colors
    methods_labels = {
        'Leiden': 'Leiden (full-fledged)',
        'Hedonic': 'Leiden (phase 1)',
        'Spectral': 'Spectral Clustering',
        'OnePass': 'OnePass',
        'Mirror': 'Mirror'
    }
    idx = 2
    color_dict = {
        'Leiden': cmap[idx],
        'Hedonic': cmap[idx+4],
        'Spectral': cmap[idx+8],
        'OnePass': cmap[idx+12],
        'Mirror': cmap[idx+16]
    }
    
    metric_names = {'duration': 'Efficiency', 'robustness': 'Robustness', 'accuracy': 'Accuracy'}
    
    figsize = (15, 3)
    fig, axs = plt.subplots(1, 3, figsize=figsize)
    bar_width = 0.15
    indices = np.arange(len(noise_levels))

    handles, labels = [], []
    for col, metric in enumerate(metrics):
        ax = axs[col]
        for i, m in enumerate(methods):
            row_list = results[metric][m]
            means = [r[1] for r in row_list]
            cis = [r[2] for r in row_list]
            
            bars = ax.bar(
                indices + i * bar_width,
                means,
                width=bar_width,
                yerr=cis,
                color=color_dict[m],
                capsize=3,
                label=methods_labels[m] if col == 0 else "_nolegend_"
            )
            if col == 0:
                handles.append(bars[0])
                labels.append(methods_labels[m])
        
        ax.set_xticks(indices + bar_width * (len(methods)-1)/2)
        ax.set_xticklabels([f"{n:.2f}" for n in noise_levels])
        ax.set_xlabel("Noise")
        ax.set_ylabel(metric_names[metric])
        ax.set_title(metric_names[metric])
    
    fig.tight_layout()
    fig.legend(handles, labels, loc='upper center', ncol=len(methods), bbox_to_anchor=(0.5, -0.1))
    fig.subplots_adjust(bottom=0.2)
    fig.savefig("figure3.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)

# =============================================================================
# 6) Figure 4: Bar Plots with Means & Confidence Intervals (split by communities)
# =============================================================================
@memory.cache
def compute_figure4_data(df: pd.DataFrame):
    """
    Similar to figure3 but splitted by #communities and noise.
    """
    methods = ['Leiden', 'Hedonic', 'Spectral', 'OnePass', 'Mirror']
    df_methods = df[df['method'].isin(methods)]
    communities = sorted(df_methods['number_of_communities'].unique())
    metrics = ['duration', 'robustness', 'accuracy']
    
    # We'll store a structure {noise=0} and {noise=1} for each metric & method & communities
    # e.g. results[noise][metric][method] -> list of (nc, mean, ci)
    results = {0: {m: {} for m in metrics}, 1: {m: {} for m in metrics}}
    
    for noise_val in [0, 1]:
        sub_noise = df_methods[df_methods['noise'] == noise_val]
        for metric in metrics:
            for meth in methods:
                row_list = []
                for nc in communities:
                    sub = sub_noise[(sub_noise['number_of_communities'] == nc) & (sub_noise['method'] == meth)]
                    if len(sub) == 0:
                        mean_val, ci = (np.nan, 0)
                    else:
                        mean_val = sub[metric].mean()
                        ci = 1.96 * sub[metric].std() / np.sqrt(len(sub))
                    row_list.append((nc, mean_val, ci))
                results[noise_val][metric][meth] = row_list

    return {
        "methods": methods,
        "communities": communities,
        "metrics": metrics,
        "results": results
    }

def plot_figure4(fig4_data: dict):
    """
    2 rows: row=0 => noise=0, row=1 => noise=1
    columns for metrics
    """
    methods = fig4_data["methods"]
    communities = fig4_data["communities"]
    metrics = fig4_data["metrics"]
    results = fig4_data["results"]
    
    cmap = plt.get_cmap('tab20b').colors
    methods_labels = {
        'Leiden': 'Leiden (full-fledged)',
        'Hedonic': 'Leiden (phase 1)',
        'Spectral': 'Spectral Clustering',
        'OnePass': 'OnePass',
        'Mirror': 'Mirror'
    }
    idx = 2
    color_dict = {
        'Leiden': cmap[idx],
        'Hedonic': cmap[idx+4],
        'Spectral': cmap[idx+8],
        'OnePass': cmap[idx+12],
        'Mirror': cmap[idx+16]
    }
    
    metric_names = {'duration': 'Efficiency', 'robustness': 'Robustness', 'accuracy': 'Accuracy'}

    figsize = (15, 6)
    fig, axs = plt.subplots(2, 3, figsize=figsize)
    bar_width = 0.15
    indices = np.arange(len(communities))

    handles, labels = [], []
    for row, noise_val in enumerate([0, 1]):
        for col, metric in enumerate(metrics):
            ax = axs[row, col]
            for i, m in enumerate(methods):
                row_list = results[noise_val][metric][m]
                means = [r[1] for r in row_list]
                cis = [r[2] for r in row_list]
                
                bars = ax.bar(
                    indices + i * bar_width,
                    means,
                    width=bar_width,
                    yerr=cis,
                    color=color_dict[m],
                    capsize=3,
                    label=methods_labels[m] if (row==0 and col==0) else "_nolegend_"
                )
                if (row==0 and col==0):
                    handles.append(bars[0])
                    labels.append(methods_labels[m])
            
            ax.set_xticks(indices + bar_width * (len(methods)-1)/2)
            ax.set_xticklabels([str(nc) for nc in communities])
            
            if row == 0:
                ax.set_title(metric_names[metric])
            if row == 1:
                ax.set_xlabel("Number of Communities")
            
            ax.set_ylabel(metric_names[metric])
    
    fig.tight_layout()
    fig.legend(handles, labels, loc='upper center', ncol=len(methods), bbox_to_anchor=(0.5, -0.1))
    fig.subplots_adjust(bottom=0.1)
    fig.savefig("figure4.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)

# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Generate cached figures from a dataset.')
    parser.add_argument('path', type=str, nargs='?', 
                        default='/Users/lucas/Databases/Hedonic/PHYSA/Synthetic_Networks/V1020/resultados.parquet',
                        help='Path to the parquet or CSV.GZ dataset.')
    args = parser.parse_args()
    data_path = args.path
    
    print(f"Loading data from {data_path} ...")
    df = load_data(data_path)
    df = include_spectral(df)
    print(f"Data loaded. Rows = {len(df)}")

    # -------------------------------------------------------------------------
    # FIGURE 1 (GroundTruth)
    # -------------------------------------------------------------------------
    print("Computing data for figure 1 ...")
    fig1_data = compute_figure1_data(df)
    print("Plotting figure1 ...")
    plot_figure1(fig1_data)
    print("Done: figure1.pdf")

    # -------------------------------------------------------------------------
    # FIGURE 2 (Robustness)
    # -------------------------------------------------------------------------
    print("Computing data for figure 2 (robustness) ...")
    fig2_rob_data = compute_figure2_robustness_data(df)
    print("Plotting figure2_robustness ...")
    plot_figure2_robustness(fig2_rob_data)
    print("Done: figure2_robustness.pdf")

    # -------------------------------------------------------------------------
    # FIGURE 2 (Efficiency)
    # -------------------------------------------------------------------------
    print("Computing data for figure 2 (efficiency) ...")
    fig2_eff_data = compute_figure2_efficiency_data(df)
    print("Plotting figure2_efficiency ...")
    plot_figure2_efficiency(fig2_eff_data)
    print("Done: figure2_efficiency.pdf")

    # -------------------------------------------------------------------------
    # FIGURE 3
    # -------------------------------------------------------------------------
    print("Computing data for figure 3 ...")
    fig3_data = compute_figure3_data(df)
    print("Plotting figure3 ...")
    plot_figure3(fig3_data)
    print("Done: figure3.pdf")

    # -------------------------------------------------------------------------
    # FIGURE 4
    # -------------------------------------------------------------------------
    print("Computing data for figure 4 ...")
    fig4_data = compute_figure4_data(df)
    print("Plotting figure4 ...")
    plot_figure4(fig4_data)
    print("Done: figure4.pdf")

    print("All figures generated successfully.")

if __name__ == "__main__":
    main()
