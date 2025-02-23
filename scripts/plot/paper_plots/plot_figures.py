#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from stopwatch import Stopwatch
from scipy.stats import gaussian_kde
from scipy.interpolate import griddata


# =============================================================================
# Figure 1: Ground Truth Histograms and Heatmaps
# =============================================================================
def plot_figure1(df: pd.DataFrame, cmap: str = "BuPu"):
    # Use only GroundTruth rows
    gt_df = df[df['method'] == 'GroundTruth'].drop_duplicates()
    communities = sorted(gt_df['number_of_communities'].unique())
    
    figsize = (15, 6)
    fig, axs = plt.subplots(2, 5, figsize=figsize)
    
    # ---------------------------
    # First pass: find the global max histogram count
    # ---------------------------
    global_max = 0
    hist_data = []
    
    for nc in communities:
        subset = gt_df[gt_df['number_of_communities'] == nc]['robustness']
        # Use np.histogram (no plotting yet) to find the bin counts
        counts, bin_edges = np.histogram(subset, bins=100, range=(0, 1))
        # Track the maximum
        if counts.max() > global_max:
            global_max = counts.max()
        # Store everything for second pass
        hist_data.append((nc, subset, counts, bin_edges))
    
    # ---------------------------
    # Second pass: plot histograms & set y-limits
    # ---------------------------
    hist_color = plt.get_cmap(cmap + '_r')(0)
    for i, (nc, subset, counts, bin_edges) in enumerate(hist_data):
        ax = axs[0, i]
        ax.hist(subset, bins=100, range=(0, 1), color=hist_color)
        ax.set_title(f"{nc} Communities")
        ax.set_xlabel("Fraction of robust nodes")
        # Set the same y-limit for all histograms
        ax.set_ylim(0, global_max)
        
        if i == 0:
            ax.set_ylabel("Ground Truth Partitions Count")
        else:
            ax.set_ylabel("")
    
    # ---------------------------
    # Second row: Heatmaps
    # ---------------------------
    for i, nc in enumerate(communities):
        ax = axs[1, i]
        subset = gt_df[gt_df['number_of_communities'] == nc]
        pivot = subset.pivot_table(
            values='robustness', 
            index='p_in', 
            columns='multiplier', 
            aggfunc='mean'
        )
        # Sort rows/columns
        pivot = pivot.sort_index(ascending=True)
        pivot = pivot.reindex(sorted(pivot.columns), axis=1)
        im = ax.imshow(pivot.values, aspect='auto', vmin=0, vmax=1, cmap=cmap)
        
        ax.set_xlabel(r"Difficulty Factor ($\lambda$)")
        if i == 0:
            ax.set_ylabel(r"Intra Edge Probability ($p$)")
        else:
            ax.set_ylabel("")
        
        # Ticks/labels
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([f"{val:.2f}" for val in pivot.columns], rotation=45, fontsize=8)
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([f"{val:.2f}" for val in pivot.index], fontsize=8)
    
    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Robustness")
    
    fig.subplots_adjust(right=0.9, hspace=0.3, wspace=0.3)
    # fig.subplots_adjust(right=0.88, wspace=0.3)
    fig.savefig("figure1.pdf", dpi=300)
    plt.close(fig)

# =============================================================================
# Figure 2 (robustness): Contour Plots for Methods (Excluding GroundTruth)
# =============================================================================
def plot_figure2_robustness(df):
    """
    Create Figure 2 with 2 rows x 5 columns of contour plots.
      - Only the 5 methods (excluding GroundTruth) are used:
          Spectral, Leiden, Hedonic, OnePass, Mirror.
      - Column subfigures represent the methods, relabeled as:
          'Spectral Clustering', 'Leiden (full-fledged)', 'Leiden (phase 1)', 'OnePass', 'Mirror'
      - First row: x-axis = Robustness (0-1), y-axis = Accuracy.
      - Second row: x-axis = Efficiency (duration), y-axis = Accuracy.
      - Only subplots in the left column show the y-axis label.
      - Each method’s contour plot uses a unique color scheme (using standard colormaps
        distinct from Fig. 1).
    """
    # Mapping between internal method names and desired labels.
    methods_mapping = {
        'Leiden': 'Leiden (full-fledged)',
        'Hedonic': 'Leiden (phase 1)',
        'Spectral': 'Spectral Clustering',
        'OnePass': 'OnePass',
        'Mirror': 'Mirror'
    }
    
    # Define distinct colormaps for each method.
    # (These are standard matplotlib colormaps that you can change as needed.)
    colormap_dict = {
        'Leiden': 'Blues',
        'Hedonic': 'Greens',
        'Spectral': 'Oranges',
        'OnePass': 'Reds',
        'Mirror': 'Purples'
    }
    
    # Filter the dataframe to include only the methods of interest.
    df_methods = df[df['method'].isin(methods_mapping.keys())]
    figsize = (15, 6)
    fig, axs = plt.subplots(2, 5, figsize=figsize)
    
    # Loop over each method/column.
    for j, (m_key, m_label) in enumerate(methods_mapping.items()):
        subset = df_methods[df_methods['method'] == m_key]
        subset_noisy = subset[subset['noise'] == 1]
        
        # --- First row: x = Robustness, y = Accuracy ---
        ax = axs[0, j]
        if not subset.empty:
            sns.kdeplot(
                x=subset['robustness'],
                y=subset['accuracy'],
                ax=ax,
                fill=True,
                levels=14,
                cmap=colormap_dict[m_key]
            )
        ax.set_title(m_label)
        ax.set_xlabel("Robustness")
        if j == 0:
            ax.set_ylabel("Accuracy")
        else:
            ax.set_ylabel("")
        
        # --- Second row: x = Robustness (robustness), y = Accuracy ---
        ax2 = axs[1, j]
        if not subset_noisy.empty:
            sns.kdeplot(
                x=subset_noisy['robustness'],
                y=subset_noisy['accuracy'],
                ax=ax2,
                fill=True,
                levels=14,
                cmap=colormap_dict[m_key]
            )
        ax2.set_xlabel("Robustness")
        if j == 0:
            ax2.set_ylabel("Accuracy")
        else:
            ax2.set_ylabel("")
    
    fig.tight_layout()
    fig.savefig("figure2_robustness.pdf", dpi=300)
    plt.close(fig)

# =============================================================================
# Figure 2 (efficiency): Contour Plots for Methods (Excluding GroundTruth)
# =============================================================================
def plot_figure2_efficiency(df):
    """
    Create Figure 2 with 2 rows x 5 columns of contour plots.
      - Only the 5 methods (excluding GroundTruth) are used:
          Spectral, Leiden, Hedonic, OnePass, Mirror.
      - Column subfigures represent the methods, relabeled as:
          'Spectral Clustering', 'Leiden (full-fledged)', 'Leiden (phase 1)', 'OnePass', 'Mirror'
      - First row: x-axis = Robustness (0-1), y-axis = Accuracy.
      - Second row: x-axis = Efficiency (duration), y-axis = Accuracy.
      - Only subplots in the left column show the y-axis label.
      - Each method’s contour plot uses a unique color scheme (using standard colormaps
        distinct from Fig. 1).
    """
    # Mapping between internal method names and desired labels.
    methods_mapping = {
        'Leiden': 'Leiden (full-fledged)',
        'Hedonic': 'Leiden (phase 1)',
        'Spectral': 'Spectral Clustering',
        'OnePass': 'OnePass',
        'Mirror': 'Mirror'
    }
    
    # Define distinct colormaps for each method.
    # (These are standard matplotlib colormaps that you can change as needed.)
    colormap_dict = {
        'Leiden': 'Blues',
        'Hedonic': 'Greens',
        'Spectral': 'Oranges',
        'OnePass': 'Reds',
        'Mirror': 'Purples'
    }
    
    # Filter the dataframe to include only the methods of interest.
    df_methods = df[df['method'].isin(methods_mapping.keys())]
    figsize = (15, 6)
    fig, axs = plt.subplots(2, 5, figsize=figsize)
    
    # Loop over each method/column.
    for j, (m_key, m_label) in enumerate(methods_mapping.items()):
        subset = df_methods[df_methods['method'] == m_key]
        subset_noisy = subset[subset['noise'] == 1]
        
        # --- First row: x = Efficiency (duration), y = Accuracy ---
        ax = axs[0, j]
        if not subset.empty:
            sns.kdeplot(
                x=subset['duration'],
                y=subset['accuracy'],
                ax=ax,
                fill=True,
                levels=14,
                cmap=colormap_dict[m_key]
            )
        ax.set_title(m_label)
        ax.set_xlabel("Efficiency")
        if j == 0:
            ax.set_ylabel("Accuracy")
        else:
            ax.set_ylabel("")
        
        # --- Second row: x = Efficiency (duration), y = Accuracy ---
        ax2 = axs[1, j]
        if not subset_noisy.empty:
            sns.kdeplot(
                x=subset_noisy['duration'],
                y=subset_noisy['accuracy'],
                ax=ax2,
                fill=True,
                levels=14,
                cmap=colormap_dict[m_key]
            )
        ax2.set_xlabel("Efficiency")
        if j == 0:
            ax2.set_ylabel("Accuracy")
        else:
            ax2.set_ylabel("")
    
    fig.tight_layout()
    fig.savefig("figure2_efficiency.pdf", dpi=300)
    plt.close(fig)

# =============================================================================
# Figure 3: Bar Plots with Means and Confidence Intervals
# =============================================================================
def plot_figure3(df):
    methods = ['Leiden', 'Hedonic', 'Spectral', 'OnePass', 'Mirror']
    cmap = plt.get_cmap('tab20b').colors
    methods_labels = {
        'Leiden': 'Leiden (full-fledged)',
        'Hedonic': 'Leiden (phase 1)',
        'Spectral': 'Spectral Clustering',
        'OnePass': 'OnePass',
        'Mirror': 'Mirror'
    }
    idx=2
    color_dict = {
        'Leiden': cmap[idx],
        'Hedonic': cmap[idx+4],
        'Spectral': cmap[idx+8],
        'OnePass': cmap[idx+12],
        'Mirror': cmap[idx+16]
    }
    df_methods = df[df['method'].isin(methods)]
    
    metrics = ['duration', 'robustness', 'accuracy']
    metric_names = {'duration': 'Efficiency', 'robustness': 'Robustness', 'accuracy': 'Accuracy'}
    
    figsize = (15, 3)
    fig, axs = plt.subplots(1, 3, figsize=figsize)
    
    noise_levels = sorted(df_methods['noise'].unique())
    bar_width = 0.15
    indices = np.arange(len(noise_levels))

    handles, labels = [], []
    
    for col in range(3):
        ax = axs[col]
        for i, m in enumerate(methods):
            means, cis = [], []
            for noise in noise_levels:
                sub = df_methods[(df_methods['noise'] == noise) & (df_methods['method'] == m)]
                mean_val, ci = (sub[metrics[col]].mean(), 1.96 * sub[metrics[col]].std() / np.sqrt(len(sub))) if len(sub) > 0 else (np.nan, 0)
                means.append(mean_val)
                cis.append(ci)
            
            bars = ax.bar(indices + i * bar_width, means, width=bar_width, yerr=cis,
                          color=color_dict[m], capsize=3, label=methods_labels[m] if col == 0 else "_nolegend_")
            if col == 0:
                handles.append(bars[0])
                labels.append(methods_labels[m])
        
        ax.set_xticks(indices + bar_width * (len(methods) - 1) / 2)
        ax.set_xticklabels([f"{noise:.2f}" for noise in noise_levels])
        ax.set_xlabel("Noise")
        ax.set_ylabel(metric_names[metrics[col]])
        ax.set_title(metric_names[metrics[col]])
    
    fig.tight_layout()
    
    # Add the single legend and adjust layout to avoid cropping
    fig.legend(handles, labels, loc='upper center', ncol=len(methods), bbox_to_anchor=(0.5, -0.1))
    fig.subplots_adjust(bottom=0.1)  # Adjust top margin

    fig.savefig("figure3.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)

# =============================================================================
# Figure 4: Bar Plots with Means and Confidence Intervals
# =============================================================================
def plot_figure4(df):
    methods = ['Leiden', 'Hedonic', 'Spectral', 'OnePass', 'Mirror']
    cmap = plt.get_cmap('tab20b').colors
    methods_labels = {
        'Leiden': 'Leiden (full-fledged)',
        'Hedonic': 'Leiden (phase 1)',
        'Spectral': 'Spectral Clustering',
        'OnePass': 'OnePass',
        'Mirror': 'Mirror'
    }
    idx=2
    color_dict = {
        'Leiden': cmap[idx],
        'Hedonic': cmap[idx+4],
        'Spectral': cmap[idx+8],
        'OnePass': cmap[idx+12],
        'Mirror': cmap[idx+16]
    }
    df_methods = df[df['method'].isin(methods)]
    
    metrics = ['duration', 'robustness', 'accuracy']
    metric_names = {'duration': 'Efficiency', 'robustness': 'Robustness', 'accuracy': 'Accuracy'}
    
    figsize = (15, 6)
    fig, axs = plt.subplots(2, 3, figsize=figsize)
    
    communities = sorted(df_methods['number_of_communities'].unique())
    bar_width = 0.15
    indices = np.arange(len(communities))

    handles, labels = [], []

    for row in range(2):
        if row == 1:
            df_methods = df_methods[df_methods['noise'] == 1.0]
        for col in range(3):
            ax = axs[row, col]
            for i, m in enumerate(methods):
                means, cis = [], []
                for nc in communities:
                    sub = df_methods[(df_methods['number_of_communities'] == nc) & (df_methods['method'] == m)]
                    mean_val, ci = (sub[metrics[col]].mean(), 1.96 * sub[metrics[col]].std() / np.sqrt(len(sub))) if len(sub) > 0 else (np.nan, 0)
                    means.append(mean_val)
                    cis.append(ci)
                
                bars = ax.bar(indices + i * bar_width, means, width=bar_width, yerr=cis,
                              color=color_dict[m], capsize=3, label=methods_labels[m] if (row, col) == (0, 0) else "_nolegend_")
                if (row, col) == (0, 0):
                    handles.append(bars[0])
                    labels.append(methods_labels[m])

            ax.set_xticks(indices + bar_width * (len(methods) - 1) / 2)
            ax.set_xticklabels([str(nc) for nc in communities])
            if row == 1:
                ax.set_xlabel("Number of Communities")
            else:
                ax.set_title(metric_names[metrics[col]])
            ax.set_ylabel(metric_names[metrics[col]])
    
    fig.tight_layout()

    # Add the single legend and adjust layout to avoid cropping
    fig.legend(handles, labels, loc='upper center', ncol=len(methods), bbox_to_anchor=(0.5, -0.1))
    fig.subplots_adjust(bottom=0)  # Adjust top margin

    fig.savefig("figure4.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)

def include_spectral(df):
    # Get unique noise values excluding 0.1
    unique_noise_values = df['noise'].unique()
    unique_noise_values = [n for n in unique_noise_values if n != 0.1]

    # Filter rows where method is 'Spectral' and noise is 0.1
    spectral_rows = df[(df['method'] == 'Spectral') & (df['noise'] == 0.1)]

    # Create new rows by copying and updating the noise values
    new_rows = []
    for noise in unique_noise_values:
        temp = spectral_rows.copy()
        temp['noise'] = noise
        new_rows.append(temp)

    # Append new rows to the original DataFrame
    df = pd.concat([df] + new_rows, ignore_index=True)
    return df


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate figures from JSON files.')
    parser.add_argument('path', type=str, help='Path to the directory containing the JSON files.', nargs='?', default='/Users/lucas/Databases/Hedonic/PHYSA/Synthetic_Networks/V1020/resultados.parquet')
    args = parser.parse_args()
    dir_path = args.path
    sw = Stopwatch()
    sw.start()
    if False:
        df = pd.read_parquet(dir_path)
    else:
        df = pd.read_csv(dir_path.replace(".parquet", ".csv.gzip"), compression="gzip")
    sw.stop()
    print(f"Loaded {dir_path} in {sw.duration}s")
    df = include_spectral(df)
    # Create the three figures
    sw.reset()
    sw.start()
    # plot_figure1(df)
    sw.stop()
    print(f"Generated figure1.pdf in {sw.duration}s")
    sw.reset()
    sw.start()
    plot_figure2(df)
    sw.stop()
    print(f"Generated figure2.pdf in {sw.duration}s")
    sw.reset()
    sw.start()
    # plot_figure3(df)
    sw.stop()
    print(f"Generated figure3.pdf in {sw.duration}s")
    sw.reset()
    sw.start()
    # plot_figure4(df)
    sw.stop()
    print(f"Generated figure4.pdf in {sw.duration}s")
    
    print("Figures generated and saved as figure1.pdf, figure2.pdf, and figure3.pdf")

