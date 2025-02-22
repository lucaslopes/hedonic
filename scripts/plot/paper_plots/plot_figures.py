#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
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
    
    fig.subplots_adjust(right=0.9, hspace=0.4)
    fig.savefig("figure1.png", dpi=300)
    plt.close(fig)

# =============================================================================
# Figure 2: Contour Plots for Methods (Excluding GroundTruth)
# =============================================================================
def plot_figure2(df):
    """
    Create Figure 2 with 2 rows x 5 columns of contour plots.
      - Only the 5 methods (excluding GroundTruth) are used:
          Spectral, Leiden, Hedonic, OnePass, Mirror.
      - Column subfigures represent the methods, relabeled as:
          'Spectral Clustering', 'Leiden Algorithm', 'Hedonic Games', 'OnePass', 'Mirror'
      - First row: x-axis = Robustness (0-1), y-axis = Accuracy.
      - Second row: x-axis = Efficiency (duration), y-axis = Accuracy.
      - Only subplots in the left column show the y-axis label.
      - Each method’s contour plot uses a unique color scheme (using standard colormaps
        distinct from Fig. 1).
    """
    # Mapping between internal method names and desired labels.
    methods_mapping = {
        'Spectral': 'Spectral Clustering',
        'Leiden': 'Leiden (full-fledged)',
        'Hedonic': 'Leiden (phase 1)',
        'OnePass': 'OnePass',
        'Mirror': 'Mirror'
    }
    # Define colormaps (avoid the color used in Fig1)
    cmap_dict = {
        'Spectral': "Reds",
        'Leiden': "Blues",
        'Hedonic': "Greens",
        'OnePass': "Purples",
        'Mirror': "Oranges"
    }
    # Filter to include only the 5 methods of interest.
    df_methods = df[df['method'].isin(methods_mapping.keys())]
    figsize = (15, 6)
    fig, axs = plt.subplots(2, 5, figsize=figsize)
    
    # Loop over each method/column.
    for j, (m_key, m_label) in enumerate(methods_mapping.items()):
        subset = df_methods[df_methods['method'] == m_key]
        
        # --- First row: x = Robustness, y = Accuracy ---
        ax = axs[0, j]
        x = subset['robustness'].values
        y = subset['accuracy'].values
        if len(x) > 0:
            # Estimate density via gaussian_kde
            xy = np.vstack([x, y])
            z = gaussian_kde(xy)(xy)
            # Create a grid for interpolation
            xi = np.linspace(0, 1, 100)
            yi = np.linspace(y.min(), y.max(), 100)
            xi, yi = np.meshgrid(xi, yi)
            zi = griddata((x, y), z, (xi, yi), method='linear')
            cont1 = ax.contourf(xi, yi, zi, levels=14, cmap=cmap_dict[m_key])
        ax.set_title(m_label)
        ax.set_xlabel("Robustness")
        if j == 0:
            ax.set_ylabel("Accuracy")
        else:
            ax.set_ylabel("")
        
        # --- Second row: x = Efficiency (duration), y = Accuracy ---
        ax2 = axs[1, j]
        x2 = subset['duration'].values
        y2 = subset['accuracy'].values
        if len(x2) > 0:
            xy2 = np.vstack([x2, y2])
            z2 = gaussian_kde(xy2)(xy2)
            xi2 = np.linspace(x2.min(), x2.max(), 100)
            yi2 = np.linspace(y2.min(), y2.max(), 100)
            xi2, yi2 = np.meshgrid(xi2, yi2)
            zi2 = griddata((x2, y2), z2, (xi2, yi2), method='linear')
            cont2 = ax2.contourf(xi2, yi2, zi2, levels=14, cmap=cmap_dict[m_key])
        ax2.set_xlabel("Efficiency")
        if j == 0:
            ax2.set_ylabel("Accuracy")
        else:
            ax2.set_ylabel("")
    
    fig.tight_layout()
    fig.savefig("figure2.png", dpi=300)
    plt.close(fig)

# =============================================================================
# Figure 3: Bar Plots with Means and Confidence Intervals
# =============================================================================
def plot_figure3(df):
    """
    Create Figure 3 with 2 rows x 3 columns of bar plots.
    
      First row (x-axis = Noise [0-1]):
        - Left: Y-axis = Efficiency
        - Middle: Y-axis = Robustness
        - Right: Y-axis = Accuracy

      Second row (x-axis = Number of Communities):
        - Left: Y-axis = Efficiency
        - Middle: Y-axis = Robustness
        - Right: Y-axis = Accuracy

      For each x-axis value, plot 5 grouped bars (one for each method).
      Methods: Spectral, Leiden, Hedonic, OnePass, Mirror (labeled as in Fig2)
      Each bar shows the mean value and 95% CI.
    """
    # Use only the 5 methods (exclude GroundTruth)
    methods = ['Spectral', 'Leiden', 'Hedonic', 'OnePass', 'Mirror']
    methods_labels = {
        'Spectral': 'Spectral Clustering',
        'Leiden': 'Leiden (full-fledged)',
        'Hedonic': 'Leiden (phase 1)',
        'OnePass': 'OnePass',
        'Mirror': 'Mirror'
    }
    # Use the same colormaps (here using solid colors) as in Fig2.
    color_dict = {
        'Spectral': "red",
        'Leiden': "blue",
        'Hedonic': "green",
        'OnePass': "purple",
        'Mirror': "orange"
    }
    df_methods = df[df['method'].isin(methods)]
    
    # Define the metrics to plot and their display names.
    # For bar plots, we use:
    # - Efficiency: duration (y-axis label "Efficiency")
    # - Robustness: robustness (y-axis label "Robustness")
    # - Accuracy: accuracy (y-axis label "Accuracy")
    metrics = ['duration', 'robustness', 'accuracy']
    metric_names = {'duration': 'Efficiency', 'robustness': 'Robustness', 'accuracy': 'Accuracy'}
    
    # Set common figure size
    figsize = (15, 3)
    fig, axs = plt.subplots(1, 3, figsize=figsize)
    
    # --- First row: Group by Noise ---
    noise_levels = sorted(df_methods['noise'].unique())
    # For each subplot (one metric per column)
    for col in range(3):
        ax = axs[col]
        # Width for each grouped bar
        bar_width = 0.15
        indices = np.arange(len(noise_levels))
        for i, m in enumerate(methods):
            means = []
            cis = []
            for noise in noise_levels:
                sub = df_methods[(df_methods['noise'] == noise) & (df_methods['method'] == m)]
                if len(sub) > 0:
                    mean_val = sub[metrics[col]].mean()
                    std_val = sub[metrics[col]].std()
                    n = len(sub)
                    ci = 1.96 * std_val / np.sqrt(n)
                else:
                    mean_val = np.nan
                    ci = 0
                means.append(mean_val)
                cis.append(ci)
            ax.bar(indices + i * bar_width, means, width=bar_width, yerr=cis,
                   label=methods_labels[m], color=color_dict[m], capsize=3)
        ax.set_xticks(indices + bar_width * (len(methods) - 1) / 2)
        ax.set_xticklabels([f"{noise:.2f}" for noise in noise_levels])
        ax.set_xlabel("Noise")
        ax.set_ylabel(metric_names[metrics[col]])
        ax.set_title(metric_names[metrics[col]])
        ax.legend(fontsize='small')
    
    fig.tight_layout()
    fig.savefig("figure3.png", dpi=300)
    plt.close(fig)

# =============================================================================
# Figure 4: Bar Plots with Means and Confidence Intervals
# =============================================================================
def plot_figure4(df):
    """
    Create Figure 4 with 2 rows x 3 columns of bar plots.
    
      First row (x-axis = Noise [0-1]):
        - Left: Y-axis = Efficiency
        - Middle: Y-axis = Robustness
        - Right: Y-axis = Accuracy

      Second row (x-axis = Number of Communities):
        - Left: Y-axis = Efficiency
        - Middle: Y-axis = Robustness
        - Right: Y-axis = Accuracy

      For each x-axis value, plot 5 grouped bars (one for each method).
      Methods: Spectral, Leiden, Hedonic, OnePass, Mirror (labeled as in Fig2)
      Each bar shows the mean value and 95% CI.
    """
    # Use only the 5 methods (exclude GroundTruth)
    methods = ['Spectral', 'Leiden', 'Hedonic', 'OnePass', 'Mirror']
    methods_labels = {
        'Spectral': 'Spectral Clustering',
        'Leiden': 'Leiden (full-fledged)',
        'Hedonic': 'Leiden (phase 1)',
        'OnePass': 'OnePass',
        'Mirror': 'Mirror'
    }
    # Use the same colormaps (here using solid colors) as in Fig2.
    color_dict = {
        'Spectral': "red",
        'Leiden': "blue",
        'Hedonic': "green",
        'OnePass': "purple",
        'Mirror': "orange"
    }
    df_methods = df[df['method'].isin(methods)]
    
    # Define the metrics to plot and their display names.
    # For bar plots, we use:
    # - Efficiency: duration (y-axis label "Efficiency")
    # - Robustness: robustness (y-axis label "Robustness")
    # - Accuracy: accuracy (y-axis label "Accuracy")
    metrics = ['duration', 'robustness', 'accuracy']
    metric_names = {'duration': 'Efficiency', 'robustness': 'Robustness', 'accuracy': 'Accuracy'}
    
    # Set common figure size
    figsize = (15, 6)
    fig, axs = plt.subplots(2, 3, figsize=figsize)
    
    # --- First row: Group by Noise ---
    communities = sorted(df_methods['number_of_communities'].unique())
    for col in range(3):
        ax = axs[0, col]
        bar_width = 0.15
        indices = np.arange(len(communities))
        for i, m in enumerate(methods):
            means = []
            cis = []
            for nc in communities:
                sub = df_methods[(df_methods['number_of_communities'] == nc) & (df_methods['method'] == m)]
                if len(sub) > 0:
                    mean_val = sub[metrics[col]].mean()
                    std_val = sub[metrics[col]].std()
                    n = len(sub)
                    ci = 1.96 * std_val / np.sqrt(n)
                else:
                    mean_val = np.nan
                    ci = 0
                means.append(mean_val)
                cis.append(ci)
            ax.bar(indices + i * bar_width, means, width=bar_width, yerr=cis,
                   label=methods_labels[m], color=color_dict[m], capsize=3)
        ax.set_xticks(indices + bar_width * (len(methods) - 1) / 2)
        ax.set_xticklabels([str(nc) for nc in communities])
        # ax.set_xlabel("Number of Communities")
        ax.set_ylabel(metric_names[metrics[col]])
        # ax.set_title(metric_names[metrics[col]])
        ax.legend(fontsize='small')
    
    # --- Second row: Group by Number of Communities ---
    communities = sorted(df_methods['number_of_communities'].unique())
    for col in range(3):
        ax = axs[1, col]
        bar_width = 0.15
        indices = np.arange(len(communities))
        for i, m in enumerate(methods):
            means = []
            cis = []
            for nc in communities:
                sub = df_methods[(df_methods['number_of_communities'] == nc) & (df_methods['method'] == m)]
                if len(sub) > 0:
                    mean_val = sub[metrics[col]].mean()
                    std_val = sub[metrics[col]].std()
                    n = len(sub)
                    ci = 1.96 * std_val / np.sqrt(n)
                else:
                    mean_val = np.nan
                    ci = 0
                means.append(mean_val)
                cis.append(ci)
            ax.bar(indices + i * bar_width, means, width=bar_width, yerr=cis,
                   label=methods_labels[m], color=color_dict[m], capsize=3)
        ax.set_xticks(indices + bar_width * (len(methods) - 1) / 2)
        ax.set_xticklabels([str(nc) for nc in communities])
        ax.set_xlabel("Number of Communities")
        ax.set_ylabel(metric_names[metrics[col]])
        # ax.set_title(metric_names[metrics[col]])
        ax.legend(fontsize='small')
    
    fig.tight_layout()
    fig.savefig("figure4.png", dpi=300)
    plt.close(fig)

# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate figures from JSON files.')
    parser.add_argument('path', type=str, help='Path to the directory containing the JSON files.', nargs='?', default='/Users/lucas/Databases/Hedonic/PHYSA/Synthetic_Networks/V1020/resultados.csv.gzip')
    args = parser.parse_args()
    dir_path = args.path
    
    df = pd.read_csv(dir_path, compression="gzip")
    
    # Create the three figures
    # plot_figure1(df)
    # plot_figure2(df)
    plot_figure3(df)
    plot_figure4(df)
    
    print("Figures generated and saved as figure1.png, figure2.png, and figure3.png")

