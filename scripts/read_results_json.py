'''
This script reads the all results.json files from a given directory and creates a dataframe with the results.
'''


import os
import json
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt


def read_json_to_dataframe(directory_path):
    data = []
    directory = os.path.expanduser(directory_path)
    files_to_process = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):# and file.startswith('hedonic'):
                files_to_process.append(os.path.join(root, file))
    
    for filepath in tqdm(files_to_process, desc="Processing JSON files"):
        with open(filepath, 'r') as f:
            d = json.load(f)
            d.pop('partition', None)
            d['path'] = filepath
            data.append(d)
    
    return pd.DataFrame(data)


def preprocess_dataframe(directory_path):
    df = pd.read_csv(directory_path + '.csv.gz')
    df = df[df['method'] != 'groundtruth']
    df = df[df['duration'] < 1]
    df = df[df['accuracy'] > 0]
    df['method'] = df['method'].astype('category')
    return df


def plot_duration_histogram(df):
    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("husl", len(df['method'].unique()))
    sns.histplot(data=df, x='duration', hue='method', palette=palette, multiple='layer', bins=500, alpha=0.5, stat="count")
    plt.title('Histogram of Duration by Method')
    plt.xlabel('Duration')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()


def create_single_kde_plot(data, ax, title):
    """Create a single KDE plot for one method."""
    sns.kdeplot(
        data=data, x='robustness', y='accuracy',
        fill=True, alpha=0.5, ax=ax, cmap='viridis'
    )
    ax.set_title(title)
    ax.set_xlabel('Robustness')
    ax.set_ylabel('Accuracy')

def plot_scatter_robustness_accuracy(df):
    methods = df['method'].unique()
    n_methods = len(methods)
    
    # Calculate number of rows and columns for subplots
    ncols = n_methods # min(3, n_methods)  # Maximum 3 plots per row
    nrows = 1 # (n_methods + ncols - 1) // ncols
    
    # Create figure with subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, 
                            figsize=(6*ncols, 5*nrows))
    
    # Flatten axes array if multiple rows
    if nrows > 1:
        axes = axes.flatten()
    elif ncols == 1:
        axes = [axes]
    
    # Create individual plots
    for i, method in enumerate(methods):
        method_data = df[df['method'] == method]
        create_single_kde_plot(method_data, axes[i], f'Method: {method}')
    
    # Hide empty subplots if any
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('scatter_robustness_accuracy_subplots.png', dpi=300)
    plt.show()


directory_path = '/Users/lucas/Databases/Hedonic/ServerResult/FINAL_1020'
# df = read_json_to_dataframe(directory_path)
# df.to_csv(directory_path + '.csv.gz', index=False, compression='gzip')
df = preprocess_dataframe(directory_path)
# plot_duration_histogram(df)
plot_scatter_robustness_accuracy(df)