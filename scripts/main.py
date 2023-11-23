import os
import json
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from experiment import run_exp

#################################################

def plot_heatmaps(df, community_size=None, value_column='accuracy'):
    if community_size:
        df = df[df["community_size"] == community_size]
    if value_column == 'duration':
        vmin, vmax = df[value_column].min(), df[value_column].max()
    else:
        vmin, vmax = 0, 1
    grouped_data = df.groupby(['method', 'p_in', 'multiplier'])[value_column].mean().reset_index()
    methods = grouped_data["method"].unique()
    fig, axes = plt.subplots(1, len(methods), figsize=(15, 5))
    for i, method in enumerate(methods):
        method_df = grouped_data[grouped_data["method"] == method]
        method_data = method_df.pivot_table(index="p_in", columns="multiplier", values=value_column)
        ax = sns.heatmap(method_data, ax=axes[i], cmap="Greens", vmin=vmin, vmax=vmax)
        ax.set_title(method)
        ax.set_xlabel("multiplier")
        ax.set_ylabel("p_in")
    return fig, axes

#################################################

run_exp(
    community_size=100,
    number_of_communities=2,
    length_of_probability_array=3)
data = []
directory = os.path.expanduser('~/Databases/hedonic')
for file in os.listdir(directory):
    file_path = os.path.join(directory, file)
    with open(file_path, 'r') as f:
        json_data = json.load(f)
    data.append(json_data)
df = pd.DataFrame(data, index=range(len(data)))  # Specify an index based on the number of elements in data
fig, axes = plot_heatmaps(df, 100)
plt.tight_layout()
plt.show()