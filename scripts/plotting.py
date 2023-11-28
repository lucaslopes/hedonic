import os
import json
import config
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

#################################################

def plot_heatmaps(df, value_column='accuracy'):
  vmin = 0
  vmax = df[value_column].max() if value_column == 'duration' else 1
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

def read_json_to_dataframe(directory_path):
  data = []
  directory = os.path.expanduser(directory_path)
  files = [f for f in os.listdir(directory) if f.endswith(".json")]
  for file in files:
    with open(os.path.join(directory, file), 'r') as f:
      json_data = json.load(f)
    data.append(json_data)
  df = pd.DataFrame(data, index=range(len(data)))  # Specify an index based on the number of elements in data
  return df

#################################################

def main():
  df = read_json_to_dataframe(config.experiment_params['output_results_path'])
  df = df[df["community_size"] == config.experiment_params['community_size']]
  fig, axes = plot_heatmaps(df)
  plt.tight_layout()
  plt.show()


__name__ == '__main__' and main()