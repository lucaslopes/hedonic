import os
import cv2
import json
import config
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import re

#################################################

def plot_heatmaps(df):
  fig, axes = plt.subplots(3, 5, figsize=(16, 9), dpi=300) # (14, 7)
  method_names = {
    'hedonic': 'Hedonic',
    'leiden': 'Leiden',
    'multilevel': 'Louvain',
    'leading': 'Spectral',
    'local': 'L. Improve',}
  value_columns = {
    'accuracy': 'Greens',
    'robustness': 'Blues',
    'duration': 'YlOrBr',}
  method_col = {
    'hedonic': 3,
    'leiden': 2,
    'multilevel': 1,
    'leading': 0,
    'local': 4,}
  value_row = {
    'accuracy': 0,
    'robustness': 1,
    'duration': 2,}
  for i, value_column in enumerate(value_columns):
    vmin = 0
    vmax = df[value_column].max() if value_column == 'duration' else 1
    grouped_data = df.groupby(['method', 'p_in', 'multiplier'])[value_column].mean().reset_index()
    methods = grouped_data["method"].unique()
    for j, method in enumerate(methods):
      method_df = grouped_data[grouped_data["method"] == method]
      method_data = method_df.pivot_table(index="p_in", columns="multiplier", values=value_column)
      row, col = value_row[value_column], method_col[method]
      ax = sns.heatmap(method_data, ax=axes[row, col], cmap=value_columns[value_column], vmin=vmin, vmax=vmax)
      ax.set_title(f"{method_names[method]} wrt {value_column}")
      ax.set_xlabel('difficulty multiplier' if i == len(value_columns)-1 else None)
      ax.set_ylabel('intra-community probability' if j == 0 else None)
  return fig, axes

#################################################

def read_json_to_dataframe(directory_path):
  data = []
  my_path = '/Users/lucas/Databases/Hedonic/Experiments/paper_exp/Clusters = 3/Size = 340/Noise = 0.1/P_in = 0.1/Multiplier = 0.1/Method = community_hedonic/hedonic_1713561021.846727.json'
  directory = os.path.expanduser(directory_path)
  for root, dirs, files in os.walk(directory):
    for file in files:
      if file.endswith(".json"):
        pth = os.path.join(root, file)
        with open(pth, 'r') as f:
          json_data = json.load(f)
        json_data['path'] = pth
        data.append(json_data)
  df = pd.DataFrame(data, index=range(len(data)))  # Specify an index based on the number of elements in data
  return df

#################################################

def main():
  frame_files = []
  pth_base = '/Users/lucas/Databases/Hedonic/Experiments/paper_exp/'
  df = read_json_to_dataframe(pth_base)#config.experiment_params['output_results_path'])
  paths = {re.sub(r'/P_in = \d+.*', '', p) for p in df['path'].values if 'Clusters = 3' in p}
  paths = sorted(paths)
  for i, pth in enumerate(paths):  # Generate 100 frames
    n_clusters = int(re.search(r'/Clusters = (\d+)/', pth).group(1))
    noise = float(re.search(r'/Noise = (\d+.*)', pth).group(1))
    df_filter = df[(df["number_of_communities"] == n_clusters) & (df["noise"] == noise)]
    fig, axes = plot_heatmaps(df_filter)
    frame_file = f"frame_{i:02d}.png"
    fig.suptitle(pth[len(pth_base):], fontsize=16)
    plt.tight_layout()
    plt.savefig(frame_file)
    plt.close()
    frame_files.append(frame_file)
  # Compose video
  frame = cv2.imread(frame_files[0])
  height, width, layers = frame.shape
  video = cv2.VideoWriter('video__.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))
  frames_per_image = 10  # Define how many frames you want for each image
  for frame_file in frame_files:
    image = cv2.imread(frame_file)
    for _ in range(frames_per_image):  # Write the same image multiple times
      video.write(image)
  video.release()
  # Optional: Cleanup frame files
  for frame_file in frame_files:
    os.remove(frame_file)


__name__ == '__main__' and main()