import os
import cv2
import json
import config
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
import re

#################################################

def plot_heatmaps(df, max_value=None):
  fig, axes = plt.subplots(3, 7, figsize=(21, 7), dpi=300) # (14, 7)
  method_names = {
    'GroundTruth': 'GroundTruth',
    'Hedonic': 'Hedonic',
    'Leiden': 'Leiden',
    'Louvain': 'Louvain',
    'Spectral': 'Spectral',
    'OnePass': 'OnePass',
    'Mirror': 'Mirror',}
  value_columns = {
    'accuracy': 'Greens',
    'robustness': 'Blues',
    'efficiency': 'YlOrBr',}
  method_col = {
    'GroundTruth': 0,
    'Spectral': 1,
    'Louvain': 2,
    'Leiden': 3,
    'Hedonic': 4,
    'OnePass': 5,
    'Mirror': 6,}
  value_row = {
    'efficiency': 0,
    'robustness': 1,
    'accuracy': 2,
    }
  for i, value_column in enumerate(value_columns):
    vmin = 0
    # vmax = df['efficiency'].max() if max_value is None else max_value if value_column == 'efficiency' else 1
    vmax = df[value_column].max() if value_column == 'efficiency' else 1
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
  # my_path = '/Users/lucas/Databases/Hedonic/Experiments/paper_exp/Clusters = 3/Size = 340/Noise = 0.1/P_in = 0.1/Multiplier = 0.1/Method = community_hedonic/hedonic_1713561021.846727.json'
  directory = os.path.expanduser(directory_path)
  for root, dirs, files in os.walk(directory):
    for file in files:
      if file.endswith(".json"):
        pth = os.path.join(root, file)
        with open(pth, 'r') as f:
          try:
            json_data = json.load(f)
          except json.JSONDecodeError:
            print(f"Error reading {pth}")
            continue
        json_data['path'] = pth
        data.append(json_data)
  df = pd.DataFrame(data, index=range(len(data)))  # Specify an index based on the number of elements in data
  return df

#################################################

def main():
  frame_files = []
  folder = 'PHYSA_1000'
  base = f'/Users/lucas/Databases/Hedonic/'
  pth_base = f'{base}/{folder}'
  df = read_json_to_dataframe(pth_base)#config.experiment_params['output_results_path'])
  df.rename(columns={'duration': 'efficiency'}, inplace=True)
  df.to_pickle(f"{base}/{folder}.pkl", compression="bz2")
  vmax = df['efficiency'].max()
  df['noise'] = df['path'].apply(lambda x: float(re.search(r'/Noise = (\d+.*)/P_in*', x).group(1)))
  paths = {re.sub(r'/P_in = \d+.*', '', p) for p in df['path'].values}# if 'Clusters = 3' in p}
  paths = sorted(paths)
  for i, pth in tqdm(enumerate(paths), total=len(paths)):  # Generate 100 frames
    n_clusters = int(re.search(r'/(\d+) Communities of', pth).group(1))
    noise = float(re.search(r'/Noise = (\d+.*)', pth).group(1))
    df_filter = df[(df["number_of_communities"] == n_clusters) & (df["noise"] == noise)]
    fig, axes = plot_heatmaps(df_filter, max_value=None)
    frame_file = f"media/{folder}_frame_{i:02d}.png"
    fig.suptitle(pth[len(pth_base)+1:].replace('/', ' | '), fontsize=16)
    plt.tight_layout()
    plt.savefig(frame_file)
    plt.close()
    frame_files.append(frame_file)
  # Compose video
  frame = cv2.imread(frame_files[0])
  height, width, layers = frame.shape
  video = cv2.VideoWriter(f"media/video_{folder}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))
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