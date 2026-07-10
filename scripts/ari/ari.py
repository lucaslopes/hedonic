import re
import os
import json
from tqdm import tqdm
from igraph import compare_communities
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd



def get_gt(s):
    """
    Extracts two integers (C, N) from a string containing a pattern like '5C_240'.
    Returns a list `gt` of community assignments for each node, where each community label
    is repeated `nodes_per_community` times.
    """
    match = re.search(r'(\d+)C_(\d+)', s)
    if not match:
        raise ValueError("Pattern like '5C_240' not found in string")
    n_communities = int(match.group(1))
    nodes_per_community = int(match.group(2))
    gt = []
    for c in range(n_communities):
        gt.extend([c] * nodes_per_community)
    return gt

def get_all_json_paths(base_path):
    all_json_paths = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.json'):
                json_path = os.path.join(root, file)
                all_json_paths.append(json_path)
    return all_json_paths

def update_ari(pth):
    with open(pth, 'r') as f:
        data = json.load(f)
        gt = None
        for d in data:
            if d["method"] == "GroundTruth":
                gt = get_gt(pth)
                break
        if gt is None:
            gt = get_gt(pth)
        for d in data:
            partition = d['partition']
            ari = compare_communities(partition, gt, method="adjusted_rand")
            d['adjusted_rand'] = ari
    output_path = pth.replace('/resultados/', '/resultados_ari/')
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f)#, indent=4)

def update_ari_all(base_path):
    all_json_paths = get_all_json_paths(base_path)
    for pth in tqdm(all_json_paths, desc="Processing JSON files"):
        update_ari(pth)


base_path = "/Users/lucas/Databases/Hedonic/PHYSA/Synthetic_Networks/V1020/resultados/"
output_path = base_path.replace('/resultados/', '/resultados_ari/')
# update_ari_all(base_path)

all_json_paths = get_all_json_paths(output_path)
map = list()
for pth in tqdm(all_json_paths, desc="Processing JSON files"):
    with open(pth, 'r') as f:
        data = json.load(f)
        for d in data:
            t = (d['accuracy'], d['adjusted_rand'])
            map.append(t)


def plot_accuracy_ari_lines(map_list):
    """
    Plots a figure where each tuple in map_list is represented as a line
    from (x=0, y=accuracy) to (x=1, y=adjusted_rand).
    """
    plt.figure(figsize=(8, 6))
    for accuracy, ari in map_list:
        plt.plot([0, 1], [accuracy, ari], color='blue', alpha=0.3)
    plt.xticks([0, 1], ['Rand Index', 'Adjusted Rand Index'])
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.title('Rand Index to Adjusted Rand Index Mapping')
    plt.ylim(-1, 1)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()



def plot_accuracy_ari_jointplot(df):
    # df = pd.DataFrame(map_list, columns=['Rand Index', 'Adjusted Rand Index'])
    sns.jointplot(data=df, x='Rand Index', y='Adjusted Rand Index', kind='hex', color='blue')
    plt.show()



df = pd.DataFrame(map, columns=['Rand Index', 'Adjusted Rand Index'])
csv_path = os.path.join(output_path, "accuracy_ari_map.csv")
df.to_csv(csv_path, index=False)



plot_accuracy_ari_jointplot(df)