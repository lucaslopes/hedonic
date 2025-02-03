import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def read_txt_files(directory):
    txt_files_content = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    txt_files_content[file_path] = f.read()
    return txt_files_content


def extract_info_from_path(file_path, content):
    pattern = re.compile(r'(\d+)C_.*?/P_in = ([\d.]+)/Difficulty = ([\d.]+)/robustness_(\d+)\.txt')
    match = pattern.search(file_path)
    if match:
        return {
            'communities': int(match.group(1)),
            'network': int(match.group(4)),
            'P_in': float(match.group(2)),
            'difficulty': float(match.group(3)),
            'robustness': float(content)
        }
    return None

directory = '/Users/lucas/Databases/Hedonic/PHYSA_2040/robustness'
txt_files_content = read_txt_files(directory)

data = [extract_info_from_path(file_path, content) for file_path, content in txt_files_content.items()]
df = pd.DataFrame(data)

# Plot individual histograms for each community value
unique_communities = df['communities'].unique()
num_communities = len(unique_communities)

plt.figure(figsize=(25, 4))
for i, community in enumerate(unique_communities, 1):
    plt.subplot(1, num_communities, community-1)
    sns.histplot(data=df[df['communities'] == community], x='robustness', bins=100, element='step', stat='density', common_norm=False, alpha=0.25, color='blue')
    plt.title(f'{community} Communities', fontsize=25)
    plt.xlabel(f'Fraction of robust nodes', fontsize=25)
    plt.ylabel('Networks', fontsize=25)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=16)

plt.tight_layout()
# plt.show()
# Save the figure as an SVG file
output_path = 'robustness_hist.pdf'
plt.savefig(output_path, format='pdf')