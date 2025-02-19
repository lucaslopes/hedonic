import os
import re
from tqdm import tqdm
from utils import get_all_subpaths

# Function to extract sorting keys from the file name
def extract_sorting_keys(filename):
    match = re.search(r'network_(\d+)\.pkl', filename)
    network_index = int(match.group(1)) if match else float('inf')
    
    difficulty_match = re.search(r'Difficulty = (\d+\.\d+)', filename)
    difficulty = float(difficulty_match.group(1)) if difficulty_match else float('inf')
    
    p_in_match = re.search(r'P_in = (\d+\.\d+)', filename)
    p_in = float(p_in_match.group(1)) if p_in_match else float('inf')
    
    n_communities_match = re.search(r'(\d+)C_', filename)
    n_communities = int(n_communities_match.group(1)) if n_communities_match else float('inf')
    
    return (network_index, n_communities, p_in, difficulty)

# Sort the list of files based on the extracted keys
def sort_files(file_list):
    return sorted(file_list, key=extract_sorting_keys)

# Configuration
# suffix = 0
main_session = 'HedonicSession'  # f"Session_{suffix}"
python_script = "scripts/exp.py"
networks_dir = "/Users/lucas/Databases/Hedonic/PHYSA/Synthetic_Networks/V1020/networks"
env_activation = "source $(conda info --base)/etc/profile.d/conda.sh && conda activate hedonic"

# Find all .pkl files
pkl_files = get_all_subpaths(networks_dir, endswith=".pkl")

# Sort the list of files
sorted_commands = sort_files(pkl_files)

# Write commands to file
community_count = 0
commands_file = networks_dir.replace("/networks", "/commands.txt")
with open(commands_file, "w") as f:
    for file in tqdm(sorted_commands[:1000], desc="Generating commands"):
        # Generate a valid window name (last 4 subdirectories)
        window_name = "/".join(file[:-4].split("/")[-4:])
        window_name = f'"{window_name}"'  # Ensure it's quoted properly

        # Create the correct command format
        command = (
            f"tmux new-window -t {main_session} -n {window_name} "
            f"\"bash -i -c 'source $(conda info --base)/etc/profile.d/conda.sh && conda activate hedonic && "
            f"exec python {python_script} \\\"{file}\\\"'\";\n"
        )

        # if not os.path.exists(file.replace('.pkl', '.completed'))
        if ('network_000.pkl' in file):
            f.write(command)
            community_count += 1

print(f"Generated {community_count} commands in {commands_file}")
