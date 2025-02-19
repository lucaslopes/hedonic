import os
import re
import argparse
from tqdm import tqdm
from utils import get_all_subpaths


def extract_sorting_keys(filename):
    """Extract sorting keys from the file name"""
    match = re.search(r'network_(\d+)\.pkl', filename)
    network_index = int(match.group(1)) if match else float('inf')
    
    difficulty_match = re.search(r'Difficulty = (\d+\.\d+)', filename)
    difficulty = float(difficulty_match.group(1)) if difficulty_match else float('inf')
    
    p_in_match = re.search(r'P_in = (\d+\.\d+)', filename)
    p_in = float(p_in_match.group(1)) if p_in_match else float('inf')
    
    n_communities_match = re.search(r'(\d+)C_', filename)
    n_communities = int(n_communities_match.group(1)) if n_communities_match else float('inf')
    
    return (network_index, n_communities, p_in, difficulty)


def sort_files(file_list):
    """Sort the list of files based on the extracted keys"""
    return sorted(file_list, key=extract_sorting_keys)


def generate_command(file, main_session, python_script, env_activation):
    """Generate a valid window name (last 4 subdirectories) and create the correct command format"""
    window_name = "/".join(file[:-4].split("/")[-4:])
    window_name = f'"{window_name}"'  # Ensure it's quoted properly

    command = (
        f"tmux new-window -t {main_session} -n {window_name} "
        f"\"{env_activation} && "
        f"exec python {python_script} \\\"{file}\\\"'\";\n"
    )

    return command


def write_commands(sorted_commands, main_session, python_script, env_activation, commands_file, filter_completed=False):
    community_count = 0
    with open(commands_file, "w") as f:
        for file in tqdm(sorted_commands, desc="Generating commands"):
            command = generate_command(file, main_session, python_script, env_activation)
            if not filter_completed or not os.path.exists(file.replace('.pkl', '.completed')):
                f.write(command)
                community_count += 1
    return community_count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter-completed", action="store_true", help="Do not generate commands for already completed communities")
    args = parser.parse_args()

    # Configuration
    suffix = 0
    main_session = f"HedonicSession_{suffix}"
    python_script = "scripts/exp.py"
    networks_dir = "/Users/lucas/Databases/Hedonic/PHYSA/Synthetic_Networks/V1020/networks"
    env_activation = "source $(conda info --base)/etc/profile.d/conda.sh && conda activate hedonic"

    # Find all .pkl files
    pkl_files = get_all_subpaths(networks_dir, endswith=".pkl")

    # Sort the list of files
    sorted_commands = sort_files(pkl_files)

    # Write commands to file
    commands_file = networks_dir.replace("/networks", "/commands.txt")
    community_count = write_commands(sorted_commands, main_session, python_script, env_activation, commands_file, filter_completed=args.filter_completed)


    print(f"Generated {community_count} commands in {commands_file}")


if __name__ == "__main__":
    main()
