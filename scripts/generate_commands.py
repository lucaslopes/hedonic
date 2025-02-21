import os
import argparse
from tqdm import tqdm
from utils import get_all_subpaths, sort_files

def generate_command(file, main_session, python_script, env_activation):
    """Generate a valid window name (last 4 subdirectories) and create the correct command format"""
    window_name = "/".join(file[:-4].split("/")[-4:])
    window_name = f'"{window_name}"'  # Ensure it's quoted properly

    command = (
        f"tmux new-window -t {main_session} -n {window_name} "
        f"\"bash -i -c '{env_activation} && exec python {python_script} \\\"{file}\\\"'\";\n"
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
    # suffix = 0
    main_session = "HedonicSession"  # f"HedonicSession_{suffix}"
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
