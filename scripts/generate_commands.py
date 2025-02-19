import os
from utils import get_all_subpaths

# Configuration
# suffix = 0
main_session = 'HedonicSession'  # f"Session_{suffix}"
python_script = "scripts/exp.py"
networks_dir = "/Users/lucas/Databases/Hedonic/PHYSA/Synthetic_Networks/V1020/networks"
env_activation = "source $(conda info --base)/etc/profile.d/conda.sh && conda activate hedonic"

# Find all .pkl files
pkl_files = get_all_subpaths(networks_dir, endswith=".pkl")

# Sort the list of files
pkl_files.sort()

# Write commands to file
commands_file = networks_dir.replace("/networks", "/commands.txt")
with open(commands_file, "w") as f:
    for file in pkl_files:
        # Generate a valid window name (last 4 subdirectories)
        window_name = "/".join(file[:-4].split("/")[-4:])
        window_name = f'"{window_name}"'  # Ensure it's quoted properly

        # Create the correct command format
        command = (
            f"tmux new-window -t {main_session} -n {window_name} "
            f"\"bash -i -c 'source $(conda info --base)/etc/profile.d/conda.sh && conda activate hedonic && "
            f"exec python {python_script} \\\"{file}\\\"'\";\n"
        )

        if not os.path.exists(file.replace('.pkl', '.completed')):
            f.write(command)

print(f"Generated {len(pkl_files)} commands in {commands_file}")
