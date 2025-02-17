#!/bin/bash

# List to store commands
declare -a commands
declare -i max_n_nodes

max_concurrent_windows=100
suffix="Nodes_$max_n_nodes"
env_name="PHYSA_$max_n_nodes"
python_script="scripts/exp.py"
networks_dir="/Users/lucas/Databases/Hedonic/PHYSA/Synthetic_Networks/V2040/networks"

pkl_files=()
while IFS= read -r file; do
    pkl_files+=("$file")
done < <(find "$networks_dir" -type f -name '*.pkl')

main_session="Session_$suffix"
# log_dir="logs"
# mkdir -p $log_dir # Ensure the log directory exists

total_commands=${#pkl_files[@]}

# Ensure conda is available
source $(conda info --base)/etc/profile.d/conda.sh

# Attempt to create or confirm the main tmux session
tmux has-session -t $main_session 2>/dev/null
if [ $? != 0 ]; then
    echo "Creating new tmux session: $main_session"
    tmux new-session -d -s $main_session
    echo "Session $main_session created."
else
    echo "Session $main_session already exists."
fi

# Generate all commands
echo "Generating commands..."
command_count=0
for file in "${pkl_files[@]}"; do
  window_name="Network_${file}"
  python_command="source $(conda info --base)/etc/profile.d/conda.sh && conda activate hedonic && python $python_script $file"
  command="tmux new-window -t $main_session -n ${window_name} \"$python_command;\""
  commands+=("$command")
  command_count=$((command_count + 1))
  percentage=$(echo "scale=2; ($command_count/$total_commands)*100" | bc -l)
  echo "Command added ($percentage%): $command "
done

# Execute commands within the main session
i=0
echo "Executing commands..."
while [ $i -lt ${#commands[@]} ]; do
  if tmux has-session -t $main_session 2>/dev/null; then
    active_windows=$(tmux list-windows -t $main_session | wc -l) # Count active windows in the main session
    if [ $active_windows -le $max_concurrent_windows ]; then
      time_now=$(date +"%Y-%m-%d %H:%M:%S")
      percentage=$(echo "scale=2; ($i/$total_commands)*100" | bc -l)
      printf "%s\n" "Running command #$((i+1)) out of $total_commands ($percentage%) at $time_now" "${commands[$i]}"
      printf "\n"
      tmux send-keys -t $main_session "${commands[$i]}" C-m
      ((i++))
    else
      sleep 2 # Wait before checking again
    fi
  else
    echo "Session $main_session has been terminated unexpectedly. Exiting..."
    break
  fi
done

# Wait for all windows to complete
while [ $(tmux list-windows -t $main_session | wc -l) -gt 1 ]; do
  active_windows=$(tmux list-windows -t $main_session | wc -l) # Count active windows in the main session
  printf "Waiting for %d windows to complete...\n" $((active_windows - 1))
  sleep 10
done

# Optional: Kill the main session after all work is done
if tmux has-session -t $main_session 2>/dev/null; then
  time_now=$(date +"%Y-%m-%d %H:%M:%S")
  printf "All work is done now $time_now. Killing session %s\n" $main_session
  tmux kill-session -t $main_session
fi

# ./scripts/run.sh 2>&1 | tee exp.log
