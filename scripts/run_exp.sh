#!/bin/bash

# List to store commands
declare -a commands
declare -i max_n_nodes
env_name="ExperimentResults"
python_script="scripts/experiment.py"
max_n_nodes=1020
seeds=(0 1 2 3 4 5 6 7 8 9)
n_communities=(2 3 4 5 6)
probabilities=(0.10 0.09 0.08 0.07 0.06 0.05 0.04 0.03 0.02 0.01)
difficulties=(0.75 0.7 0.65 0.6 0.55 0.5 0.4 0.3 0.2 0.1)
max_concurrent_windows=50
main_session="ExperimentSession"

# Create the main tmux session
tmux new-session -d -s $main_session

# Generate all commands
for n_community in "${n_communities[@]}"; do
  for seed in "${seeds[@]}"; do
    for p_in in "${probabilities[@]}"; do
      for difficulty in "${difficulties[@]}"; do
        session_name="Communities_${n_community}-Seed_${seed}-p_in_${p_in}-difficulty_${difficulty}"
        python_command="python $python_script --folder_name $env_name --max_n_nodes $max_n_nodes --n_communities $n_community --seed $seed --p_in $p_in --difficulty $difficulty"
        command="tmux new-window -t $main_session -n ${session_name} \"$python_command; tmux kill-window\""
        commands+=("$command")
      done
    done
  done
done

# Execute commands within the main session
i=0
while [ $i -lt ${#commands[@]} ]; do
  active_windows=$(tmux lsw -t $main_session | wc -l) # Count active windows in the main session
  if [ $active_windows -le $max_concurrent_windows ]; then
    eval "${commands[$i]}"
    ((i++))
  else
    sleep 1 # Wait before checking again
  fi
done

# Wait for all windows to complete
while [ $(tmux lsw -t $main_session | wc -l) -gt 1 ]; do
  sleep 10
done

# Optional: Kill the main session after all work is done
tmux kill-session -t $main_session
