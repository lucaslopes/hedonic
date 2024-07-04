#!/bin/bash

# List to store commands
declare -a commands
declare -i max_n_nodes
n_comm=6
max_n_nodes=600
suffix="Nodes_$max_n_nodes-Communities_$n_comm"
env_name="Results_$max_n_nodes"
python_script="scripts/experiment.py"
probabilities=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10)
difficulties=(0.1 0.2 0.3 0.4 0.5 0.55 0.6 0.65 0.7 0.75)
seeds=(0 1 2 3 4 5 6 7 8 9)
n_communities=($n_comm)
# seeds=(0 1 2 3 4 5 6 7 8 9)
# n_communities=(2 3 4 5 6)

max_concurrent_windows=4
main_session="Session_$suffix"
# log_dir="logs"
# mkdir -p $log_dir # Ensure the log directory exists

# Attempt to create or confirm the main tmux session
tmux has-session -t $main_session 2>/dev/null
if [ $? != 0 ]; then
    echo "Creating new tmux session: $main_session"
    tmux new-session -d -s $main_session
else
    echo "Session $main_session already exists."
fi

# Generate all commands
for n_community in "${n_communities[@]}"; do
  for seed in "${seeds[@]}"; do
    for p_in in "${probabilities[@]}"; do
      for difficulty in "${difficulties[@]}"; do
        window_name="Communities_${n_community}-Seed_${seed}-p_in_${p_in}-difficulty_${difficulty}"
        # log_file="${log_dir}/${window_name}.log"
        python_command="python $python_script --folder_name $env_name --max_n_nodes $max_n_nodes --n_communities $n_community --seed $seed --p_in $p_in --difficulty $difficulty"
        # command="tmux new-window -t $main_session -n ${window_name} \"$python_command 2>&1 | tee $log_file; tmux kill-window\""
        command="tmux new-window -t $main_session -n ${window_name} \"$python_command; tmux kill-window\""
        commands+=("$command")
      done
    done
  done
done

# Execute commands within the main session
i=0
while [ $i -lt ${#commands[@]} ]; do
  if tmux has-session -t $main_session 2>/dev/null; then
    active_windows=$(tmux lsw -t $main_session | wc -l) # Count active windows in the main session
    if [ $active_windows -le $max_concurrent_windows ]; then
      time_now=$(date +"%Y-%m-%d %H:%M:%S")
      printf "%s\n" "Running command #$i $time_now" "${commands[$i]}"
      eval "${commands[$i]}"
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
while tmux has-session -t $main_session 2>/dev/null && [ $(tmux lsw -t $main_session | wc -l) -gt 1 ]; do
  sleep 10
done

# Optional: Kill the main session after all work is done
if tmux has-session -t $main_session 2>/dev/null; then
  time_now=$(date +"%Y-%m-%d %H:%M:%S")
  printf "All work is done now $time_now. Killing session %s\n" $main_session
  tmux kill-session -t $main_session
fi

# ./scripts/run_exp.sh 2>&1 | tee exp.log