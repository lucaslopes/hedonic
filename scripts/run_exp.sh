#!/bin/bash

# List to store commands
declare -a commands
declare -i max_n_nodes

max_concurrent_windows=10


n_comm=2
max_n_nodes=1000
# suffix="Nodes_$max_n_nodes-Communities_$n_comm"
suffix="Nodes_$max_n_nodes"
env_name="PHYSA_$max_n_nodes"
python_script="scripts/experiment.py"
probabilities=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10)
difficulties=(0.1 0.2 0.3 0.4 0.5 0.55 0.6 0.65 0.7 0.75)
seeds=(0 1 2 3 4 5 6 7 8 9) # unique networks
n_partitions=10 # different initial partitions
n_communities=($n_comm)
noises=(0.01 0.25 0.5 0.6 0.7 0.75 0.8 0.85 0.9 0.95 1.0 1.1)



# n_comm=2
# max_n_nodes=100
# # suffix="Nodes_$max_n_nodes-Communities_$n_comm"
# suffix="Nodes_$max_n_nodes"
# env_name="PHYSA_$max_n_nodes"
# python_script="scripts/experiment.py"
# probabilities=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10)
# difficulties=(0.1 0.2 0.3 0.4 0.5 0.55 0.6 0.65 0.7 0.75)
# seeds=(0) # unique networks
# n_partitions=1 # different initial partitions
# n_communities=($n_comm)
# noises=(0.7)

main_session="Session_$suffix"
# log_dir="logs"
# mkdir -p $log_dir # Ensure the log directory exists

# Calculate total number of commands
total_commands=$(( ${#seeds[@]} * ${#n_communities[@]} * ${#probabilities[@]} * ${#difficulties[@]} * ${#noises[@]} * n_partitions ))

# Attempt to create or confirm the main tmux session
tmux has-session -t $main_session 2>/dev/null
if [ $? != 0 ]; then
    echo "Creating new tmux session: $main_session"
    tmux new-session -d -s $main_session
else
    echo "Session $main_session already exists."
fi

# Generate all commands
echo "Generating commands..."
command_count=0
for seed in "${seeds[@]}"; do
  for n_community in "${n_communities[@]}"; do
    for p_in in "${probabilities[@]}"; do
      for difficulty in "${difficulties[@]}"; do
        for noise in "${noises[@]}"; do
          window_name="Communities_${n_community}-Seed_${seed}-p_in_${p_in}-difficulty_${difficulty}-noise_${noise}"
          # log_file="${log_dir}/${window_name}.log"
          python_command="python $python_script --folder_name $env_name --max_n_nodes $max_n_nodes --n_communities $n_community --seed $seed --p_in $p_in --difficulty $difficulty --noises $noise --n_partitions $n_partitions"
          # command="tmux new-window -t $main_session -n ${window_name} \"$python_command 2>&1 | tee $log_file;\""
          command="tmux new-window -t $main_session -n ${window_name} \"$python_command;\""
          commands+=("$command")
          command_count=$((command_count + 1))
          percentage=$(echo "scale=2; ($command_count/$total_commands)*100" | bc -l)
          echo "Command added ($percentage%): $command "
        done
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
      printf "%s\n" "Running command #$i out of $total_commands at $time_now" "${commands[$i]}"
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
while [ $(tmux lsw -t $main_session | wc -l) -gt 1 ]; do
  active_windows=$(tmux lsw -t $main_session | wc -l) # Count active windows in the main session
  printf "Waiting for %d windows to complete...\n" $((active_windows - 1))
  sleep 10
done

# Optional: Kill the main session after all work is done
if tmux has-session -t $main_session 2>/dev/null; then
  time_now=$(date +"%Y-%m-%d %H:%M:%S")
  printf "All work is done now $time_now. Killing session %s\n" $main_session
  tmux kill-session -t $main_session
fi

# ./scripts/run_exp.sh 2>&1 | tee exp.log