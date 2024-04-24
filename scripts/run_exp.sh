#!/bin/bash

# List to store commands
declare -a commands
declare -i max_n_nodes
env_name="MainResult"
python_script="scripts/experiment.py"
max_n_nodes=60
seeds=(0 1 2 3 4 5 6 7 8 9)
n_communities=(2 3 4 5 6)
probabilities=(0.10 0.09 0.08 0.07 0.06 0.05 0.04 0.03 0.02 0.01)
difficulties=(0.75 0.7 0.65 0.6 0.55 0.5 0.4 0.3 0.2 0.1)

max_concurrent_windows=1
main_session="ExperimentSession"
log_dir="logs"
mkdir -p $log_dir # Ensure the log directory exists

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
        session_name="Communities_${n_community}-Seed_${seed}-p_in_${p_in}-difficulty_${difficulty}"
        log_file="${log_dir}/${session_name}.log"
        python_command="python $python_script --folder_name $env_name --max_n_nodes $max_n_nodes --n_communities $n_community --seed $seed --p_in $p_in --difficulty $difficulty"
        command="tmux new-window -t $main_session -n ${session_name} \"$python_command 2>&1 | tee $log_file; tmux kill-window\""
        commands+=("$command")
      done
    done
  done
done

# Execute commands within the main session
i=0
while [ $i -lt ${#commands[@]} ]; do
  if tmux has-session -t $main_session 2>/dev/null; then
    active_windows=$(tmux lsw -t $main_session | grep -c "windows")
    while [ $active_windows -ge $max_concurrent_windows ]; do
      sleep 1
      active_windows=$(tmux lsw -t $main_session | grep -c "windows")
    done
    eval "${commands[$i]}"
    ((i++))
  else
    echo "Session $main_session has been terminated unexpectedly. Exiting..."
    break
  fi
done

# Wait for all windows to complete
while tmux has-session -t $main_session 2>/dev/null && [ $(tmux lsw -t $main_session | grep -c "windows") -gt 1 ]; do
  sleep 10
done

# Optional: Kill the main session after all work is done
if tmux has-session -t $main_session 2>/dev/null; then
  tmux kill-session -t $main_session
fi


# #!/bin/bash

# # List to store commands
# declare -a commands
# declare -i max_n_nodes
# env_name="MainResult"
# python_script="scripts/experiment.py"
# max_n_nodes=1020
# seeds=(0 1 2 3 4 5 6 7 8 9)
# n_communities=(2 3 4 5 6)
# probabilities=(0.10 0.09 0.08 0.07 0.06 0.05 0.04 0.03 0.02 0.01)
# difficulties=(0.75 0.7 0.65 0.6 0.55 0.5 0.4 0.3 0.2 0.1)

# max_concurrent_windows=2
# main_session="ExperimentSession"
# # log_dir="logs"
# # mkdir -p $log_dir # Ensure the log directory exists

# # Create the main tmux session
# tmux new-session -d -s $main_session

# # Generate all commands
# for n_community in "${n_communities[@]}"; do
#   for seed in "${seeds[@]}"; do
#     for p_in in "${probabilities[@]}"; do
#       for difficulty in "${difficulties[@]}"; do
#         window_name="Communities_${n_community}-Seed_${seed}-p_in_${p_in}-difficulty_${difficulty}"
#         # log_file="${log_dir}/${window_name}.log"
#         python_command="python $python_script --folder_name $env_name --max_n_nodes $max_n_nodes --n_communities $n_community --seed $seed --p_in $p_in --difficulty $difficulty"
#         # command="tmux new-window -t $main_session -n ${window_name} \"$python_command 2>&1 | tee $log_file; tmux kill-window\""
#         command="tmux new-window -t $main_session -n ${window_name} \"$python_command; exit\""
#         commands+=("$command")
#       done
#     done
#   done
# done

# # Execute commands within the main session with controlled concurrency
# # i=0
# # while [ $i -lt ${#commands[@]} ]; do
# #   active_windows=$(tmux lsw -t $main_session | wc -l) # Count active windows in the main session
# #   while [ $active_windows -ge $max_concurrent_windows ]; do
# #     sleep 1
# #     active_windows=$(tmux lsw -t $main_session | wc -l) # Count active windows in the main session
# #   done
# #   eval "${commands[$i]}"
# #   ((i++))
# # done


# # Execute commands within the main session
# i=0
# while [ $i -lt ${#commands[@]} ]; do
#   active_windows=$(tmux lsw -t $main_session | wc -l) # Count active windows in the main session
#   printf "%s\n" "$active_windows"
#   if [ $active_windows -le $max_concurrent_windows ]; then
#     eval "${commands[$i]}"
#     ((i++))
#     printf "%s\n" "$i" "${commands[$i]}"
#   else
#     sleep 2 # Wait before checking again
#   fi
# done

# # Wait for all windows to complete
# while [ $(tmux lsw -t $main_session | wc -l) -gt 1 ]; do
#   sleep 10
# done

# # Optional: Kill the main session after all work is done
# tmux kill-session -t $main_session
