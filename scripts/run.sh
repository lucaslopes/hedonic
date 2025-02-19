#!/bin/bash

main_session="HedonicSession"
commands_file="/Users/lucas/Databases/Hedonic/PHYSA/Synthetic_Networks/V1020/commands.txt"
max_concurrent_windows=4

# Ensure conda is available
source "$(conda info --base)/etc/profile.d/conda.sh"

# Create the tmux session if it doesn’t exist
tmux has-session -t "$main_session" 2>/dev/null
if [ $? != 0 ]; then
  echo "Creating new tmux session: $main_session"
  tmux new-session -d -s "$main_session"
else
  echo "Session $main_session already exists."
fi

# Load commands into an array using a while loop instead of mapfile
commands=()
while IFS= read -r line; do
  commands+=("$line")
done < "$commands_file"

total_commands=${#commands[@]}
i=0

echo "Executing commands..."

# Process commands array until empty
while [ ${#commands[@]} -gt 0 ]; do
  # Check if the tmux session is still active
  if ! tmux has-session -t "$main_session" 2>/dev/null; then
    echo "Session $main_session has been terminated unexpectedly. Exiting..."
    break
  fi

  # Get the current number of active windows
  active_windows=$(tmux list-windows -t "$main_session" 2>/dev/null | wc -l)
  
  # If there is room for another window, execute the next command
  if [ "$active_windows" -lt "$max_concurrent_windows" ]; then
    command="${commands[0]}"
    commands=("${commands[@]:1}")  # Remove the first element
    time_now=$(date +"%Y-%m-%d %H:%M:%S")
    percentage=$(echo "scale=2; ($i/$total_commands)*100" | bc -l)
    echo ""
    echo "[$time_now] Running command #$((i+1)) out of $total_commands ($percentage%): $command"
    eval "$command"
    ((i++))
  else
    # No free slot yet, wait a bit
    sleep 2
  fi
done

# Wait for all windows (other than the initial window) to complete
while [ $(tmux list-windows -t "$main_session" 2>/dev/null | wc -l) -gt 1 ]; do
  active_windows=$(tmux list-windows -t "$main_session" 2>/dev/null | wc -l)
  echo "Waiting for $((active_windows - 1)) window(s) to complete..."
  sleep 10
done

# Kill the session after completion
if tmux has-session -t "$main_session" 2>/dev/null; then
  time_now=$(date +"%Y-%m-%d %H:%M:%S")
  echo "[$time_now] All work is done. Killing session $main_session."
  tmux kill-session -t "$main_session"
fi
