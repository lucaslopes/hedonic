#!/bin/bash

# suffix=0
main_session="HedonicSession"  # "Session_$suffix"
commands_file="/Users/lucas/Databases/Hedonic/PHYSA/Synthetic_Networks/V2040/commands.txt"
max_concurrent_windows=4

# Ensure conda is available
source $(conda info --base)/etc/profile.d/conda.sh

# Create the tmux session if it doesn’t exist
tmux has-session -t $main_session 2>/dev/null
if [ $? != 0 ]; then
  echo "Creating new tmux session: $main_session"
  tmux new-session -d -s $main_session
else
  echo "Session $main_session already exists."
fi

# Execute commands from file
echo "Executing commands..."
total_commands=$(wc -l < "$commands_file")
i=0

while IFS= read -r command; do
  if tmux has-session -t $main_session 2>/dev/null; then
    active_windows=$(tmux list-windows -t $main_session | wc -l)
    if [ $active_windows -le $max_concurrent_windows ]; then
      time_now=$(date +"%Y-%m-%d %H:%M:%S")
      percentage=$(echo "scale=2; ($i/$total_commands)*100" | bc -l)
      printf "%s\n" "Running command #$((i+1)) out of $total_commands ($percentage%) at $time_now" "$command"
      printf "\n"
      eval "$command"
      ((i++))
    else
      sleep 2
    fi
  else
    echo "Session $main_session has been terminated unexpectedly. Exiting..."
    break
  fi
done < "$commands_file"

# Wait for all windows to complete
while [ $(tmux list-windows -t $main_session | wc -l) -gt 1 ]; do
  active_windows=$(tmux list-windows -t $main_session | wc -l)
  printf "Waiting for %d windows to complete...\n" $((active_windows - 1))
  sleep 10
done

# Kill the session after completion
if tmux has-session -t $main_session 2>/dev/null; then
  time_now=$(date +"%Y-%m-%d %H:%M:%S")
  printf "All work is done now %s. Killing session %s\n" "$time_now" "$main_session"
  tmux kill-session -t $main_session
fi
