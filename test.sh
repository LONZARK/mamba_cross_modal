weep_id="$1" # as an input argument
 
# Loop through the tasks
for i in {1..18}; do # open sessions in parallel
  # Define a session name for each task
  session_name="Ses_$i"
  mod_res=$(($i % 10))
  prefix="export CUDA_DEVICE_ORDER=PCI_BUS_ID; conda activate env_torch"
  task="CUDA_VISIBLE_DEVICES=$mod_res wandb agent kangshuoli/Fairness_Aware_ENN_OOD_v3/$sweep_id"
 
  if [ "$mod_res" -eq 0 ] || [ "$mod_res" -eq 6 ]; then
    echo "Skipping index $i."
    continue # skip
  else
    # Create a new tmux session, detached (-d), and start the task
    tmux new-session -d -s "$session_name" "$prefix; $task"
  
    # Optionally, you can also split windows, create panes, and run different commands in the same session
    # For example, to split the window in the current session and run another command:
    # tmux split-window -h -t "$session_name" "another_command"
  
    echo "Started task $i in tmux session $session_name, using GPU $mod_res"
  fi
done
 
echo "All tasks started in separate tmux sessions."
 
# chmod +x tmux_pipeline.sh