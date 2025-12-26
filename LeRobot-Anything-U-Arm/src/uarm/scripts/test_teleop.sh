#!/bin/bash

# === Setup ROS environment ===
source /opt/ros/noetic/setup.bash

# Directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source "$SCRIPT_DIR/../devel/setup.bash"

echo "[INFO] ROS environment loaded, starting nodes..."


# === Start servo_reader.py ===
rosrun UArm servo_reader.py &
PID1=$!
echo "[INFO] servo_reader.py started with PID $PID1"

# === Start teleoperation controller ===
rosrun UArm servo2arm.py &
PID2=$!
echo "[INFO] servo2arm.py started with PID $PID2"





# === Setup cleanup logic for Ctrl+C ===
trap "echo '[INFO] Ctrl+C received. Shutting down all nodes...'; kill $PID1 $PID2; exit" SIGINT

# === Wait for all child processes to finish ===
wait
