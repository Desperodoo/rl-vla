# 🔧 System Setup

## Prerequisites

- **Ubuntu 20.04**
- [**ROS Noetic**](https://wiki.ros.org/noetic/Installation/Ubuntu)
- **Python 3.9+**

---

## Step-by-Step Setup

1. **Install Python Dependencies**

   ```sh
   # install both ros1 and simulation requirements
   pip install -r overall_requirements.txt 
   ```
   
   If your system doesn't support ROS1, you can install the dependencies without ROS1 with the following command which supports simulation teleoperation and check [this note](https://github.com/MINT-SJTU/Lerobot-Anything-U-arm/blob/main/src/simulation/README.md) . 
   ```sh
   pip install -r requirements.txt
   ```

2. **Build Catkin Workspace**

   **Method 1: Using the build script (Recommended)**
   ```sh
   ./build.sh
   ```
   
   **Method 2: Manual build**
   ```sh
   # Configure environment
   source /opt/ros/noetic/setup.bash
   conda activate arx-py310  # if using conda
   export CMAKE_PREFIX_PATH=/opt/ros/noetic:$CMAKE_PREFIX_PATH
   
   # Build workspace
   catkin_make -DCMAKE_POLICY_VERSION_MINIMUM=3.5
   ```
   
   Then source the workspace:
   ```sh
   source devel/setup.bash
   ```

3. **Verify Installation**

   ```sh
   # Test if ROS can find the package
   rospack find uarm
   ```

---

## Environment Setup

For convenience, you can use the provided setup script to configure your environment:

```sh
source setup_env.sh
```

This script will:
- Load ROS Noetic environment
- Activate conda environment (if available)
- Fix CMAKE_PREFIX_PATH for proper compilation
- Source workspace devel environment (if built)

**Note**: If you encounter ROS environment variable conflicts after activating conda, ensure that your conda environment doesn't have ROS2 packages installed. The conda environment should only contain Python dependencies, while ROS1 Noetic should be installed system-wide via apt.

---

# 🤖 Plug-and-Play with Real Robot with ROS1

## 0. Setup Serial Port Permissions (First Time Only)

Before using the real robot, you need to grant your user access to the serial port:

```sh
./add_serial_permissions.sh
```

After running this script, **log out and log back in** for the changes to take effect.

Alternatively, for the current terminal session only:
```sh
newgrp dialout
```

Verify the permission:
```sh
groups  # Should include 'dialout'
```

---

## 1. Start ROS Core

Open a terminal and run:

```sh
roscore
```

## 2. Verify Teleop Arm Output

In a new terminal, check servo readings:

```sh
rosrun uarm servo_zero.py
```

This will display real-time angles from all servos. You should check whether `SERIAL_PORT` is available on your device and modify the variable if necessary. 

## 3. Publish Teleop Data

Still in the second terminal, start the teleop publisher:

```sh
rosrun uarm servo_reader.py
```

Your teleop arm now publishes to the `/servo_angles` topic.

## 4. Control the Follower Arm

Choose your robot and run the corresponding script:

- **For Dobot CR5:**
  ```sh
  rosrun uarm scripts/Follower_Arm/Dobot/servo2Dobot.py
  ```

- **For xArm:**
  ```sh
  rosrun uarm scripts/Follower_Arm/xarm/servo2xarm.py
  ```

- **For ARX5 (X5/L5):**
  
  First, test the robot connection:
  ```sh
  rosrun uarm scripts/Follower_Arm/ARX/test_joint_control.py X5 can0
  ```
  
  Then run teleoperation:
  ```sh
  rosrun uarm scripts/Follower_Arm/ARX/arx_teleop.py
  ```
  
  **Note**: Make sure you have:
  - Installed the ARX5 SDK from [arx5-sdk](https://github.com/real-stanford/arx5-sdk)
  - Configured CAN bus interface (e.g., `can0`)
  - Updated the `ROOT_DIR` path in the script to point to your arx5-sdk installation

---

# 🖥️ Try It Out in Simulation

If you do not have robot hardware, you can try teleoperation in simulation.  
See detailed guidance [here](https://github.com/MINT-SJTU/Lerobot-Anything-U-arm/blob/feat/simulation/src/simulation/README.md).
