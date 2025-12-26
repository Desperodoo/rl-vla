# 🤖 Lerobot Anything

[![en](https://img.shields.io/badge/lang-en-blue.svg)](README.md)
[![ROS Noetic](https://img.shields.io/badge/ROS-Noetic-brightgreen.svg)](https://www.ros.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Ubuntu](https://img.shields.io/badge/Ubuntu-20.04-orange.svg)](https://ubuntu.com/)
[![Arxiv](https://img.shields.io/badge/arXiv-2509.02437-b31b1b.svg
)](https://arxiv.org/abs/2509.02437)
[![Apache License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<p align="center">
  <img src="pics/Xarm.gif" width="30%" alt="xArm Demo" />
  <img src="pics/Dobot.gif" width="30%" alt="Dobot Demo" />
  <img src="pics/Arx.gif" width="30%" alt="ARX Demo" />
</p>

---

> **🚀 Bringing Leader-Follower teleoperation system to every real robot and robot arm -- Cheaper, Smoother, Plug-and-Play**
> **💵 Starts from $60 cost!! Then controls any robot arm system!!**

*Built upon the giants: [LeRobot](https://github.com/huggingface/lerobot), [SO-100/SO-101](https://github.com/TheRobotStudio/SO-ARM100), [XLeRobot](https://github.com/Vector-Wangel/XLeRobot#), [Gello](https://github.com/wuphilipp/gello_mechanical/tree/main)*

# 📰 News

- 2025-09-10: STEP files of 3 configs are uploaded.
- 2025-09-08: Hardware Assembly Video Uploaded! We also open-sourced 5 datasets collected by UArm with XArm6 at [our huggingface page](https://huggingface.co/MINT-SJTU)
- 2025-08-15: **LeRobot Anything UArm 0.1.0** hardware setup, the 1st version fully capable for three major robot arm configurations, starts from 60$.

---

# 📋 Table of Contents

- [Overview](#-overview)
- [✨ Features](#-features)
- [💵 Total Cost](#-total-cost-)
- [🤖 Supported Robots (find your robot in the list!)](#-supported-robots)
- [🚀 Quick Start](#-quick-start)
- [🔮 Roadmap](#-roadmap)
- [🤝 Contributing](#-contributing)
- [🔧 Tired of DIY? Order Now](#-Order-Link)
---

## 🎯 Overview

LeRobot Anything is a **low-cost, universal, leader-follower teleoperation system** for any commercial robot arms through 3 hardware configurations. Designed for researchers, educators, and robotics enthusiasts, it provides a standardized interface for diverse robot platforms. This project focus on extending the Lerobot to control any real robot in both real scene and simulation. 

For detail, see our technical report [here](https://arxiv.org/abs/2509.02437).

### 🎯 Target Environment (Docker coming soon)

- **OS**: Ubuntu 20.04
- **ROS**: Noetic
- **Simulation**: SAPIEN integration (Built upon [ManiSkill](https://github.com/haosulab/ManiSkill))

---

## ✨ Features

| Feature                             | Description                                                                   |
| ----------------------------------- | ----------------------------------------------------------------------------- |
| 🔄**Universal Compatibility** | Four teleop configurations covering **most (95%) commercial robot arms** |
| 📡**ROS Integration**         | Native ROS1 support with `/servo_angles` topic publishing                   |
| 🎮**Real-time Control**       | Low-latency joint-space control                                        |
| 🔌**Plug & Play**             | Easy follower-arm integration with provided examples                          |
| 🛠️**Extensible**            | Simple API for adding new robot support                                       |
| 💰**Cost-effective**          | Ultra low-cost hardware solution                                              |
| 🎯**Optimized Hardware**      | Move smoothly and flexibly                                                    |
| 💻**Simulation Test**         | Support teleoperation test in simulation environment                                  |

### 🎮 Ready-to-Use Examples

**Real Robot Examples:**

<p align="center">

| Dobot CR5 | xArm Series | ARX5 |
|-----------|-------------|------|
| <img src="pics/Dobot.gif" width="200" alt="Dobot CR5 Real Robot" /> | <img src="pics/Xarm.gif" width="200" alt="xArm Series Real Robot" /> | <img src="pics/Arx.gif" width="200" alt="ARX5 Real Robot" /> |

</p>

**Simulation Examples:**

<p align="center">

| SO100 | ARX-X5 | XLeRobot |
|-------|--------|----------|
| <img src="pics/so100-sim.gif" width="200" alt="SO100 Simulation" /> | <img src="pics/arx-x5-sim.gif" width="200" alt="ARX-X5 Simulation" /> | <img src="pics/x_fetch-sim.gif" width="200" alt="XLeRobot Simulation" /> |

| xArm Series | Franka Panda | Piper |
|-------------|--------------|-------------|
| <img src="pics/xarm6_robotiq-sim.gif" width="200" alt="xArm Series Simulation" /> | <img src="pics/panda-sim.gif" width="200" alt="Franka Panda Simulation" /> | <img src="pics/piper-sim.gif" width="200" alt="Piper Simulation" /> |

</p> 

## 💵 Total Cost 💵

> [!NOTE]
> Cost excludes 3D printing, tools, shipping, and taxes.

| Price                             | US             | EU              | CN               |
| --------------------------------- | -------------- | --------------- | ---------------- |
| **Basic** (use your laptop) | **~$60** | **~€60** | **~¥360** |
| ↑ Servos                         | +$60           | +€60           | +¥405           |

---

## 🤖 Supported Robots (find your robot in the list!)

| Configuration                                                                                                         | Compatible Robot Arms                                                                      | Status   |
| --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ | -------- |
| [**Config 1**](https://github.com/MINT-SJTU/LeRobot-Anything-U-Arm/tree/main/mechanical/Config1_STL) | Xarm6, Fanuc LR Mate 200iD, Trossen ALOHA, Agile PiPER, Realman RM65B, KUKA LBR iiSY Cobot | ✅ Ready |
| [**Config 2**](https://github.com/MINT-SJTU/LeRobot-Anything-U-Arm/tree/main/mechanical/Config2_STL) | Dobot CR5, UR5, ARX R5*, AUBO i5, JAKA Zu7                                                 | ✅ Ready |
| [**Config 3**](https://github.com/MINT-SJTU/LeRobot-Anything-U-Arm/tree/main/mechanical/Config3_STL) | Franka FR3, Franka Emika Panda, Flexiv Rizon, Realman RM75B , Xarm7                               | ✅ Ready |

> 💡 **This is only part of our supported robots! For more robots, you can compare their mechanical configuration with each U-Arm to confirm which one fits your need (in most cases it will).

---

## 🚀 Quick Start

> [!NOTE]
> If you are totally new to programming, please spend at least a day to get yourself familiar with basic Python, Ubuntu and GitHub (with the help of Google and AI). At least you should know how to set up Ubuntu system, git clone, pip install, use interpreters (VS Code, Cursor, PyCharm, etc.) and directly run commands in the terminals.

1. 💵 **Buy your parts**: [Bill of Materials](https://docs.google.com/document/d/1TjhJOeJXsD5kmoYF-kuWfPju6WSUeSnivJiU7TH4vWs/edit?tab=t.0#heading=h.k991lzlarfb8)
2. 🖨️ **Print your stuff**: [3D printing](https://github.com/MINT-SJTU/Lerobot-Anything-U-arm/tree/main/mechanical)
3. 🔨 **Assemble**! [Assembly Guide](https://www.youtube.com/watch?v=-gpYuN2LlVs)
4. 💻 **Software Env Set up & Real-world Teleop**: [Get your robot moving!](https://github.com/MINT-SJTU/Lerobot-Anything-Uarm/blob/main/howtoplay.md)
5. 🎮 **Simulation**: [Try it out in SAPIEN!](https://github.com/MINT-SJTU/Lerobot-Anything-U-arm/blob/main/src/simulation/README.md)

For detailed hardware guide, check  [Hardware Guide](https://docs.google.com/document/d/1TjhJOeJXsD5kmoYF-kuWfPju6WSUeSnivJiU7TH4vWs/edit?tab=t.0#heading=h.k991lzlarfb8)



---

## 🔮 Roadmap

### 🎯 TO-DO List

- [X] **SAPIEN Simulation Environment**: Install and Play!

  - Virtual teleop setup mirroring physical hardware
  - Rapid prototyping and testing capabilities
  - Integration with existing SAPIEN workflows
- [ ] **ROS2 Support**
- [ ] **Docker Image**
- [ ] **Humanoid System: Config4**

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### 💡 Feature Requests

### 🔧 Code Contributions

### 🤖 Adding New Robot Support

---

## ❓ FAQ

### 1. Is this project compatible with my XXX robot arm?
**A:** Please first refer to the **Supported Robots** section above, where part of compatible arms are listed along with corresponding config.  Note that this project aims to provide a **universal teleoperation framework** and example control code. Here, “compatibility” means that the operator can intuitively control the corresponding follower arm on hardware side by using UArm.  
This is independent of different brands’ software APIs — it only depends on the **joint topology** of the robot.  To find your robot’s joint topology, check [Hardware Guide](https://docs.google.com/document/d/1TjhJOeJXsD5kmoYF-kuWfPju6WSUeSnivJiU7TH4vWs/edit?tab=t.0#heading=h.k991lzlarfb8).

---

### 2. How can I develop based on my own follower arm?
**A:** Start by reading [`LeRobot-Anything-U-Arm/src/uarm/scripts/Uarm_teleop/servo_reader`](https://github.com/MINT-SJTU/LeRobot-Anything-U-Arm/blob/main/src/uarm/scripts/Uarm_teleop/servo_reader.py).  
This script reads all UArm joint angles and is wrapped as a ROS node.  
You need to write a **subscriber** based on your follower arm’s API that receives joint commands from the `'/servo_angles'` topic and sends them to your arm.  [`Dobot Controller`](https://github.com/MINT-SJTU/LeRobot-Anything-U-Arm/blob/main/src/uarm/scripts/Follower_Arm/Dobot/servo2Dobot.py) is a simple example.

If you prefer not to use ROS communication, you can directly read the UArm’s servo command data and send them to your follower arm.  
See this [`ARX example`](https://github.com/MINT-SJTU/LeRobot-Anything-U-Arm/blob/main/src/uarm/scripts/Follower_Arm/ARX/arx_teleop.py) for reference.


## 🔧 Order Link
If you are tired of DIY, buy it on [JD 京东链接](https://item.jd.com/10170154240149.html) or contact business@evomind-tech.com for purchase.
Please note your Robotic Arm brand and model. 下单/咨询请注明使用的机械臂具体品牌型号。如有技术问题需要解答，请通过下方二维码进入社区群里咨询，京东上不便于回答讨论详细技术问题。


## 👥 Contributors

- **Yanwen Zou** - Hardware&Software System Design
- **Zhaoye Zhou** - Hareware Assemble and Adjustment
- **Chenyang Shi** - SAPIEN and website setup
- **Jiaqi Lu** - Real-world Experiment
- **Zewei Ye** - Real-world Experiment
- **Jie Yi** - Real-world Experiment
- **Junda Huang** - Idea Discussion and website setup
- **Gaotian Wang** - Idea Discussion

This project builds upon the excellent work of:

- [LeRobot](https://github.com/huggingface/lerobot) - The foundation for robot learning
- [SO-100/SO-101](https://github.com/TheRobotStudio/SO-ARM100) - Hardware inspiration
- [XLeRobot](https://github.com/Vector-Wangel/XLeRobot) - Extended robot support
- [Gello](https://github.com/wuphilipp/gello_mechanical/tree/main) - Hardware inspiration

Thanks to all the talented contributors behind these detailed and professional projects! You are also welcomed to join our WeChat Community for discussion and questions:
<div align="center">
<img src="pics/uarm_community.jpg" width="200" />
</div>
---

<div align="center">

**Made with ❤️ for the robotics community**

[![GitHub stars](https://img.shields.io/github/stars/MINT-SJTU/Lerobot-Anything-U-arm?style=social)](https://github.com/MINT-SJTU/LeRobot-Anything-U-Arm)
[![GitHub forks](https://img.shields.io/github/forks/MINT-SJTU/Lerobot-Anything-U-arm?style=social)](https://github.com/MINT-SJTU/LeRobot-Anything-U-Arm)

</div>
