"""
Consistency Policy 真机推理框架

基于 UMI-ARX 框架设计，针对关节空间控制优化。

模块结构:
├── config.py                    # 路径配置
├── policy_inference.py          # 策略推理节点 (ZMQ 服务端)
├── robot_controller_mp.py       # 机械臂控制器 (多进程，200Hz)
├── realsense_camera.py          # RealSense 相机模块 (多进程)
├── joint_trajectory_interpolator.py  # 关节空间轨迹插值器
├── eval_real_mp.py              # 主评估脚本 (多进程)
└── tests/                       # 测试脚本
    ├── test_offline_inference.py  # 离线推理测试
    ├── test_robot_replay.py       # 轨迹回放测试
    ├── test_camera.py             # RealSense 相机测试
    ├── test_controller.py         # 多进程控制器测试
    └── README.md                  # 测试说明文档

架构说明:
- 控制器在独立进程中以 200Hz 运行，使用 JointTrajectoryInterpolator 插值
- 相机在独立进程中以 30Hz 采集，通过 SharedMemoryRingBuffer 传递帧
- 策略推理通过 ZMQ 通信，可在同一机器或不同机器运行
- 使用 umi-arx 的 shared_memory 模块实现进程间无锁通信

使用示例:
    # 1. 启动策略推理服务
    python -m consistency_policy.policy_inference
    
    # 2. 运行评估 (多进程版本)
    python -m consistency_policy.eval_real_mp -o ./output
    
    # 3. 运行测试
    python -m consistency_policy.tests.test_camera --detect
    python -m consistency_policy.tests.test_controller --state-only
    python -m consistency_policy.tests.test_robot_replay
"""

__version__ = "0.2.1"
