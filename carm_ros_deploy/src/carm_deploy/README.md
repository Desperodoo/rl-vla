# CARM ROS 部署包

基于 ROS1 原生通信的 CARM 机械臂部署框架。

## 📁 目录结构

```
carm_deploy/
├── core/                       # 核心模块
│   ├── __init__.py
│   ├── env_ros.py             # 机械臂环境封装 (RealEnvironment)
│   └── safety_controller.py   # 安全控制器 (SafetyController)
│
├── inference/                  # 推理模块
│   ├── __init__.py
│   ├── inference_ros.py       # 策略推理主程序
│   └── inference_logger.py    # 推理日志记录器
│
├── data/                       # 数据模块
│   ├── __init__.py
│   ├── record_data_ros.py     # 数据记录程序
│   └── analyze_dataset.py     # 数据集分析脚本
│
├── tools/                      # 工具脚本
│   ├── __init__.py
│   ├── offline_test.py        # 离线测试脚本
│   ├── verify_safety_config.py # 安全配置验证
│   ├── record_workspace.py    # 工作空间记录
│   └── arm_test/              # 机械臂测试脚本
│       ├── test_connection.py # 测试连接
│       ├── test_motion.py     # 测试运动
│       ├── test_gripper.py    # 测试夹爪
│       └── safe_shutdown.py   # 安全关闭
│
├── utils/                      # 工具模块
│   ├── __init__.py
│   ├── image_sync.py          # 多相机图像同步
│   ├── trajectory_interpolator.py  # 轨迹插值工具
│   └── paths.py               # 路径配置
│
├── camera/                     # 相机工具
│   ├── test_realsense.py      # 相机测试 (pyrealsense2)
│   └── ros_camera_subscriber.py  # ROS 相机订阅
│
├── config/                     # 配置文件
│   └── default.yaml           # 默认配置
│
├── launch/                     # ROS Launch 文件
│   ├── camera.launch          # 相机启动（统一入口）
│   ├── realsense_d405.launch  # 兼容旧入口（已废弃，内部转发）
│   ├── inference.launch       # 推理节点
│   ├── record.launch          # 数据记录
│   └── full_system.launch     # 完整系统
│
├── CMakeLists.txt
├── package.xml
└── README.md
```

## 🚀 快速开始

### 依赖安装

```bash
# 系统依赖
sudo apt install ros-noetic-realsense2-camera ros-noetic-cv-bridge

# Python 依赖
conda activate carm
pip install numpy scipy h5py opencv-python einops

# CARM SDK
# 编译并安装 Python 绑定（需要 pybind11）：
# cd /path/to/arm_control_sdk/python
# python build_carm.py --Release
# python install_carm.py
```

### 编译 ROS 包

```bash
cd /path/to/rl-vla
./scripts/build_catkin.sh
source carm_ros_deploy/devel/setup.bash
```

### 环境变量（可选）

```bash
# 设置项目根目录
export RL_VLA_ROOT=/path/to/rl-vla
```

## 📷 相机操作

```bash
# 启动相机
roslaunch carm_deploy camera.launch
```

## 📝 数据采集

### Teleop 录制模式

当前 `record_data_ros` 已支持两种 teleop bridge 模式：

- `passive_shadow`
  - recorder 只录制 teleop shadow / candidate 观测
  - 不切 backend owner
  - 不由上位机真实接管 teleop 控制权
- `upper_control`
  - 启用 upper bridge
  - 是否真实接管由 `upper_control_enabled` 决定
  - `upper_control_enabled:=false` 时是 candidate-only
  - `upper_control_enabled:=true` 时会在 `recording=True` 期间切 owner 并由上位机 live execute

当前推荐默认模式仍是：

- 常规采集: `teleop_bridge_mode:=passive_shadow`
- 真机验证: `teleop_bridge_mode:=upper_control upper_control_enabled:=false`
- 最小 live 接管验证: `teleop_bridge_mode:=upper_control upper_control_enabled:=true`

### 单相机录制

```bash
# 启动相机
roslaunch carm_deploy camera.launch

# 启动录制（推荐默认：passive shadow）
roslaunch carm_deploy record.launch \
    output_dir:=~/recorded_data \
    teleop_bridge_mode:=passive_shadow

# 如果相机已在别处启动
roslaunch carm_deploy record.launch \
    output_dir:=~/recorded_data \
    use_camera:=false \
    teleop_bridge_mode:=passive_shadow
```

### 双相机录制

```bash
# 腕部 + 第三视角一体化录制（默认透传到 record.launch）
roslaunch carm_deploy dual_camera.launch \
    output_dir:=~/recorded_data \
    wrist_serial:=218622279840 \
    third_serial:=<D455_SERIAL> \
    teleop_bridge_mode:=passive_shadow
```

### Candidate-only / Live execute 示例

```bash
# upper_control candidate-only：启动 50Hz 候选控制线程，但不真实下发
roslaunch carm_deploy record.launch \
    output_dir:=~/recorded_data \
    use_camera:=false \
    teleop_bridge_mode:=upper_control \
    upper_control_enabled:=false

# upper_control live execute：仅建议在真机小幅验证时使用
roslaunch carm_deploy record.launch \
    output_dir:=~/recorded_data \
    use_camera:=false \
    teleop_bridge_mode:=upper_control \
    upper_control_enabled:=true \
    vis:=false
```

可选 teleop 参数：

- `backend_url_v2`: 显式指定 `teleop_target_v2` 接口
- `events_v2_url`: 显式指定 `events_v2` 接口
- `pred_horizon`: shadow chunk horizon
- `act_horizon`: action horizon metadata
- `teleop_candidate_control_freq`: candidate/live 控制线程频率，默认 `50Hz`
- `teleop_signal_timeout_ms`: stale timeout，默认 `150ms`
- `enable_teleop_sse`: 是否启用 SSE 监控
- `return_to_zero`: `upper_control` 下建议保持 `false`

控制键:

- `s`: 开始/停止录制
- `y`: 保存当前 episode
- `n`: 丢弃当前 episode
- `q`: 退出 recorder

### 当前 teleop 落盘内容

在保留旧字段兼容性的同时，teleop uplift 已额外写入：

- `teleop_processed_target_abs`
- `teleop_human_chunk_abs`
- `teleop_human_chunk_rel`
- `teleop_reconstructed_target_abs`
- `teleop_active`
- `teleop_processed_sequence`
- `teleop_raw_sequence`
- `teleop_signal_age_ms`
- `teleop_abs_reconstruction_pos_error`
- `teleop_abs_reconstruction_rot_error`
- `upper_candidate_target_abs`
- `upper_candidate_pos_error`
- `upper_candidate_rot_error`
- `teleop_candidate_loop_dt_ms`
- `teleop_candidate_stale`
- `teleop_candidate_applied`

当前 HDF5 metadata 也会记录：

- `teleop_bridge_mode`
- `backend_url_v2`
- `events_v2_url`
- `teleop_signal_timeout_ms`
- `control_owner_at_start`
- `upper_control_enabled_at_start`

### 数据分析

```bash
python data/analyze_dataset.py --data_dir ~/recorded_data/mix
```

## 🤖 策略推理

### 测试模式

```bash
# 干运行模式（不执行动作，最安全）
rosrun carm_deploy inference_ros.py --pretrain /path/to/model.pt --dry_run

# 慢速模式（5Hz）
rosrun carm_deploy inference_ros.py --pretrain /path/to/model.pt --slow_mode
```

### 正常推理

```bash
# 使用 launch 文件
roslaunch carm_deploy inference.launch pretrain:=/path/to/model.pt safety_config:=$(find carm_deploy)/safety_config.json

# 或直接运行
rosrun carm_deploy inference_ros.py --pretrain /path/to/model.pt --safety_config ~/rl-vla/carm_ros_deploy/src/carm_deploy/safety_config.json
```

> 推理必须提供 `safety_config.json`，默认路径为 `carm_deploy/safety_config.json`。
>
> 当前现役 `inference_ros` pipeline 不再接入 `KeyboardInterventionHandler` /
> `InterventionApplier` 这条旧键盘干预链路；相关实现仅保留作历史参考。

### 离线测试

```bash
# 使用数据集评估模型（不需要机械臂）
python tools/offline_test.py \
    --model_path /path/to/model.pt \
    --data_dir ~/rl-vla/recorded_data/mix \
    --compare_ema
```

## 🔒 安全控制

### 工作空间记录

```bash
# 拖动示教模式记录安全边界
python tools/record_workspace.py --output ~/rl-vla/carm_ros_deploy/src/carm_deploy/safety_config.json
```

### 安全配置验证

```bash
# 验证安全配置
python tools/verify_safety_config.py --config ~/rl-vla/carm_ros_deploy/src/carm_deploy/safety_config.json
```

## 🔧 机械臂调试

```bash
# 测试连接
python tools/arm_test/test_connection.py

# 测试关节运动
python tools/arm_test/test_motion.py

# 测试夹爪
python tools/arm_test/test_gripper.py

# 安全关闭
python tools/arm_test/safe_shutdown.py
```

## ⚙️ 配置说明

### 机械臂参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `robot_ip` | 10.42.0.101 | 机械臂 IP |
| `robot_mode` | 2 (MIT) | 控制模式 (禁用 mode=1) |
| `robot_tau` | 10.0 | 夹爪力矩 |

### 控制模式

| 模式 | 值 | 说明 |
|------|-----|------|
| IDLE | 0 | 空闲 |
| POSITION | 1 | **禁用** (危险) |
| MIT | 2 | 推荐 |
| DRAG | 3 | 拖动示教 |
| PF | 4 | 力位混合 |

### 推理参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_inference_steps` | 10 | 推理步数 |
| `use_ema` | False | 使用 EMA 模型 |
| `temporal_factor_k` | 0.05 | 时序融合因子 |
| `desire_inference_freq` | 30 | 推理频率 (Hz) |

> 图像尺寸：推理时会根据训练的 `args.json` 自动选择。ResNet 编码器默认使用 224x224，其它编码器使用 128x128。

## 📊 支持的算法

- **Consistency Flow** (推荐): 快速、高质量
- **Flow Matching**: 标准 Flow Matching
- **Diffusion Policy**: DDPM-based

## 📋 数据格式

记录的数据保存为 HDF5 格式。当前 teleop uplift 版本会在旧 schema 上增量扩展 teleop shadow / candidate 字段：

```
episode_0001_20240108_120000.hdf5
├── observations/
│   ├── images          # [T, H, W, 3] uint8, RGB 格式
│   ├── qpos_joint      # [T, 7] float64 (6 joints + gripper)
│   ├── qpos_end        # [T, 8] float64 (xyz + quat + gripper)
│   ├── qpos            # [T, 15] float64 (兼容格式)
│   ├── gripper         # [T] float64
│   └── timestamps      # [T] float64
├── action                          # [T, 8/15] 兼容字段；当前 teleop uplift 中表示 processed absolute target fallback
├── teleop_processed_target_abs     # [T, 8]
├── teleop_human_chunk_abs          # [T, H, 8]
├── teleop_human_chunk_rel          # [T, H, 7]
├── teleop_reconstructed_target_abs # [T, H, 8]
├── teleop_active                   # [T]
├── teleop_signal_age_ms            # [T]
├── upper_candidate_target_abs      # [T, 8]
├── teleop_candidate_loop_dt_ms     # [T]
├── teleop_candidate_stale          # [T]
├── teleop_candidate_applied        # [T]
└── attrs/
    ├── num_steps
    ├── record_freq
    ├── teleop_bridge_mode
    ├── control_owner_at_start
    ├── upper_control_enabled_at_start
    └── ...
```

## 🔗 相关链接

- [根目录 README](../../../README.md)
- [数据集 README](../../../recorded_data/README.md)
- [训练代码](../../../rlft/diffusion_policy/README.md)
