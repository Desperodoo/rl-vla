# 数据集对比报告：ManiSkill vs 真机数据集

## 概述

本报告对比了ManiSkill官方仿真数据集与ARX5真机采集数据集的结构差异，为数据加载pipeline适配提供依据。

## 1. 数据集基本信息

| 属性 | ManiSkill 官方数据集 | ARX5 真机数据集 |
|------|---------------------|----------------|
| **路径** | `~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.rgb.pd_joint_pos.physx_cuda.h5` | `~/.arx_demos/processed/pick_cube/20251218_235920/trajectory.h5` |
| **轨迹数量** | ~1000条 | ~50条 |
| **平均轨迹长度** | ~50-100步 | ~400-600步 |
| **控制频率** | 20Hz (仿真) | 30Hz |
| **控制模式** | `pd_joint_pos` | `joint_pos` |

## 2. HDF5结构对比

### 2.1 顶层结构

**ManiSkill 数据集:**
```
trajectory.h5
├── traj_0/
│   ├── obs/
│   ├── actions
│   ├── terminated
│   ├── truncated
│   ├── success
│   ├── env_states/
│   └── rewards
├── traj_1/
...
```

**真机数据集:**
```
trajectory.h5
├── traj_0/
│   ├── obs/
│   └── actions
├── traj_1/
...
```

**差异:** 真机数据集没有`rewards`, `terminated`, `truncated`, `success`, `env_states`字段（真机无仿真环境状态）。

### 2.2 观测数据结构 (`obs/`)

#### ManiSkill 数据集

```
obs/
├── agent/
│   ├── qpos: float32, (T+1, 9)     # 关节位置（Panda 7DoF + 2夹爪）
│   └── qvel: float32, (T+1, 9)     # 关节速度
├── extra/
│   ├── is_grasped: bool, (T+1,)    # 是否抓取
│   ├── tcp_pose: float32, (T+1, 7) # 末端位姿 (xyz + quat)
│   └── goal_pos: float32, (T+1, 3) # 目标位置
├── sensor_param/
│   └── base_camera/
│       ├── extrinsic_cv: float32, (T+1, 3, 4)
│       ├── cam2world_gl: float32, (T+1, 4, 4)
│       └── intrinsic_cv: float32, (T+1, 3, 3)
└── sensor_data/
    └── base_camera/
        └── rgb: uint8, (T+1, 128, 128, 3)
```

#### 真机数据集

```
obs/
├── joint_pos: float32, (T, 6)      # 关节位置（ARX5 6DoF）
├── joint_vel: float32, (T, 6)      # 关节速度
├── gripper_pos: float32, (T, 1)    # 夹爪位置
├── timestamps: float64, (T,)       # 时间戳
└── images/
    ├── external/                   # Eye-to-hand 相机
    │   ├── rgb: uint8, (T, 256, 256, 3)
    │   └── depth: uint16, (T, 256, 256)
    └── wrist/                      # Eye-in-hand 相机
        ├── rgb: uint8, (T, 256, 256, 3)
        └── depth: uint16, (T, 256, 256)
```

### 2.3 关键差异总结

| 属性 | ManiSkill | 真机 | 说明 |
|------|-----------|------|------|
| **State路径** | `obs/agent/qpos`, `obs/agent/qvel`, `obs/extra/*` | `obs/joint_pos`, `obs/joint_vel`, `obs/gripper_pos` | 路径结构不同 |
| **State维度** | qpos: 9, qvel: 9 | joint_pos: 6, joint_vel: 6, gripper: 1 | 机器人DoF不同 |
| **图像路径** | `obs/sensor_data/base_camera/rgb` | `obs/images/wrist/rgb`, `obs/images/external/rgb` | 路径结构不同 |
| **图像分辨率** | 128×128 | 256×256 | 真机分辨率更高 |
| **相机数量** | 1 (base_camera) | 2 (wrist + external) | 真机有双相机 |
| **Depth数据** | 无 | 有 (uint16) | 真机支持深度 |
| **数据对齐** | obs比actions多1帧 (T+1 vs T) | obs与actions相同帧数 (T) | 帧对齐方式不同 |

## 3. 数据类型详情

### 3.1 State数据

| 字段 | ManiSkill | 真机 |
|------|-----------|------|
| **关节位置** | `float32, (T+1, 9)` | `float32, (T, 6)` |
| **关节速度** | `float32, (T+1, 9)` | `float32, (T, 6)` |
| **夹爪** | 包含在qpos最后2维 | `float32, (T, 1)` 单独字段 |
| **末端位姿** | `float32, (T+1, 7)` 在extra | 无（需从关节角计算） |

### 3.2 图像数据

| 属性 | ManiSkill | 真机 |
|------|-----------|------|
| **RGB dtype** | `uint8` | `uint8` |
| **RGB shape** | `(T+1, 128, 128, 3)` | `(T, 256, 256, 3)` |
| **Depth dtype** | N/A | `uint16` |
| **Depth shape** | N/A | `(T, 256, 256)` |
| **Depth scale** | N/A | 1000.0 (mm → m) |

### 3.3 动作数据

| 属性 | ManiSkill | 真机 |
|------|-----------|------|
| **dtype** | `float32` | `float32` |
| **shape** | `(T, 8)` | `(T, 7)` |
| **内容** | 位置+姿态增量 (7D) + 夹爪 (1D) | 关节位置 (6D) + 夹爪 (1D) |

## 4. 相机命名映射

| 功能 | ManiSkill命名 | 真机命名 |
|------|--------------|---------|
| **Eye-to-hand** | `base_camera` | `external` |
| **Eye-in-hand** | N/A | `wrist` |

**注意:** 根据需求，训练时只使用 `wrist` (eye-in-hand) 相机的RGB/D数据。

## 5. 数据加载适配建议

### 5.1 State提取器

```python
# ManiSkill state提取
def maniskill_state_extractor(obs):
    return [obs["agent"]["qpos"], obs["agent"]["qvel"]] + list(obs["extra"].values())

# 真机 state提取
def real_robot_state_extractor(obs):
    return [obs["joint_pos"], obs["joint_vel"], obs["gripper_pos"]]
```

### 5.2 图像提取

```python
# ManiSkill RGB
rgb = obs["sensor_data"]["base_camera"]["rgb"]  # (T, 128, 128, 3)

# 真机 RGB (只用wrist)
rgb = obs["images"]["wrist"]["rgb"]  # (T, 256, 256, 3)
depth = obs["images"]["wrist"]["depth"]  # (T, 256, 256), uint16
```

### 5.3 帧对齐处理

- **ManiSkill:** obs有T+1帧，actions有T帧，需要截断obs最后一帧
- **真机:** obs和actions都是T帧，直接对齐

## 6. 配置文件

### 真机数据集 `config.yaml` 关键信息

```yaml
cameras:
  external:  # Eye-to-hand
    resolution: [640, 480]
    enable_depth: true
  wrist:     # Eye-in-hand
    resolution: [640, 480]
    enable_depth: true

robot:
  model: X5
  joint_dof: 6
  gripper_range: [0.0, 0.08]

control_freq: 30
depth_scale: 1000.0  # mm to m
```

### 真机数据集 `trajectory.json` 关键信息

```json
{
  "env_info": {
    "env_id": "20251218_235920",
    "max_episode_steps": 661
  },
  "source_type": "teleoperation",
  "episodes": [
    {
      "episode_id": 0,
      "control_mode": "joint_pos",
      "elapsed_steps": 461,
      "info": {"success": true}
    }
  ]
}
```

## 7. 统计信息 (`stats.json`)

真机数据集提供了归一化统计信息：

| 字段 | Mean | Std | Min | Max |
|------|------|-----|-----|-----|
| joint_pos[0] | -0.168 | 0.264 | -0.809 | 0.206 |
| joint_pos[1] | 1.026 | 0.695 | -0.004 | 2.066 |
| ... | ... | ... | ... | ... |
| gripper_pos | 0.065 | 0.015 | 0.012 | 0.080 |

这些统计信息可用于训练时的数据归一化。
