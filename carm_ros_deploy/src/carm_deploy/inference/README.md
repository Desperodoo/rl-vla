# Inference 模块

CARM 机械臂真机推理管线。将策略模型与 ROS 环境、安全控制器、动作后处理、日志记录串联为完整的推理节点。

---

## 模块结构

```
inference/
├── inference_ros.py       # ROS CLI 入口 (rosrun 启动点)
├── inference_node.py      # 核心控制器 (InferenceNode)
├── config.py              # 类型化配置 (InferenceConfig)
├── action_processor.py    # 动作后处理管线 (ActionProcessor)
├── policy_loader.py       # 策略模型加载与推理 (RealPolicy)
├── inference_logger.py    # 推理日志 HDF5 + run_info.json
├── inference_recorder.py  # 干预数据采集 (用于后续训练)
├── dry_run.py             # 离线测试环境 (无需真机)
├── LOG_FORMAT.md          # 日志格式规范
└── _archive/              # 历史文档 (CODE_REVIEW, REFACTORING_MAP)
```

### 职责划分

| 模块 | 职责 |
|------|------|
| `inference_ros.py` | argparse + rospy.init_node + main() |
| `inference_node.py` | 推理线程 + 控制循环 + episode 生命周期 |
| `config.py` | 所有参数的单一来源 (InferenceConfig dataclass) |
| `action_processor.py` | 速度缩放 → 安全裁剪 → 相对→绝对坐标转换 |
| `policy_loader.py` | 模型加载、图像预处理、推理调用 |
| `inference_logger.py` | HDF5 数据记录 + run_info.json 元数据 |
| `inference_recorder.py` | 干预采集 (--intervention --record_inference) |
| `dry_run.py` | 合成/HDF5 回放环境，用于离线验证 |

---

## 快速启动

```bash
# 基本推理 (30Hz, PF 模式)
rosrun carm_deploy inference_ros.py --pretrain /path/to/model.pt safety_config:=~/rl-vla/safety_config.json --intervention

# 启用干预采集
rosrun carm_deploy inference_ros.py \
    --pretrain /path/to/model.pt \
    --intervention \
    --record_inference

# 自定义参数
rosrun carm_deploy inference_ros.py \
    --pretrain /path/to/model.pt \
    --robot_ip 10.42.0.101 \
    --robot_mode 4 \
    --control_freq 50 \
    --execution_mode receding_horizon \
    --act_horizon 8
```

---

## 关键参数

### 机械臂

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--robot_ip` | `10.42.0.101` | 机械臂 IP |
| `--robot_mode` | `4` | 控制模式 (2=MIT, 4=PF，**禁止使用 1=Position**) |
| `--arm_init_pose` | `[0.2475, 0.0014, 0.3251, ...]` | 初始末端位姿 |
| `--init_speed` | `2.0` | 初始化移动速度 (0-10) |
| `--skip_init_confirm` | `False` | 跳过初始化确认提示 (脚本化启动用) |

### 推理

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--execution_mode` | `receding_horizon` | `receding_horizon` / `temporal_ensemble` |
| `--act_horizon` | `8` | 每次执行的动作步数 |
| `--control_freq` | `50` | 控制循环频率 (Hz) |
| `--desire_inference_freq` | `30` | 推理频率 (Hz) |
| `--inference_speed_scale` | `1.0` | 动作幅度缩放 |

### 安全

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--safety_config` | 自动查找 `safety_config.json` | 工作空间边界配置 |
| `--robot_mode 1` | **禁用** | Position 模式被强制切换为 MIT 模式 |

---

## 推理流程

```
每个推理步 (~30Hz):
  1. get_observation()          → 图像 + 关节状态
  2. policy(qpos, image)        → all_actions [pred_horizon, action_dim]
  3. apply_speed_scale()        → 幅度缩放
  4. apply_safety_checks()      → 工作空间裁剪
  5. apply_intervention()       → 键盘干预叠加 (可选)
  6. convert_to_absolute()      → 相对 delta → 绝对末端位姿
  7. ActionChunkManager.post()  → chunk 管理 (receding_horizon / temporal_ensemble)

每个控制步 (~50Hz):
  8. ActionChunkManager.query() → 当前时刻的融合动作
  9. env.end_control_nostep()   → track_pose() + set_gripper()
```

---

## 离线测试

无需真机，用 `dry_run.py` 验证推理管线：

```bash
# 合成数据测试
python -c "
from inference.dry_run import SyntheticEnvironment
env = SyntheticEnvironment()
obs = env.get_observation()
print(obs.keys())
"

# HDF5 回放测试
python -c "
from inference.dry_run import HDF5ReplayEnvironment
env = HDF5ReplayEnvironment('/path/to/data.hdf5')
for i in range(10):
    obs = env.get_observation()
"
```

---

## 日志格式

见 [LOG_FORMAT.md](LOG_FORMAT.md)。

每次推理运行生成：
- `run_info.json` — 运行配置和元数据
- `*.hdf5` — 逐步数值数据 (qpos, actions, images, timestamps)
- `timeline_*.jsonl` — 时序事件 (推理延迟、控制频率等)

---

## 历史文档

`_archive/` 目录保存重构过程文档：
- `CODE_REVIEW.md` — 重构前代码审查报告 (2026-03-13)
- `REFACTORING_MAP.md` — 重构设计文档 (2026-03-16)
- `analyze_inference_data.py` — 旧版离线分析脚本 (已被 dry_run.py 替代)
