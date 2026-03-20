# Inference 模块重构文档

> 日期: 2026-03-16
> 范围: `carm_ros_deploy/src/carm_deploy/inference/`
> 基于: `CODE_REVIEW.md` (2026-03-13) 发现的 8 Bug + 6 冗余 + 5 可读性 + 2 安全问题

---

## 一、总览

重构前 `inference_ros.py` 是一个 **1256 行的单体文件**，包含 ROS CLI 入口、配置解析、推理循环、控制循环、动作后处理、安全检查等所有逻辑。重构后拆分为 **6 个职责单一的模块 + 1 个共享工具**。

### 行数变化

| 文件 | 重构前 | 重构后 | 变化 |
|------|--------|--------|------|
| `inference_ros.py` | 1256 | 262 | **-994** (仅保留 CLI 入口) |
| `inference_node.py` | — | 782 | **新建** (核心控制器) |
| `config.py` | — | 132 | **新建** (类型化配置) |
| `action_processor.py` | — | 201 | **新建** (动作管线) |
| `dry_run.py` | — | 189 | **新建** (离线测试环境) |
| `utils/log_compat.py` | — | 37 | **新建** (日志兼容层) |
| `policy_loader.py` | 614 | 617 | +3 (BUG-1/7 修复) |
| `inference_recorder.py` | 463 | 451 | -12 (改用 log_compat) |
| `__init__.py` | 12 | 23 | +11 (新增导出) |

**生产代码**: 1256 + 614 + 463 + 12 = **2345 行** → 262 + 782 + 132 + 201 + 189 + 37 + 617 + 451 + 23 = **2694 行** (+349 行，净增均为新功能代码)

### 测试

| 测试文件 | 数量 | 状态 |
|----------|------|------|
| `test_inference_node.py` | 7 | 已有 → 修复 mock 路径 |
| `test_config.py` | 16 | **新建** |
| `test_action_processor.py` | 12 | **新建** |
| `test_dry_run.py` | 12 | **新建** |
| 其他已有测试 | 141 | 无影响 |
| **Total** | **188** | 全部通过 |

### 调试脚本

| 脚本 | 行数 | 用途 |
|------|------|------|
| `scripts/test_mock_arm_motion.py` | 402 | **新建** — 5 项真机离线测试（初始位姿/XYZ平移/夹爪/安全边界/轨迹回放） |
| `scripts/live_inference_test.py` | 277 | **新建** — 7 步人工配合联调检查表 |

---

## 二、文件映射关系

### 2.1 `inference_ros.py` → 拆分为 4 个文件

```
inference_ros.py (原 1256 行)
│
├── inference_ros.py (262 行)     ← CLI 入口、argparse、ROS param overlay
├── inference_node.py (782 行)    ← InferenceNode 类（推理循环 + 控制循环）
├── config.py (132 行)            ← InferenceConfig dataclass
└── action_processor.py (201 行)  ← ActionProcessor + ActionIndices + SafetyResult
```

### 2.2 行级映射（原 inference_ros.py → 新文件）

| 原文件行范围 | 原功能 | → 新位置 | 备注 |
|-------------|--------|----------|------|
| 1-40 | 文件头、imports | `inference_ros.py:1-35` | 精简 imports |
| 41-180 | `parse_args()` | `inference_ros.py:46-164` | 保留原位 |
| 181-210 | `main()` 前半：ROS init + config 构建 | `inference_ros.py:171-220` | 改用 `InferenceConfig.from_argparse()` |
| 211-230 | config dict 构建（50+ key） | `config.py` 全文 | 替换为 `InferenceConfig` dataclass |
| 231-420 | `InferenceNode.__init__()` | `inference_node.py:44-190` | 拆分为 `__init__` + 4 个 factory 方法 |
| 421-500 | Episode 生命周期管理 | `inference_node.py:350-399` | `_start_new_episode`, `_stop_current_episode`, `_confirm_save_episode` |
| 501-543 | `_preprocess_image()`, `_normalize_images()` | **已删除** | BUG-1 修复：图像预处理完全由 `RealPolicy` 负责 |
| 544-800 | `_inference_loop()` | `inference_node.py:548-668` + `action_processor.py` | 循环体拆为 ~60 行主循环 + 6 个 helper 方法；动作处理提取至 `ActionProcessor` |
| 620-650 | speed_scale 应用 | `action_processor.py:94-105` | `apply_speed_scale()` |
| 650-750 | 安全检查（平移/工作空间/夹爪） | `action_processor.py:109-179` | `apply_safety_checks()` — BUG-3 修复 |
| 750-810 | 相对→绝对动作转换 | `action_processor.py:183-201` | `convert_to_absolute()` |
| 810-850 | chunk post + VecTF | `inference_node.py:462-517` | `_post_action_chunk()` |
| 850-1000 | `control_loop()` | `inference_node.py:674-747` | 保持逻辑，提取 Hz 估算 |
| 1000-1060 | `shutdown()` | `inference_node.py:753-782` | 增加重入保护 |
| 1060-1200 | `main()` 后半：ROS 启动 + signal | `inference_ros.py:220-262` | 保留原位 |
| — | is_full_mode + index 计算（散布 3 处）| `action_processor.py:29-49` | R-2 修复：`ActionIndices` 一次计算 |
| — | try/except rospy 日志（散布多处）| `utils/log_compat.py` 全文 | R-3 修复：统一日志兼容层 |

### 2.3 未改动文件

| 文件 | 行数 | 说明 |
|------|------|------|
| `inference_logger.py` | 630 | 日志模块，无需改动 |
| `analyze_inference_data.py` | 777 | 离线分析脚本，独立 |

---

## 三、新增模块详解

### 3.1 `config.py` — InferenceConfig

**解决的问题**: 原 `inference_ros.py` 的 `__init__` 内手工构建 50+ key 的 untyped dict，无补全、无校验、key 写错不报错。

```python
@dataclass
class InferenceConfig:
    # Robot
    robot_ip: str = "10.42.0.101"
    arm_init_pose: list = ...
    gripper_open_value: float = 0.078
    # Camera
    camera_topics: list = ...
    image_size: int = 128
    # Policy
    pretrain: str = ""
    algorithm: str = "consistency_flow"
    # Action execution
    pred_horizon: int = 16
    action_dim_full: int = 15
    inference_speed_scale: float = 1.0
    # ... 共 25 个类型化字段

    @property
    def timeline_enabled(self) -> bool:    # BUG-6 fix: 只读派生
        return not self.timeline_disabled

    @property
    def teleop_scale(self) -> float:       # GAP-2 fix: 固定为 1.0
        return 1.0
```

**兼容性**: `InferenceNode.__init__` 同时接受 `InferenceConfig` 和 `dict`，通过 `InferenceConfig.from_dict(config)` 自动转换。

### 3.2 `action_processor.py` — 动作处理管线

**解决的问题**: 原 `_inference_loop()` 内动作后处理散布在 ~200 行中，speed_scale/安全检查/坐标转换耦合；`is_full_mode` 索引重复计算 3 次 (R-2)；夹爪安全检查通过构造 dummy joints 传给 SafetyController (BUG-3)。

```
ActionProcessor pipeline:
  raw_actions → apply_speed_scale() → apply_safety_checks() → convert_to_absolute()
                                        ├── 平移幅度裁剪
                                        ├── 工作空间边界检查
                                        └── 夹爪值直接 np.clip (BUG-3 fix)
```

**`ActionIndices`**: 冻结 dataclass，从 `action_dim_full` 一次性计算所有列索引：
```python
ActionIndices.from_action_dim(15)
# → is_full_mode=True, rel_pose_start=7, rel_pose_end=14, gripper_idx=14
```

### 3.3 `dry_run.py` — 离线测试环境

**解决的问题**: 原来测试 InferenceNode 必须连接真机 + ROS + 相机，无法离线验证。

`DryRunEnvironment` 是 `RealEnvironment` 的 drop-in 替换，接口完全一致：

| 方法 | RealEnvironment | DryRunEnvironment |
|------|----------------|-------------------|
| `get_observation()` | 读 ROS 话题 + SDK | 回放 HDF5 / 合成数据 |
| `end_control_nostep(action)` | 发送给机械臂 | 记录到 `actions_sent` |
| `init_status()` | SDK 初始化 | no-op |
| `shutdown()` | SDK 关闭 | no-op, 打印统计 |

两种工厂方法：
- `DryRunEnvironment.from_hdf5(path)` — 回放录制的 episode
- `DryRunEnvironment.synthetic(num_frames=100)` — 生成正弦扰动合成数据

### 3.4 `utils/log_compat.py` — 日志兼容层

**解决的问题**: `inference_ros.py`, `policy_loader.py`, `inference_recorder.py` 中散布多处相同的 try/except rospy 日志模式 (R-3)。

```python
# 之前（每个文件中重复）:
try:
    import rospy
    rospy.loginfo(msg)
except ImportError:
    import logging
    logging.getLogger(__name__).info(msg)

# 之后（一处定义，处处调用）:
from utils.log_compat import log_info, log_warn, log_err
log_info(msg)
```

---

## 四、Bug 修复清单

| Bug ID | 问题描述 | 原位置 | 修复方式 | 新位置 |
|--------|---------|--------|---------|--------|
| **BUG-1** | 图像双重预处理（InferenceNode 做一次 resize+HWC→CHW，RealPolicy 又做一次） | `inference_ros.py:501-543` | 删除 `_preprocess_image()` 和 `_normalize_images()`，InferenceNode 只传裸 HWC 图像 | `inference_node.py` 不含图像预处理；`policy_loader.py:412-418` 是唯一预处理点 |
| **BUG-3** | 夹爪安全检查构造 dummy 14D joints 传给 SafetyController | `inference_ros.py:718` | 夹爪值直接 `np.clip(gripper, min, max)` | `action_processor.py:163-168` |
| **BUG-6** | `--timeline_enabled` argparse flag 永远为 False（`store_true` 但无人传参） | `inference_ros.py` argparse | 改为 `timeline_disabled` + 只读 property `timeline_enabled = not timeline_disabled` | `config.py:80-82` |
| **BUG-7** | `torch.load()` 缺少 `weights_only` 参数（PyTorch 2.6+ 警告） | `policy_loader.py:298` | 添加 `weights_only=False` | `policy_loader.py:298` |
| **BUG-8** | `truncate_at_act_horizon` 无法禁用（argparse 只有 `store_true`） | `inference_ros.py` argparse | 改为 `--no_truncate_at_act_horizon` + `store_false` | `inference_ros.py:135` + `config.py:62` |
| **SEC-1** | 推理循环 `except Exception` 吞掉所有异常，死循环不退出 | `inference_ros.py:~600` | 新增连续错误计数器，10 次连续失败后终止 | `inference_node.py:563-570`, `MAX_CONSECUTIVE_ERRORS=10` |

### 未修复（设计正确或需进一步确认）

| Bug ID | 说明 | 处置 |
|--------|------|------|
| BUG-2 | `apply_relative_transform` 语义 — 所有 step 以同一 `qpos_end` 为基准 | 确认正确（训练 label 是 `inv(T_obs)@T_target`），已添加注释 |
| BUG-4 | 夹爪 horizon vote tie-breaking 偏向 open | 设计行为，非 bug |
| BUG-5 | `_apply_gripper_hysteresis` 仅看最近 2 帧 | 可调优但非必要 |

---

## 五、冗余 / 可读性改善

| Issue ID | 问题 | 修复 |
|----------|------|------|
| **R-2** | `is_full_mode` + 索引计算散布 3 处（inference loop、safety、convert） | `ActionIndices` 冻结 dataclass 一次计算 |
| **R-3** | try/except rospy 日志模式重复 4+ 处 | `utils/log_compat.py` 统一 |
| **S-1** | `inference_ros.py` 单文件 1256 行 | 拆分为 4 个模块（262+782+132+201） |
| **S-4** | 无 config dataclass，50+ key untyped dict | `InferenceConfig` 25 个类型化字段 |
| **SEC-1** | 异常吞没导致潜在死循环 | 连续错误计数器 + 自动退出 |

---

## 六、功能变化总结

### 6.1 行为不变（向后兼容）

| 功能 | 说明 |
|------|------|
| ROS 启动方式 | `rosrun carm_deploy inference_ros.py --pretrain ...` 完全不变 |
| argparse 参数 | 所有原有参数保持兼容，可能新增但不删除 |
| 推理循环 | observe → infer → process → chunk → control 流程不变 |
| 控制循环 | 50Hz 查询 fused action → 下发机械臂，EMA Hz 估算不变 |
| 人工干预 | KeyboardInterventionHandler 逻辑不变 |
| 数据录制 | InferenceRecorder start/stop/save/discard 生命周期不变 |
| HDF5 输出格式 | 录制数据格式完全不变 |
| 安全控制器 | SafetyController JSON 配置加载和工作空间检查不变 |
| 日志输出 | InferenceLogger 行为不变 |

### 6.2 行为改变

| 变化 | 旧行为 | 新行为 | 影响 |
|------|--------|--------|------|
| **图像预处理** (BUG-1) | InferenceNode 做 resize+CHW，RealPolicy 再检查一次 | InferenceNode 传裸 HWC，RealPolicy 是唯一预处理点 | 无功能影响（之前因判断条件不会重复执行），但消除了潜在风险 |
| **夹爪裁剪** (BUG-3) | 构造 14D dummy joints → SafetyController.check_safety() | 直接 `np.clip(gripper, min, max)` | 无功能影响，结果相同但更高效清晰 |
| **timeline_enabled** (BUG-6) | argparse `--timeline_enabled` 无法通过 CLI 激活 | `--timeline_disabled` 反转，默认启用 | **行为变化**: timeline 默认从不启用 → 默认启用 |
| **truncate_at_act_horizon** (BUG-8) | 只能设为 True | 新增 `--no_truncate_at_act_horizon` 可设为 False | 新增能力，默认行为不变 |
| **异常处理** (SEC-1) | 推理循环 except 吞掉所有异常，永不退出 | 连续 10 次错误后自动 shutdown | **行为变化**: 持续出错时会退出而非死循环 |
| **teleop_scale** (GAP-2) | 可配置，默认 0.6 | 固定为 1.0，只读 property | **行为变化**: 不再支持配置此值 |

### 6.3 新增功能

| 功能 | 文件 | 说明 |
|------|------|------|
| **类型化配置** | `config.py` | IDE 补全、默认值校验、序列化/反序列化 |
| **离线测试环境** | `dry_run.py` | 无需真机/ROS/相机即可测试推理管线 |
| **动作管线独立测试** | `action_processor.py` | speed_scale / safety / convert 可独立单元测试 |
| **47 个新增单测** | `test_config.py` / `test_action_processor.py` / `test_dry_run.py` | 覆盖所有新模块和 bug 修复 |
| **真机离线测试** | `scripts/test_mock_arm_motion.py` | 5 项安全 gated 机械臂运动测试 |
| **人工联调检查表** | `scripts/live_inference_test.py` | 7 步交互式端到端验证 |

---

## 七、依赖关系图

```
inference_ros.py (CLI 入口)
│
├── InferenceConfig                     ← config.py
│
└── InferenceNode                       ← inference_node.py
    ├── RealEnvironment / DryRunEnvironment  ← core/env_ros.py / dry_run.py
    ├── RealPolicy                     ← policy_loader.py
    ├── SafetyController               ← core/safety_controller.py
    ├── ActionProcessor                ← action_processor.py
    │   └── ActionIndices
    ├── ActionChunkManager             ← utils/trajectory_interpolator.py
    ├── InferenceLogger                ← inference_logger.py
    ├── InferenceRecorder              ← inference_recorder.py
    ├── KeyboardInterventionHandler    ← utils/keyboard_intervention.py
    └── log_info / log_warn / log_err  ← utils/log_compat.py
```

---

## 八、向后兼容性

`inference_ros.py` 第 39 行保留了 re-export：

```python
from inference.inference_node import InferenceNode
```

任何 `from inference.inference_ros import InferenceNode` 的外部代码不会 break。

`InferenceNode.__init__` 接受 `dict` 参数，内部自动 `InferenceConfig.from_dict(config)` 转换：

```python
# 旧代码（继续工作）
node = InferenceNode({"pretrain": "/path/to/model.pt", ...})

# 新代码（推荐）
cfg = InferenceConfig(pretrain="/path/to/model.pt")
node = InferenceNode(cfg)
```
