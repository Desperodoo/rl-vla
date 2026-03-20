# Inference 模块代码审查报告

> 审查日期: 2026-03-13
> 审查范围: `carm_ros_deploy/src/carm_deploy/inference/` 全部 Python 文件
> 审查维度: Bug、代码可读性、代码冗余、安全与健壮性

---

## 文件概览

| 文件 | 行数 | 职责 |
|------|------|------|
| `inference_ros.py` | 1257 | 主推理节点（ROS 入口、推理循环、控制循环） |
| `policy_loader.py` | 615 | 策略模型加载与推理 |
| `inference_logger.py` | 630 | 推理日志记录（HDF5 + run_info.json） |
| `inference_recorder.py` | 464 | 推理干预数据采集（用于后续训练） |
| `analyze_inference_data.py` | 777 | 离线分析脚本 |
| `__init__.py` | 13 | 包定义 |

---

## 一、Bug 与潜在问题

### BUG-1: `inference_ros.py:511-543` — 图像双重预处理

`InferenceNode._normalize_images()` 对图像做了 resize + HWC→CHW 转换，但 `RealPolicy.__call__()` (policy_loader.py:495-496) 又检查 `image.shape[0] != 3` 后再次调用 `self._preprocess_image(image)` 做一模一样的事。

**实际效果**：因为 `_normalize_images` 已经输出 CHW 格式且 `shape[0]==3`，所以 RealPolicy 的重复检查不会二次执行。但这个逻辑**脆弱**：如果 image 恰好是 3xHxW 已是 CHW 但不是 128x128，就会跳过 resize。

**建议**：在 `inference_ros.py` 中只传裸 RGB 图像给 policy，让预处理完全由 `RealPolicy` 负责（单一责任原则）。

### BUG-2: `inference_ros.py:694` — `apply_relative_transform` 语义需确认

```python
# Line 694: 安全检查阶段
target_pose = apply_relative_transform(relative_pose, qpos_end[:7], grip)
# Line 777: 实际转换阶段
target_pose = apply_relative_transform(relative_pose, qpos_end[:7], grip)
```

所有 16 步预测都以相同的 `qpos_end`（当前末端位姿）为基准做 `apply_relative_transform`，而非链式累积（step 2 以 step 1 的目标位姿为基准）。

- 如果训练 label 是 `inv(T_obs) @ T_target`（每帧都相对于该帧的观测） → 当前实现**正确**
- 如果训练 label 是链式增量 → 存在语义错误

**建议**：确认训练时 action label 的语义并添加注释说明。

### BUG-3: `inference_ros.py:718` — 夹爪安全检查使用 dummy joints

```python
gripper_action = np.array([0, 0, 0, 0, 0, 0, grip])  # dummy joints + gripper
clipped_gripper, grip_warnings = self.safety_controller.check_joint_limits(gripper_action)
```

用 0 填充的 dummy joints 可能触发 `check_joint_limits` 的关节限位警告（如果 0 不在某些关节的合法范围内），产生**误报安全事件**。

**建议**：单独实现 gripper 限位检查，或传入真实的关节角度。

### BUG-4: `inference_logger.py:252-258` — 潜在竞态条件

```python
start_idx = self.step_count - len(self.current_episode)
```

如果 `log_step()` 在 flush 期间被并发调用（推理线程），存在竞态条件。当前代码没有锁保护。

**风险等级**：低（当前 `log_step` 仅在推理线程单线程调用）。

**建议**：添加线程锁或添加注释明确单线程约束。

### BUG-5: `inference_recorder.py:168` — 浅拷贝 list

```python
self.pending_data = {k: v.copy() if isinstance(v, list) else v
                     for k, v in self.episode_data.items()}
```

`list.copy()` 是浅拷贝，内部 np.ndarray 元素共享引用。功能上没问题（`_reset_buffer` 创建新 list，不修改旧 list 元素），但语义上容易误导后续开发者。

### BUG-6: `inference_ros.py:1008-1011` — `--timeline_enabled` argparse 逻辑矛盾

```python
parser.add_argument('--timeline_enabled', action='store_true', ...)
parser.add_argument('--timeline_disabled', action='store_true', ...)
```

`main()` 的 line 1141-1144 无条件覆盖：

```python
if config.get('timeline_disabled', False):
    config['timeline_enabled'] = False
else:
    config['timeline_enabled'] = True  # 永远为 True
```

**结果**：`--timeline_enabled` 参数永远无效，timeline 始终启用（除非传 `--timeline_disabled`）。

**建议**：删除 `--timeline_enabled`，只保留 `--timeline_disabled`。

### BUG-7: `policy_loader.py:307` — `torch.load` 缺少 `weights_only` 参数

```python
ckpt = torch.load(model_path, map_location=self.device)
```

PyTorch 2.6+ 默认 `weights_only=True`，会导致加载失败（checkpoint 通常包含非 tensor 数据如 dict）。

**建议**：显式指定 `weights_only=False`。

### BUG-8: `inference_ros.py:1045` — `truncate_at_act_horizon` 无法关闭

```python
parser.add_argument('--truncate_at_act_horizon', action='store_true', default=True, ...)
```

`action='store_true'` + `default=True` = 无论传不传此参数，值都是 True，无法关闭。

**建议**：改用 `--no_truncate_at_act_horizon` (store_false, dest='truncate_at_act_horizon')。

---

## 二、代码冗余

### R-1: 图像预处理三重实现

| 方法 | 文件 | 功能 |
|------|------|------|
| `InferenceNode._preprocess_image()` | inference_ros.py:511-523 | resize |
| `InferenceNode._normalize_images()` | inference_ros.py:525-545 | resize + HWC→CHW |
| `RealPolicy._preprocess_image()` | policy_loader.py:421-427 | resize + HWC→CHW |

三个方法做同一件事。`InferenceNode` 应直接传原始图像，让 `RealPolicy` 内部处理。

### R-2: `is_full_mode` 索引计算重复 3 次

`_inference_loop` 中 line 648-650、671-674、767-770 三处几乎相同：

```python
is_full_mode = (self._action_dim_full == 15)
rel_pose_start = 7 if is_full_mode else 0
rel_pose_end = 14 if is_full_mode else 7
gripper_idx = 14 if is_full_mode else 7
```

**建议**：在 `__init__` 中一次性计算，存为实例属性。

### R-3: `_log_info` / `_log_warn` 日志工具重复定义

- `inference_recorder.py` lines 44-55
- `policy_loader.py` lines 38-47

两处实现几乎相同（判断 rospy 可用性后选择 rospy.loginfo 或 print）。

**建议**：提取到 `utils/log_compat.py`，项目统一使用。

### R-4: `InferenceRecorder` 中 `qpos` 冗余存储

`qpos`（15D）= `concat(qpos_joint(7D), qpos_end(8D))`，三者同时存储。与 `record_data_ros.py` 同一问题。

### R-5: `inference_recorder.py` 中 gripper 双重存储

`gripper` 单独存一份，但 `qpos_joint[6]` 和 `qpos_end[7]` 中各已包含 gripper 值。

### R-6: `analyze_inference_data.py` 硬编码 15D action 维度

多处硬编码维度索引（7,8,9 for XYZ, 14 for gripper），8D (ee_only) 模式下会得到错误结果。应根据 `action_dim` 动态确定。

---

## 三、可读性建议

### S-1: `inference_ros.py` 过长（1257 行）

`_inference_loop` 方法长达 320+ 行（line 547-870），包含：获取观测、预处理、推理、安全检查、干预应用、动作转换、chunk 管理。

**建议**拆分为：
- `_run_policy_inference(obs)` → 返回 `all_actions`
- `_apply_safety_checks(all_actions, qpos_end)` → 安全裁剪后的 actions
- `_apply_intervention(all_actions)` → 干预后的 actions
- `_create_and_post_chunk(all_actions, ...)` → 创建 VecTF 并 post 到 manager

### S-2: `inference_ros.py:35-44` — import 顺序不规范

`import sys, os` 出现在 line 35-36（numpy, cv2, rospy 之后）。约定：stdlib 在最前，`sys.path.insert` 在所有本地模块 import 之前。

### S-3: `policy_loader.py` 中 debug counter 可清理

Lines 522-541 的 gripper debug logging 是临时调试代码（20 行），建议用 logging level 控制或移除。

### S-4: 缺少 config 类型定义

`InferenceNode` 的 config 字典接受 50+ 个 key，无 dataclass/TypedDict 定义，全靠 `config.get('key', default)` 散落在构造函数中。

**建议**：用 `@dataclass` 或 TypedDict 集中定义。

### S-5: deprecated 参数静默忽略

`--joint_cmd_mode` 和 `--teleop_scale` 标记 deprecated，但使用时没有实际警告。

**建议**：添加 `warnings.warn()` 提示。

---

## 四、安全与健壮性

### SEC-1: `inference_ros.py:862-865` — 异常吞没

推理循环中 `except Exception` 打印错误后继续。持续性错误（如 CUDA OOM）会产生大量重复日志。

**建议**：添加连续错误计数器，超过阈值（如 10 次）时优雅退出。

### SEC-2: shutdown 可能重复执行

`signal_handler` 和 `rospy.on_shutdown` 都调用 `node.shutdown()`。虽有 `_shutdown_called` 防重入，但并发场景下仍可能竞争。

---

## 五、修改优先级

| 优先级 | 问题 | 影响 |
|--------|------|------|
| **P0** | BUG-7 torch.load weights_only | PyTorch 2.6+ 崩溃 |
| **P0** | BUG-8 truncate_at_act_horizon 无法关闭 | 配置项无效 |
| **P1** | BUG-1 图像双重预处理 | 脆弱逻辑 |
| **P1** | BUG-3 dummy joints 安全检查误报 | fake safety 事件 |
| **P1** | BUG-6 timeline_enabled 死代码 | 用户困惑 |
| **P2** | R-1/R-2 代码冗余 | 可读性/维护性 |
| **P2** | S-1 inference_loop 过长 | 可读性 |
| **P2** | R-3 日志工具重复 | DRY 原则 |
| **P3** | R-4/R-5 数据冗余存储 | 磁盘空间 |
| **P3** | S-4 缺少 config dataclass | 长期可维护性 |

---

## 六、实施计划

### Batch 1: P0 修复（2 项）

1. `policy_loader.py:307` — 添加 `weights_only=False`
2. `inference_ros.py:1045` — 改为 `--no_truncate_at_act_horizon` (store_false)

### Batch 2: P1 修复（3 项）

1. `inference_ros.py` — 删除 `_preprocess_image`/`_normalize_images`，传裸图像给 RealPolicy
2. `inference_ros.py:718` — 夹爪安全检查传入真实 qpos_joint
3. `inference_ros.py:1008-1011` — 删除 `--timeline_enabled`，仅保留 `--timeline_disabled`

### Batch 3: P2 清理（3 项）

1. `inference_ros.py` — `is_full_mode` 等索引移到 `__init__`
2. `inference_ros.py` — `_inference_loop` 拆分为 4 个私有方法
3. 新建 `utils/log_compat.py` 统一日志工具

### 暂不修改

- R-4/R-5：保持兼容性
- BUG-2：当前语义正确，仅需添加注释
- BUG-4：单线程调用，风险低
- BUG-5：功能正确
- S-4：范围太大，后续单独重构
