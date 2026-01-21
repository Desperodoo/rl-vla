# 数据预处理对齐分析报告

## 概述

本报告对比分析了 `train_carm.py` (CARMDataset) 和 `train_finetune_from_realbot.py` (InferenceDataset) 两个训练脚本的数据预处理流程，确认它们是否对齐以支持混合训练。

**分析日期**: 2025-01-20  
**Checkpoint**: `/home/lizh/rl-vla/rlft/diffusion_policy/runs/consistency_flow_discrete_gripper_weight_0.02/checkpoints/latest.pt`

---

## 1. 数据格式对比

### 1.1 观测数据 (Observations)

| 组件 | CARMDataset | InferenceDataset | 状态 |
|------|-------------|------------------|------|
| RGB shape | `[obs_horizon, 3, 128, 128]` | `[obs_horizon, 3, 128, 128]` | ✅ 一致 |
| RGB dtype | `uint8` | `uint8` | ✅ 一致 |
| RGB range | `[0, 255]` | `[0, 255]` | ✅ 一致 |
| State shape | `[obs_horizon, 7]` | `[obs_horizon, 7]` | ✅ 一致 |
| State content | `qpos_joint` (6 joints + 1 gripper) | `qpos_joint` (6 joints + 1 gripper) | ✅ 一致 |

### 1.2 动作数据 (Actions)

| 组件 | CARMDataset | InferenceDataset | 状态 |
|------|-------------|------------------|------|
| actions_cont shape | `[pred_horizon, 13]` | `[pred_horizon, 13]` | ✅ 一致 |
| gripper_label shape | `[pred_horizon]` | `[pred_horizon]` | ✅ 一致 |
| 动作格式 | joint(6) + rel_pose(7) | joint(6) + rel_pose(7) | ✅ 一致 |
| Gripper 离散化阈值 | 0.05 | 0.05 | ✅ 一致 |

### 1.3 观测处理函数

两个 Dataset 使用**相同的** `create_carm_obs_process_fn()`：

```python
obs_process_fn = create_carm_obs_process_fn(
    output_format="NCHW",
    target_size=(128, 128),  # 或 (224, 224) 对于 ResNet
    normalize_images=True,
)
```

---

## 2. 关键差异分析

### 2.1 相对位姿计算方式

| 数据源 | 相对位姿来源 | 说明 |
|--------|--------------|------|
| **CARMDataset** | 动态计算 `compute_relative_pose_transform(ref_pose, target_pose)` | 从绝对位姿计算相对变换 |
| **InferenceDataset** | 直接使用 `action_intervened[:, 0, 7:14]` | 模型输出已经是相对位姿 |

这是**预期行为**：
- 遥操作数据记录的是**绝对目标位姿**，需要转换为相对位姿
- 推理数据记录的是**模型输出的相对位姿**，可以直接使用

### 2.2 数值范围统计

基于 200 个样本的统计：

| 维度 | Teleop Mean | Teleop Std | Inference Mean | Inference Std | Ratio |
|------|-------------|------------|----------------|---------------|-------|
| Joints [0:6] | 0.3086 | 0.8303 | 0.2742 | 0.8584 | ~1.0 |
| ‖rel_xyz‖ (m) | 0.0077 | 0.0062 | 0.0065 | 0.0034 | ~1.2 |
| rel_qw | 0.9998 | 0.0001 | **1.0145** | 0.0019 | ⚠️ |

### 2.3 四元数归一化问题

**发现问题**: 推理数据中的四元数 `qw > 1`，四元数范数 `‖quat‖ > 1`。

```
inference_episode_0001: qw range [1.0065, 1.0197], ‖quat‖ range [1.0067, 1.0198]
inference_episode_0003: qw range [0.9973, 1.0215], ‖quat‖ range [1.0009, 1.0219]
```

**原因**: 模型输出的四元数没有被强制归一化。

**影响**: 
- 推理时会在 `apply_relative_transform()` 中进行归一化处理，不影响执行
- Finetune 时，模型会学习到略大于 1 的 qw 值

**建议**: 
- 如果需要严格归一化，可以在 `InferenceDataset` 中添加四元数归一化步骤
- 当前偏差很小 (~1.5%)，通常不会显著影响训练

---

## 3. 预测时域分析

分析不同 `pred_horizon` 下相对位姿的数值范围：

| Horizon | Mean ‖xyz‖ (m) | Std ‖xyz‖ (m) | Max ‖xyz‖ (m) |
|---------|----------------|---------------|---------------|
| 1 | 0.0091 | 0.0097 | 0.0383 |
| 4 | 0.0107 | 0.0114 | 0.0531 |
| 8 | 0.0128 | 0.0138 | 0.0691 |
| 16 | 0.0170 | 0.0185 | 0.0959 |

**说明**: 随着预测时域增大，相对位姿的数值范围会增大，因为目标帧离观测帧越远，位姿变化越大。

---

## 4. 模型兼容性测试

### 4.1 前向传播测试

使用指定 checkpoint 加载模型，分别测试两种数据的前向传播：

```
Teleop obs_features shape: [1, 2, 512]
  visual_feat: [1, 2, 256], state_feat: [1, 2, 256]

Inference obs_features shape: [1, 2, 512]
  visual_feat: [1, 2, 256], state_feat: [1, 2, 256]

Teleop visual_feat stats: mean=0.4907, std=0.9995
Inference visual_feat stats: mean=0.4639, std=0.9639
```

**结论**: ✅ 两种数据源的特征提取流程完全兼容。

---

## 5. 结论与建议

### 5.1 对齐状态

| 检查项 | 状态 |
|--------|------|
| 数据格式 | ✅ 完全一致 |
| 观测处理 | ✅ 使用相同的 obs_process_fn |
| 动作格式 | ✅ 13D continuous + discrete gripper |
| 数值范围 | ⚠️ 在同一数量级，存在约 1.2x 差异 |
| 四元数归一化 | ⚠️ 推理数据 qw 略大于 1 |

### 5.2 建议

1. **可以安全进行混合训练**: 数据格式完全一致，数值范围差异在可接受范围内。

2. **使用 intervention weighting**: `train_finetune_from_realbot.py` 已实现基于 intervention 的样本加权，推荐使用。

3. **四元数处理** (可选):
   ```python
   # 在 InferenceDataset._load_inference_episode 中添加归一化
   rel_quat = actions[:, 10:14]
   rel_quat = rel_quat / np.linalg.norm(rel_quat, axis=1, keepdims=True)
   actions[:, 10:14] = rel_quat
   ```

4. **Action Normalization** (可选): 如果需要更严格的数值对齐，可以使用 `ActionNormalizer`。

---

## 附录：数据流对比

### CARMDataset (遥操作数据)

```
HDF5 file
├── action: [T, 15]         # 绝对位姿
│   ├── [0:6]   joints
│   ├── [6]     gripper
│   ├── [7:14]  absolute_end_pose  ← 需要转换
│   └── [14]    gripper
├── observations/
│   ├── images: [T, H, W, C]
│   ├── qpos_joint: [T, 7]
│   └── qpos_end: [T, 8]    ← 作为参考位姿

处理流程:
1. ref_pose = qpos_end[obs_frame, :7]
2. target_pose = action[t, 7:14]
3. relative_pose = compute_relative_pose_transform(ref_pose, target_pose)
4. actions_cont[t] = [joints(6), relative_pose(7)]
```

### InferenceDataset (推理数据)

```
HDF5 file
├── action_intervened: [T, pred_horizon, 15]  # 模型输出/干预后
│   └── [:, 0, :]   第一个预测步的动作
│       ├── [0:6]   joints
│       ├── [6]     gripper
│       ├── [7:14]  relative_pose  ← 已经是相对位姿！
│       └── [14]    gripper
├── intervention_mask: [T, pred_horizon, 15]
├── observations/
│   ├── images: [T, H, W, C]
│   ├── qpos_joint: [T, 7]
│   └── qpos_end: [T, 8]

处理流程:
1. actions = action_intervened[:, 0, :]
2. relative_pose = actions[t, 7:14]  # 直接使用
3. actions_cont[t] = [joints(6), relative_pose(7)]
```
