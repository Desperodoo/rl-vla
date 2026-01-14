# CARM 推理调试指南

本指南详细说明如何进行 **离线测试** 和 **真机渐进测试**，确保模型部署安全可靠。

## 📁 文件结构

```
carm_deploy/
├── inference_ros.py        # 主推理脚本（已集成安全控制和日志）
├── offline_test.py         # 离线测试脚本
├── safety_controller.py    # 安全控制器
├── inference_logger.py     # 推理日志记录器
└── INFERENCE_DEBUG_GUIDE.md # 本文档
```

---

## 🔬 第一阶段：离线测试（无需机器人）

离线测试在没有机器人的情况下验证模型推理流水线是否正确。

### 1.1 环境准备

```bash
# 使用训练环境 (包含 PyTorch)
conda activate arx-py310

# 进入部署目录
cd ~/rl-vla/catkin_ws/src/carm_deploy
```

### 1.2 运行离线测试

```bash
# 基本用法：评估模型在数据集上的表现
python offline_test.py \
    --model_path ~/rl-vla/rlft/diffusion_policy/runs/consistency_flow/checkpoints/latest.pt \
    --data_dir ~/rl-vla/recorded_data

# 指定输出目录
python offline_test.py \
    --model_path ~/rl-vla/rlft/diffusion_policy/runs/consistency_flow/checkpoints/latest.pt \
    --data_dir ~/rl-vla/recorded_data \
    --output_dir ~/rl-vla/offline_eval_results

# 只测试前 5 个 episode
python offline_test.py \
    --model_path ~/rl-vla/rlft/diffusion_policy/runs/consistency_flow/checkpoints/latest.pt \
    --data_dir ~/rl-vla/recorded_data \
    --max_episodes 5
```

### 1.3 输出内容

离线测试会生成：

1. **误差指标**（终端打印）：
   - 关节 MSE/MAE
   - 夹爪 MSE/MAE
   - 末端位姿 MSE/MAE

2. **可视化图表**（保存到 output_dir）：
   - `episode_XX_comparison.png` - 预测 vs 真值对比曲线
   - `error_distribution.png` - 误差分布直方图
   - `cumulative_error.png` - 累积误差曲线
   - `metrics.json` - 所有指标汇总

### 1.4 分析结果

**理想情况**：
- 关节 MAE < 0.05 rad
- 夹爪 MAE < 0.01
- 预测曲线与真值曲线高度吻合

**需要注意的问题**：
- 曲线有明显相位延迟 → 检查 obs_horizon 设置
- 全局偏移 → 检查 action 归一化/反归一化
- 某些关节误差大 → 检查该关节数据质量

---

## 🤖 第二阶段：真机渐进测试

真机测试分三个阶段，逐步增加风险：

### 阶段 2.1：干运行模式（Dry Run）

**特点**：
- 只进行推理，**不发送任何控制命令**
- 用于验证 ROS 通信、图像预处理、模型加载

```bash
# 确保 ROS 已启动
# Terminal 1: ROS core
roscore

# Terminal 2: 启动相机
roslaunch realsense2_camera rs_camera.launch

# Terminal 3: 干运行测试
conda activate carm
rosrun carm_deploy inference_ros.py \
    --pretrain ~/rl-vla/rlft/diffusion_policy/runs/consistency_flow/checkpoints/latest.pt \
    --dry_run \
    --data_dir ~/rl-vla/recorded_data
```

**检查点**：
- [ ] 模型加载成功
- [ ] 图像接收正常（Inference time 输出）
- [ ] 无异常错误日志

### 阶段 2.2：慢速模式（Slow Mode）

**特点**：
- 5Hz 推理频率（正常为 30Hz）
- 机器人移动更慢，有更多时间响应问题
- 日志记录所有动作

```bash
# 确保机器人已开机并处于安全位置
# 手握急停开关

rosrun carm_deploy inference_ros.py \
    --pretrain ~/rl-vla/rlft/diffusion_policy/runs/consistency_flow/checkpoints/latest.pt \
    --slow_mode \
    --data_dir ~/rl-vla/recorded_data \
    --log_dir ~/rl-vla/inference_logs/slow_test_1
```

**安全检查**：
- [ ] 机器人能正常移动到初始位置
- [ ] 关节运动范围合理（无碰撞）
- [ ] 夹爪开合正常
- [ ] 无安全警告日志

### 阶段 2.3：正常模式

**特点**：
- 30Hz 推理频率
- 全速执行

```bash
rosrun carm_deploy inference_ros.py \
    --pretrain ~/rl-vla/rlft/diffusion_policy/runs/consistency_flow/checkpoints/latest.pt \
    --data_dir ~/rl-vla/recorded_data \
    --log_dir ~/rl-vla/inference_logs/normal_test_1
```

---

## 📊 第三阶段：日志分析

### 3.1 查看推理日志

```python
from inference_logger import InferenceLogAnalyzer

# 加载日志
analyzer = InferenceLogAnalyzer("~/rl-vla/inference_logs/slow_test_1/inference_20250108_123456.h5")

# 打印统计信息
stats = analyzer.get_statistics()
print(f"总步数: {stats['total_steps']}")
print(f"平均推理时间: {stats['avg_inference_time']:.4f}s")
print(f"安全事件数: {stats['safety_event_count']}")

# 绘制关节轨迹
analyzer.plot_joint_trajectory(save_path="joint_trajectory.png")

# 绘制动作对比（如有真值）
analyzer.plot_action_comparison(save_path="action_comparison.png")
```

### 3.2 对比训练数据

```python
import h5py
import numpy as np
import matplotlib.pyplot as plt

# 加载推理日志
with h5py.File("inference_log.h5", 'r') as f:
    infer_qpos = f['qpos'][:]
    infer_actions = f['actions'][:]

# 加载训练数据（某个 episode）
with h5py.File("~/rl-vla/recorded_data/episode_0.hdf5", 'r') as f:
    train_qpos = f['observations/qpos'][:]
    train_actions = f['action'][:]

# 对比可视化
fig, axes = plt.subplots(7, 1, figsize=(12, 14))
for i in range(7):
    axes[i].plot(train_qpos[:, i], label='Training', alpha=0.7)
    axes[i].plot(infer_qpos[:len(train_qpos), i], label='Inference', alpha=0.7)
    axes[i].set_ylabel(f'Joint {i}')
    axes[i].legend()
plt.tight_layout()
plt.savefig("qpos_comparison.png")
```

---

## ⚠️ 安全注意事项

### 必须准备

1. **急停开关** - 随时可按下
2. **工作空间清理** - 移除危险物品
3. **操作人员** - 至少两人（一人操作，一人监控）

### 安全参数

安全控制器会自动从 `dataset_info.json` 加载关节限位，并扩展 10% 余量：

```json
{
  "joint_limits": {
    "min": [-0.1, -2.09, ...],
    "max": [3.24, 2.09, ...]
  }
}
```

### 异常处理

| 问题 | 解决方案 |
|------|----------|
| 关节超限警告 | 检查模型输出，可能需要 clip |
| 动作幅度过大 | 降低 `temporal_factor_k` |
| 推理延迟高 | 检查 GPU 使用，考虑减少 flow steps |
| 图像异常 | 检查相机话题，确认 RGB 格式 |

---

## 🛠️ 命令行参数参考

```bash
rosrun carm_deploy inference_ros.py --help
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--pretrain` | '' | 模型 checkpoint 路径 |
| `--dry_run` | False | 干运行模式，不执行动作 |
| `--slow_mode` | False | 慢速模式，5Hz |
| `--data_dir` | '' | 数据目录（用于加载安全限位） |
| `--safety_config` | '' | 自定义安全配置文件 |
| `--log_dir` | '~/rl-vla/inference_logs' | 日志保存目录 |
| `--desire_inference_freq` | 30 | 推理频率 |
| `--joint_cmd_mode` | False | 关节控制模式 |

---

## 📋 测试 Checklist

### 离线测试 ✅
- [ ] 模型加载成功
- [ ] 关节 MAE < 0.05 rad
- [ ] 预测曲线与真值吻合
- [ ] 无 NaN/Inf 输出

### 干运行测试 ✅
- [ ] ROS 通信正常
- [ ] 图像预处理正确
- [ ] 推理时间稳定 (<50ms)
- [ ] 无异常日志

### 慢速测试 ✅
- [ ] 机器人移动平稳
- [ ] 无关节超限
- [ ] 夹爪动作正常
- [ ] 任务基本完成

### 正常测试 ✅
- [ ] 任务完成率达标
- [ ] 动作流畅
- [ ] 无安全事件

---

## 🔗 相关文件

- [训练脚本](../../../rlft/diffusion_policy/train_carm.py)
- [数据采集脚本](../../../carm_real/record_data_surreal3576.py)
- [ROS 环境接口](./env_ros.py)
