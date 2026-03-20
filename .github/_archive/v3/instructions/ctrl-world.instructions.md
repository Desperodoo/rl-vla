---
name: 'Ctrl-World Code Standards'
description: 'Ctrl-World 子模块的修改规范'
applyTo: 'ctrl_world/**/*.py'
---

# Ctrl-World 代码修改规范

## 修改原则
- **最小修改**: Ctrl-World 是外部代码，应尽量少改动原始文件
- 优先通过新增文件(adapter/wrapper)来适配，而非直接修改核心模型代码
- 必须修改时，用 `# VLAW MODIFICATION:` 注释标记改动位置

## 需要修改的文件
1. `config.py` → 新增 `maniskill` task_type 和对应配置
2. `dataset/` → 新增 `dataset_maniskill.py` 数据加载器

## 不应修改的文件
- `models/ctrl_world.py` — 核心模型架构 (SVD UNet 本身不限定分辨率)
- `models/pipeline_stable_video_diffusion.py` — 扩散管线

## 关键参数
```python
# ManiSkill 适配参数
width = 192          # 或 128 (降级)
height = 384         # 2cam × 192 垂直拼接 (或 256)
num_frames = 5
num_history = 4      # 可从 6 降为 4 节省显存
action_dim = 7       # pd_ee_delta_pose
down_sample = 3      # 根据 ManiSkill 帧率调整
```

## 数据格式
- 输入: HDF5 (VAE latent + action + text)
- Action 编码: ManiSkill delta pose (xyz + euler + gripper) = 7D
- Action 归一化: 使用 ManiSkill 自己的统计量，不复用 DROID stat.json
