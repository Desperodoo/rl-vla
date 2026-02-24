# rl-vla 项目全局指令

## 项目概述
这是一个机器人学习项目 (rl-vla)，当前正在复现 VLAW 论文 (arXiv:2602.12063)。
项目核心是：在 ManiSkill3 仿真环境中，用 ShortCut Flow 策略 + Ctrl-World 视频扩散世界模型 + VLM 奖励模型实现策略与世界模型的迭代共同改进。

## 复现计划
完整计划见 [VLAW_REPRODUCTION_PLAN.md](VLAW_REPRODUCTION_PLAN.md)，涵盖 8 个阶段 (P0-P7)、GPU 分配、技术方案等。

## 项目状态
实时状态跟踪见 [vlaw-status.md](vlaw-status.md)。每个 Agent 在完成任务后应更新此文件。

## 代码库结构
```
rlft/                    ← 主代码包
  algorithms/il/         ← 模仿学习 (ShortCut Flow, Flow Matching)
  algorithms/online_rl/  ← 在线 RL (PLD-SAC, DSRL-SAC)
  buffers/               ← 数据缓冲区
  datasets/              ← 数据集加载
  envs/                  ← 环境封装 (ManiSkill)
  networks/              ← 网络模块
  roboreward/            ← RoboReward 模块 (参考用)
  vlaw/                  ← VLAW 新模块 (待实现)
  online/                ← 训练入口脚本

ctrl_world/              ← Ctrl-World 代码 (git submodule)
scripts/                 ← 辅助脚本
```

## 编码规范
- **语言**: Python 3.10+
- **类型提示**: 所有函数签名使用 type hints
- **配置**: 使用 `tyro` dataclass 管理超参数
- **日志**: 使用 `wandb` 记录实验
- **GPU**: 10 × RTX 4090 (24GB each)，使用 `CUDA_VISIBLE_DEVICES` 分配
- **训练框架**: PyTorch 2.x, HuggingFace Accelerate (多 GPU)
- **数据格式**: HDF5 (轨迹数据), safetensors (模型权重)
- **环境**: conda `rlft_ms3`

## 核心技术栈
- **策略**: ShortCut Flow (1D U-Net, flow matching, PlainConv 视觉编码器)
- **世界模型**: Ctrl-World (SVD UNet + VAE + CLIP + Action Encoder MLP)
- **奖励模型**: Qwen3-VL-4B/8B (二分类 P('yes') > α=0.8)
- **仿真环境**: ManiSkill3 (GPU 向量化, obs_mode="rgbd")
- **RL 基线**: PLD-SAC, DSRL-SAC (已实现)

## 协作约定
- 每个模块完成后，更新 `.github/vlaw-status.md` 中对应的状态
- checkpoint 保存路径: `checkpoints/vlaw/{module_name}/`
- 数据保存路径: `data/vlaw/{data_type}/`
- 代码修改前先阅读已有模块的接口定义
- 新增代码放在 `rlft/vlaw/` 下，入口脚本放在 `rlft/online/`
