# ctrl_world — Ctrl-World 修改规范

这是外部代码库（原始 Ctrl-World 项目），遵循**最小修改原则**。Claude Code 进入此目录时自动加载本文件。

## 核心原则

**不是本项目代码，改动要最小化。** 适配 ManiSkill 的逻辑尽量写在 `rlft/vlaw/ctrl_world_adapter.py`（封装层），而非直接修改 ctrl_world 内部。

## 允许修改的文件

| 文件 | 允许的改动 |
|------|----------|
| `config.py` | 添加 `maniskill` task type 及对应参数 |
| `dataset/dataset_maniskill.py` | **新建**：ManiSkill HDF5 数据加载器 |

## 禁止修改的文件

- `models/ctrl_world.py` — SVD UNet 核心模型
- `models/pipeline_stable_video_diffusion.py` — 推理 pipeline
- 任何已有 dataset 文件（`dataset_droid.py` 等）
- 训练脚本（`train.py`）— 通过配置或封装调用，不改脚本

## 必须标注修改

所有必要改动行必须注释：

```python
# VLAW MODIFICATION: <原因简述>
```

## ManiSkill 关键参数

```python
# ctrl_world/config.py 中的 ManiSkill 配置
MANISKILL_CONFIG = {
    "width": 192,
    "height": 384,          # 双相机竖拼（ADR-002）
    "num_frames": 5,        # 每次预测帧数
    "num_history": 4,       # 历史帧数
    "action_dim": 7,        # delta pose: xyz(3) + euler(3) + gripper(1)
    "down_sample": 3,       # ManiSkill 15Hz → ~5Hz
    "fps": 5,
}
```

## 数据格式（与 DROID 的关键差异）

| 参数 | DROID（原始） | ManiSkill（本项目） |
|------|-------------|-------------------|
| 分辨率 | 320×240 | 192×384（竖拼） |
| Action | 7D absolute | 7D **delta** pose |
| Action stats | `stat.json`（DROID） | ManiSkill 自己计算 |
| 相机数 | 多相机 | 2 个固定相机 |

`dataset/dataset_maniskill.py` 从 `data/vlaw/encoded/*.h5` 读取 `latent_concat`（已预处理的 VAE latent），**不**在线做 VAE 编码。

## 训练启动（参考）

```bash
# Phase A：只训练 Action Encoder + temporal attention（~10K steps）
conda activate ctrl_world
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 \
  rlft/vlaw/train_world_model.py --phase A --max_steps 10000

# Phase B：全量 finetune（DeepSpeed ZeRO-2，~20K-50K steps）
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 \
  --deepspeed_config ctrl_world/deepspeed_zero2.json \
  rlft/vlaw/train_world_model.py --phase B --max_steps 50000
```

内存优化（Phase B 必须）：`gradient_checkpointing=True`，fp16，`gradient_accumulation_steps=8`，`decode_chunk_size=4`。预期显存：~20-22GB / 4090。
