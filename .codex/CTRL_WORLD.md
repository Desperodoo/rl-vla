# Ctrl-World Modification Rules

`ctrl_world/` 是外部代码库。默认原则不是“在里面自由开发”，而是“最小修改、明确标注、优先封装”。

## 1. 核心原则

1. 不是本项目原生代码，改动要最小化。
2. ManiSkill / VLAW 适配逻辑优先写在 `rlft/vlaw/ctrl_world_adapter.py` 或等价封装层。
3. 只有当外部接口确实不够用时，才进入 `ctrl_world/` 内部修改。

## 2. 允许修改的文件

| 文件 | 允许的改动 |
|------|-----------|
| `ctrl_world/config.py` | 添加 `maniskill` task type 及其参数 |
| `ctrl_world/dataset/dataset_maniskill.py` | 新建 ManiSkill HDF5 数据加载器 |

## 3. 禁止修改的文件

- `ctrl_world/models/ctrl_world.py`
- `ctrl_world/models/pipeline_stable_video_diffusion.py`
- 任何已有 dataset 文件，如 `dataset_droid.py`
- 训练脚本 `train.py`

如需调整训练行为，应通过：
- 新配置
- 封装适配层
- 仓库侧调用脚本

不要直接改动 Ctrl-World 主干训练入口。

## 4. 修改标注要求

所有必要的 `ctrl_world/` 内部改动都要带清晰注释：

```python
# VLAW MODIFICATION: <reason>
```

## 5. ManiSkill 配置规格

```python
MANISKILL_CONFIG = {
    "width": 192,
    "height": 384,
    "num_frames": 5,
    "num_history": 4,
    "action_dim": 7,
    "down_sample": 3,
    "fps": 5,
}
```

关键含义：
- 双相机竖拼后分辨率为 `192 x 384`
- 历史帧数 `num_history=4`
- action 是 7D
- 频率从 ManiSkill 15Hz 下采样到约 5Hz

## 6. 数据格式差异

| 维度 | DROID 原始设定 | ManiSkill / 本项目 |
|------|----------------|--------------------|
| 分辨率 | 320x240 | 192x384 |
| action 语义 | 7D absolute | 7D delta pose |
| action stats | `stat.json` | 项目侧自行统计 |
| 相机数 | 多相机 | 两个固定相机 |

`dataset_maniskill.py` 应从：
- `data/vlaw/encoded/*.h5`

读取：
- `latent_concat`

并且：
- 不在 DataLoader 内在线跑 VAE 编码

## 7. 训练策略

### 7.1 Phase A

目标：
- 只训练 Action Encoder + temporal attention
- 让动作条件分支先收敛

参考命令：

```bash
conda activate ctrl_world
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 \
  rlft/vlaw/train_world_model.py --phase A --max_steps 10000
```

### 7.2 Phase B

目标：
- 进行全量微调

参考命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 \
  --deepspeed_config ctrl_world/deepspeed_zero2.json \
  rlft/vlaw/train_world_model.py --phase B --max_steps 50000
```

必须启用的显存优化：
- `gradient_checkpointing=True`
- fp16
- `gradient_accumulation_steps=8`
- `decode_chunk_size=4`

预期单卡显存：
- 约 20-22GB / RTX 4090

## 8. 质量评估

Action replay 只是最低层门槛，不等价于 imagination 可用。

Action replay 指标：
- PSNR > 18，目标 > 20
- SSIM > 0.7，目标 > 0.8
- LPIPS < 0.3，目标 < 0.2

额外要求：
- 必须生成 imagination 可视化
- 必须做人审
- 人审未通过时，不得把世界模型当作下游 imagination / policy 更新的 Go 信号

## 9. 与 BUG-D 相关的额外注意

当前已知最重要的结构性矛盾：
- 世界模型训练使用 absolute EE pose
- policy 输出 delta action

因此所有看起来“简单补一个转换”的方案都需要高度谨慎。以下方案已经被历史验证为不可靠或失败：
- 直接 delta 积分
- `pd_ee_pose` 控制模式迁移

当前更现实的方向：
- 动力学适配器
- 必要时小模型做 delta -> ee 映射
- 或保留最小 sim-in-loop 成本
