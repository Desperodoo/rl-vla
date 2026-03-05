# 模块间接口契约

---

## HDF5 轨迹格式

```
traj_XXXX/
  rgb_base:       (T, 192, 192, 3) uint8
  rgb_render:     (T, 192, 192, 3) uint8
  state:          (T, D)           float32   ← D 因任务而异 (25/29/25)
  actions:        (T, 7)           float32
  env_success:    (T,)             bool
  latent_concat:  (T, 4, 48, 24)  float16   ← data_pipeline 写入
  attrs: task_instruction(str), source("real"|"synthetic"), vlm_reward(0|1), success(bool)
```

## CtrlWorldAdapter.rollout()

```python
def rollout(obs_latents: Tensor,  # (num_history+num_frames, 4, 48, 24)
            actions: ndarray,     # (num_history+num_frames, 7)
            instruction: str = "") -> Tensor:  # (2, num_frames, 4, 24, 24)
```

## VLAWRewardModel.score_trajectory()

```python
def score_trajectory(frames: List[PIL.Image] | ndarray,  # 16帧
                     instruction: str) -> dict:  # {"p_yes": float, "reward": 0|1}
```

## 动作归一化

- 统计量: `data/vlaw/meta_info/maniskill/stat.json` → `{state_01, state_99}` (7D p01/p99)
- 公式: `norm = 2*(x - p01) / (p99 - p01 + 1e-8) - 1`

## 路径约定

| 类型 | 路径 |
|------|------|
| Rollout (高成功率) | `data/vlaw/rollouts/high_suc/{task}/` → `encoded/train/{task}/` |
| Rollout (混合) | `data/vlaw/rollouts/mixed/{task}/` → `encoded/train/{task}/` |
| Rollout (评估) | `data/vlaw/rollouts/eval/{task}/` → `encoded/eval/eval_set.h5` |
| 合成 | `data/vlaw/synthetic/iter{N}/` |
| 标注 | `data/vlaw/labeled/iter{N}/` |
| 统计量 | `data/vlaw/meta_info/maniskill/stat.json` |
| WM ckpt | `checkpoints/vlaw/world_model/{pretrained,iter1}/` |
| VLM ckpt | `checkpoints/vlaw/reward_model/{qwen_vl,lora_iter1}/` |
| Policy ckpt | `checkpoints/vlaw/policy/iter{N}/` |

> ⚠️ 旧路径 `demos/`, `encoded/demos/` 已废弃 (BUG-020). 详见 [VLAW_FRESH_START_PLAN.md](../VLAW_FRESH_START_PLAN.md).

---

## VAE 编码流程 & 验证标准

**编码脚本**: `scripts/vlaw/encode_v2_data.py` (conda env: `rlft_ms3`)

**流程**: 
1. 加载 HDF5 中 `rgb_base` (T,192,192,3) + `rgb_render` (T,192,192,3)
2. Resize 128→192 (若需要), 垂直拼接 → (T, 384, 192, 3)
3. VAE encode → latent (T, 4, 48, 24) fp16
4. 保存为 `.pt` 文件 (含 `latent_concat`, `actions`, `state`, `env_success`)

**验证标准** (编码后必检):
- latent shape: `(T, 4, 48, 24)`, dtype=fp16
- **top-bot-diff > 0.5**: `(latent[:, :, :24, :] - latent[:, :, 24:, :]).abs().mean()`.  正常值 ~0.85-0.90. 若 <0.1 说明双相机坍塌 (BUG-020)
- 文件大小: 每条轨迹 ~0.5-1.5MB (取决于 T)

---

## VLM 训练/评估配置 (v3 已验证)

**训练脚本**: `rlft/vlaw/reward/train_reward_model.py`
**评估 (threshold sweep)**: `rlft/vlaw/reward/eval_threshold_ablation.py`

**v3 生产配置 (ADR-028 验证)**:
```bash
# 训练
--train-steps 300 --per-device-batch-size 1 --gradient-accumulation-steps 16
--lora-r 16 --lora-alpha 32 --num-frames 16 --video-fps 2.0
--lr 2e-5 --warmup-steps 20 --eval-steps 50 --eval-ratio 0.15 --threshold 0.8

# 推理 (Imagination 标注)
# α=0.5 (平衡): recall=85.9%, FP=12.6%
# α=0.8 (保守): recall=61.2%, FP=0%
```

**数据要求**: eval 集正样本比例 30%-60% (ADR-029).  v3: 85正/95负 (47%)

---

## v3 数据收集参数 (ADR-026/027 验证)

```bash
# collector.py 配置
frame_skip=4          # 20Hz/4 = 5Hz, 精确匹配 WM 预训练 (DROID 15Hz/3)
max_episode_steps=200 # 覆盖 ManiSkill 默认 50
num_envs=64           # GPU vec env 并行
min_traj_length=5     # frame_skip=4 下短成功轨迹可能 T 很小
num_episodes=1200     # 消除 selection bias (BUG-024)
```

**预期分布**: success_at_end ~46% (AWSC checkpoint).  若 100% 或 0% 说明有问题。
