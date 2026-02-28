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
| 演示 | `data/vlaw/demos/{task}/` → `encoded/demos/{task}/` |
| Rollout | `data/vlaw/rollouts/iter{N}/` → `encoded/rollouts/iter{N}/` |
| 合成 | `data/vlaw/synthetic/iter{N}/` |
| 统计量 | `data/vlaw/meta_info/maniskill/stat.json` |
| WM ckpt | `checkpoints/vlaw/world_model/{pretrained,iter1}/` |
| VLM ckpt | `checkpoints/vlaw/reward_model/{qwen_vl,lora_iter1}/` |
| Policy ckpt | `checkpoints/vlaw/policy/iter{N}/` |
