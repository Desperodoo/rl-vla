# 模块间接口契约

> 记录各模块的输入输出格式。修改接口时必须更新此文件。

---

## HDF5 轨迹数据格式（核心数据格式）

由 `rlft/vlaw/data_collector.py` 写入，所有模块均读此格式。

```
HDF5 文件
├── traj_0000/
│   ├── rgb_base:       (T, 192, 192, 3) uint8   — base_camera RGB
│   ├── rgb_render:     (T, 192, 192, 3) uint8   — hand_camera RGB
│   ├── state:          (T, D)           float32  — agent qpos+qvel（⚠️ D 因任务而异: LiftPegUpright=25, PickCube=29, StackCube=25）
│   ├── obs_agent:      (T, D)           float32  — 同 state，obs dict 里的 agent 字段（D 同上，按任务动态推断）
│   ├── actions:        (T, 7)           float32  — delta pose (xyz+euler+gripper)
│   ├── env_success:    (T,)             bool     — ManiSkill GT success
│   └── latent_concat:  (T, 4, 48, 24)  float16  — VAE latent（data_pipeline.py 写入）
│   attrs:
│       task_instruction: str
│       source_tag:       "real" | "synthetic"  (attrs key 实为 "source")
│       vlm_reward:       0 | 1（Reward-Agent 标注后写入）
│       success:          bool（env_success 的 OR reduce）
├── traj_0001/ ...
```

**关键约定**:
- `latent_concat` 由 `data_pipeline.py` 后处理写入，原始 rollout HDF5 中可能不存在
- `vlm_reward` 由 `reward_model.py` 标注后写入，imagination 生成的 HDF5 中可能不存在

---

## CtrlWorldAdapter.rollout() 接口

```python
# rlft/vlaw/ctrl_world_adapter.py
def rollout(
    obs_latents: Tensor,    # (num_history + num_frames, 4, 48, 24) — 历史帧 latent
    actions:     ndarray,   # (num_history + num_frames, 7)          — delta pose（未归一化）
    instruction: str = "",
) -> Tensor:                # (N_CAMS=2, num_frames, 4, 24, 24) float32
```

**注意**:
- `obs_latents` 前 `num_history(=4)` 帧作为历史，第 `num_history` 帧作为当前条件帧
- 返回值已按相机拆分（rearrange m=2,n=1），每个相机的 latent 尺寸是 24×24（非 48×24）
- 动作在内部归一化，需要 `data/vlaw/meta_info/maniskill/stat.json` 存在

---

## VLAWRewardModel.score_trajectory() 接口

```python
# rlft/vlaw/reward_model.py
def score_trajectory(
    frames:      List[PIL.Image] | ndarray,  # VLM 输入帧（均匀采样 16 帧）
    instruction: str,
) -> dict:
# 返回: {"p_yes": float, "reward": int(0或1), "threshold": 0.8}
```

**阈值**: `reward = 1 if p_yes > 0.8 else 0`（VLAW 论文 α=0.8）

---

## ShortCutFlowPolicy.get_actions() 接口

```python
# rlft/vlaw/data_collector.py
def get_actions(
    obs_features: Tensor,   # (N_env, obs_horizon=2, feat_dim)
) -> ndarray:               # (N_env, action_dim=7)
```

**obs_features 组成**: `concat([PlainConv(rgb), agent_state])` 沿 feat_dim 维度

---

## ImaginationEngine.rollout_single() 接口

```python
# rlft/vlaw/imagination.py
def rollout_single(
    initial_latent: Tensor,   # (4, 48, 24) — 真实帧 VAE encode 后的起始 latent
    initial_state:  ndarray,  # (D,) — 各任务不同: LiftPegUpright=25, PickCube=29, StackCube=25
    instruction:    str,
    task_id:        str,
) -> SyntheticTrajectory:
# SyntheticTrajectory.latents: (T, 4, 48, 24) float16
# SyntheticTrajectory.actions: (T, 7) float32
```

---

## 动作归一化约定

- 统计量路径: `data/vlaw/meta_info/maniskill/stat.json`
- 格式: `{"state_01": [7个p1分位数], "state_99": [7个p99分位数]}`
- 归一化公式: `norm = 2*(x - p01) / (p99 - p01 + 1e-8) - 1` → 值域 [-1, 1]
- 所有模块（dataset_maniskill, ctrl_world_adapter, policy_updater）均使用此文件

---

## 路径约定

| 数据类型 | 路径模板 |
|---------|---------|
| 演示 HDF5 (原始) | `data/vlaw/demos/{task_id}/` |
| 演示 HDF5 (VAE编码) | `data/vlaw/encoded/demos/{task_id}/` |
| 真实 Rollout | `data/vlaw/rollouts/iter{N}/` |
| 合成数据 | `data/vlaw/synthetic/iter{N}/` |
| 动作统计量 | `data/vlaw/meta_info/maniskill/stat.json` |
| WM 权重 (pretrained) | `checkpoints/vlaw/world_model/pretrained/` |
| WM 权重 (finetuned) | `checkpoints/vlaw/world_model/phase_a/` 或 `phase_b/` |
| VLM 权重 | `checkpoints/vlaw/reward_model/qwen_vl/` |
| State Predictor | `checkpoints/vlaw/state_predictor/` |
| 策略权重 (base) | `checkpoints/il/best_eval_success_once.pt` |
| 策略权重 (VLAW iter) | `checkpoints/vlaw/policy/iter{N}/` |
