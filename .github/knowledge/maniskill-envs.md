# ManiSkill 环境速查

> 精简版。详细 evaluate() 源码见 ManiSkill3 源码，直接 grep 即可。

---

## 任务与维度

| 任务 | obs_agent_dim (HDF5) | full_state_dim (env) | act_dim | T_mean |
|------|---------------------|---------------------|---------|--------|
| LiftPegUpright-v1 | **25** | 32 | 7 | 46.1 |
| PickCube-v1 | **29** | 42 | 7 | 42.6 |
| StackCube-v1 | **25** | 48 | 7 | 42.6 |

> ⚠️ **state_dim 因任务而异，代码中必须动态获取，不要硬编码！**
> HDF5 `obs_agent` 仅含 robot qpos+qvel，不含 extra。

---

## HDF5 数据字段

| 字段 | shape | dtype | 说明 |
|------|-------|-------|------|
| `rgb_base` | (T, 192, 192, 3) | uint8 | base_camera |
| `rgb_render` | (T, 192, 192, 3) | uint8 | render_camera |
| `state` / `obs_agent` | (T, D) | float32 | D 因任务而异 |
| `actions` | (T, 7) | float32 | delta pose + gripper |
| `env_success` | (T,) | bool | ManiSkill GT |
| `latent_concat` | (T, 4, 48, 24) | float16 | VAE latent (后处理写入) |

---

## 关键约定

- **success 语义**: `success_at_end = env_success[-1]`（VLAW 论文语义）
- **control_mode**: `pd_ee_delta_pose` → 7D action (xyz+euler+gripper)
- **相机**: 128×128, base_camera + render_camera, 垂直拼接 → VAE latent (4,32,16)  
  > ⚠️ 实际分辨率以 CollectorConfig.camera_width/height 为准
- **obs_horizon**: 2 | **act_steps**: 8 | **max_episode_steps**: 200

## ⚠️ Termination 行为 (BUG-024 / ADR-027)

> **所有 ManiSkill3 任务: success → terminated (early termination)**

- `BaseEnv.step()` L1054: `terminated = info["success"].clone()`
- 成功的 episode 在 success 瞬间立即结束 (terminated=True)
- 失败的 episode 跑满 max_episode_steps 后 truncated=True
- **结果**: 成功轨迹时长 << 失败轨迹时长
- **影响 data collection**:
  - 向量化并行 env 中，成功 ep 先完成 → 小量采集时 selection bias (全是成功)
  - 解决: 大量采集 (num_episodes >> num_envs)，让失败 ep 也有时间完成
- **LiftPegUpright-v1 具体数据** (AWSC policy):
  - 成功: ~10-120 步 (T=5-35 @ frame_skip=4)
  - 失败: 固定 200 步 (T=51 @ frame_skip=4)
  - 真实成功率: success_once=80%, success_at_end(200步)=46%

## VLM Prompt 要点

| 任务 | 判断要点 |
|------|---------|
| LiftPegUpright | peg 是否竖直站立在桌面 |
| PickCube | 红色方块是否在目标位置，机器人静止 |
| StackCube | 红方块叠在绿方块上，夹爪已松开，方块静止 |
