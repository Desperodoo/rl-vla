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
- **相机**: 192×192, base_camera + render_camera, 垂直拼接 → VAE latent (4,48,24)
- **obs_horizon**: 2 | **act_steps**: 8 | **max_episode_steps**: 200

## VLM Prompt 要点

| 任务 | 判断要点 |
|------|---------|
| LiftPegUpright | peg 是否竖直站立在桌面 |
| PickCube | 红色方块是否在目标位置，机器人静止 |
| StackCube | 红方块叠在绿方块上，夹爪已松开，方块静止 |
