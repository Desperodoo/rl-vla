# ManiSkill 环境详细信息

> 最后更新: 2026-02-25 | 数据来源: HDF5 实测 + ManiSkill 源码分析 + data_collector.py

---

## 任务一览

| 任务 ID | 类别 | 难度 | 描述 |
|---------|------|------|------|
| LiftPegUpright-v1 | Tabletop | 中 | 将 peg 竖直放置到桌面上（正立） |
| PickCube-v1 | Tabletop | 低 | 将红色方块移动到目标位置 |
| StackCube-v1 | Tabletop | 高 | 将红色方块叠放在绿色方块上（需松手） |

---

## 观测空间与 state_dim（实测）

### 两种 state_dim 的区别

⚠️ **重要**: HDF5 中存储的 `state`/`obs_agent` 字段只包含 **robot qpos+qvel**，
而 ManiSkill `obs_mode="state"` 环境返回的完整 state 还包含 `extra`（任务相关信息）。

| 任务 | obs_agent_dim（HDF5实测） | full_state_dim（ManiSkill env） | 差值（extra部分） |
|------|--------------------------|--------------------------------|-------------------|
| LiftPegUpright-v1 | **25** | 32 | 7 |
| PickCube-v1 | **29** | 42 | 13 |
| StackCube-v1 | **25** | 48 | 23 |

> **策略训练使用 `obs_agent_dim`（HDF5字段）**，不含 extra，因为 collector 只保存 agent 状态。
> 代码中必须动态获取，**不要硬编码**！

### 详细 obs 结构（ManiSkill obs_mode="state"）

```
obs = {
  "agent": {
    "qpos": (N, qpos_dim),   # 关节位置
    "qvel": (N, qvel_dim),   # 关节速度
  },
  "extra": {
    "tcp_pose": (N, 7),      # 工具中心点（EE）位姿 [xyz + quat]
    ...                       # 任务相关额外信息（goal_pos, is_grasped 等）
  }
}
```

> 注：full_state = concat(qpos, qvel, extra)，维度见上表

---

## HDF5 数据集结构（实测）

### 路径约定

```
data/vlaw/rollouts/iter{i}/{task_id}/{task_id}_real_{timestamp}.h5
data/vlaw/encoded/rollouts/iter{i}/{task_id}/{task_id}_real_{timestamp}.h5  # 编码后
data/vlaw/labeled/iter{i}/{task_id}/{task_id}_vlm_rewards.h5               # VLM标注后
```

### traj 组字段（实测，iter1）

| 字段 | shape | dtype | 含义 |
|------|-------|-------|------|
| `rgb_base` | (T, 192, 192, 3) | uint8 | base_camera RGB |
| `rgb_render` | (T, 192, 192, 3) | uint8 | render_camera（env.render()输出） |
| `state` | (T, D) | float32 | robot agent qpos+qvel，D 因任务而异 |
| `obs_agent` | (T, D) | float32 | 同 state（两个键相同内容） |
| `actions` | (T, 7) | float32 | delta pose + gripper（7维） |
| `env_success` | (T,) | bool | 每步 success 标志 |

> T = 轨迹步数（含最后一步），T 因轨迹而异（min~1, max~200）

### 各任务实测 T 范围（iter1，50条轨迹）

| 任务 | state_dim | act_dim | T_min | T_max | T_mean |
|------|-----------|---------|-------|-------|--------|
| LiftPegUpright-v1 | 25 | 7 | 7 | 67 | 46.1 |
| PickCube-v1 | 29 | 7 | 1 | 66 | 42.6 |
| StackCube-v1 | 25 | 7 | 1 | 66 | 42.6 |

---

## success 定义

### 两种统计口径

- **success_once** (`np.any(env_success)`)：轨迹中**任意步骤**达成目标
- **success_at_end** (`env_success[-1]`)：**最终步骤**达成目标

### VLAW 语义

- **论文语义**：使用 `success_at_end`（任务完成 = 最终状态达标）
- **VLM 标注应参考**：`success_at_end`，prompt 应聚焦最终帧
- **注意**：`env_success` 为 `bool` 数组，不是 float

### 各任务实测成功率（Iter 1，50条轨迹）

> ⚠️ Iter 1 是策略初始迭代，成功率极低属正常现象

| 任务 | success_once | success_at_end |
|------|-------------|----------------|
| LiftPegUpright-v1 | 8/50 (16%) | 8/50 (16%) |
| PickCube-v1 | 0/50 (0%) | 0/50 (0%) |
| StackCube-v1 | 0/50 (0%) | 0/50 (0%) |

> LiftPegUpright-v1 的 success_once == success_at_end，说明成功后策略会持续保持成功状态（peg 已竖立后不会倒下）。

---

## 相机配置

| 相机名 | HDF5 键 | 分辨率 | 视角 | 来源 |
|--------|---------|--------|------|------|
| base_camera | `rgb_base` | 192×192 | 固定侧视/俯视 | ManiSkill obs sensor_data |
| render_camera | `rgb_render` | 192×192 | 渲染视角（env.render） | render_mode="rgb_array" |

> 配置来源：`rlft/vlaw/data_collector.py` L124-L127, L214-L220
> VAE latent 对应：192×192 → (4, 48, 24)（参见 ctrl_world VAE 配置）

---

## 控制配置

| 参数 | 值 | 来源 |
|------|-----|------|
| control_mode | `pd_ee_delta_pose` | data_collector.py L139 |
| action_dim | **7**（xyz_delta×3 + euler_delta×3 + gripper×1） | HDF5实测 |
| obs_horizon | 2（策略输入历史帧数） | data_collector.py L142 |
| act_steps | 8（每次策略调用执行步数） | data_collector.py L145 |
| max_episode_steps | 200 | data_collector.py L130 |
| obs_mode | `rgbd` | data_collector.py L357 |
| camera 分辨率 | 192×192 | data_collector.py L124-L127 |

> ⚠️ 注意：`gym.make(task, obs_mode="state")` 默认返回 **8维** action space（使用默认 control_mode），
> 而实际数据收集使用 `pd_ee_delta_pose`（**7维**）。两者不同，以 HDF5 数据为准。

---

## success 源码定义（精简）

### LiftPegUpright-v1

```python
def evaluate(self):
    q = self.peg.pose.q
    qmat = rotation_conversions.quaternion_to_matrix(q)
    euler = rotation_conversions.matrix_to_euler_angles(qmat, "XYZ")
    is_peg_upright = (
        torch.abs(torch.abs(euler[:, 2]) - np.pi / 2) < 0.08
    )  # 允许 0.08 rad 误差
    close_to_table = torch.abs(self.peg.pose.p[:, 2] - self.peg_half_length) < 0.005
    return {"success": is_peg_upright & close_to_table}
```

> **含义**：peg 沿 Z 轴方向接近竖直（绕 Z 轴转角接近 ±π/2），且底部靠近桌面

### PickCube-v1

```python
def evaluate(self):
    is_obj_placed = (
        torch.linalg.norm(self.goal_site.pose.p - self.cube.pose.p, axis=1)
        <= self.goal_thresh
    )
    is_grasped = self.agent.is_grasping(self.cube)
    is_robot_static = self.agent.is_static(0.2)
    return {
        "success": is_obj_placed & is_robot_static,
        "is_obj_placed": is_obj_placed,
        "is_robot_static": is_robot_static,
        "is_grasped": is_grasped,
    }
```

> **含义**：方块到达目标位置（goal_site 附近）且机器人静止。**注意：到达目标后需松开并静止**，
> 单纯抓住不算成功！

### StackCube-v1

```python
def evaluate(self):
    # cubeA（红方块）在 cubeB（绿方块）上方
    pos_A = self.cubeA.pose.p
    pos_B = self.cubeB.pose.p
    offset = pos_A - pos_B
    xy_flag = (
        torch.linalg.norm(offset[..., :2], axis=1)
        <= torch.linalg.norm(self.cube_half_size[:2]) + 0.005
    )
    z_flag = torch.abs(offset[..., 2] - self.cube_half_size[..., 2] * 2) <= 0.005
    is_cubeA_on_cubeB = torch.logical_and(xy_flag, z_flag)
    is_cubeA_static = self.cubeA.is_static(lin_thresh=1e-2, ang_thresh=0.5)
    is_cubeA_grasped = self.agent.is_grasping(self.cubeA)
    success = is_cubeA_on_cubeB * is_cubeA_static * (~is_cubeA_grasped)
    return {
        "is_cubeA_grasped": is_cubeA_grasped,
        "is_cubeA_on_cubeB": is_cubeA_on_cubeB,
        "is_cubeA_static": is_cubeA_static,
        "success": success.bool(),
    }
```

> **含义**：红方块叠在绿方块正上方，**且已松开夹爪**，且方块保持静止。
> 这是三任务中最难的：需精确叠放 + 松手 + 等待稳定。

---

## VLM 标注 Prompt 设计建议

基于 evaluate() 源码，各任务的 VLM 判断要点：

| 任务 | 关键视觉判断 | prompt 要点 |
|------|-------------|------------|
| LiftPegUpright-v1 | peg 是否竖直站立在桌面 | "Is the peg standing upright on the table?" |
| PickCube-v1 | 红色方块是否到达目标位置（goal marker） | "Is the red cube placed at the goal position and the robot is static?" |
| StackCube-v1 | 红方块是否叠在绿方块上，夹爪是否已松开 | "Is the red cube stacked on top of the green cube with the gripper open?" |

---

## 已知注意事项

1. **state_dim 因任务而异**：LiftPegUpright=25, PickCube=29, StackCube=25，代码中必须动态获取
2. **HDF5 state vs ManiSkill full_state**：HDF5 `obs_agent` 仅包含 robot qpos+qvel（25/29/25），不含 extra；ManiSkill obs_mode="state" 返回 32/42/48（含extra）
3. **action_dim=7**（来自 pd_ee_delta_pose），而 `gym.make(..., obs_mode="state")` 默认 action_space 为8维（不同 control_mode），以数据为准
4. **env_success** 为 `bool` 数组，不是 `float`
5. **推荐使用 success_at_end** 作为 Ground Truth 标签
6. **VLM prompt 应聚焦最终帧**（对应 success_at_end 语义）
7. **相机固定分辨率 192×192**，对应 VAE latent (4, 48, 24)
8. **StackCube 最难**：需叠放 + 松手 + 等待稳定，三个条件同时满足
9. **PickCube success 需松手**：`is_robot_static` 隐含需要完成放置动作后停止
10. **Iter 1 成功率极低**（0-16%）属于正常初始状态，VLAW 迭代训练后应显著提升
