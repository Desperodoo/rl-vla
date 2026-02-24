# ManiSkill Demo 数据使用规范

适用范围: `rlft/vlaw/demo_prep.py`, `rlft/datasets/maniskill_dataset.py`, `rlft/online/train_rlpd.py`

---

## 1. Demo 数据存储路径

ManiSkill 官方 demo 存储在:
```
~/.maniskill/demos/<task_id>/rl/
    trajectory.none.<control_mode>.<backend>.h5       ← 原始轨迹 (无观测)
    trajectory.rgb.<control_mode>.<backend>.h5        ← 含 RGB 观测 (需先 replay)
    trajectory.state.<control_mode>.<backend>.h5      ← 含 state 观测 (需先 replay)
    trajectory.rgbd.<control_mode>.<backend>.h5       ← 含 RGBD 观测 (需先 replay)
    ppo_<control_mode>_ckpt.pt                        ← PPO 预训练权重
```

已默认下载的任务 (截至 2026-02):
```
AnymalC-Reach-v1  DrawTriangle-v1  LiftPegUpright-v1  PegInsertionSide-v1
PickCube-v1  PlugCharger-v1  PokeCube-v1  PullCube-v1  PullCubeTool-v1
PushCube-v1  PushT-v1  RollBall-v1  StackCube-v1  StackPyramid-v1
TwoRobotPickCube-v1  TwoRobotStackCube-v1
```

---

## 2. 文件格式说明

### 关键规则
- **`trajectory.none.*` 文件无 `obs` 键**，只含 actions、env_states，**不能直接用于策略训练**
- **必须使用 `trajectory.rgb.*` 或 `trajectory.rgbd.*`** 才能获取图像观测
- LiftPegUpright-v1 已有 `trajectory.rgb.*` — 其他任务只有 `trajectory.none.*` 需先 replay

### trajectory.rgb.*.h5 内部结构
```
traj_i/
    obs/
        agent/
            qpos:        (T, N_joints)      # 关节位置
            qvel:        (T, N_joints)      # 关节速度
        extra/
            tcp_pose:    (T, 7)             # 工具中心点姿态 [xyz + quat]
            goal_pos:    (T, 3)             # 目标位置 (可选，任务相关)
            is_grasped:  (T, 1)             # 抓取状态 (可选)
        sensor_data/
            base_camera/
                rgb:     (T, 128, 128, 3)   # 只有 base_camera！无 hand_camera
    actions:             (T-1, 7)           # 注意: 比 obs 少一帧！
    terminated:          (T-1,)
    truncated:           (T-1,)
    success:             (T-1,)             # 每步成功标志
    rewards:             (T-1,)
    env_states:          (T, ...)           # 环境状态 (用于 reset)
```

### 重要注意事项
- **T vs T-1 不对齐**: obs 有 T 步，actions 有 T-1 步，使用时需截断对齐
- **单相机**: 标准 replay 得到的 demo 只有 `base_camera`(128×128)，**没有 hand_camera**
- **默认分辨率**: 128×128。VLAW 需要 192×192 → 需做 PIL bilinear resize
- **任务数量**: LiftPegUpright 有 669 条轨迹，远超 VLAW 论文所需 25 条

---

## 3. 生成 rgb demo (已有 none 文件 → replay)

```bash
# 下载 (若未下载)
python -m mani_skill.utils.download_demo PickCube-v1

# Replay → trajectory.rgb.*.h5
CUDA_VISIBLE_DEVICES=4 python -m mani_skill.trajectory.replay_trajectory \
    --traj-path ~/.maniskill/demos/PickCube-v1/rl/trajectory.none.pd_ee_delta_pose.physx_cuda.h5 \
    -o rgb \
    -c pd_ee_delta_pose \
    -b physx_cuda \
    -n 64 \
    --record-rewards \
    --reward-mode dense \
    --use-first-env-state \
    --save-traj
# 输出: ~/.maniskill/demos/PickCube-v1/rl/trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5
```

或使用项目脚本:
```bash
bash scripts/replay_demos.sh PickCube-v1 pd_ee_delta_pose 64
```

---

## 4. 转换为 VLAW HDF5 格式 (demo_prep.py)

```bash
# 直接转换 (LiftPegUpright 已有 rgb demo)
CUDA_VISIBLE_DEVICES=4 python -m rlft.vlaw.demo_prep \
    --env_id LiftPegUpright-v1 \
    --num_trajs 25 \
    --target_hw 192 \
    --frame_skip 3 \
    --output_dir data/vlaw/demos

# 自动 replay 后转换 (PickCube-v1 等)
CUDA_VISIBLE_DEVICES=4 python -m rlft.vlaw.demo_prep \
    --env_id PickCube-v1 \
    --num_trajs 25 \
    --target_hw 192 \
    --auto_replay \
    --num_envs 64 \
    --output_dir data/vlaw/demos
```

### 转换后 VLAW HDF5 格式
```
data/vlaw/demos/<task_id>/<task_id>_demo_<ts>.h5
    traj_0000/
        rgb_base    (T, 192, 192, 3) uint8   — base_camera (resize 自 128)
        rgb_render  (T, 192, 192, 3) uint8   — 同 rgb_base (demo 单相机)
        state       (T, 25) float32          — qpos+qvel+tcp_pose
        obs_agent   (T, 25) float32          — 同 state
        actions     (T, 7) float32
        env_success (T,) bool
    meta/
        num_trajectories  int
        success_rate      float
        env_id            str
        camera_hw         "192,192"
        source            "demo"
        original_h5       str               — 源文件路径 (可追溯)
```

---

## 5. 现有管线对 demo 的使用方式

### train_rlpd.py — offline demo mixing (RLPD)
```python
from rlft.datasets import OfflineRLDataset
offline_dataset = OfflineRLDataset(
    data_path="~/.maniskill/demos/LiftPegUpright-v1/rl/trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5",
    include_rgb=True,
    num_traj=None,           # None = 全部
    obs_horizon=2,
    pred_horizon=8,
    act_horizon=8,
    control_mode="pd_ee_delta_pose",
    env_id="LiftPegUpright-v1",
    rgb_format="NCHW",       # 内部自动转换
    gamma=0.9,
    device=device,
)
```
- 使用 `ManiSkillDataset` / `OfflineRLDataset` 加载 `trajectory.rgb.*.h5`
- `obs_process_fn` 由 `create_obs_process_fn(env_id)` 生成，自动拼接所有相机

### train_pld.py — offline demo collection (PLD)
- PLD 使用 base policy rollout 作为 offline demo（不加载已有 demo 文件）
- `_collect_offline_demos()` 在真实环境中采集 50 条轨迹存入 replay buffer

---

## 6. VLAW P1.3 所需任务列表

| 任务 | rgb demo 状态 | 轨迹数 | 备注 |
|------|--------------|--------|------|
| LiftPegUpright-v1 | **✅ 已就绪** | 669 条 | base_camera 128×128 |
| PickCube-v1 | ⚠️ 仅 none | 需 replay | — |
| StackCube-v1 | ⚠️ 仅 none | 需 replay | — |

**P1.3 建议**: 先以 LiftPegUpright-v1 为首要任务，转换 25 条 demo 即可。

---

## 7. 关键坑记录

| 问题 | 解决方案 |
|------|---------|
| `trajectory.none.*.h5` → `KeyError: 'obs'` | 必须用 `trajectory.rgb.*.h5`，先 replay |
| obs 有 T 步，actions 有 T-1 步 | 转换时统一截为 T-1 步对齐 |
| demo 图像 128×128 vs VLAW 需要 192×192 | `demo_prep.py` 自动 PIL bilinear resize |
| demo 只有 `base_camera`，无 `hand_camera` | VLAW 格式中 `rgb_render = rgb_base.copy()` |
| `OfflineRLDataset` 要求 obs_mode != "none" | 只用 trajectory.rgb/rgbd/state 文件 |
