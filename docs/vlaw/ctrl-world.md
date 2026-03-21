# Ctrl-World Action Adapter 技术报告

> 本文档整理 Ctrl-World 世界模型的 Action Conditioning 机制和 Dynamics Adapter 架构对比分析。

---

## 1. Ctrl-World Action Conditioning 架构

### 1.1 核心设计

Ctrl-World 使用 **7D 笛卡尔空间条件 (Cartesian-space condition)** 作为 action conditioning：

```
action_cond = [tcp_x, tcp_y, tcp_z, euler_rx, euler_ry, euler_rz, gripper_norm]
```

- **不是** joint-space action
- **不是** delta action
- **是** 绝对末端执行器位姿 + 夹爪开度

### 1.2 WM Action Encoder 架构

源码位置: `ctrl_world/models/ctrl_world.py:71-107`

```python
class Action_encoder2(nn.Module):
    def __init__(self, action_dim, action_num, hidden_size, text_cond=True):
        input_dim = int(action_dim)  # 7
        self.action_encode = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
            nn.SiLU(),
            nn.Linear(1024, 1024)
        )
```

| 参数 | 值 | 说明 |
|------|----|----|
| `action_dim` | 7 | EE 笛卡尔位姿 + gripper |
| `action_num` | `num_history + num_frames` | 条件帧数 |
| `hidden_size` | 1024 | 编码维度 |
| 支持 CFG | ✅ | 5% 概率 zero out |

### 1.3 条件注入方式

```python
# ctrl_world/models/ctrl_world.py:170-173
action = batch['action']  # (B, f, 7)
action_hidden = self.action_encoder(action, texts, ...)  # (B, f, 1024)
# 注入 UNet
model_pred = self.unet(..., encoder_hidden_states=action_hidden, ...)
```

条件通过 `encoder_hidden_states` 传入 SVD UNet 的 cross-attention 层。

---

## 2. DROID Action Adapter (原始实现)

### 2.1 问题背景

DROID 数据集的 policy 输出是 **joint velocity** (7D)，而 WM 需要 **Cartesian EE pose** (7D)。需要一个适配器进行转换。

### 2.2 DROID Dynamics MLP 架构

源码位置: `ctrl_world/models/action_adapter/train2.py:38-95`

```python
class Dynamics(nn.Module):
    def __init__(self, action_dim=7, action_num=15, hidden_size=512):
        input_dim = int(action_dim * (action_num + 1))  # 7 * 16 = 112
        output_dim = int(action_num * action_dim)       # 15 * 7 = 105

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, output_dim),
        )
```

### 2.3 DROID Adapter 数据流

```
┌─────────────────────────────────────────────────────────────────────┐
│                   DROID Action Adapter Pipeline                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Policy Output       Dynamics MLP          FK            WM Input   │
│  ─────────────       ────────────       ────────       ──────────   │
│                                                                     │
│  joint_vel (7D)  ──► future_joint_pos ──► xyz+euler ──► action_cond │
│  ↓                    (15 steps)          (15 x 6D)    (15 x 7D)    │
│  gripper (1D)   ─────────────────────────────────────► gripper_norm │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**关键点**:
1. Adapter **只处理 arm 部分** (joint velocity → joint delta → forward kinematics)
2. Gripper **绕过 Adapter**，直接从 policy 输出拼接
3. FK 将 joint positions 转换为 xyz + euler

### 2.4 输入/输出规格

| 属性 | 值 | 说明 |
|------|----|----|
| Input | `(B, 1, 7)` + `(B, 15, 7)` | current_joint + joint_velocities |
| Output | `(B, 15, 7)` | future joint positions (deltas) |
| 归一化 | 1%/99% percentile | `joint_vel_01/99`, `joint_delta_01/99` |
| 训练数据 | DROID parquet | 150K samples |

---

## 3. ManiSkill Dynamics Adapter (本项目实现)

### 3.1 问题背景 (BUG-D)

ManiSkill 使用 **pd_ee_delta_pose** 控制模式:
- Policy 输出: `delta_xyz, delta_euler, gripper` (7D)
- WM 需要: `abs_xyz, abs_euler, gripper_norm` (7D)

**BUG-D 根本矛盾**: 从 delta action 到绝对 EE pose 的转换需要 PD 控制器仿真，而 Imagination 的目标正是绕开仿真。

### 3.2 我们的 Dynamics Adapter 方案

源码位置: `rlft/vlaw/world_model/dynamics_adapter.py`

```python
class DynamicsAdapter(nn.Module):
    def __init__(
        self,
        state_dim: int = 25,    # qpos(9) + qvel(9) + tcp_pose(7)
        action_dim: int = 7,    # delta pose
        act_steps: int = 5,     # chunk size
        hidden_dim: int = 512,
    ):
        input_dim = state_dim + action_dim * act_steps   # 25 + 35 = 60
        output_dim = 10 * act_steps                       # 10 * 5 = 50

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
```

### 3.3 ManiSkill Adapter 数据流

```
┌─────────────────────────────────────────────────────────────────────┐
│               ManiSkill Dynamics Adapter Pipeline                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Policy Output        Dynamics Adapter              WM Input        │
│  ─────────────        ────────────────             ──────────       │
│                                                                     │
│  delta_actions  ──┐                                                 │
│  (5 x 7D)         ├──► MLP ──► sin/cos (5x10D) ──► euler (5x7D)    │
│  current_state ──┘     (60→512→512→512→50)        (atan2)           │
│  (25D)                                                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.4 Sin/Cos 编码方案

**问题**: Euler angle 存在 ±π wrapping 问题 (e.g., 3.14 和 -3.14 本质相同)

**方案**: 输出 `sin(θ), cos(θ)` 对，推理时用 `atan2` 恢复

```python
# 输出格式 (10D per step)
output = [tcp_x, tcp_y, tcp_z,
          sin_rx, cos_rx, sin_ry, cos_ry, sin_rz, cos_rz,
          gripper_norm]

# 恢复 euler
euler_rx = torch.atan2(sin_rx, cos_rx)
euler_ry = torch.atan2(sin_ry, cos_ry)
euler_rz = torch.atan2(sin_rz, cos_rz)
```

---

## 4. 架构对比分析

### 4.1 DROID vs ManiSkill Adapter 对比

| 属性 | DROID Adapter | ManiSkill Adapter |
|------|--------------|-------------------|
| **输入空间** | Joint space | Cartesian space |
| **输入** | joint_pos(7) + joint_vel(15×7) | state(25) + delta_action(5×7) |
| **输入维度** | 112D | 60D |
| **输出** | joint_delta(15×7) | EE_pose_sincos(5×10) |
| **后处理** | 需要 FK | 仅 atan2 |
| **参数量** | ~580K | ~584K |
| **Hidden dim** | 512 | 512 |
| **Layers** | 3层 (无 norm) | 3层 + LayerNorm |
| **激活** | SiLU | SiLU |

### 4.2 为什么 ManiSkill 不用 DROID 方案?

1. **控制模式不同**: ManiSkill pd_ee_delta_pose 直接在笛卡尔空间，无需 FK
2. **State 信息更丰富**: 25D state 包含 qpos, qvel, tcp_pose
3. **直接端到端**: 省略 FK 环节，减少误差传播

### 4.3 实验版本演进

| 版本 | 特点 | 结果 |
|------|------|------|
| **V1** (baseline) | 3层MLP, sin/cos输出 | ✅ +0.92 dB PSNR |
| **V2** (更深) | 6层MLP, 3.77M params | +5% MAE改善, 不值得 |
| **V3** (delta target) | 预测 EE delta | ❌ euler wrapping bug |
| **Exp A** (single-step) | 自回归预测 | ❌ MAE↓37% 但 PSNR+0.00 |
| **Exp D** (step-weight) | 渐进权重损失 | ❌ +0.25 vs V1 +0.36 |

---

## 5. V1 最佳模型详情

### 5.1 模型规格

```yaml
架构: 3-layer MLP (60→512→512→512→50)
激活: SiLU + LayerNorm
参数量: 584,242 (584K)
输入: state(25D) + delta_actions(5×7D) = 60D
输出: future_ee_poses(5×10D) = 50D (sin/cos编码)
checkpoint: checkpoints/vlaw/dynamics_adapter/best.pt
```

### 5.2 WM PSNR 验证结果

| 条件 | PSNR (dB) | 说明 |
|------|-----------|------|
| **GT EE** | 31.77 ± 3.18 | Oracle 上限 |
| **Tiled EE** | 29.59 ± 2.57 | BUG-D baseline |
| **Adapter EE** | **30.51 ± 2.46** | **+0.92 dB** ✅ |

**结论**: Adapter 恢复了 BUG-D 造成的 2.18 dB gap 中的 **42%**。

### 5.3 EE 预测精度

| 指标 | 训练集 | 分布外 |
|------|--------|--------|
| pos_mae | 14.5mm | 50-75mm |
| euler_mae | 0.116 rad | 0.18-0.23 rad |

### 5.4 关键发现

1. **更低 MAE ≠ 更好 PSNR**: Exp A 的 pos_mae=8.68mm (比 V1 低 37%)，但 PSNR 提升为零
2. **Chunk prediction 优于 single-step**: Autoregressive chaining 可能引入 WM 敏感的相关误差
3. **Frame skip 影响显著**: 训练数据 fs=4，测试数据 fs=3 导致 pos_mae 增加 1.9-4x
4. **误差累积为线性**: step1=18.5mm → step5=33.6mm (1.8x)，非指数爆炸

---

## 6. Imagination 集成方式

### 6.1 配置项

```python
@dataclass
class ImaginationEnvConfig:
    dynamics_adapter_ckpt: str = ""  # 空=禁用, 路径=启用

@dataclass
class ImaginationRLEnvConfig:
    dynamics_adapter_ckpt: str = ""
```

### 6.2 调用方式

```python
# imagination_env.py / imagination_rl_env.py
if self.dynamics_adapter is not None:
    future_ee = self.dynamics_adapter.predict(
        state=current_state,           # (25,)
        action_chunk=delta_actions,    # (5, 7)
    )  # → (5, 7) world frame EE poses
else:
    future_ee = np.tile(current_ee, (5, 1))  # BUG-D fallback
```

### 6.3 运行命令

```bash
# 启用 Adapter 的 Imagination
PYTHONPATH="${PYTHONPATH}:$(pwd)/ctrl_world" \
conda run -n rlft_ms3 python rlft/vlaw/scripts/run_imagination.py \
    --dynamics_adapter_ckpt checkpoints/vlaw/dynamics_adapter/best.pt \
    --num_trajectories 50 \
    --output_dir results/vlaw/imagination_v1_adapter
```

---

## 7. 与 Ctrl-World 论文的对齐情况

| 论文描述 | 我们的实现 | 状态 |
|---------|-----------|------|
| 7D Cartesian condition | 7D (xyz + euler + gripper) | ✅ 对齐 |
| Action encoder MLP | 3层 MLP + text conditioning | ✅ 对齐 |
| Frame-level conditioning | Per-frame action embedding | ✅ 对齐 |
| Dynamics adapter | 从 joint_vel→FK 改为 delta→EE | 🔄 适配 |
| Percentile normalization | 使用 stat.json | ✅ 对齐 |

---

## 8. 未来改进方向

### 8.1 数据层面
- [ ] 采集更多样化的训练数据 (不同成功率/噪声分布)
- [ ] 统一 frame_skip (当前 mixed=4, adapter=3)

### 8.2 模型层面
- [ ] 尝试 Transformer 替代 MLP (更好的序列建模)
- [ ] 更细粒度预测 (单步而非 5 步 chunk)
- [ ] Quaternion 替代 euler 避免 wrapping

### 8.3 Pipeline 层面
- [ ] 全量 Imagination (50 条) + VLM 标注对比 D_syn+ 率
- [ ] 端到端 RL 微调中使用 Adapter

---

## 附录 A: 关键文件路径

| 文件 | 用途 |
|------|------|
| `ctrl_world/models/ctrl_world.py` | WM 模型定义 |
| `ctrl_world/models/action_adapter/train2.py` | DROID Adapter |
| `ctrl_world/dataset/dataset_maniskill.py` | ManiSkill 数据集 |
| `rlft/vlaw/world_model/dynamics_adapter.py` | 我们的 Adapter |
| `rlft/vlaw/world_model/imagination_env.py` | Imagination 集成 |
| `scripts/vlaw/diagnostic/test_adapter_psnr.py` | PSNR 验证脚本 |

## 附录 B: stat.json 格式

```json
{
  "state_01": [-0.123, -0.456, 0.789, -3.14, -1.57, -3.14, 0.0],
  "state_99": [ 0.123,  0.456, 0.987,  3.14,  1.57,  3.14, 1.0]
}
```

字段含义: `[tcp_x, tcp_y, tcp_z, euler_rx, euler_ry, euler_rz, gripper_norm]` 的 1%/99% percentiles

---

*最后更新: 2026-03-18*
