# BUG-D 深度解析：WM Imagination 质量退化的根因

本报告从 World Model 的基本原理出发，逐步解释 BUG-D——为什么 Imagination 生成的视频中 peg 几乎不动。

---

## 1. World Model 是什么，做什么

World Model (WM) 是一个**视频预测模型**：给定历史观测和动作，预测未来会看到什么。

在我们的项目中，WM 基于 **Stable Video Diffusion (SVD)**，一个视频扩散模型。它在潜在空间 (latent space) 工作：

```
输入：
  - 6 帧历史 latent（机器人之前在做什么）
  - 5 帧未来 action conditioning（机器人接下来要做什么动作）
  - 文本指令（"Lift the peg upright"）

输出：
  - 5 帧未来 latent（预测的未来画面）
```

每帧 latent 的形状是 `(4, 48, 24)`，可以通过 VAE decoder 解码成 RGB 图像。

### 1.1 WM 的核心能力：条件生成

WM 不是"凭空想象"，而是**根据你告诉它的动作来预测对应的画面**。

打个比方：
- 你说"机械臂向右移动 5cm"→ WM 预测画面中机械臂向右移动
- 你说"机械臂夹紧 peg"→ WM 预测画面中 gripper 闭合抓住 peg
- 你说"机械臂不动"→ WM 预测画面中什么都不变

这就是所谓的 **action conditioning**——动作是 WM 最关键的输入信号。

---

## 2. Action 的两种表示：绝对位姿 vs 增量动作

这是理解 BUG-D 的关键。

### 2.1 绝对 EE 位姿 (Absolute EE Pose)

"End-Effector (EE) Pose" 是机械臂末端执行器（手爪）在三维空间中的**绝对位置和朝向**：

```
ee_pose = [tcp_x, tcp_y, tcp_z, euler_rx, euler_ry, euler_rz, gripper_norm]
           ←── 位置 (3D) ──→  ←── 朝向 (3D 欧拉角) ──→  ←── 夹爪开度 ──→
```

每帧都有一个确定的 ee_pose，描述"此刻手爪在哪里"。举例：

```
时刻 t=0: ee_pose = [0.10, 0.20, 0.30, 0.0, 0.0, 0.0, 1.0]  ← 手爪在 (0.1, 0.2, 0.3)，张开
时刻 t=1: ee_pose = [0.12, 0.20, 0.30, 0.0, 0.0, 0.0, 1.0]  ← 手爪右移了 2cm
时刻 t=2: ee_pose = [0.14, 0.20, 0.35, 0.1, 0.0, 0.0, 0.5]  ← 手爪继续右移+上升+旋转+半闭合
```

**WM 在训练时接收的就是这种绝对位姿**。每一帧都有独特的、真实的 ee_pose。

### 2.2 增量动作 (Delta Action / pd_ee_delta_pose)

这是**策略网络 (Policy) 输出的动作**。策略不直接输出"手爪应该在哪里"，而是输出"手爪应该怎么移动"：

```
delta_action = [Δx, Δy, Δz, Δrx, Δry, Δrz, gripper_target]
               ←── 位移增量 ──→ ←── 旋转增量 ──→  ←── 夹爪目标 ──→
```

举例：

```
delta = [+0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  ← "向右移 2cm，夹爪保持张开"
```

### 2.3 为什么有两种表示

```
┌─────────────┐     delta_action      ┌──────────────┐     actual EE pose
│   Policy    │ ──────────────────→  │ PD Controller │ ──────────────────→  物理仿真
│  (神经网络)  │   "我想向右移2cm"     │  + 物理引擎    │   经过物理计算后的
└─────────────┘                      └──────────────┘    实际手爪位姿
```

- **策略输出** delta：语义是"我想让手爪做这个运动"
- **PD 控制器 + 物理引擎**执行这个命令：但由于惯性、碰撞、关节限制等，实际移动量 ≠ delta 命令
- **仿真器反馈**真实的 absolute ee_pose：这是手爪实际到达的位置

**关键差异**：`delta_action ≠ 实际位移`。特别是当手爪碰到物体、到达关节极限、或物理仿真引入摩擦和延迟时，差距可以很大。

---

## 3. WM 训练时发生了什么

WM 训练使用从 ManiSkill 仿真器收集的**真实轨迹数据**。

### 3.1 数据准备流程

```
ManiSkill 仿真器
  │
  │  运行策略，记录真实轨迹
  │
  ▼
HDF5 文件 (每帧记录):
  ├── image_base, image_hand    ← RGB 图像 (384×192 双相机)
  ├── state (25维)               ← [关节角度, 关节速度, tcp位置, tcp四元数]
  └── action (7维)               ← delta action (策略输出)

  │
  │  VAE 编码: image → latent
  │  state_to_ee_pose_7d: state → absolute EE pose
  │
  ▼
训练数据 (Dataset_ManiSkill.__getitem__):
  ├── latent: (T=11, 4, 48, 24)  ← 6帧历史 + 5帧未来
  ├── action: (T=11, 7)          ← 每帧的【真实绝对EE位姿】已归一化
  └── text: "Lift the peg upright"
```

关键转换函数 `state_to_ee_pose_7d`：
```python
def state_to_ee_pose_7d(state):
    tcp_pos = state[:, 18:21]           # 真实 TCP 位置 (x,y,z)
    tcp_quat = state[:, 21:25]          # 真实 TCP 四元数
    euler = Rotation.from_quat(...).as_euler("xyz")  # 转欧拉角
    gripper_norm = (state[:, 7] / 0.04).clip(0, 1)   # 夹爪归一化开度
    return concat(tcp_pos, euler, gripper_norm)  # (N, 7)
```

**重点**：训练数据中的 action **不是策略输出的 delta**，而是**从仿真器状态中提取的真实绝对位姿**。每帧都不同，因为每帧手爪确实在不同的位置。

### 3.2 WM 如何使用 Action

Action 通过 **Action Encoder** (一个 3 层 MLP) 编码后，注入 UNet 的 cross-attention：

```
action: (B, T=11, 7)
  │
  │  Action_encoder2: Linear(7→1024) → SiLU → Linear(1024→1024) → SiLU → Linear(1024→1024)
  │                   + CLIP text embedding 融合
  │
  ▼
action_hidden: (B, T=11, 1024)
  │
  │  逐帧注入 UNet: reshape → (B*T, 1, 1024)
  │  作为 cross-attention 的 key/value
  │
  ▼
UNet 的每个 down/mid/up block 都通过 cross-attention 接收对应帧的 action
```

**每帧有独立的 action embedding**（`frame_level_cond=True`）：
- 第 1 帧的 UNet 处理看到的是第 1 帧的 EE pose
- 第 5 帧的 UNet 处理看到的是第 5 帧的 EE pose
- 不同帧看到不同的 action → WM 学会"根据每帧 EE pose 的变化来预测物体运动"

### 3.3 WM 学到了什么

训练好的 WM 学会了一个映射：

```
如果 EE pose 从帧1到帧5 逐渐向右+向上移动
  → 预测画面中手爪向右上方移动，可能带着 peg 一起

如果 EE pose 从帧1到帧5 都不变（值相同）
  → 预测画面中什么都不动（机械臂保持静止）
```

---

## 4. Imagination 推理时发生了什么

Imagination 是 WM 的"部署"阶段：不再有真实仿真器反馈，WM 自己生成视频。

### 4.1 Imagination 循环

Imagination 跑 12 个 chunk，每个 chunk 预测 5 帧：

```
Chunk 0: 真实初始帧 → WM 预测 5 帧
Chunk 1: 用 Chunk 0 的预测作为历史 → WM 再预测 5 帧
Chunk 2: 用之前的预测作为历史 → WM 再预测 5 帧
...
Chunk 11: → 总共 60 帧的合成视频
```

每个 chunk 需要两个输入：
1. **History latents**: 从之前的预测结果中采样 6 帧
2. **Future actions**: 告诉 WM 接下来 5 帧机械臂要做什么

### 4.2 Future Actions 的构造——这就是 BUG-D 的核心

策略网络输出的是 **delta actions**，但 WM 需要的是**绝对 EE poses**。

这就产生了一个关键问题：**如何把 delta 转成 absolute？**

#### 原始做法（BUG-D 行为）——Tiled

```python
current_ee = state_to_ee_pose_7d(current_state)  # 当前手爪位姿
future_ee = np.tile(current_ee, (5, 1))           # 5 帧全部复制成一样的！
```

这意味着 5 帧 future 的 action conditioning 全是**相同的值**。

对 WM 来说，这等价于告诉它："接下来 5 帧手爪都在同一个位置" → WM 预测"什么都不动"。

**即使策略实际上输出了"向右移、向上抬"等动作，WM 完全不知道**——因为这些 delta 信息没有被传递给 WM。

```
实际情况：
  策略说: "向右移 2cm" → "向上抬 3cm" → "旋转 10°" → "闭合夹爪" → "向上抬 5cm"

WM 收到的（tiled）:
  帧1: [0.10, 0.20, 0.30, 0.0, 0.0, 0.0, 1.0]  ← 当前位姿
  帧2: [0.10, 0.20, 0.30, 0.0, 0.0, 0.0, 1.0]  ← 完全相同
  帧3: [0.10, 0.20, 0.30, 0.0, 0.0, 0.0, 1.0]  ← 完全相同
  帧4: [0.10, 0.20, 0.30, 0.0, 0.0, 0.0, 1.0]  ← 完全相同
  帧5: [0.10, 0.20, 0.30, 0.0, 0.0, 0.0, 1.0]  ← 完全相同

WM 理解: "手爪不动" → 预测静态画面 → peg 不动
```

这就是为什么 Imagination 视频中 **peg 几乎不动** 的原因。

#### Fix1 尝试——Delta 积分

```python
def integrate_delta_to_ee_poses(current_ee, action_chunk):
    """把 delta actions 积分成绝对 EE poses"""
    ee = current_ee.copy()
    results = []
    for t in range(T):
        ee[:3] += delta[:3]                          # 位置 += 位移
        ee[3:6] = (delta_rot * cur_rot).as_euler()   # 旋转组合
        ee[6] = clip(delta[6], 0, 1)                 # 夹爪
        results.append(ee.copy())
    return results
```

思路是：既然策略说"每步向右移 2cm"，那我从当前位姿开始一步步加上去，不就得到每帧的绝对位姿了吗？

```
Fix1 试图构造:
  帧1: [0.10, 0.20, 0.30, ...] + delta1 → [0.12, 0.20, 0.30, ...]
  帧2: [0.12, 0.20, 0.30, ...] + delta2 → [0.12, 0.20, 0.33, ...]
  帧3: ...
```

**但 Fix1 失败了。** 原因在 §2.3 已提到：**delta_action ≠ 实际位移**。

```
策略输出:  delta = [+0.02, 0.0, 0.0, ...]   "想向右移 2cm"
PD 控制器执行后:
  实际位移 = [+0.015, +0.001, -0.002, ...]    由于惯性、摩擦等，实际只移了 1.5cm
                                                 且 y/z 方向有微小偏移

Fix1 积分:  假设移了 2cm → [0.12, 0.20, 0.30]
实际位姿:   只移了 1.5cm → [0.115, 0.201, 0.298]
```

5 帧累积下来，积分误差越来越大，产生的 EE pose 序列**偏离 WM 训练时见过的分布** →  WM 预测质量反而更差。

---

## 5. 实验如何证实了 BUG-D

### 5.1 Alpha Sweep 实验 (Group C)

我们做了一个关键实验：在 tiled 和 GT 之间**线性插值**。

```
future_ee = current_ee + α × (gt_future_ee - current_ee)

α=0.0 → tiled（全相同，BUG-D 行为）
α=1.0 → GT（训练时的真实值）
α=0.5 → 一半一半
```

结果（在 peg 动态大的样本上）：

```
α=0.0 (tiled):  PSNR = 27.23 dB  ← BUG-D 行为
α=0.25:         PSNR = 27.54 dB
α=0.5:          PSNR = 27.76 dB
α=1.0 (GT):     PSNR = 34.12 dB  ← 完美！比 tiled 好 6.9 dB
α=1.5:          PSNR = 27.41 dB  ← 超调了，也变差
α=2.0:          PSNR = 26.77 dB  ← 更差
```

**α=1.0 是明确的最优点**，而且是**单调升再单调降**。这种 V 字形曲线完美证明了 WM 需要准确的 GT EE poses。

### 5.2 排除其他因素

| 疑似因素 | 实验 | 结论 |
|---------|------|------|
| 自回归误差累积 | D_MC: 6 chunk AR vs oracle | gap = -0.2 dB → **完全无影响** |
| Inference steps (50→25) | Group B | ~0.8 dB → 次要 |
| History 采样策略 | Group E: 4 种策略 | <0.5 dB → 无影响 |
| History 噪声敏感度 | Group D3: σ=0~0.5 | PSNR 不降反升 → 无影响 |

所有其他因素加起来不超过 1 dB，而 **BUG-D 单独造成 4.5~8.5 dB 的质量下降**。

---

## 6. 总结：BUG-D 的本质

```
┌────────────────────────────────────────────────────────────────┐
│                    训练时（一切正常）                            │
│                                                                │
│  仿真器 → 真实状态 → state_to_ee_pose_7d → 每帧不同的绝对位姿   │
│                                                                │
│  WM 学到：ee_pose 逐帧变化 → 预测对应的物体运动                  │
│                                                                │
├────────────────────────────────────────────────────────────────┤
│                    推理时（BUG-D）                               │
│                                                                │
│  策略 → delta actions → ???如何转换???                          │
│                                                                │
│  ❌ Tiled: 5帧全一样 → WM 认为手爪不动 → peg 不动               │
│  ❌ Fix1 积分: delta ≠ 实际位移 → 积分偏离分布 → 预测更差        │
│  ✅ GT: 从仿真器获取真实位姿（但需要 env 交互，失去想象意义）     │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

**根本矛盾**：
- WM 需要 **absolute EE pose**（训练时就是这样）
- Policy 输出 **delta action**（这是 ManiSkill 的 action space）
- 两者之间的转换需要经过 PD 控制器 + 物理仿真，而这恰恰是 Imagination 想要绕开的

---

## 7. 可能的修复方向

### Plan A: ManiSkill env 内执行 + 读回真实 EE pose

```
策略 → delta actions → env.step(delta) → 读取 env 状态 → state_to_ee_pose_7d → 送入 WM
```

- 优点：EE pose 完全准确，在 WM 训练分布内
- 缺点：需要 env 交互，Imagination 不再完全脱离仿真器
- 可行性：最简单最稳，立即可实施

### Plan B: 训练轻量 Delta→EE 映射网络

```
训练数据：(current_state, delta_action) → next_ee_pose（从真实轨迹中提取）
推理时：用这个小网络替代物理仿真进行转换
```

- 优点：不需要 env 交互
- 缺点：需要额外训练，泛化性未知

### Plan C: 将 WM 改为 Delta Action Conditioning

```
修改 WM：不再接收绝对 EE pose，改为接收 delta actions
需要重新编码训练数据 + 重训 WM（~56 小时）
```

- 优点：彻底消除推理时的转换问题
- 缺点：需要完整重训周期
- 风险：delta action 信息量可能不如 absolute pose 丰富（delta 不包含当前位置信息）
