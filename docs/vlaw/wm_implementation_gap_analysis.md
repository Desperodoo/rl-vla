# WM 实现差异分析：VLAW/Ctrl-World 官方 vs 当前复现

**日期**：2026-03-08
**目的**：系统对比 Ctrl-World 官方代码与 VLAW 论文中 WM 的使用方式，找出我们复现中可能遗漏的关键细节。

---

## 0. 总结：发现 3 个关键问题

| 编号 | 问题 | 严重程度 | 影响 |
|------|------|---------|------|
| **BUG-A** | **动作空间错配**：DROID 用**绝对笛卡尔位姿**做 action conditioning，我们用 **delta pose**（且 delta 在 [-1,1] 内已归一化，又被 state percentile 二次归一化） | **🔴 严重** | WM 的 action conditioning 可能完全失效，导致生成的视频不受动作控制 |
| **BUG-B** | **相机拼接方式不同**：DROID 各相机**独立 VAE 编码**后在 latent 空间拼接；我们在**像素空间拼接**后联合 VAE 编码 | **🟡 中等** | VAE 边界处可能产生跨相机伪影，但实际影响可能有限 |
| **BUG-C** | **相机数量差异**（2 vs 3）导致的模型适配问题 | **🟢 已知/低** | 已在 ADR-002 中处理，latent shape 调整正确 |

---

## 1. 动作空间错配（BUG-A）—— 最关键问题

### 1.1 DROID 官方做法

**Ctrl-World 在 DROID 上训练时，"action" 实际上是绝对末端执行器位姿**，而非动作增量。

来源：`ctrl_world/dataset/dataset_droid_exp33.py:189-193`
```python
# prepare action cond data
cartesian_pose = np.array(label['observation.state.cartesian_position'])[state_id]
gripper_pose = np.array(label['observation.state.gripper_position'])[state_id][..., np.newaxis]
action = np.concatenate((cartesian_pose, gripper_pose), axis=-1)   # 7D 绝对位姿
action = self.normalize_bound(action, state_p01, state_p99)        # percentile 归一化到 [-1,1]
```

- `cartesian_position` = `[x, y, z, rx, ry, rz]` — **机器人末端的绝对空间位置**
- `gripper_position` = `[0~1]` — 夹爪开合度
- `state_01/state_99` = 位姿的 1%/99% 分位数 — 本质上是**工作空间包围盒**
- 归一化后的 "action" 表示：当前帧中末端执行器在工作空间中的**相对位置**

### 1.2 我们当前做法

我们的数据存储的是 ManiSkill 的 `pd_ee_delta_pose` 动作：

```
actions: (T, 7) float32, 值域 = [-1.0, 1.0]（ManiSkill Box action space 原生范围）
```

ManiSkill `pd_ee_delta_pose` 的 7 维含义：`[delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz, gripper]`
- 这是**动作增量**（delta），不是绝对位姿
- 值已经在 [-1, 1] 内（ManiSkill 原生）

然后在 WM 训练/推理时（`dataset_maniskill.py:170-180`），又用 `state_01/state_99` 做了一次归一化：
```python
def _normalize_action(self, action):
    p01, p99 = self.norm_stats
    ndata = 2.0 * (action - p01) / (p99 - p01 + 1e-8) - 1.0
    return np.clip(ndata, -1.0, 1.0)
```

问题是 `state_01/state_99` 来自 **关节角度位置** 的百分位数（`scripts/vlaw/generate_stat_json.py:81-84`）：
```python
state_for_pct = states[:, :action_dim]   # 取 state 前 7 列 = 关节角度
```

stat.json 实际值：
```
state_01: [-0.2228, -0.4856, -0.7541, -2.7566, -1.6692,  0.1227, -1.5755]
state_99: [ 0.6521,  1.6764,  0.3604, -0.1063,  0.2425,  2.7781,  1.4918]
```

**这对一个已经在 [-1,1] 的 delta pose 做归一化，结果完全错误**：

| 维度 | delta 原始范围 | state_01 | state_99 | 归一化效果 |
|------|-----------|---------|---------|-----------|
| dim 0 (delta_x) | [-1, 1] | -0.223 | 0.652 | delta=0 → 归一化后 ≈ -0.49（严重偏移） |
| dim 3 (delta_rx) | [-1, 1] | -2.757 | -0.106 | delta=0 → 归一化后 ≈ 1.08（严重偏移+截断） |

### 1.3 三重问题

1. **语义错配**：DROID 的 "action" 是**绝对位姿**（当前在哪），我们的是**增量动作**（要移动多少）。Ctrl-World 预训练学到的是"位姿→视频"的映射，不是"增量→视频"的映射。
2. **二次归一化**：delta pose 已经在 [-1,1]，再用关节角度范围归一化导致值严重失真。
3. **归一化基底错误**：即使需要归一化，也应该用 action 本身的统计量，而非毫不相关的关节角度百分位。

### 1.4 影响评估

这可能是 WM imagination **缺乏操作动态**的根本原因之一。如果 action conditioning 信号被严重扭曲，WM 就无法正确理解要生成什么样的动作，退化为基于初始帧的无条件视频扩散。

### 1.5 修复方案

**方案 A（推荐）：对齐 DROID 做法——使用绝对 EE 位姿**

将 HDF5 中存储的数据从 delta pose 切换为绝对末端执行器位姿：
- 从 `state` 字段提取（ManiSkill state 的前 7 维是关节角度，需要 FK 变换为 EE pose）
- 或直接从 obs 中提取 EE pose（如果 ManiSkill 支持 `tcp_pose` obs）
- 重新计算 `state_01/state_99` 为 EE 位姿的百分位

这样 WM 的 action encoder 接收到的信号语义与预训练一致。

**方案 B（最小改动）：跳过二次归一化**

既然 delta pose 已经在 [-1,1]，可以直接跳过 `_normalize_action`：
```python
def _normalize_action(self, action):
    return action   # delta pose 已在 [-1,1]，无需额外归一化
```

这不能解决"语义错配"问题（delta vs absolute），但至少消除了二次变形。WM 在微调过程中可能学会适应 delta 语义。

**方案 C（完整对齐）：改用绝对位姿 + 重新编码数据 + 重训 WM**

最合理但工作量最大。

---

## 2. 相机 VAE 编码方式差异（BUG-B）

### 2.1 DROID 官方做法

3 个相机**各自独立**通过 VAE 编码，然后在 **latent 空间**拼接：

```python
# dataset_droid_exp33.py:177-186
latnt_cond1 = self._get_obs(label, rgb_id, cond_cam_id1, pre_encode=True)  # (T, 4, 24, 40)
latnt_cond2 = self._get_obs(label, rgb_id, cond_cam_id2, pre_encode=True)  # (T, 4, 24, 40)
latnt_cond3 = self._get_obs(label, rgb_id, cond_cam_id3, pre_encode=True)  # (T, 4, 24, 40)
latent = torch.zeros((T, 4, 72, 40))
latent[:,:, 0:24]  = latnt_cond1   # cam 0 → latent H[0:24]
latent[:,:,24:48]  = latnt_cond2   # cam 1 → latent H[24:48]
latent[:,:,48:72]  = latnt_cond3   # cam 2 → latent H[48:72]
```

每个相机的 latent block 是**独立自包含**的——VAE encoder 只看到单个相机的 192×320 图像。

### 2.2 我们当前做法

2 个相机先在**像素空间**拼接，再联合 VAE 编码：

```python
# data/pipeline.py:163-189
concat_frames = concat_cameras(rgb_base, rgb_render, mode="vertical")  # (T, 384, 192, 3)
# 然后整张 384x192 图像一起送入 VAE
z = vae.encode(concat_frames).latent_dist.sample() * scaling_factor    # (T, 4, 48, 24)
```

### 2.3 差异影响

| 方面 | 独立编码（DROID） | 联合编码（我们） |
|------|----------------|---------------|
| latent 语义 | 每个 24-row block 是独立相机的完整编码 | 48-row latent 是两相机联合的编码 |
| 跨相机信息 | 完全隔离 | VAE 卷积核可以在边界处跨相机感受 |
| 推理解码 | `rearrange m=3` 分离后可独立解码 | `rearrange m=2` 分离后解码可能产生边界伪影 |
| 与预训练一致性 | ✅ Ctrl-World 预训练就是独立编码 | ⚠️ 不一致 |

**实际影响评估**：由于 VAE 是全卷积网络，且 SD VAE 的感受野有限（约 64 像素），在 192 像素高的边界处跨相机影响较小。但这仍然是一个不一致：

- 训练时 latent 是联合编码的
- 推理时 `rearrange m=2` 把联合 latent 按 m=2 拆分后分别解码——但这些 24-row blocks 不是独立编码的，拆分后的语义可能有微小偏差

### 2.4 修复方案

**推荐**：改为独立编码，与 DROID 一致。修改 `data/pipeline.py` 中 `encode_frames_batch` 的调用方式：
```python
# 不拼接，分别编码
latent_base   = encode_frames_batch(vae, rgb_base,   batch_size, device)   # (T, 4, 24, 24)
latent_render = encode_frames_batch(vae, rgb_render,  batch_size, device)   # (T, 4, 24, 24)
latent_concat = np.concatenate([latent_base, latent_render], axis=2)        # (T, 4, 48, 24)
```

---

## 3. 相机数量差异（已知/低）

| | DROID | ManiSkill |
|--|-------|-----------|
| 相机数 | 3（exterior_1, exterior_2, wrist） | 2（base, hand） |
| 单相机像素 | 192×320 | 192×192 |
| 拼接后像素 | 576×320 | 384×192 |
| latent shape | (T, 4, 72, 40) | (T, 4, 48, 24) |
| rearrange m | m=3 | m=2 |

**评估**：这已经在 ADR-002 中处理。UNet 是全卷积的，可以处理不同空间尺寸。推理时 rearrange 的 m 参数也做了相应调整（`ctrl_world_adapter.py` 中用 `m=2`）。宽度从 320→192 意味着分辨率更低，但 SVD 架构支持这种变化。

**潜在问题**：Ctrl-World 预训练在 DROID 上是 3 相机，微调时改为 2 相机，UNet spatial layers 需要适应新的空间分布。这也是为什么 Phase-B 全量微调（`freeze_unet_spatial=False`）是必要的。

---

## 4. 其他对比细节

### 4.1 数据规模差异

| | DROID (预训练) | ManiSkill (微调) |
|--|--------------|----------------|
| 轨迹数 | ~95,000 | ~235 |
| 场景多样性 | 564 场景 | 1 个任务 |
| 训练步数 | 500,000（预训练）| ~1600（当前） |

比例 = 235/95000 ≈ 0.25%。数据量极小，WM 主要依赖预训练知识迁移。

### 4.2 Action Conditioning 架构

`Action_encoder2` 架构（`ctrl_world.py:71-107`）：
```
Input: (B, T, 7)
  → Linear(7→1024) → SiLU → Linear(1024→1024) → SiLU → Linear(1024→1024)
  → + CLIP text embedding (broadcast)
  → Output: (B, T, 1024) 作为 encoder_hidden_states 注入 UNet cross-attention

frame_level_cond=True 时: 每帧独立 action embedding
frame_level_cond=False 时: 所有帧共享同一个 action embedding
```

我们使用 `frame_level_cond=True`，与官方一致。

### 4.3 History 采样方式

**官方 rollout 脚本**（`rollout_replay_traj.py`）：
```python
history_idx = [0, 0, -8, -6, -4, -2]
# pos 0,1 = 始终锚定到第一帧（真实帧）
# pos 2-5 = 相对当前位置的负偏移（越老的帧越早）
```

**我们的 P4.3 imagination engine**（`imagination_env.py`）：
```python
history_idx = [0, 0, -12, -9, -6, -3]
```

**差异**：偏移值不同。DROID 用 `-8,-6,-4,-2`（每 2 帧一跳），我们用 `-12,-9,-6,-3`（每 3 帧一跳）。后者间隔更大,覆盖更长时间窗口。具体哪种更好需要看数据频率和任务特性，但与预训练不一致可能是个问题。

### 4.4 VAE 编码器/解码器一致性

| 环节 | 使用的 VAE |
|------|-----------|
| 数据编码（`data/pipeline.py`） | `AutoencoderKL`（sd-vae-ft-mse） |
| WM 训练（`CrtlWorld.forward`） | 不涉及 VAE（纯 latent 空间操作） |
| WM 验证/Imagination 解码 | `AutoencoderKLTemporalDecoder`（SVD VAE） |

两者共享相同的 spatial encoder，编码结果在数学上等价（`scaling_factor=0.18215` 一致）。解码器架构不同但这只影响可视化/VLM 输入，不影响训练。**这个差异是可接受的**。

### 4.5 推理参数对比

| 参数 | DROID 官方 | 我们当前 | 说明 |
|------|-----------|---------|------|
| `num_inference_steps` | 50 | 25 | 我们减半了去噪步数 |
| `max_guidance_scale` | 2.0（train）/ 1.0（eval） | 2.0（adapter 默认） | eval 时官方用 1.0 |
| `motion_bucket_id` | 127 | 127 | 一致 |
| `fps` | 7 | 7 | 一致 |
| `num_frames` | 5 | 5 | 一致 |
| `num_history` | 6 | 6 | 一致 |

---

## 5. 优先级排序与行动建议

### P0：修复动作空间（BUG-A）

这是最紧急的问题。**可能是 WM imagination 缺乏操作动态的根因**。

**推荐路径**（增量验证）：

1. **立即验证**（不重训 WM）：修改 `dataset_maniskill.py` 跳过 `_normalize_action`（方案 B），用当前 WM ckpt-1600 跑一次 short-horizon imagination，对比可视化。如果 action conditioning 信号恢复正常，应该能看到一些改善。

2. **正式修复**：切换到绝对 EE 位姿（方案 A）。需要：
   - 从 ManiSkill obs 中提取 `tcp_pose`（末端绝对位姿）
   - 重新编码数据（或在 dataset 层在线计算）
   - 重新计算 `state_01/state_99` 为 EE 位姿百分位
   - 重训 WM

3. **完整重训**（方案 C）：改 data pipeline + 重编码 + 重训。

### P1：修复相机编码方式（BUG-B）

改为独立 VAE 编码。需要重新编码所有数据。可以与 P0 合并执行。

### P2：验证 history 采样偏移

将 `[-12,-9,-6,-3]` 改为与 DROID 一致的 `[-8,-6,-4,-2]`，或测试两者对比。

### P3：推理参数调整

已通过消融实验验证 `steps=50` 更好。考虑 eval 时用 `guidance_scale=1.0`。

---

## 6. 代码引用

| 文件 | 关键行 | 内容 |
|------|-------|------|
| `ctrl_world/dataset/dataset_droid_exp33.py` | 189-193 | DROID 用绝对位姿作为 action |
| `ctrl_world/dataset/dataset_droid_exp33.py` | 177-186 | DROID latent 空间相机拼接 |
| `ctrl_world/dataset/dataset_maniskill.py` | 148-151 | ManiSkill 数据加载 |
| `ctrl_world/dataset/dataset_maniskill.py` | 170-180 | 二次归一化代码 |
| `rlft/vlaw/data/pipeline.py` | 163-193 | 像素空间相机拼接 |
| `rlft/vlaw/data/pipeline.py` | 196-230 | 联合 VAE 编码 |
| `scripts/vlaw/generate_stat_json.py` | 81-84 | stat.json 用关节角度计算 |
| `data/vlaw/meta_info/maniskill/stat.json` | 全文 | 实际归一化参数 |
| `ctrl_world/models/ctrl_world.py` | 71-107 | Action Encoder 架构 |
| `ctrl_world/scripts/rollout_replay_traj.py` | — | 官方 history 采样 |
