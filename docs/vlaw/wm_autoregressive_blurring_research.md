# WM 自回归 Rollout 图像模糊问题调研报告

> **创建**: 2026-03-02 | **状态**: 调研完成，改进方向已确认
> **背景**: BUG-019 修复后，合成帧初始质量恢复正常，但 WM 自回归 rollout 中图像随时间推移仍有模糊趋势

---

## 一、问题本质

Ctrl-World 是基于 SVD (Stable Video Diffusion) 的视频扩散模型，每次调用预测 `num_frames=5` 帧的未来 latent。要生成长轨迹，需要将上一轮的 **预测 latent 作为下一轮的 history conditioning 输入**，自回归地迭代多轮（`interact_num` 次）。这就产生了经典的 **compounding error / error accumulation** 问题：每轮 WM 输出都有微小偏差，而这些偏差通过自回归链不断放大，宏观表现为 **图像逐步模糊、细节丢失**。

---

## 二、Ctrl-World 官方的处理方式

### 2.1 History Buffer 机制

官方代码 (`ctrl_world/scripts/rollout_interact_pi.py`) 使用一个 **列表式 history buffer** (`his_cond`)，每次 WM 推理时通过稀疏采样构建历史条件：

```python
history_idx = [0, 0, -12, -9, -6, -3]  # 6个历史帧的索引
his_latent = torch.cat([his_cond[idx] for idx in history_idx], dim=0)  # (6, 4, 72, 40)
```

关键设计：
- **`idx=0` 重复两次**：始终锚定到第一帧（真实 VAE 编码），意味着 **第一帧的真实 latent 永远作为条件参与每次推理**，为后续生成提供固定锚点，一定程度上抑制漂移
- **`idx=-12, -9, -6, -3`**：稀疏采样最近的历史帧，覆盖较长的时间窗口（间隔 3 帧 = 跨越 ~12 帧历史）
- **`num_history=6`**：总共 6 帧历史条件

### 2.2 Image Conditioning（当前帧条件）

每次 WM 推理都传入 `current_latent = his_cond[-1]` 作为 `image_cond`，在 pipeline 内部会被重复到所有帧并 concat 到 latent 通道上（SVD 架构标准做法）。这以一张"参考帧"锚定生成，约束输出不偏离太远。

### 2.3 Pipeline 内部处理

在 `CtrlWorldDiffusionPipeline.__call__()` 中：

```python
# image_cond 作为 latent 条件 (L400-420)
if history is not None:
    B, num_his, C, H, W = history.shape
    num_frames_all = num_frames + num_his
    image_latents = image_latents.unsqueeze(1).repeat(1, num_frames_all, 1, 1, 1)

# denoising loop 中 history 拼接到 latent (L480)
if history is not None:
    latent_model_input = torch.cat([history, latent_model_input], dim=1)

# 推理后移除 history 对应部分 (L520)
if history is not None:
    noise_pred = noise_pred[:, num_his:, :, :, :]
```

注意：`noise_aug_strength = 0.02` 被 **注释掉了**（`# image = image + noise_aug_strength * noise`），实际上没有对 conditioning image 加噪。

### 2.4 官方没有做的事

- **没有 re-anchoring**：不会在中途用真实帧重新校准，rollout 完全在 WM 预测的 latent 空间内迭代
- **没有 noise augmentation**：conditioning image 无加噪
- **没有 latent 正则化**：无显式的 latent norm clipping 或 EMA 平滑

### 2.5 官方默认 Rollout 长度

| 参数 | 默认值 | pickplace | 含义 |
|------|--------|-----------|------|
| `interact_num` | 12 | 15 | 自回归迭代次数 |
| `pred_step` | 5 | 5 | 每次预测帧数 |
| `policy_skip_step` | 2 | 2 | 策略执行跳步 |
| 总帧数 | ~48 | ~60 | 约 10-12 秒 @5Hz |

官方 **跑 12-15 轮自回归**，总共 48-60 帧，模糊问题同样存在但被认为在可接受范围内。

---

## 三、VLAW 论文的考虑

1. **短 Horizon 策略**：VLAW 不试图消除模糊，而是控制 imagination horizon 在合理范围内
2. **VLM 语义级过滤**：合成轨迹即使有模糊，只要 **语义上正确**（物体位置、任务完成状态），VLM 就能判定为成功。α 阈值筛选会自动过滤掉因模糊导致 VLM 无法判断的轨迹
3. **Action Supervision**：策略更新使用的是 **action supervision**（以 latent 对应的 action 为训练信号），不是直接从模糊图像学习视觉特征，因此图像质量退化对策略学习的直接影响有限
4. **WM 迭代微调缓解**：每轮用策略 rollout 的真实数据微调 WM，让 WM 更适应实际策略产生的分布，间接减缓误差累积

---

## 四、与我们当前实现的对比

| 维度 | 官方 Ctrl-World | 我们的实现 (imagination.py) | 差异分析 |
|------|----------------|---------------------------|---------|
| **History 构建** | 稀疏采样 `[0,0,-12,-9,-6,-3]` 共 6 帧 | 滑动窗口 `lat_buf[-window_len:]` | ⚠️ 关键差异 |
| **第一帧锚定** | **是** — `history_idx[:2]` 始终指向 `his_cond[0]`（真实帧） | **否** — 纯滑动窗口，初始真实帧会被挤出 buffer | ❌ 必须修复 |
| **num_history** | 6 | 4（为省显存降低） | ⚠️ 需恢复 |
| **history 稀疏采样** | 间隔 3 (`-12,-9,-6,-3`)，覆盖宽时间窗口 | 连续帧 | ⚠️ 差异 |
| **总 rollout 长度** | 12×5=60 (pickplace 15×5=75) | 12×5=60 | ✅ 一致 |
| **image_cond 来源** | `his_cond[-1]`（WM 预测帧 latent） | `lat_buf[-1]`（WM 预测帧 latent） | ✅ 一致 |
| **history buffer 更新** | 追加到列表：`his_cond.append(predict_latents[pred_step-1])` | 滑动窗口替换：`lat_buf = new_latents[-window_len:]` | ⚠️ 结构差异 |

### 关键代码对比

**官方** (`rollout_interact_pi.py` L340-380):
```python
# 初始化: 用真实帧填充整个 buffer
for i in range(Agent.args.num_history * 4):
    his_cond.append(first_latent)  # 真实帧 VAE 编码

# 每轮推理: 稀疏采样历史
history_idx = [0, 0, -12, -9, -6, -3]  # idx=0 始终是真实帧
his_latent = torch.cat([his_cond[idx] for idx in history_idx], dim=0)

# 更新: 追加 (列表不断增长, idx=0 永远可达)
his_cond.append(predict_latents[pred_step-1])
```

**我们** (`imagination.py` L280-360):
```python
# 初始化: 用真实帧填充 window_len 帧
lat_buf = initial_latent.unsqueeze(0).expand(window_len, -1, -1, -1).clone()

# 每轮推理: 整个 buffer 作为输入
wm_input = lat_buf.clone()

# 更新: 滑动窗口 (旧帧被丢弃)
lat_buf = new_latents[-window_len:].clone()
```

---

## 五、核心发现

1. **最关键差异 — 第一帧锚定缺失**：官方的 `history_idx = [0, 0, ...]` 中前两个元素始终指向 **真实初始帧**，为整个自回归链提供固定的"视觉锚点"。我们的滑动窗口实现没有这个设计，初始帧会在几轮后从 buffer 中消失，**加速模糊退化**

2. **num_history 不一致**：官方 6 帧 vs 我们 4 帧。更长的历史窗口为 WM 提供更丰富的上下文，有助于稳定生成质量

3. **稀疏 vs 连续**：官方间隔 3 帧的稀疏采样覆盖了更宽的时间窗口（约 12 帧跨度），而我们的连续帧窗口覆盖范围更窄

4. **图像模糊是固有特性**：即使完全对齐官方实现，autoregressive video diffusion 的误差累积仍然存在。VLAW 和 Ctrl-World 均 **不试图从根本上消除它**，而是通过 (a) 初始帧锚定、(b) 控制 rollout 长度、(c) VLM 质量筛选三重手段将其影响限制在可接受范围内

---

## 六、改进方向

| # | 改进 | 优先级 | 说明 |
|---|------|--------|------|
| 1 | **对齐 History 构建方式** | 🔴 必做 | 改用列表式 history buffer + 稀疏采样（`history_idx = [0, 0, -12, -9, -6, -3]`） |
| 2 | **保留第一帧锚点** | 🔴 必做 | 真实初始帧 latent 始终作为 history 的前两帧参与条件生成 |
| 3 | **num_history 恢复为 6** | 🔴 必做 | 与官方保持一致，4→6 帧，4090 24GB 显存应可承载 |
| 4 | **总 rollout 长度不下降** | ✅ 保持 | 当前 `num_interact=12, pred_step=5` 与官方一致 |
| 5 | **VLM 自然过滤** | ✅ 保持 | 依靠 VLM 奖励模型筛选质量，模糊轨迹被低分过滤 |

---

## 七、涉及文件

| 文件 | 角色 | 需要修改 |
|------|------|---------|
| `rlft/vlaw/utils/imagination.py` | Imagination engine (无 env step) | ✅ history buffer 重构 |
| `rlft/vlaw/world_model/imagination_env.py` | Imagination engine (有 env step) | ✅ history buffer 重构 |
| `ctrl_world/config.py` | `wm_args_maniskill.num_history` | ✅ 4 → 6 |
| `ctrl_world/scripts/rollout_interact_pi.py` | 官方参考代码 | ❌ 不修改，仅参考 |

---

## 参考资料

- **Ctrl-World 官方代码**: `ctrl_world/scripts/rollout_interact_pi.py` (L330-390)
- **Ctrl-World Pipeline**: `ctrl_world/models/pipeline_ctrl_world.py` (L250-550)
- **Ctrl-World 配置**: `ctrl_world/config.py` — `history_idx = [0,0,-12,-9,-6,-3]`, `num_history=6`, `interact_num=12`
- **我们的 Imagination**: `rlft/vlaw/utils/imagination.py` (L253-370)
- **VLAW 论文**: arXiv:2602.12063
