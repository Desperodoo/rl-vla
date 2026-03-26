# VLAW 复现计划 — ManiSkill + ShortCut Flow + Ctrl-World

> **论文**: VLAW: Iterative Co-Improvement of Vision-Language-Action Policy and World Model (arXiv:2602.12063)
> **目标**: 在当前 rl-vla 代码库基础上，尽可能忠实地复现 VLAW 框架
> **创建时间**: 2026-02-24

---

## 一、核心替换映射

| VLAW 原版 | 本次复现 | 说明 |
|-----------|---------|------|
| π₀.₅ (VLA, Transformer + Flow Matching) | ShortCut Flow (1D U-Net, `obs_mode="rgb"`) | 保留 flow matching 本质，用 PlainConv 视觉编码器处理图像 |
| DROID 真机 (Franka + Robotiq, 3 相机) | ManiSkill3 GPU 仿真 (1-2 相机, 128×128) | 保留 RGB 观测空间 |
| Ctrl-World (SVD, 320×192, 3 相机拼接) | Ctrl-World (SVD, 降低分辨率, 适配 ManiSkill 相机配置) | **保留完整视频扩散架构** |
| Qwen3-VL-4B 二分类 | Qwen3-VL-4B/8B 二分类 + 概率阈值 | 保持一致 |
| DROID 数据集 (95K 轨迹) | ManiSkill 演示数据 + Rollout 数据 | 规模更小，需适配数据格式 |
| ManiSkill 内置 Reward | **不使用** — 仅用 VLM 奖励模型 | 内置 success 信号仅作为 ground truth 验证 |

---

## 二、算力资源规划 (10 × RTX 4090, 24GB each)

### 2.1 显存分析

**Ctrl-World 模型组成** (原版 A100 上的参数量):
| 组件 | 参数量 | 显存 (fp16) | 可训练 |
|------|--------|------------|--------|
| UNet (SVD) | ~1.5B | ~3GB | ✅ |
| VAE | ~83M | ~170MB | ❌ 冻结 |
| Image Encoder (CLIP-ViT) | ~86M | ~170MB | ❌ 冻结 |
| Text Encoder (CLIP) | ~63M | ~130MB | ❌ 冻结 |
| Action Encoder | ~3M | ~6MB | ✅ |

**训练显存估算** (含梯度 + 优化器状态 + 激活值):
- 原版 (320×192×3cam, batch=4): ~60-80GB → 需 A100
- **本次方案** (128×128×2cam, batch=1-2, gradient_checkpointing + fp16):
  - 模型权重: ~4GB
  - 梯度: ~4GB
  - 优化器 (AdamW): ~8GB
  - 激活值 (with grad ckpt): ~4-6GB
  - **合计: ~20-22GB → 可放入单张 4090**

### 2.2 GPU 分配方案

```
GPU 0-3: Ctrl-World 训练 (4 GPU DDP, accelerate)
          - 分辨率: 128×128×2cam = 256×128 拼接, latent ~32×16 → 4×32×16
          - batch_size_per_gpu=2, gradient_accumulation=4 → effective batch=32
          - fp16, gradient_checkpointing

GPU 4-5: ManiSkill 数据收集 (2 GPU 并行)
          - num_envs=64 per GPU (GPU 向量化环境)
          - 渲染 RGB 帧 + 记录状态/动作

GPU 6-7: VLM 奖励模型 (2 GPU)
          - GPU 6: Qwen3-VL-4B 推理/微调 (~10GB)
          - GPU 7: 批量推理备用 / 渲染

GPU 8-9: ShortCut Flow 策略训练 + 评估
          - GPU 8: 策略更新 (Weighted Flow Matching)
          - GPU 9: 评估环境
```

> **注**: 阶段间 GPU 可复用。数据收集完成后，GPU 4-5 可加入 WM 训练。

---

## 三、详细技术方案

### 3.1 数据格式设计

#### 3.1.1 ManiSkill RGB 观测适配

ManiSkill3 默认提供 `base_camera` 和 `hand_camera` 两个视角 (类比 DROID 的第三人称 + 腕部相机)。

```python
# ManiSkill 环境配置
env = gym.make(
    "LiftPegUpright-v1",
    obs_mode="rgbd",             # 获取 RGB+Depth 观测
    render_mode="rgb_array",     # 允许渲染
    sensor_configs=dict(
        base_camera=dict(width=128, height=128),
        hand_camera=dict(width=128, height=128),
    ),
)
```

**与 Ctrl-World 的分辨率适配方案**:
- 原版 Ctrl-World: 3 相机 × (320×192) → 垂直拼接为 (320×576) → VAE latent (40×72×4)
- **本次方案**: 2 相机 × (128×128) → 垂直拼接为 (128×256) → VAE latent (16×32×4)
- 或: 2 相机水平拼接 (256×128) → VAE latent (32×16×4)
- **推荐**: 使用 **2 相机 × (192×192)** → 垂直拼接 (192×384) → VAE latent (24×48×4)
  - 更接近原版的宽高比，且 192 是 8 的整倍数 (VAE 要求)
  - 显存增加可控 (相比 128×128 约增加 2.25x latent size)

**备选降级方案** (显存不足时):
- 单相机 128×128 → latent 16×16×4 (最节省)
- 可动态选择：训练用低分辨率，评估用高分辨率

#### 3.1.2 轨迹数据结构

```python
# 每条轨迹的数据结构 (保存为 HDF5)
trajectory = {
    # --- 原始数据 (ManiSkill 直出) ---
    "rgb_base": np.array([T, H, W, 3], dtype=uint8),     # base_camera RGB
    "rgb_hand": np.array([T, H, W, 3], dtype=uint8),     # hand_camera RGB
    "state": np.array([T, state_dim], dtype=float32),     # 完整物理状态
    "obs_agent": np.array([T, agent_dim], dtype=float32), # agent 本体感知
    "actions": np.array([T, action_dim], dtype=float32),  # action chunks
    "env_success": np.array([T], dtype=bool),             # ManiSkill GT (仅验证用)

    # --- VAE 编码 (预计算, 加速训练) ---
    "latent_concat": np.array([T, 4, lat_h, lat_w], dtype=float16),  # 2cam 拼接后的 VAE latent

    # --- 元信息 ---
    "task_instruction": str,                              # 任务语言描述
    "vlm_reward": int,                                    # VLM 标记的 0/1
    "vlm_prob": float,                                    # VLM P('yes') 概率
    "source": str,                                        # "real" or "synthetic"
}
```

#### 3.1.3 从 ManiSkill 到 Ctrl-World 的数据管线

```
ManiSkill rollout
    ↓ 每步保存 RGB + state + action
    ↓ (ManiSkill 15Hz 控制频率 → 下采样到 ~5Hz 匹配 Ctrl-World)
    ↓
VAE 离线编码 (批量处理)
    ↓ 2 相机垂直拼接 → (192, 384, 3) 
    ↓ VAE encode → (4, 24, 48)
    ↓
保存 HDF5: {latent, action, text, ...}
    ↓
Ctrl-World 训练数据加载器
```

### 3.2 Ctrl-World 适配

#### 3.2.1 模型适配要点

**需要修改的核心参数**:
```python
# 原版 config
width = 320         # → 192 (或 128)
height = 192 * 3    # → 192 * 2 = 384 (2 相机) 或 128 * 2 = 256
num_frames = 5      # → 保持 5 (每次预测 5 帧 = 1 秒)
num_history = 6     # → 可降为 4 (减少历史帧数节省显存)
action_dim = 7      # → 7 (ManiSkill pd_ee_delta_pose, 保持不变)
down_sample = 3     # → 根据 ManiSkill 控制频率调整
```

**Ctrl-World 代码层面修改**:
1. `config.py` → 新增 `maniskill` task_type，适配分辨率和相机数
2. `models/ctrl_world.py` → 无需修改模型架构 (SVD UNet 本身不限定分辨率)
3. `dataset/` → 新增 ManiSkill 数据加载类，从 HDF5 读取 latent + action
4. `scripts/train_wm.py` → 适配新数据格式
5. `scripts/rollout_interact_pi.py` → 替换 π₀.₅ 为 ShortCut Flow

**关键技术点 — Action Space 映射**:
- 原版 Ctrl-World 使用 DROID 的 cartesian pose (xyz + euler + gripper) = 7D
- ManiSkill 使用 `pd_ee_delta_pose` (delta_xyz + delta_euler + gripper) = 7D
- 维度匹配但语义不同 — Ctrl-World 的 action 是绝对位姿，ManiSkill 是增量
- **解决方案**: 在数据处理层做绝对/增量转换，或直接训练 Ctrl-World 接受增量动作
  - 推荐后者: Ctrl-World 的 Action Encoder 是 MLP，可以学习任意 action 编码
  - 在数据归一化时使用 ManiSkill 的 action 统计量替换 DROID 的 `stat.json`

#### 3.2.2 预训练权重使用策略

Ctrl-World 提供了在 DROID 上预训练的权重 (~8GB), 包含:
- UNet: 已学习通用的视频预测能力
- VAE: 通用的图像编解码
- Image Encoder: CLIP 视觉特征
- Action Encoder: DROID 特定的动作编码

**微调策略**:
```
Phase A (WM 预热): 仅微调 Action Encoder + UNet 的部分层
  - 冻结 VAE, Image Encoder, Text Encoder, UNet 大部分层
  - 解冻 Action Encoder + UNet 的 temporal attention 层
  - 用 ManiSkill 演示数据训练 ~10K steps
  - 目的: 适配 ManiSkill 的动作空间和视觉风格

Phase B (VLAW 迭代中的 WM 微调):
  - 解冻 UNet 全部 (与 VLAW 论文一致)
  - 冻结 VAE, Image Encoder, Text Encoder
  - gradient_checkpointing=True
  - 用 D_real + λ·D_demo 联合训练 50K steps/轮
```

#### 3.2.3 显存优化技术

```python
# 1. Gradient Checkpointing (已内置于 Ctrl-World)
self.unet.enable_gradient_checkpointing()

# 2. 混合精度 (fp16)
accelerator = Accelerator(mixed_precision='fp16')

# 3. Gradient Accumulation (batch 效果放大)
gradient_accumulation_steps = 8  # batch_per_gpu=1 × 4gpu × 8acc = 32

# 4. DeepSpeed ZeRO-2 (如 4GPU DDP 仍不够)
# accelerate config → deepspeed_zero2.json

# 5. 推理时的优化
#    - VAE decode chunk: decode_chunk_size=4 (而非全部一次性解码)
#    - 减少 num_inference_steps: 50 → 25 (imagination 阶段速度翻倍)
#    - 使用 torch.compile 加速 UNet forward

# 6. xFormers memory-efficient attention (如果 SVD UNet 支持)
# self.unet.enable_xformers_memory_efficient_attention()
```

### 3.3 奖励模型

> **⚠️ 关键澄清 (BUG-011 根因)**: VLAW 论文明确指出零样本 VLM 效果不足，必须在第一轮迭代用真实 rollout 数据微调后才能使用 α=0.8 阈值。见论文 Section 4.1 + Appendix C。

#### 3.3.0 论文原文要求 (Appendix C)

> **原文**: "Each trajectory is **temporally downsampled into a 16-frame video** before being fed to the model. We **fine-tune the Qwen3-VL-4B-Instruct model for 200 steps with batch size 128**."
>
> "We observe that directly prompting the reward model to output a **binary yes/no decision can be overly optimistic**. We instead examine the model-assigned **probability of the 'yes' token** and only label a trajectory as successful when this probability exceeds a **threshold of 0.8**"
>
> Section 4.1: "we find that the **zero-shot VLM is not accurate enough**, so in the first iteration, we **fine-tune the VLM** with the success labels r_τ in D_real."

关键要点：
- **零样本不可用**（论文原话："zero-shot VLM is not accurate enough"）
- **16帧**均匀下采样（论文明确指定，当前 `reward_model.py` 配置 `num_frames=16` 已正确）
- **先微调，后使用 α=0.8 阈值**（阈值是微调后模型的配置，零样本 p_yes < 0.15 无意义）
- Table 3 数据：微调+阈值后 FP=2，直接生成二值答案 FP=8（减少 75%）
- **微调数据**: K=50 条/任务 的真实 rollout + ManiSkill `info["success"]` 标签

#### 3.3.1 两阶段设计（零样本过渡 → 微调正式）

```
阶段 0 (Iter-0, 零样本, 当前状态):
  - 仅用于 sanity check / 相关性验证，不作为正式奖励信号
  - p_yes 可作为连续软权重（不做二值化），方向上有 3x ratio (0.0009 vs 0.0003)
  - D_real 奖励标注: 使用 env_success_at_end 替代（ManiSkill 完全可信）
  - 不使用 α=0.8 阈值（零样本 p_yes < 0.15，阈值将导致 vlm_success=0%）

阶段 1 (Iter-1, 微调后 — 论文实际做法):
  - 收集 D_real (50条/任务) → 用 ManiSkill env_success 构造 (video16帧, instruction, yes/no)
  - fine-tune Qwen3-VL-4B-Instruct: 200 steps, batch=128 (gradient_accumulation 实现)
  - 微调后验证: confusion matrix on held-out ~40 条轨迹，目标 FP < 20%
  - 微调后使用 α=0.8 → 标注 D_syn（否则 D_syn 全部被标为失败）
```

**当前项目状态 (2026-02-25)**:
- ✅ `rlft/vlaw/reward_model.py` — 零样本推理已实现
- ❌ `rlft/vlaw/train_reward_model.py` — **待实现** (P3.2)
- Iter-1 过渡方案: `vlm_reward = env_success_at_end`（绕过零样本 VLM 问题）

#### 3.3.2 架构设计 — VLAW 式二分类

严格遵循 VLAW 论文 (Section 4.1, Eq. 3):

```python
class VLAWRewardModel:
    """
    VLAW 论文式 VLM 奖励模型:
    - 输入: 轨迹视频帧 + 任务指令 (16帧均匀采样, 论文 Appendix C)
    - 输出: P('yes') 概率  (提取 'yes' token logit, 非生成)
    - 判定: R(τ) = 1[P('yes'|τ, I) > α], α=0.8  (仅微调后有效)
    """
    def __init__(self, model_name="Qwen/Qwen3-VL-4B-Instruct"):
        self.model = load_qwen3vl(model_name)  # ~10GB on 4090
        self.threshold = 0.8  # 仅在 fine-tuned 版本中有意义

    def score_trajectory(self, frames: List[Image], instruction: str) -> dict:
        # 16 帧均匀采样（论文 Appendix C 明确）
        logits = self.model.forward(frames, prompt)
        prob_yes = softmax(logits)['yes']  # P('yes' token), 非生成概率
        return {
            "success": prob_yes > self.threshold,  # 零样本时 p_yes < 0.15, 不可信
            "prob": prob_yes,
        }
```

#### 3.3.3 VLM 微调 (P3.2 — 论文必要步骤)

**时机**: **第一轮迭代收集 D_real 后立即微调**，微调后的模型用于 D_syn 标注。

**标签来源**:
- K=50 条/任务 rollout → ManiSkill `info["success"]` 作为 ground truth (r_τ)
- ManiSkill 仿真优势: 精确的 success 信号，无需人工标注
- 构造 (视频16帧, 任务指令, yes/no) 三元组

**微调超参** (论文 Appendix C):
```python
# 核心超参（与 VLAW 论文 Appendix C 完全一致）
# 训练步数: 200 steps
# 批大小:   batch_size=128 (gradient_accumulation_steps=128 on 单张 4090)
# LoRA:     r=16, alpha=32, target_modules=["q_proj", "v_proj"]
from peft import LoraConfig, get_peft_model
lora_config = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
# 单任务: 50 条 rollout × 16 帧/条 = ~50 训练样本，200 steps 约等于 training 4 epochs
# 多任务: 50条/任务 × N_tasks 条，200 steps 仍足够（每步 batch=128 覆盖大部分样本）
```

**微调验证** (对照 VLAW Table 3):
- 准备 40 条 held-out 轨迹，人工确认 ground truth
- 计算 confusion matrix (TP/FP/TN/FN)
- 目标: FP < 20%（论文 Table 3 微调+阈值后 FP=2/40=5%）

**合成轨迹评估**:
- 世界模型 latent → VAE decode → RGB 帧 → 均匀下采样 **16 帧** → 微调后 VLM 评估
- decode 分块执行 (decode_chunk_size=4)

#### 3.3.4 Iter-1 权宜方案（train_reward_model.py 未就绪时）

```python
# 方案 A（推荐）: 使用 env_success_at_end 作为 vlm_reward
# ManiSkill 仿真中 env_success 完全可信，与论文 r_τ 等价，精度 > VLM
for traj in d_real:
    traj["vlm_reward"] = int(traj["env_success_at_end"])

# 方案 B: 连续 p_yes 作为软权重（不二值化）
# 即使零样本 p_yes=0.03, 成功/失败间仍有正向 ratio (~3x)，可加权 FM loss
for traj in d_real:
    traj["vlm_weight"] = traj["p_yes"]  # soft weight
```

> **重要结论**: 本项目用 ManiSkill 仿真，env_success 完全可用。  
> VLM fine-tuning 的**主要价值在于 D_syn 标注**（世界模型生成的合成轨迹无 env_success 可用）。  
> Iter-1 D_real 策略更新可直接用 env_success 方案 A；D_syn 标注必须等 fine-tuned 奖励模型就绪。

#### 3.3.5 与 RoboReward 的关系

VLAW 论文参考文献中引用了 RoboReward (Lee et al., 2026, arXiv:2601.00675)。  
设计思想一致（VLM二分类 + P('yes') token），实现上：
- **不使用** `rlft/roboreward/` 的1-5分连续评分体系
- **新建** 独立的 VLAW 式二分类模块 (`rlft/vlaw/reward_model.py` + `train_reward_model.py`)
- **可复用**:
  - `roboreward/config.py` 中的模型加载逻辑
  - `roboreward/dataset_converter.py` 中的帧采样工具
  - `roboreward/labeler.py` 中的 Qwen3-VL 推理管线

### 3.4 Imagination 引擎 — 策略 ↔ Ctrl-World 闭环

#### 3.4.1 Policy-in-the-Loop Rollout

这是 VLAW 的核心创新之一 — 在世界模型中做闭环 rollout:

```
1. 从真实轨迹采样初始帧 → VAE encode → latent_0
2. 循环 K_interact 步:
   a. VAE decode latent → RGB images (2 相机)
   b. ShortCut Flow 策略:
      - PlainConv(image) → visual_feature
      - [visual_feature, agent_state] → obs
      - ShortCut Flow inference → action_chunk (H=8 步)
   c. Ctrl-World 前向:
      - 输入: current_latent + history_latents + action_chunk + instruction
      - 输出: predicted_future_latents (5 帧)
   d. 更新 history buffer
3. 收集完整 latent 序列 → 解码 → VLM 评估
```

**关键技术挑战与解决方案**:

**(a) 策略输入的 agent_state 问题**:
- ShortCut Flow 的 obs = [visual_feature, agent_state]
- 在 imagination 中,我们有 predicted image (→ visual_feature) 但无 ground_truth agent_state

| 方案 | 描述 | 状态 |
|------|------|------|
| **方案 A（最终）**: `env.step()` | ManiSkill 仿真直接调用，精确获取 $s_{t+1}$，支持 `num_envs=1..N` 并行 | **P4.3 必须实现** |
| **方案 B（临时）**: State Predictor MLP | 残差 MLP $\hat{s}_{t+1} = s_t + f(s_t, a_t)$，~0.1MB，无需环境 | **当前 P4.1/P4.2，仅用于跑通流程** |
| 方案 C（放弃）: vision-only obs | 忽略 agent_state | 信息损失，不采用 |

> **⚠️ 重要**: 本项目用 ManiSkill **仿真**替代真机，`env.step()` 完全可用且精确。State Predictor MLP 仅是"先跑通代码再优化"的临时脚手架。P4.3 必须完成从方案 B → 方案 A 的迁移。迁移后可通过控制 `num_envs` 和合成数据数量来系统测试数据效率（这正是使用仿真的最大优势）。

**(b) Imagination 速度**:
- 原版 Ctrl-World: A100 上 ~10s/step → H100 上 ~5s/step
- 4090 上预估: ~8-12s/step (取决于分辨率)
- 每条轨迹: 12 steps × ~10s = ~2 分钟
- 500 条合成轨迹: ~16 小时 (单 GPU) → ~4 小时 (4 GPU 并行)
- **优化**: 减少 num_inference_steps 50→25 → 时间减半 (~2 小时/500 条)

**(c) 多 GPU 并行 Imagination**:
```python
# 4 张 GPU 各加载一份 Ctrl-World + ShortCut Flow
# 每张 GPU 生成 125 条轨迹
# 并行执行, 结果汇总
```

#### 3.4.2 Imagination 数据质量控制

参照 VLAW 论文:
- **Ensemble Sampling**: 每个初始帧生成多条轨迹 (diversity from diffusion sampling)
- **Variance Filtering**: 丢弃视觉质量过差的轨迹 (LPIPS > threshold)
- **VLM 概率阈值**: α=0.8 (保守筛选, 减少 false positive)
- **预期成功率**: ~20-40% 的合成轨迹被标记为成功 (与 VLAW Table 3 一致)

### 3.5 策略更新 — Weighted Flow Matching

#### 3.5.1 损失函数 (VLAW Eq. 4)

```python
# VLAW 的策略更新本质是 Filtered BC:
# 只用成功轨迹做 flow matching supervision
L = E_{(o,a) ~ D_syn+ ∪ D_real+} [L_FM(θ; o, a)]

# 等价于: w(o,a) = 1 if from success traj, else 0
# 所以只需把成功轨迹放入训练集即可
```

**实现**: 在 `ShortCutFlowAgent` 中新增 `compute_weighted_loss()`:
```python
def compute_weighted_loss(self, actions, obs, weights=None):
    """
    与 compute_loss 相同, 但支持 per-sample 权重.
    当 weights=None 时退化为标准 loss (等价于 Filtered BC).
    """
    # ... 标准 flow matching 前向 ...
    loss = mse(predicted_velocity, target_velocity)
    if weights is not None:
        loss = (loss * weights.unsqueeze(-1).unsqueeze(-1)).mean()
    return loss
```

#### 3.5.2 训练细节 (遵循 VLAW 论文)

```python
# 策略更新超参 (与 VLAW 论文一致)
policy_update_steps = 2000
policy_batch_size = 256       # gradient_accumulation 在 4090 上实现
policy_lr = 1e-5              # 与 Ctrl-World lr 一致, 小心不要破坏预训练
warmup_steps = 100            # linear warmup
data_mix_ratio = 0.5          # real:synthetic = 1:1 (每 batch 各一半)
```

#### 3.5.3 数据来源

```
D_real+: 真实 rollout 中 VLM 标记为成功的轨迹
  - 来源: ManiSkill 环境
  - 格式: (image_obs, action_chunk) pairs

D_syn+: Imagination 中 VLM 标记为成功的轨迹
  - 来源: Ctrl-World 生成 → VLM 筛选
  - 格式: (decoded_image, action_chunk) pairs
  - 注意: 合成图像可能有伪影, 但 flow matching 训练对此有一定鲁棒性
```

### 3.6 完整迭代算法 (VLAW Algorithm 1 适配版)

```
Algorithm: VLAW-ManiSkill

输入:
  - 预训练 ShortCut Flow 策略 π_θ (checkpoint: best_eval_success_once.pt)
  - 预训练 Ctrl-World 世界模型 M_ϕ (checkpoint: ctrl-world DROID pretrained)
  - ManiSkill 演示数据 D_demo (用于 WM 正则化)
  - 超参: K_iter=2, K_real=50/task, N_syn=500/task, α=0.8

Phase 0 — 预热:
  - 在 ManiSkill 演示数据上预训练 Ctrl-World action encoder + temporal attention
  - 微调 VLM 奖励模型 (用 D_demo 的 success 标签)

for i = 1 to K_iter:
    === Step 1: 真实环境 Rollout ===
    for each task in tasks:
        τ_real = rollout(π_θ, ManiSkill, K=50 条)
        D_real.append(τ_real)  # 数据累积

    === Step 2: VAE 离线编码 ===
    for τ in D_real_new:
        τ.latent = VAE_encode(concat_cameras(τ.rgb_base, τ.rgb_hand))

    === Step 3: VLM 奖励标注 (真实数据) ===
    for τ in D_real_new:
        render 16 帧 → VLM → P('yes'|τ, I)
        τ.label = 1 if P('yes') > α else 0
    D_real+ = {τ ∈ D_real : τ.label = 1}

    === Step 4: 微调世界模型 ===
    Train Ctrl-World on D_real + λ·D_demo
    50K steps, fp16, gradient_checkpointing
    保存 checkpoint

    === Step 5: Imagination (Policy-in-the-Loop) ===
    for j = 1 to N:
        s₀ ~ sample_initial_frame(D_real)
        τ_syn_j = rollout_in_WM(π_θ, M_ϕ, s₀)
    D_syn = {τ_syn_1, ..., τ_syn_N}

    === Step 6: VLM 奖励标注 (合成数据) ===
    for τ in D_syn:
        VAE_decode(τ.latents) → RGB 帧 → VLM → P('yes')
        τ.label = 1 if P('yes') > α else 0
    D_syn+ = {τ ∈ D_syn : τ.label = 1}

    === Step 7: 策略更新 ===
    Train π_θ on D_real+ ∪ D_syn+
    Weighted Flow Matching loss
    2000 steps, batch 256

    === Step 8: 评估 ===
    Eval π_θ in ManiSkill (50 episodes per task)
    Record: success_rate, success_at_end, reward

end for
```

---

## 四、文件结构规划

### 4.1 新增文件

```
rlft/vlaw/                           ← 核心新模块
├── __init__.py
├── config.py                        ← VLAW 超参配置 (tyro dataclass)
├── data_collector.py                ← ManiSkill rollout + RGB/state 记录
├── data_pipeline.py                 ← VAE 编码 + 数据格式转换
├── ctrl_world_adapter.py            ← Ctrl-World 封装 (适配 ManiSkill 分辨率/相机)
├── train_world_model.py             ← WM 训练 (对 Ctrl-World train_wm.py 的封装)
├── reward_model.py                  ← VLAW 式 VLM 二分类 + 概率阈值
├── train_reward_model.py            ← VLM LoRA 微调
├── imagination.py                   ← Policy-in-the-Loop Rollout 引擎
├── state_predictor.py               ← 轻量 state predictor (MLP, 用于 imagination)
├── policy_updater.py                ← Weighted FM loss + 策略微调
└── evaluation.py                    ← 评估 + Baseline 对比

rlft/online/
└── train_vlaw.py                    ← 主训练脚本 (完整迭代循环, tyro CLI)

scripts/sweep_vlaw/                  ← Sweep 基建 (复用 pld 模板)
├── config.sh
├── configs/vlaw.sh
├── sweep.sh
└── analyze_sweep.py

ctrl_world/                          ← Ctrl-World 代码 (git submodule 或 copy)
├── models/
│   ├── ctrl_world.py
│   ├── pipeline_ctrl_world.py
│   ├── pipeline_stable_video_diffusion.py
│   └── unet_spatio_temporal_condition.py
├── dataset/
│   └── dataset_maniskill.py         ← 新增: ManiSkill 数据加载
├── config.py                        ← 修改: 新增 ManiSkill 配置
└── scripts/
    ├── train_wm.py
    └── rollout_maniskill.py         ← 新增: ManiSkill imagination rollout
```

### 4.2 修改的已有文件

```
rlft/algorithms/il/shortcut_flow.py  ← 新增 compute_weighted_loss()
rlft/algorithms/il/flow_matching.py  ← 新增 compute_weighted_loss()
rlft/envs/make_env.py               ← 新增 RGB 帧记录/保存工具
rlft/__init__.py                     ← 注册 vlaw 模块
```

---

## 五、分阶段实施计划

### Phase 0: 环境搭建与验证 (3-4 天)

**P0.1 — Ctrl-World 环境搭建** (1-2 天)
- [ ] 克隆 Ctrl-World repo → `ctrl_world/`
- [ ] 安装依赖: `diffusers==0.34.0`, `transformers==4.48.1`, `decord`, `einops`, ...
- [ ] 下载预训练权重: SVD (~8GB), CLIP (~600MB), Ctrl-World checkpoint (~8GB)
- [ ] 在 4090 上验证推理: `python scripts/rollout_replay_traj.py` (用 DROID 子集)
- [ ] 测量单卡显存: 推理时 ~12-16GB 预期

**P0.2 — ManiSkill RGB 数据验证** (1 天)
- [ ] 确认 ManiSkill `obs_mode="rgbd"` 输出格式和分辨率
- [ ] 测试 2 相机图像拼接 → VAE 编码 → latent shape
- [ ] 验证 VAE decode(encode(image)) 重建质量 (PSNR > 25 即可)
- [ ] 确认 `env.get_state()` / `env.set_state()` 可用性 (用于 state predictor)

**P0.3 — VLM 模型获取** (1 天)
- [ ] 下载 Qwen3-VL-4B-Instruct 或 Qwen3-VL-8B-Instruct
- [ ] 在 4090 上验证加载和推理 (单张图像 → 文本)
- [ ] 测试对 ManiSkill 渲染图像的零样本质量评估

### Phase 1: 数据收集与管线 (4-5 天)

**P1.1 — ManiSkill Rollout 收集器** (2 天)
- [ ] 实现 `rlft/vlaw/data_collector.py`
  - 用现有 ShortCut Flow 策略在 ManiSkill 中 rollout
  - 同时记录: RGB 帧 (2 相机), agent state, actions
  - 保存为 HDF5 格式
  - 同时记录 ManiSkill `info["success"]` (仅作对照)
- [ ] 验证: 收集 10 条轨迹, 检查数据完整性

**P1.2 — VAE 编码管线** (1-2 天)
- [ ] 实现 `rlft/vlaw/data_pipeline.py`
  - 2 相机图像拼接 → Ctrl-World VAE 编码 → latent
  - 批量处理, 支持多进程
  - Action 归一化 (计算 ManiSkill action 统计量)
- [ ] 验证: 编码 → 解码 → 视觉质量检查

**P1.3 — 演示数据准备** (1 天)
- [ ] 收集 ManiSkill 演示数据 (D_demo)
  - 使用已有 HDF5 demos 或通过 scripted policy 生成
  - 每任务 ~25 条 (与 VLAW 论文一致)
- [ ] 转换为 Ctrl-World 训练格式 (latent + action + text)

### Phase 2: Ctrl-World 适配与训练 (5-7 天)

**P2.1 — Ctrl-World 代码适配** (2-3 天)
- [ ] 修改 `config.py`: 新增 ManiSkill 相关配置
  - 分辨率, 相机数, action_dim, 帧率
- [ ] 新增 `dataset/dataset_maniskill.py`: ManiSkill HDF5 数据加载
  - 加载 latent + action + text
  - 数据增强: 时间偏移, 随机裁剪 history
- [ ] 适配 action encoder: ManiSkill delta pose vs DROID absolute pose
- [ ] 适配 action 归一化: 用 ManiSkill 动作统计量替换 DROID stat.json

**P2.2 — WM 预热训练** (2-3 天)
- [ ] Phase A: 仅微调 action encoder + temporal attention
  - 4 GPU DDP, fp16, gradient_checkpointing
  - ~10K steps on D_demo
  - 验证: action replay → 视频质量 (visual inspection + PSNR)
- [ ] Phase B: 解冻 UNet 全部, 与 D_demo 联合训练
  - ~20K steps
  - 验证: 长 horizon rollout 稳定性

**P2.3 — 世界模型验证** (1 天)
- [ ] Action Replay 测试: 真实动作序列 → WM 预测 → 与 GT 对比
- [ ] 视频质量指标: PSNR, SSIM, LPIPS (参照 VLAW Table 1)
- [ ] 定性分析: 成功/失败轨迹的预测准确度

### Phase 3: VLM 奖励模型 (3-4 天)

**P3.1 — 奖励模型实现** (1-2 天)
- [ ] 实现 `rlft/vlaw/reward_model.py`
  - Qwen3-VL 加载 + 二分类推理
  - P('yes') 概率提取 + 阈值过滤 (α=0.8)
  - 批量推理接口 (多条轨迹)
- [ ] 实现 `rlft/vlaw/train_reward_model.py`
  - LoRA 微调管线
  - 训练数据: ManiSkill rollout 视频 + success 标签

**P3.2 — 奖励模型微调与验证** (2 天)
- [ ] 收集训练数据: 50 条 rollout × 5 任务, ManiSkill success 标签
- [ ] LoRA 微调: 200 steps, batch 128
- [ ] 验证: Confusion Matrix (TP/FP/TN/FN)
  - 目标: FP < 10% (保守筛选)
  - 对比: zero-shot vs finetuned (参照 VLAW Table 3)
- [ ] 在合成图像上测试 (模拟 WM 输出质量)

### Phase 4: Imagination 引擎 (4-5 天)

**P4.1 — State Predictor** (1 天)
- [ ] 实现 `rlft/vlaw/state_predictor.py`
  - 2-layer MLP: (state + action) → next_state
  - 训练数据: 真实 rollout 轨迹
  - 用于 imagination 中补充策略需要的 agent_state

**P4.2 — Policy-in-the-Loop 引擎** (2-3 天)
- [ ] 实现 `rlft/vlaw/imagination.py`
  - ShortCut Flow + Ctrl-World 闭环推理
  - History buffer 管理 (history latents, states)
  - 多 GPU 并行生成
- [ ] 流程验证:
  - 单条轨迹闭环 rollout → 视频保存 → 人工检查
  - 长 horizon (12 步, ~20 秒) 稳定性

**P4.3 — 大规模合成数据生成** (1 天)
- [ ] 500 条轨迹/任务的生成管线
- [ ] VLM 批量评估
- [ ] 数据统计: 成功率, 轨迹长度, VLM 置信度分布

### Phase 5: 策略更新 (2-3 天)

**P5.1 — Weighted Flow Matching** (1-2 天)
- [ ] 在 ShortCut Flow 中新增 `compute_weighted_loss()`
- [ ] 实现 `rlft/vlaw/policy_updater.py`
  - 数据混合: D_real+ ∪ D_syn+
  - Flow matching 监督训练
  - 2000 steps, batch 256

**P5.2 — 策略更新验证** (1 天)
- [ ] 用少量合成数据验证训练管线 (防止 loss 爆炸)
- [ ] 更新前后策略在 ManiSkill 中的 success_rate 对比

### Phase 6: 完整迭代循环 (3-4 天)

**P6.1 — 主训练脚本** (1-2 天)
- [ ] 实现 `rlft/online/train_vlaw.py`
  - 完整 Algorithm 1 循环
  - tyro CLI 参数管理
  - WandB 日志
  - Checkpoint 管理 (每轮保存策略 + WM)

**P6.2 — 2 轮迭代训练** (2-3 天)
- [ ] 第 1 轮: VLM 微调 → 数据收集 → WM 微调 → Imagination → 策略更新
- [ ] 第 2 轮: 数据收集 → WM 微调 → Imagination → 策略更新
- [ ] 监控: WandB dashboard, 每步 success_rate

### Phase 7: 评估与对比 (3-5 天)

**P7.1 — Baselines** (2 天)
| 方法 | 说明 |
|------|------|
| Base Policy | ShortCut Flow 预训练, 不做任何更新 |
| Filtered BC | 直接在真实成功轨迹上微调 (不用世界模型, 2 轮各 50 条) |
| PLD-SAC | 现有 PLD 残差在线 RL (已调优 baseline) |
| DSRL-SAC | 现有 DSRL 噪声空间在线 RL |
| VLAW (ours) | 完整 VLAW: Ctrl-World + VLM Reward + Imagination + Filtered BC |

**P7.2 — 消融实验** (1-2 天)
| 消融 | 说明 |
|------|------|
| VLAW w/o WM grounding | 不微调世界模型, 直接用预热后的 WM |
| VLAW w/o synthetic data | 只用真实成功轨迹, 不做 imagination |
| VLAW fewer synthetic | 减少合成轨迹 (500 → 250) |
| VLAW w/o demo co-training | WM 训练不混合演示数据 (λ=0) |
| VLAW w/ env reward | 用 ManiSkill GT success 替代 VLM reward (上界参考) |

**P7.3 — 评估指标**
- `success_rate`: 主指标 (ManiSkill 原生 success 判定)
- `success_at_end`: 终态成功率
- `reward_mean`: ManiSkill reward (作为参考)
- `vlm_accuracy`: VLM reward model vs ManiSkill GT 的一致率
- `wm_fidelity`: 世界模型视频质量 (PSNR, SSIM, LPIPS)

**P7.4 — 结果呈现** (1 天)
- 成功率对比表 (类似 VLAW Table 2)
- 迭代曲线图 (Base → Iter 1 → Iter 2)
- WM 质量可视化 (action replay 对比图)
- VLM reward confusion matrix

---

## 六、关键技术风险与缓解

| # | 风险 | 严重度 | 概率 | 缓解策略 |
|---|------|-------|------|---------|
| 1 | **Ctrl-World 在 4090 上训练 OOM** | 高 | 中 | ① 降分辨率至 128×128; ② 减少 num_history 至 3-4; ③ DeepSpeed ZeRO-2; ④ 梯度累积代替大 batch |
| 2 | **ManiSkill 渲染图风格与 DROID 差异过大, WM 迁移差** | 高 | 中 | ① Phase A 预热训练充分; ② 数据增强 (颜色抖动, 亮度); ③ 如实在不行, 考虑只用单相机简化问题 |
| 3 | **VLM 在 ManiSkill 渲染图上判别不准** | 中 | 中 | ① 第一轮用 GT 标签微调; ② 调高 α 阈值 (0.85-0.9); ③ 同时用多个 prompt 做集成判断 |
| 4 | **Imagination 中策略输出的 action 让 WM 发散** | 中 | 高 | ① 限制 imagination horizon; ② action clipping; ③ 监控 latent norm, 超阈值截断 |
| 5 | **合成数据质量不足导致策略退化** | 中 | 中 | ① 保守阈值 α=0.8; ② real 数据占比不低于 50%; ③ 策略更新步数保守 (2K, 不多不少) |
| 6 | **VAE 对 ManiSkill 图像编码/解码质量差** | 低 | 低 | SVD VAE 是通用视频 VAE, 泛化能力强。如有问题: 在 ManiSkill 数据上 finetune VAE decoder |
| 7 | **State predictor 不准导致策略在 imagination 中行为异常** | 中 | 中 | ① 增加 state predictor 训练数据; ② 定期用真实 env 验证预测准确度; ③ 备选: obs_mode="state" 单独训策略 |

---

## 七、时间表总览

```
Week 1 (Day 1-5):  P0 环境搭建 + P1 数据管线
Week 2 (Day 6-12): P2 Ctrl-World 适配与训练 + P3 VLM 奖励模型
Week 3 (Day 13-17): P4 Imagination 引擎 + P5 策略更新
Week 4 (Day 18-24): P6 完整迭代 + P7 评估
Week 5 (Day 25-28): P7 补充实验 + 文档整理 (buffer)

预计总工期: 4-5 周
```

---

## 八、依赖项清单

### 8.1 Python 包 (新增)
```
diffusers==0.34.0        # Ctrl-World 依赖
transformers>=4.48.1     # VLM + CLIP
accelerate               # 多 GPU 训练
decord                   # 视频解码
einops                   # 张量操作
mediapy                  # 视频保存
swanlab                  # 可选, 实验跟踪
peft                     # LoRA 微调
scipy                    # 旋转变换
qwen-vl-utils            # Qwen3-VL 工具
```

### 8.2 模型权重
```
stable-video-diffusion-img2vid    # ~8GB (HuggingFace)
clip-vit-base-patch32             # ~600MB (HuggingFace)
Ctrl-World checkpoint             # ~8GB (作者提供)
Qwen3-VL-4B-Instruct             # ~10GB (HuggingFace)
ShortCut Flow checkpoint          # ~40MB (已有)
```

### 8.3 数据
```
ManiSkill3 演示数据               # ~2-5GB (已有或可快速生成)
DROID 子集 (可选, WM 验证用)     # ~5GB
```

---

## 九、成功标准

| 指标 | 最低要求 | 目标值 | VLAW 论文 |
|------|---------|--------|----------|
| Base → VLAW success_rate 提升 | > 10% abs | > 20% abs | 39.2% abs |
| WM 合成数据贡献 | > 5% abs | > 10% abs | 11.6% abs |
| WM PSNR (action replay) | > 18 | > 20 | 21.77 |
| VLM reward FP rate | < 20% | < 10% | 2/18 = 11% |
| 完整 2 轮迭代 | ✓ | ✓ | ✓ |

> **注**: 由于用 ManiSkill 替代真机, ShortCut Flow 替代 π₀.₅, 绝对数值可能与论文差异较大。重点在于验证 **VLAW 框架的有效性** — 即迭代改进趋势和合成数据对策略的正向贡献。
