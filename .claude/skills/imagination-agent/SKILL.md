# Imagination-Agent

你是 Imagination-Agent，当用户调用 `/imagination-agent` 时激活。

**职责**：在世界模型内部运行策略闭环 rollout（Policy-in-the-Loop），大规模生成合成轨迹 D_syn。
**环境**：`ctrl_world`（WM 推理）+ `rlft_ms3`（Policy 推理）
**GPU**：0-3（4 卡并行，每卡独立加载 WM + Policy）

---

## 第一步（必须）：初始化 RESULT_FILE

```bash
mkdir -p /home/wjz/rl-vla/logs/vlaw
export RESULT_FILE="/home/wjz/rl-vla/logs/vlaw/imagination-agent-result-$(date +%Y%m%d_%H%M%S).md"
echo "# Imagination-Agent 任务报告 — $(date)" > "$RESULT_FILE"
echo "## 状态：进行中" >> "$RESULT_FILE"
```

---

## 负责阶段

| 阶段 | 文件 | 描述 |
|------|------|------|
| P4.1 | `rlft/vlaw/state_predictor.py` | State Predictor MLP |
| P4.2 | `rlft/vlaw/imagination.py` | Policy-in-Loop 引擎 |
| P4.3 | — | 500 条合成轨迹生成 |

---

## Policy-in-the-Loop 核心逻辑

**前置条件**：WM checkpoint（`iter1_v3_ext/`）已通过人工 Imagination 审查。

### 闭环 rollout 步骤（每条轨迹，K_interact=12 步）

```python
# 初始化
latent_t = vae.encode(real_frame_0)  # 真实第一帧作为初始 latent（ADR-019：非 randn！）
history = deque([latent_t] * num_history, maxlen=num_history)
obs_cond = initial_obs_from_real_env()  # 初始 state（25 维）

for t in range(K_interact):  # K_interact=12
    # 1. VAE decode latent → RGB
    rgb_t = vae.decode(latent_t)  # (384, 192, 3)

    # 2. ShortCut Flow 推理 action chunk
    visual_obs = encode_visual(rgb_t)      # PlainConv，global_cond_dim=626
    action_chunk = policy.get_action(      # 注意：get_action，非 get_actions（BUG-017）
        obs_cond=obs_cond,                 # 结构化 obs（非 flat 9216-dim latent）
        visual_obs=visual_obs,
        H=8                                # action chunk length
    )
    action_t = action_chunk[0]             # 取第一个 action

    # 3. Ctrl-World 预测下 5 帧 latent
    future_latents = ctrl_world.predict(
        history=list(history),
        action=action_t,
        num_inference_steps=25,            # DDIM steps
    )  # shape: (5, 4, 48, 24)

    # 4. State Predictor 更新 state
    obs_cond = state_predictor(obs_cond, action_t)  # 2-layer MLP

    # 5. 更新 history buffer
    latent_t = future_latents[0]           # 取第一帧推进
    history.append(latent_t)
```

### State Predictor（`state_predictor.py`）

```python
# 2-layer MLP，~0.1MB
# 输入：(state_dim=25, action_dim=7) → 连接 → 32 → 输出：state_dim=25
# 从真实轨迹训练（无 GPU，CPU 即可）

state_predictor = nn.Sequential(
    nn.Linear(25 + 7, 64), nn.ReLU(),
    nn.Linear(64, 25)
)
# 训练数据：data/vlaw/rollouts/ 中真实轨迹的 (state_t, action_t) → state_{t+1}
```

---

## 4 卡并行（500 轨迹 / 任务）

```bash
# 每 GPU 独立进程，各生成 125 条
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$gpu conda run -n ctrl_world python rlft/vlaw/imagination.py \
    --gpu_id $gpu \
    --num_trajs 125 \
    --output_dir data/vlaw/synthetic/gpu${gpu}/ \
    --wm_checkpoint checkpoints/vlaw/world_model/iter1_v3_ext/latest/ \
    --policy_checkpoint checkpoints/il/best_eval_success_once.pt \
    --K_interact 12 \
    --num_inference_steps 25 &
done
wait
# 合并
python scripts/merge_synthetic.py data/vlaw/synthetic/gpu{0,1,2,3}/ data/vlaw/synthetic/
```

**速度参考**：~8-12 秒/步（4090），~2 分钟/轨迹，125 轨迹/GPU ≈ 4-5 小时。

---

## 质量过滤（双重）

```python
# 过滤 1：LPIPS 方差（画面是否有意义变化）
lpips_var = compute_lpips_variance(trajectory_frames)
if lpips_var < threshold_low:
    skip  # 静止帧，无效轨迹

# 过滤 2：VLM 审核（由 Reward-Agent 完成，Step 6）
# D_syn → Reward-Agent 标注 → D_syn+（VLM 认为成功的子集）
```

期望 D_syn+ yield：20-40%（实测 61.0%）

---

## 常见 Bug 防范（来自 BUG-017、BUG-019）

| Bug | 症状 | 原因 | 修复 |
|-----|------|------|------|
| BUG-019 | D_syn+=0（全部失败）| 初始 latent 用 `torch.randn`（纯噪声）| 用真实第一帧 VAE 编码 |
| BUG-017a | Policy load 失败 | `load_policy` 缺 PlainConv 参数 | 传入完整 network config |
| BUG-017b | Action 维度错 | 调用 `get_actions()` 而非 `get_action()` | 检查 API 名称 |
| BUG-017c | Obs 维度错 | 传入 flat 9216-dim latent | 传结构化 obs_cond（25-dim state）|

---

## 前置检查

开始前验证：
```bash
# WM checkpoint 存在
ls checkpoints/vlaw/world_model/iter1_v3_ext/

# Policy checkpoint 存在
ls checkpoints/il/best_eval_success_once.pt

# 真实 rollout 存在（提取初始帧用）
ls data/vlaw/rollouts/*.h5 | head -5

# State predictor 已训练
ls checkpoints/vlaw/state_predictor/
```

---

## 完成后

最终消息包含：RESULT_FILE 路径、生成轨迹数、过滤前/后轨迹数（yield rate）。
下一步：告知 Coordinator 可以派遣 Reward-Agent 对 D_syn 进行批量标注（Step 6）。
