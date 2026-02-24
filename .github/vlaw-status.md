# VLAW 复现项目 — 实时状态跟踪

> **最后更新**: 2026-02-24 (WM-Agent + Data-Agent)
> **当前迭代**: P0.1 ⚠️ 部分就绪 / P0.2 ✅ / P0.3 ✅

---

## 阶段状态总览

| 阶段 | 状态 | 负责 Agent | 最后更新 | 备注 |
|------|------|-----------|---------|------|
| **P0.1** Ctrl-World 环境搭建 | 🔄 进行中 | WM-Agent | 2026-02-24 | conda env + 全部权重就绪；待运行推理验证 |
| **P0.2** ManiSkill RGB 验证 | ✅ 已完成 | Data-Agent | 2026-02-24 | obs/concat/state ✅；VAE PSNR=27.83 dB ✅；Latent (1,4,48,24)；代理 10.20.93.149:7890 下载 sd-vae-ft-mse |
| **P0.3** VLM 模型获取 | ✅ 已完成 | Reward-Agent | 2026-02-24 | conda env `vlaw_reward` (Python 3.10, torch 2.8+cu128, transformers 5.2.0, peft 0.18.1); Qwen2.5-VL-7B-Instruct (16GB) @ `checkpoints/vlaw/reward_model/qwen_vl`; 推理验证通过 (VRAM 16.6/25GB, 推理 1.8s); flash-attn 待装 |
| **P1.1** ManiSkill Rollout收集器 | ⬜ 未开始 | Data-Agent | — | — |
| **P1.2** VAE 编码管线 | ⬜ 未开始 | Data-Agent | — | — |
| **P1.3** 演示数据准备 | ⬜ 未开始 | Data-Agent | — | — |
| **P2.1** Ctrl-World 代码适配 | ⬜ 未开始 | WM-Agent | — | — |
| **P2.2** WM 训练 (Phase A+B) | ⬜ 未开始 | WM-Agent | — | — |
| **P2.3** WM 验证 | ⬜ 未开始 | WM-Agent | — | — |
| **P3.1** 奖励模型实现 | ⬜ 未开始 | Reward-Agent | — | — |
| **P3.2** 奖励模型微调验证 | ⬜ 未开始 | Reward-Agent | — | — |
| **P4.1** State Predictor | ⬜ 未开始 | Imagination-Agent | — | — |
| **P4.2** Imagination 引擎 | ⬜ 未开始 | Imagination-Agent | — | — |
| **P4.3** 大规模合成数据 | ⬜ 未开始 | Imagination-Agent | — | — |
| **P5.1** Weighted FM Loss | ⬜ 未开始 | Policy-Agent | — | — |
| **P5.2** 策略更新验证 | ⬜ 未开始 | Policy-Agent | — | — |
| **P6.1** 主训练脚本 | ⬜ 未开始 | Coordinator | — | — |
| **P6.2** 2 轮迭代训练 | ⬜ 未开始 | Coordinator | — | — |
| **P7.1** Baselines | ⬜ 未开始 | Eval-Agent | — | — |
| **P7.2** 消融实验 | ⬜ 未开始 | Eval-Agent | — | — |
| **P7.3** 评估指标 | ⬜ 未开始 | Eval-Agent | — | — |
| **P7.4** 结果呈现 | ⬜ 未开始 | Eval-Agent | — | — |

**状态图例**: ⬜ 未开始 | 🔄 进行中 | ✅ 已完成 | ❌ 阻塞 | ⚠️ 需要修复

---

## 模型 Checkpoints

| 模型 | 路径 | 状态 | 指标 |
|------|------|------|------|
| ShortCut Flow (Base) | `checkpoints/il/best_eval_success_once.pt` | ✅ 已有 | Base 策略 |
| Ctrl-World (DROID pretrained) | `checkpoints/vlaw/world_model/pretrained/` | ✅ 已就绪 (CLIP 581MB + SVD 7GB + CW checkpoint-10000.pt 8.7GB) | — |
| Ctrl-World (ManiSkill finetuned) | `checkpoints/vlaw/world_model/` | ⬜ 待训练 | PSNR: — |
| VLM Reward (Qwen3-VL) | `checkpoints/vlaw/reward_model/qwen_vl` | ✅ 已下载 (Qwen2.5-VL-7B) | FP: — |
| State Predictor | `checkpoints/vlaw/state_predictor/` | ⬜ 待训练 | — |
| ShortCut Flow (VLAW Iter 1) | `checkpoints/vlaw/policy/iter1/` | ⬜ 待训练 | SR: — |
| ShortCut Flow (VLAW Iter 2) | `checkpoints/vlaw/policy/iter2/` | ⬜ 待训练 | SR: — |

---

## 数据状态

| 数据集 | 路径 | 状态 | 数量 |
|--------|------|------|------|
| P0.2 验证结果 | `data/vlaw/validation/p0_2_validation_results.json` | ✅ 已完成 | ManiSkill ✅; VAE PSNR=27.83 dB ✅; Latent (1,4,48,24) |
| ManiSkill 演示 (D_demo) | `data/vlaw/demos/` | ⬜ 待收集 | 目标: 25条/任务 |
| 真实 Rollout (D_real) Iter 1 | `data/vlaw/rollouts/iter1/` | ⬜ 待收集 | 目标: 50条/任务 |
| 合成数据 (D_syn) Iter 1 | `data/vlaw/synthetic/iter1/` | ⬜ 待生成 | 目标: 500条/任务 |
| VAE 编码数据 | `data/vlaw/encoded/` | ⬜ 待编码 | — |
| 真实 Rollout (D_real) Iter 2 | `data/vlaw/rollouts/iter2/` | ⬜ 待收集 | 目标: 50条/任务 |
| 合成数据 (D_syn) Iter 2 | `data/vlaw/synthetic/iter2/` | ⬜ 待生成 | 目标: 500条/任务 |

---

## GPU 使用状态

| GPU | 当前分配 | 状态 |
|-----|---------|------|
| GPU 0-3 | WM-Agent (Ctrl-World 训练) | 🟢 空闲 |
| GPU 4-5 | Data-Agent (ManiSkill Rollout) | 🟢 空闲 |
| GPU 6-7 | Reward-Agent (VLM: Qwen2.5-VL-7B) | 🔵 P0.3 完成，等待 P3.1 |
| GPU 8-9 | Policy-Agent / Eval-Agent | 🟢 空闲 |

---

## 迭代历史

### 预热 (P0-P3)
- 开始时间: —
- 完成时间: —
- 备注: —

### Iteration 1
- 开始时间: —
- D_real 收集: —
- WM 训练: —
- Imagination: — (成功率: —)
- 策略更新: — (SR 变化: — → —)

### Iteration 2
- 开始时间: —
- D_real 收集: —
- WM 训练: —
- Imagination: — (成功率: —)
- 策略更新: — (SR 变化: — → —)

---

## 问题日志

| # | 日期 | 问题 | 状态 | 解决方案 |
|---|------|------|------|---------|
| 1 | 2026-02-24 | 根目录磁盘 100% 满，flash-attn 编译失败 | ⚠️ 部分解决 | 清理 pip cache (22GB) + conda tarballs (2.5GB)，释放 25GB；flash-attn 可在磁盘充裕时重装 |
| 2 | 2026-02-24 | 服务器无法直接访问 HuggingFace，VAE 权重无法下载 | ✅ 已解决 | 设置代理 `http_proxy=http://10.20.93.149:7890` 后正常下载；已写入 copilot-instructions.md |
| 3 | 2026-02-24 | ManiSkill3 标准任务仅有 `base_camera` 入 sensor_data，无 hand_camera | ⚠️ 已知设计限制 | P0.2 用 `env.render()` 获取第二视角做 shape 验证；P1.1 需自定义 env 子类覆盖 `_default_sensor_configs` 增加手腕相机 |
| 4 | 2026-02-24 | `yjguo/Ctrl-World` HF 仓库下载后只有 README.md/config.json，无 .safetensors 权重文件 | ✅ 已解决 | 实际权重文件为 `checkpoint-10000.pt`；改用 `hf_hub_download` 直接下载单文件，8.7GB 下载完成 |
| 5 | 2026-02-24 | SVD 下载产生 17G .cache 冗余(blob 重复存储) | ✅ 已解决 | 删除 `.cache/` 释放 16G；修正 download_weights.py 下载后自动清理 .cache |

---

## 更新规则
- 每个 Agent 在完成分配的子任务后，更新对应行的状态
- 格式: `| **PX.X** 任务名 | ✅ 已完成 | Agent名 | YYYY-MM-DD | 备注 |`
- 遇到问题时添加到"问题日志"
- 迭代完成后填写"迭代历史"
