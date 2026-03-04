# VLAW 复现项目 — 状态仪表盘

> **最后更新**: 2026-03-04 (T+8, Fresh Start)
> **重做计划**: [`VLAW_FRESH_START_PLAN.md`](VLAW_FRESH_START_PLAN.md)
> **核心参考**: [`VLAW_REPRODUCTION_PLAN.md`](VLAW_REPRODUCTION_PLAN.md) | [`knowledge/`](knowledge/)
> **v1 归档**: `_archive/v1/vlaw-status_v1.md`

---

## ⚠️ Fresh Start — BUG-020 数据重置

**触发原因**: `demo_prep.py` L296 `rgb_render = rgb_base.copy()` 导致全部 demo 数据双相机坍塌。
WM/VLM/Imagination/Policy 全链路基于污染数据的结果全部作废。详见 [VLAW_FRESH_START_PLAN.md](VLAW_FRESH_START_PLAN.md)。

**当前阶段**: Phase 0 — 数据准备（待启动）

---

## 干净资产

| 资产 | 路径 | 状态 |
|------|------|------|
| 预训练策略 | `checkpoints/il/best_eval_success_once.pt` | ✅ baseline ~78% |
| WM 预训练 | `checkpoints/vlaw/world_model/pretrained/Ctrl-World/checkpoint-10000.pt` | ✅ 8.7GB |
| SVD / CLIP | `checkpoints/vlaw/world_model/pretrained/{svd,clip}/` | ✅ |
| VLM 基座 | `checkpoints/vlaw/reward_model/qwen_vl/` | ✅ 8.3GB |
| State Predictor | `checkpoints/vlaw/state_predictor/` | ✅ |
| 代码库 | `rlft/vlaw/`, `scripts/vlaw/`, `ctrl_world/` | ✅ 多轮 debug 完成 |
| stat.json | `data/vlaw/meta_info/maniskill/stat.json` | ⚠️ 待从新数据重新生成 |

---

## 数据目录

| 目录 | 状态 | 说明 |
|------|------|------|
| `data/vlaw/rollouts/` | 空 | 待 Phase 0 收集 |
| `data/vlaw/encoded/` | 空 | 待 Phase 0 编码 |
| `data/vlaw/synthetic/` | 不存在 | 待 Phase 3 生成 |
| `data/vlaw/labeled/` | 不存在 | 待 Phase 3 标注 |
| `data/vlaw/_archive/v1_contaminated/` | 📦 | 全部旧数据 (~2.1GB) |

---

## Checkpoints

| 目录 | 状态 | 说明 |
|------|------|------|
| `checkpoints/vlaw/world_model/pretrained/` | ✅ | SVD + CLIP + Ctrl-World |
| `checkpoints/vlaw/world_model/` (其他) | 空 | 待 Phase 1 训练 |
| `checkpoints/vlaw/reward_model/qwen_vl/` | ✅ | Qwen3-VL-4B |
| `checkpoints/vlaw/reward_model/` (其他) | 空 | 待 Phase 2 训练 |
| `checkpoints/vlaw/policy/` | 空 | 待 Phase 4 训练 |
| `checkpoints/vlaw/_archive/v1/` | 📦 | 全部旧 ckpt (~168GB) |

---

## GPU 状态

| GPU | 分配 | 状态 |
|-----|------|------|
| 0-3 | WM 训练 | 🟢 空闲 |
| 4-5 | 数据收集 / VAE 编码 | 🟢 空闲 |
| 6-7 | VLM 微调 / 推理 | 🟢 空闲 |
| 8-9 | 策略训练 + 评估 | 🟢 空闲 |

---

## v1 关键经验总结（完整记录见 `knowledge/`）

> 以下是 v1 中值得保留的核心结论，避免 Fresh Start 中重蹈覆辙。

### Bugs — 必须记住
- **BUG-019**: Imagination 初始 latent 用了 `torch.randn` → 必须用真实帧 VAE 编码
- **BUG-020**: `demo_prep.py` `rgb_render = rgb_base.copy()` → 全数据链污染
- **BUG-017**: Imagination 三合一 bug (PlainConv参数 + API + obs格式) → 已修复在代码中

### Decisions — 已验证的结论
- **ADR-008/015**: VLM video 模式 AUC=0.83 >> multi-image 0.72，`use_video_format=True` 正确
- **ADR-010**: WM 4000步 vs 2000步无额外收益
- **ADR-011**: VLM 16帧最优 (vs 4帧 acc=0.706 / 8帧 acc=0.735)
- **ADR-016/021**: History buffer 对齐消融无显著差异，保留 sliding window
- **ADR-025**: `num_interact=4` 方案 A No-Go (20帧太短)，保持 `num_interact=12`
- **VLM LoRA**: r=16 最优，r=32/64 在 accum=128 下反而退化
- **VLM gradient_accumulation**: 128 导致 full-batch GD + 负样本主导 → 改用 8~16
- **BC 飞轮**: D_syn+ 在小数据场景价值巨大 (+240%)，1条 D_syn+ ≈ 28条 demo

### 可直接复用的结论
- WM: 从 pretrained 起步微调，2000步足够
- VLM: LoRA r=16, 200步, 16帧, video 模式, accum=8~16
- Imagination: `num_interact=12`, BUG-019 已修复在代码中
- 评估: 收集后立即验证 `rgb_base` vs `rgb_render` diff > 30
