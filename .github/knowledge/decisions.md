# 架构决策记录 (ADR)

> 已固化到代码的决策用一行摘要。仍活跃影响决策的保留详情。

---

## 已固化（一行摘要）

| ADR | 决策 | 日期 |
|-----|------|------|
| ADR-001 | VLM: Qwen3-VL-4B-Instruct (8.3GB)，替换 Qwen2.5-VL-7B | 02-25 |
| ADR-002 | 2 相机**垂直拼接** → (384×192) → latent (4,48,24) | 02-24 |
| ADR-003 | Phase-A 仅训练 AE + temporal attn（已废弃，Iter1 改全量微调） | 02-24 |

---

## 活跃决策

### ADR-004: State Predictor — 临时脚手架

- **⚠️ 临时**：仅跑通 Imagination 流程用。ManiSkill `env.step()` 精确可用，P4.3 已替换为 env.step() 版本。
- **当前状态**: `imagination_env.py` 已实现 env.step() 模式 ✅

### ADR-005: VLAWSuccessDataset 三级成功识别

按优先级过滤: ① `vlm_reward==1` → ② `success==True` → ③ `env_success.any()`

### ADR-006: ManiSkill 仿真定位

- `env.step()` = 本项目的"真实环境"
- Imagination + WM 的价值 = 评估 WM 质量 + Model-based RL 扩展
- 可通过 `num_envs=1..64` 系统测试数据效率

### ADR-007: WM Iter1 从 pretrained 开始（而非 Phase-A）

- pretrained H20: PSNR=22.35, SSIM=0.79
- Phase-A H20: PSNR=21.70, **SSIM=0.58** (退化严重, temporal attn 过拟合)
- Iter1 全量微调 1.5B UNet, DeepSpeed ZeRO-2, 2000 steps

### ADR-008: VLM 必须用 16 帧多图输入

- 单帧 AUC=0.58 (接近随机) → 16帧 images AUC=0.82
- images > video (可能因 Qwen3-VL 视频降采样丢帧)
- α=0.8 阈值仅在 LoRA 微调后有效，zero-shot p_yes < 0.01
