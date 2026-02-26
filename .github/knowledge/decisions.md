# 架构决策记录 (ADR)

> 记录关键技术决策的背景、理由和被放弃的替代方案。

---

## ADR-001: ~~使用 Qwen2.5-VL-7B~~ → 已迁移至 Qwen3-VL-4B-Instruct

- **日期**: 2026-02-24（创建）/ 2026-02-25（更新）
- **决策**: 使用 `Qwen/Qwen3-VL-4B-Instruct`（8.3GB）作为奖励模型
- **背景（原）**: 2026-02-24 下载时 Qwen3-VL-4B 在 HuggingFace 上不可用，临时使用 Qwen2.5-VL-7B-Instruct
- **更新原因**: 2026-02-25 确认 `Qwen/Qwen3-VL-4B-Instruct` 已在 HuggingFace 上线发布，恢复原计划
- **理由**: 4B 模型 VRAM 仅 8.88GB（vs 7B 的 16.6GB），节省 7.7GB 显存，GPU 6 可与其他任务共存
- **影响**: VRAM 从 16.6GB → 8.88GB；类名从 `Qwen2_5_VLForConditionalGeneration` 更新为 `Qwen3VLForConditionalGeneration`
- **放弃方案**: Qwen2.5-VL-7B-Instruct（已删除磁盘文件，释放 16GB）

---

## ADR-002: 2 相机垂直拼接方案（vs 水平拼接）

- **日期**: 2026-02-24
- **决策**: base_camera (192×192) + hand_camera (192×192) **垂直拼接** → (384×192)
- **理由**:
  1. 与 Ctrl-World 原版 DROID 3-cam 垂直拼接保持一致，修改量最小
  2. VAE 对 tall 图像处理更稳定（latent 48×24 vs 水平的 24×48）
  3. PSNR=27.83 dB 验证质量可接受
- **放弃方案**: 水平拼接 → (192×384)；单相机 192×192（信息量不足）

---

## ADR-003: Phase-A 仅训练 Action Encoder + Temporal Attention

- **日期**: 2026-02-24
- **决策**: Phase-A 冻结 UNet 空间层，仅训练 Action Encoder (~3M) + temporal attention
- **理由**:
  1. 直接全量微调 1.5B UNet 在 3×25=75 条数据上极易过拟合
  2. Action Encoder 负责动作条件编码，需要适配 ManiSkill delta pose（vs DROID 绝对位姿）
  3. Temporal attention 控制时序一致性，需要适配新的轨迹长度
- **步数分配**: Phase-A 10K steps（热身），Phase-B 20-50K steps（全量）

---

## ADR-004: State Predictor 残差 MLP（⚠️ 临时脚手架，P4.3 必须替换）

- **日期**: 2026-02-25
- **决策**: StatePredictor 预测 Δstate 而非直接预测 state_{t+1}
- **实现**: `state_{t+1} = state_t + MLP(concat(state_t, action_t))`
- **⚠️ 性质：临时脚手架**。此方案仅用于"跑通 Imagination 代码流程"，不是最终方案
- **原因**: ManiSkill 是仿真环境，`env.step(a)` 可以精确返回 $s_{t+1}$，不需要学一个近似 MLP
- **最终目标 (P4.3)**: 将 `imagination.py` 中的 Step 5 从 `state_predictor.predict_sequence()` 替换为 `env.step()` 调用
  - 可通过控制 `num_envs=1` 到 N 来调节并行规模，系统测试合成数据量 vs 策略提升的数据效率
  - 替换后 `state_predictor.py` 模块可降为可选依赖
- **放弃方案**: vision-only obs（丢失 29D 本体状态信息，策略质量下降）

---

## ADR-006: ManiSkill 仿真替代真机——Imagination 的根本定位

- **日期**: 2026-02-25
- **背景**: 原版 VLAW 面向 DROID 真实机器人，Imagination（世界模型推理）的意义在于"无需真实环境就能生成合成数据"
- **本项目定位**: 用 ManiSkill 仿真**暂时替代真机**复现 VLAW，目的是验证算法机制和数据效率
- **核心结论**:
  1. ManiSkill `env.step()` 就是本项目的"真实环境"，精确且免费
  2. Imagination 在本项目中的作用是**评估世界模型质量** + 与 env.step() 结果对比验证 WM 预测能力
  3. 最终 Imagination 应使用 `env.step()` 获取精确状态，State Predictor MLP 只是流程调通前的占位
  4. 可通过控制 ManiSkill 并行规模（`num_envs=1` 到 64）和合成数据数量来对比数据效率
- **影响**: P4.3 任务从"调参 State Predictor"变为"将 Imagination 改为 env.step() 版本"；WM 仍有价值（作为 Model-based RL 的预测器和数据扩充来源）

---

## ADR-005: VLAWSuccessDataset 三级成功识别策略

- **日期**: 2026-02-25
- **决策**: 按优先级三级过滤成功轨迹
- **背景**: 不同阶段产生的 HDF5 数据，成功标记字段名称不一致
- **策略**:
  1. `grp.attrs["vlm_reward"] == 1`（VLM 标注，最权威）
  2. `grp.attrs["success"] == True`（data_collector 写入的 env_success 汇总）
  3. `grp["env_success"].any()`（逐帧 OR 聚合，降级兜底）
- **原因**: 向前兼容，避免因字段名变化导致 dataset 返回 0 samples
