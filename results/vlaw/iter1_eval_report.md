# VLAW Iter-1 评估报告

**日期**: 2026-03-01
**评估条件**: LiftPegUpright-v1, 50 episodes, 16 envs, seed=42, GPU 9

## 1. 评估结果对比

| 方法 | Checkpoint | success_once | success_at_end | mean_reward |
|------|-----------|:------------:|:--------------:|:-----------:|
| **Baseline (Base Policy)** | `checkpoints/il/best_eval_success_once.pt` | **78.1%** | 0.0% | 1.7030 |
| **Iter-1 (修复前, 无EMA)** | `policy_iter1.pt` (无 `ema_agent`) | 10.9% | 0.0% | — |
| **Iter-1 (修复后, EMA)** | `policy_iter1.pt` (含 `ema_agent`) | **17.2%** | 0.0% | 1.5216 |

### 性能变化
- **Baseline → Iter-1 (EMA)**: -60.9% (严重退化)
- **修复前 → 修复后 (EMA fix)**: +6.3% (EMA 修复有帮助但效果有限)

## 2. 问题诊断

### 2.1 EMA 权重问题（已修复）
- **根因**: `_save_checkpoint()` 未提取 `ema_agent` 顶级键
- **影响**: eval 加载时回退到 online 权重（非 EMA）
- **修复**: 在 `_save_checkpoint()` 中添加 EMA 提取逻辑
- **效果**: 10.9% → 17.2% (+6.3%)

### 2.2 微调导致性能退化（核心问题 — 未解决）
即使使用正确的 EMA 权重，Iter-1 策略（17.2%）仍远低于 base policy（78.1%）。

**可能原因**:
1. **学习率过高** (lr=1e-5): 2000 步微调可能过度拟合到少量合成数据
2. **合成数据质量不足**: WM 合成的轨迹可能引入了分布偏移
3. **数据混合比例不当**: real/syn 混合比可能需要调优
4. **EMA 衰减率不匹配**: EMA 权重与 online 权重差异仅 ~0.002，说明 EMA 衰减太快（几乎等于 online）
5. **缺少演示数据共训练**: VLAW 论文中建议混合原始演示数据防止灾难性遗忘

## 3. 修复内容

### 3.1 `rlft/vlaw/policy/policy_updater.py` — `_save_checkpoint()`
```python
# 新增：提取 EMA 权重（与 base ckpt 格式一致）
ema_agent = {
    k.replace("velocity_net_ema.", "velocity_net."): v
    for k, v in agent_sd.items()
    if k.startswith("velocity_net_ema.")
}
if ema_agent:
    ckpt["ema_agent"] = ema_agent
```

### 3.2 Iter-1 Checkpoint 重新保存
- 备份: `policy_iter1.pt.bak`
- 新增 `ema_agent` 键: 154 个参数
- EMA vs online 权重差异: max_diff ~0.001-0.002

## 4. 下一步建议

> **核心问题不是 EMA 格式，而是 Iter-1 微调策略本身退化了。**

建议的优先调查方向：
1. **降低学习率**: 1e-5 → 1e-6 或更低
2. **减少训练步数**: 2000 → 500
3. **加入演示数据共训练**: 混合原始 demo 数据防止遗忘
4. **检查合成数据质量**: VLM 过滤是否有效
5. **检查 EMA 衰减率**: 当前值可能太接近 1.0，导致 EMA ≈ online

## 5. 文件清单

| 文件 | 描述 |
|------|------|
| `checkpoints/vlaw/policy/iter1/policy_iter1.pt` | 修复后的 Iter-1 checkpoint（含 `ema_agent`） |
| `checkpoints/vlaw/policy/iter1/policy_iter1.pt.bak` | 修复前备份 |
| `rlft/vlaw/policy/policy_updater.py` | 修复后的 `_save_checkpoint` 方法 |
| `results/vlaw/pretrained_policy_eval.json` | 最后一次 eval 的 JSON 结果 |
