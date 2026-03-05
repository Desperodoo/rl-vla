# Iter-1 Imagination 合成数据评估报告

- **合成数据**: synthetic_iter1_merged.h5 (200 trajs, 96.1MB)
- **真实数据**: LiftPegUpright-v1_real_1772643507.h5 (1200 trajs)
- **VLM 标注**: LiftPegUpright-v1_vlm_rewards.json (122/200 = 61.0% success, α=0.5)
- **评估脚本**: `rlft/vlaw/scripts/eval_imagination.py`
- **评估时间**: 2026-03-05

---

## 1. 逐轮解码质量 (Per-round VAE Decode)

每条轨迹 60 帧 = 12 轮 × 5 帧/轮。取每轮第一帧做 VAE decode → RGB，计算 no-reference 指标。

| Round | Brightness (mean±std) | Color Var | Sharpness |
|-------|----------------------|-----------|-----------|
| 0 | 124.7 ± 0.2 | 1234 ± 22 | 169 ± 9 |
| 1 | 124.6 ± 0.5 | 1156 ± 46 | 139 ± 13 |
| 2 | 124.8 ± 0.6 | 1190 ± 73 | 141 ± 18 |
| 3 | 125.1 ± 0.7 | 1193 ± 69 | 141 ± 16 |
| 4 | 125.4 ± 0.7 | 1189 ± 67 | 137 ± 16 |
| 5 | 125.4 ± 0.8 | 1193 ± 62 | 138 ± 16 |
| 6 | 125.3 ± 0.8 | 1184 ± 60 | 136 ± 16 |
| 7 | 125.2 ± 0.8 | 1168 ± 65 | 132 ± 15 |
| 8 | 125.2 ± 0.8 | 1155 ± 56 | 129 ± 13 |
| 9 | 125.2 ± 1.0 | 1150 ± 59 | 129 ± 15 |
| 10 | 124.9 ± 0.9 | 1136 ± 61 | 126 ± 16 |
| 11 | 124.8 ± 0.9 | 1124 ± 56 | 123 ± 14 |

**观察**:
- **Brightness 稳定** (~124-125)，无明显偏移
- **Color Variance 衰减 ~9%** (1234 → 1124)：后续轮图像色彩多样性降低
- **Sharpness 稳减 ~27%** (169 → 123)：自回归累积导致图像逐渐模糊，Round 0→1 最大跌幅 (169→139)
- 衰减速率在 R3 后趋缓（color_var/sharpness 变化变小）

**Strip 可视化** (每条 12 帧，每轮首帧):
- `traj_0000_suc_p0.78_strip.png` — 成功
- `traj_0001_suc_p0.78_strip.png` — 成功
- `traj_0004_suc_p0.73_strip.png` — 成功
- `traj_0002_fail_p0.38_strip.png` — 失败
- `traj_0003_fail_p0.12_strip.png` — 失败

---

## 2. Latent 统计 (Synthetic vs Real)

### Per-channel 分布

| Channel | Syn Mean | Syn Std | Real Mean | Real Std | Δ Mean |
|---------|----------|---------|-----------|----------|--------|
| 0 | 0.3149 | 0.9316 | 0.3269 | 0.9415 | -0.0120 |
| 1 | -0.3918 | 1.0794 | -0.3794 | 1.0707 | -0.0124 |
| 2 | -0.1946 | 0.8206 | -0.1905 | 0.8161 | -0.0041 |
| 3 | 0.2513 | 0.5776 | 0.2452 | 0.5847 | +0.0061 |

**观察**:
- **分布非常接近**: 所有 channel 的 Δ Mean 均 < 0.02，Δ Std < 0.01
- WM 生成的 latent 没有显著偏离真实分布 ✅

### L2 Norm 趋势

- **Real L2 norm**: 62.48 ± 0.30
- **Syn R0**: 62.23, **Syn R11**: 62.79
- **L2 drift (R11-R0)**: 0.56 (< 1%)

**观察**:
- L2 drift 极小 (< 1%)，latent 没有发散 ✅
- 轻微上移趋势（R0 略低于 Real, R11 略高于 Real）

### Channel Mean Drift

Channel 0-3 的 per-round mean shift (relative to real):
- 各 channel 偏移量稳定在 ±0.02 以内，无发散趋势 ✅

---

## 3. Action 分析

### 分布对比

| Dim | Syn Mean | Syn Std | Real Mean | Real Std | Δ Mean | Δ Std |
|-----|----------|---------|-----------|----------|--------|-------|
| x | -0.0998 | 0.2066 | -0.0353 | 0.2098 | -0.064 | -0.003 |
| y | 0.0109 | 0.2314 | 0.0161 | 0.2373 | -0.005 | -0.006 |
| z | -0.0998 | 0.5452 | -0.0610 | 0.3948 | -0.039 | **+0.150** |
| rx | -0.0045 | 0.4449 | 0.0393 | 0.3434 | -0.044 | **+0.102** |
| ry | **-0.7626** | 0.3941 | -0.2469 | 0.4693 | **-0.516** | -0.075 |
| rz | -0.0488 | 0.5000 | -0.0458 | 0.3356 | -0.003 | **+0.164** |
| gripper | -0.1476 | 0.5582 | -0.3831 | 0.5045 | **+0.236** | +0.054 |

**关键发现**:
- **ry 维度偏移显著** (syn mean -0.76 vs real -0.25): 合成轨迹旋转 bias 较大
- **z, rx, rz 方差偏大** (syn std > real std ~15-49%): WM 生成的动作噪声更大
- **gripper 偏移**: syn mean -0.15 vs real -0.38, 合成轨迹的夹爪更倾向打开
- x, y 维度匹配良好 ✅

### Action 平滑度

- **Mean L2 diff**: 0.8056 ± 0.1168
- 平滑度在 round boundary (每 5 帧) 处无明显突变

### 异常检测

- **11/200 (5.5%) 轨迹有 Frozen_dim4 问题**: ry 维度 >80% 帧近乎不变
- 无 NaN, 无 Inf, 无超范围 ✅

---

## 4. State 轨迹分析

### 成功 vs 失败

- Success: 122 trajs (61.0%)
- Fail: 78 trajs (39.0%)
- State NaN: False ✅, Inf: False ✅

### Terminal State 对比

| Dim | Success (Mean ± Std) | Fail (Mean ± Std) | Δ |
|-----|---------------------|-------------------|---|
| ee_x | -0.005 ± 0.075 | 0.016 ± 0.070 | -0.021 |
| ee_y | 1.064 ± 0.150 | 1.068 ± 0.151 | -0.004 |
| ee_z | **0.117 ± 0.077** | **0.098 ± 0.082** | **+0.019** |
| gripper | 0.010 ± 0.008 | 0.011 ± 0.008 | -0.001 |
| dim_9 (obj) | 0.021 ± 0.062 | 0.031 ± 0.097 | -0.010 |
| dim_10 (obj) | **0.052 ± 0.519** | **0.004 ± 0.357** | **+0.048** |

**观察**:
- **ee_z 差异最明显**: 成功轨迹终态 EE 高度略高 (0.117 vs 0.098)，与 "lift peg upright" 任务一致
- dim_10 (可能是物体角度/位置) 在成功轨迹中方差更大
- 其他维度差异较小

---

## 5. VLM 标注分解

### p_yes 分布

- **Mean**: 0.5596 ± 0.1888
- **Median**: 0.5622
- **Range**: [0.0953, 0.9399]

| p_yes Range | Label | Count | Fraction |
|-------------|-------|-------|----------|
| [0.0, 0.3) | low | 19 | 9.5% |
| [0.3, 0.5) | medium-low | 42 | 21.0% |
| [0.5, 0.7) | medium-high | 93 | **46.5%** |
| [0.7, 1.0) | high | 46 | 23.0% |

**Threshold sensitivity**:
- α=0.5 → 122/200 pass (61.0%)
- α=0.8 → 19/200 pass (9.5%)

### 相关性分析

| Factor | Correlation (r) | Interpretation |
|--------|-----------------|----------------|
| p_yes vs Action Smoothness | 0.165 | 弱正相关 — 更平滑的动作轨迹略倾向被判为成功 |
| p_yes vs Last-round L2 | 0.061 | 几乎无相关 |
| p_yes vs Terminal EE Height | 0.098 | 极弱正相关 — EE 更高略有助于成功判定 |

**观察**:
- VLM 判断与单一物理指标相关性均较弱，说明 VLM 是从视觉整体判断，而非简单依赖某个物理量 ✅
- p_yes 分布集中在 0.5-0.7 区间 (46.5%)，说明很多轨迹处于模糊地带

---

## 综合结论

### ✅ 正面发现
1. **Latent 分布匹配良好**: syn vs real 的 per-channel Δ Mean < 0.02, L2 drift < 1%
2. **无数据异常**: 无 NaN/Inf, state 物理合理
3. **VLM 标注合理**: p_yes 分布合理, 与物理量弱相关 (非过拟合)
4. **解码质量衰减可控**: brightness 稳定, sharpness 衰减 ~27% 但趋缓

### ⚠️ 注意事项
1. **Action ry 偏移大** (Δ=-0.516): WM 生成的旋转动作有 bias, 可能影响策略学习
2. **z/rx/rz 方差偏大**: 合成动作比真实数据更 noisy
3. **11/200 有 frozen dim4**: 5.5% 轨迹的 ry 维度几乎不变 (可能是初始帧依赖问题)
4. **α=0.5 到 α=0.8 过滤 drop 严重** (61% → 9.5%): 大量轨迹的 p_yes 在 0.5-0.7, 质量信号不强

### 建议
- 策略更新时使用 α=0.5 以获取足够 positive 数据量 (122 条)
- 监控 ry bias 对策略学习的影响
- Iter-2 WM 训练可考虑加入 action loss regularization 减少旋转 bias

---

## 输出文件索引

| 路径 | 内容 |
|------|------|
| `results/vlaw/imagination_eval/report.md` | 本报告 |
| `results/vlaw/imagination_eval/full_results.json` | 完整 JSON 结果 |
| `results/vlaw/imagination_eval/decode/` | VAE 解码可视化 + 质量统计 |
| `results/vlaw/imagination_eval/decode/per_round_quality_decay.png` | 逐轮质量衰减曲线 |
| `results/vlaw/imagination_eval/decode/traj_*_strip.png` | 5 条代表性轨迹 strip 可视化 |
| `results/vlaw/imagination_eval/latent/` | Latent 统计分析 |
| `results/vlaw/imagination_eval/latent/latent_channel_distribution.png` | Per-channel 分布对比 |
| `results/vlaw/imagination_eval/latent/latent_l2_drift.png` | L2 norm drift 曲线 |
| `results/vlaw/imagination_eval/latent/latent_channel_drift.png` | Channel mean drift |
| `results/vlaw/imagination_eval/action/` | Action 分析 |
| `results/vlaw/imagination_eval/action/action_distribution.png` | 7 维 action 分布对比 |
| `results/vlaw/imagination_eval/action/action_smoothness.png` | 平滑度分布 + 时序曲线 |
| `results/vlaw/imagination_eval/state/` | State 轨迹分析 |
| `results/vlaw/imagination_eval/state/state_trajectories.png` | 成功 vs 失败 state 走势 |
| `results/vlaw/imagination_eval/vlm/` | VLM 标注分析 |
| `results/vlaw/imagination_eval/vlm/p_yes_distribution.png` | p_yes 直方图 + CDF |
| `results/vlaw/imagination_eval/vlm/threshold_sensitivity.png` | 阈值敏感度 |
| `results/vlaw/imagination_eval/vlm/corr_pyes_*.png` | p_yes 与各因素相关性 scatter |
