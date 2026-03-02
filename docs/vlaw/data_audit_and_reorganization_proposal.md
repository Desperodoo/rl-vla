# 数据审计与规整方案

> **生成时间**: 2026-03-02 | **请求人**: 用户 | **状态**: 方案已拟，待审批后执行

---

## 一、当前编码数据集清单

| # | 路径 (相对 `data/vlaw/encoded/`) | 轨迹数 | 成功率 | T 范围 | T 均值 | 源 | camera_hw | 备注 |
|---|------|--------|--------|--------|--------|-----|-----------|------|
| 1 | `demos/.../demo_1771951465.h5` | 25 | 100% | 7-17 | 10.2 | demo | 192×192 | ✅ 干净，25 条官方 demo |
| 2 | `rollouts/iter1/.../real_1772017887.h5` | 50 | 16% | 7-67 | 46.1 | real | 192×192 | ✅ 干净，iter1 policy rollout |
| 3 | `reencode/.../real_1772087806.h5` | 50 | 6% | 17-67 | 56.3 | real | 192×192 | ⚠️ 大量长失败轨迹 |
| 4 | `reencode/.../real_1772098799.h5` | 50 | 70% | **1-8** | **2.3** | real | **128×128** | ❌ **T≤8, 全部无法用于 WM** |
| 5 | `reencode/.../real_1772108871.h5` | 20 | 65% | **1-5** | **2.8** | real | **128×128** | ❌ **T≤5, 全部无法用于 WM** |
| 6 | `reencode/.../real_1772109015.h5` | 20 | 30% | **1-4** | **2.9** | real | **128×128** | ❌ **T≤4, 全部无法用于 WM** |
| 7 | `reencode/.../real_1772109028.h5` | 20 | 30% | **1-4** | **2.9** | real | **128×128** | ❌ **T≤4, 全部无法用于 WM** |

> `reencode_highsuc_inc20` 的完整目录名为 `reencode_highsuc_inc20/LiftPegUpright-v1/`

### T 分布详情 (reencode_highsuc_inc20)

| 文件 | T=1 | T=2 | T=3 | T=4 | T=5-8 | T≥9 | **有效占比** |
|------|-----|-----|-----|-----|-------|-----|------------|
| `1772087806` (50t) | 0 | 0 | 0 | 0 | 0 | **50** | **100%** |
| `1772098799` (50t) | **29** | 3 | 2 | 11 | 5 | **0** | **0%** |
| `1772108871` (20t) | **8** | 0 | 1 | 10 | 1 | **0** | **0%** |
| `1772109015` (20t) | 3 | 0 | 14 | 3 | 0 | **0** | **0%** |
| `1772109028` (20t) | 3 | 0 | 14 | 3 | 0 | **0** | **0%** |
| **合计** | **43** | 3 | 31 | 27 | 6 | **50** | **31%** |

> "有效" = `T ≥ window_len` = 至少 `T ≥ 9` (num_history=4, num_frames=5)。
> 对于 `num_frames=15` 的训练配置, 门槛为 `T ≥ 19`, 仅 `1772087806` 中 49/50 条有效 (T=17 那条也不够)。

---

## 二、核心问题

### 问题 1: reencode 数据集 110/160 条完全无效

**原因**: 文件 #4-#7 (110 条) 的轨迹长度 T ≤ 8, 小于最小 `window_len=9`。  
WM `_index_hdf5()` 会跳过 `T < window_len` 的轨迹, 因此这 110 条轨迹在训练中**完全不使用**。

**验证**: 
- ablation_4000 (num_frames=5, window_len=9): train=2172, val=241, total=2413 → 全部来自 `1772087806` 单个文件
- optimal_steps (num_frames=15, window_len=19): train=2082, val=231, total=2313 → 同上, 窗口更少是因为 window_len 更大

### 问题 2: 相机分辨率混合

| 文件 | 原始分辨率 | 编码后 latent |
|------|-----------|--------------|
| 1772087806 | 192×192 | (T,4,48,24) |
| 1772098799 | **128×128** | (T,4,48,24) |
| 1772108871~109028 | **128×128** | (T,4,48,24) |

128×128 → 192×384 的上采样会引入与 192×192 不同的 artifact。即使 latent shape 相同, **视觉分布不一致**。

### 问题 3: 三组 WM 实验配置完全不可比

| 实验 | 训练数据 | num_frames | action shape | 有效训练窗口 |
|------|---------|-----------|-------------|------------|
| **iter1** | demos only (25t) | ? (→[1,16,7]) | 16 | ~1 (极少) |
| **ablation_4000** | reencode (实际仅 50t) | 5 (→[2,9,7]) | 9 | 2172 |
| **optimal_steps** | reencode (实际仅 49t) | 15 (→[2,11,7]) | 11 | 2082 |

> **训练数据不同 + num_frames 不同 + 有效数据量不同** → PSNR 差异 (23.40 vs 24.11 vs 25.80) 不能归因于 "训练步数", 而是**综合因素的混淆**。

### 问题 4: 训练 val split ≠ 评估 val split

| 场景 | 切分对象 | 比例 | 方向 |
|------|---------|------|------|
| WM 训练 (Dataset_ManiSkill) | **窗口** | 10% | **前 10%** = val |
| Eval 脚本 (load_trajectories) | **轨迹** | 20% | **后 20%** = val |

完全不同的切分逻辑, 但当训练数据和评估数据来自**不同 H5 文件**时 (ablation/optimal 训练用 reencode, eval 用 demos+rollouts), 这个问题不影响结果正确性 (两组数据完全独立)。

仅 **iter1** 实验 (训练用 demos, eval 也用 demos val split) 存在轻微数据泄漏风险 (训练窗口可能包含 val 轨迹的子窗口, 因为切分维度不同)。

### 问题 5: ADR-018 ("1000 步最优") 结论不可靠

ADR-018 的实验 (T-EXP-WM-05) 使用了：
- `--num_frames 15` (与 iter1 的配置不同)
- reencode 数据 (与 iter1 的 demos 不同)
- 仅 4 个 checkpoint (500/1000/1500/2000), 采样过稀疏

因此 **"1000 步最优" 的结论不能直接推广到所有 WM 训练配置**, 它仅在特定 (reencode + num_frames=15) 条件下成立。

---

## 三、规整方案

### Phase 1: 数据标准化 (必做)

#### 1a. 清理 reencode 数据集

```
# 当前: data/vlaw/encoded/reencode_highsuc_inc20/ (5 files, 160 trajs)
# 动作:
1. 将有效数据 (1772087806.h5, 50 trajs, T=17-67) 划为 "reencode_valid/"
2. 将无效数据 (4 files, 110 trajs, T≤8) 移至 "_archive/encoded_reencode_invalid/"
3. 在 valid/ 中添加 README.md 说明来源和数据特征
```

#### 1b. 建立标准评估集 (eval_fixed/)

```
# 目标: 一套固定评估数据, 所有实验共享, 永不改变
# 来源:
#   - demos: 最后 5 条 (traj_0020..0024), T=12-17, 100%成功
#   - rollouts: 最后 10 条 (traj_0040..0049), T=7-67, 混合成败
# 格式: 新建 data/vlaw/encoded/eval_fixed/eval_set.h5 (或软链接)
# 规则: 任何 WM 训练的 dataset_names 禁止包含 eval_fixed
```

#### 1c. 标准化训练 num_frames

```
# 决定: 所有 ManiSkill WM 训练统一使用 num_frames=5
# 理由:
#   1. ManiSkill 轨迹普遍较短 (T=7-67), num_frames=15 浪费大量短轨迹
#   2. Imagination 推理时每步生成 5 帧, 训练和推理一致
#   3. num_frames=5 能利用最多训练数据
# 如需实验 num_frames=15, 作为明确的消融实验 (T-EXP-WM-06)
```

### Phase 2: 补充高质量数据 (推荐)

#### 2a. 重新收集分辨率统一的 rollout 数据

```
# 当前 reencode 里混有 128×128 和 192×192, 需统一:
# - 所有 rollout 统一使用 192×192 (与 demos 和 WM 推理一致)
# - 或对现有 128×128 数据标注清楚, 不混入训练
```

#### 2b. WM 训练混合数据

```
# 推荐混合方案 (用于后续 WM 训练):
dataset_names = "demos+rollouts_clean"
# 其中 rollouts_clean = rollouts/iter1/ 全量 (50 trajs, T=7-67)
# 或 demos+reencode_valid (仅 1772087806, 50 trajs)
# 总计: 75-100 条轨迹, 全部 192×192, T≥7
```

### Phase 3: 评估标准化 (必做)

#### 3a. 统一评估脚本

```python
# 创建 scripts/vlaw/eval/eval_wm_standard.py
# 要求:
# 1. 评估集: 固定使用 eval_fixed/ (不再依赖 val_split)
# 2. 必须包含 pretrained 模型作为 baseline
# 3. 同一脚本评估任意 checkpoint 列表
# 4. 输出标准化 JSON + Markdown 报告
# 5. 逐帧时序衰减曲线 (frame_0 到 frame_4 的 PSNR/SSIM)
```

#### 3b. T-EXP-WM-05 v2 重做 (见下方)

---

## 四、目录结构 (规整后)

```
data/vlaw/encoded/
├── demos/                          # 25 条官方 demo, 100%成功, 192×192
│   └── LiftPegUpright-v1/
│       └── LiftPegUpright-v1_demo_1771951465.h5
├── rollouts/                       # 50 条 Policy rollout, 16%成功, 192×192
│   └── iter1/
│       └── LiftPegUpright-v1/
│           └── LiftPegUpright-v1_real_1772017887.h5
├── reencode_valid/                 # ← NEW: 50 条长轨迹, 6%成功, 192×192
│   └── LiftPegUpright-v1/
│       └── LiftPegUpright-v1_real_1772087806.h5
├── eval_fixed/                     # ← NEW: 标准评估集 (15 trajs, 永不训练)
│   └── README.md
│   └── eval_set.h5                 # 或软链接到 demos + rollouts
└── _archive/
    └── reencode_invalid/           # T≤8 的无效数据 (110 trajs)
        └── LiftPegUpright-v1/
            ├── real_1772098799.h5  # T=1-8, 128×128
            ├── real_1772108871.h5  # T=1-5, 128×128
            ├── real_1772109015.h5  # T=1-4, 128×128
            └── real_1772109028.h5  # T=1-4, 128×128
```

---

## 五、影响评估

| 变更 | 受影响的实验 | 影响程度 |
|------|-----------|---------|
| 清理 reencode | ablation_4000, optimal_steps 的**已完成结果不受影响** (它们本来就只用了 1772087806) | 无 |
| 标准化 num_frames=5 | 后续新实验 | 中 (需重新确认最优步数) |
| 固定评估集 | 所有未来 eval | 高 (保证可比性) |
| 标准化评估脚本 | 所有未来 eval | 高 (统一流程) |
| ADR-018 降级 | 1000 步结论 | **ADR-018 降级为"仅在 reencode+num_frames=15 下成立"**, 需 T-EXP-WM-05 v2 重新验证 |

---

## 六、执行优先级

| 优先级 | 任务 | 阻塞了什么 |
|--------|------|-----------|
| P0 | 建立 eval_fixed 评估集 | 所有未来 WM eval |
| P0 | 创建标准评估脚本 | 所有未来 WM eval |
| P1 | 清理 reencode → reencode_valid | 后续 WM 训练数据选择 |
| P1 | T-EXP-WM-05 v2 | 确定真实最优步数 |
| P2 | 补充统一分辨率数据 | 扩大训练集 |

> **预计工作量**: Phase 1 (1a+1b+1c) < 1h, Phase 3 (3a+3b) < 2h
