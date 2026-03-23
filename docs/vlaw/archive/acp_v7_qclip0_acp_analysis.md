# ACP v7 qclip0 专项诊断报告

**日期**：2026-03-23

## 依据
- 正式对照报告：`docs/vlaw/figures/rlpd_acp_v7_qclip0_acp/diagnosis_report.md`
- 诊断汇总：`docs/vlaw/figures/rlpd_acp_v7_qclip0_acp/diagnosis_summary.json`
- 共享 drift 结论：`docs/vlaw/archive/acp_v7_drift_regression_analysis.md`

---

## 1. 这轮实验回答了什么

这轮 `qclip0_acp_diag` 的目标非常明确：

> 在已经去掉共享 drift（`q_target_clip=20`）之后，PLD/DSRL 的 ACP 镜像是否还能恢复到和 sim baseline 接近的水平？

答案现在已经非常清楚：

> **不能。**

也就是说：
- `q_target_clip=20` 确实是此前 sim baseline 失效的第一根因；
- 但把它去掉以后，**ACP 仍然单独失败**；
- 这证明 ACP 存在**第二层独立问题**，而不只是被共享 drift 连带拖垮。

---

## 2. 结果总表

来自 `docs/vlaw/figures/rlpd_acp_v7_qclip0_acp/diagnosis_report.md`：

| Algo | Reward | Best SO | Best SAE | Final SAE | SAE Retention | SO-SAE Gap |
|---|---:|---:|---:|---:|---:|---:|
| DSRL | sim | 0.940 | **0.660** | **0.660** | 0.702 | 0.280 |
| DSRL | acp | 0.940 | 0.060 | 0.000 | 0.064 | 0.880 |
| PLD | sim | 0.980 | **0.820** | **0.820** | 0.837 | 0.160 |
| PLD | acp | 0.700 | 0.020 | 0.000 | 0.029 | 0.680 |

这个表说明了最核心的事实：

### PLD
- sim baseline (`qclip0`) 恢复到 `0.82 / 0.82`
- ACP mirror 只有 `0.02 / 0.00`

### DSRL
- sim baseline (`qclip0`) 恢复到 `0.66 / 0.66`
- ACP mirror 只有 `0.06 / 0.00`

因此：

> **在去掉 qclip 之后，sim vs ACP 的差距不仅没有消失，反而被完整暴露出来了。**

---

## 3. 现在可以明确拆分成两层问题

### 第一层：共享 drift（已经基本解决）
- `q_target_clip=20` 破坏了历史 sim baseline 的复现
- 这件事已经通过上一轮 drift regression 被强证实

### 第二层：ACP 专属失效（这轮被单独证实）
- 即使 sim baseline 已恢复
- PLD/DSRL 的 ACP mirror 仍然基本不工作

这意味着：

> 现在已经不需要再纠结“是不是 code-path drift 导致 ACP 看起来差”。
>
> **不是。ACP 在去掉共享 drift 后仍然明显差。**

---

## 4. 从算法视角重新看 ACP 失效

这轮结果与之前的深层分析完全吻合，但证据更强了。

### 4.1 PLD + ACP
`pld_v7_qclip0_acp_mirror_s42`：
- best SO = 0.70
- best SAE = 0.02
- final SAE = 0.00
- actor / exploration 都不健康于 sim baseline 对照

而 `pld_v7_qclip0_sim_baseline_s42`：
- best SO = 0.98
- best SAE = 0.82
- final SAE = 0.82

这说明：

> PLD 在当前 qclip0 条件下完全有能力学出高质量 hold behavior，但 ACP reward 没有把它引导到这个策略族上。

也就是说，PLD 的问题已经不能再解释成“critic 被 qclip 压住了”或“sim baseline 本来就不健康”。

现在最合理的解释是：
- ACP reward 对 progress / grasp 有信号；
- 但对 **hold-until-end** 的信号仍然不足；
- PLD 没有 BC anchor 去弥补这个 reward semantics 缺口；
- 所以仍然学成“能抓但留不住”。

### 4.2 DSRL + ACP
`dsrl_v7_qclip0_acp_mirror_s42`：
- best SO = 0.94
- best SAE = 0.06
- final SAE = 0.00

而 `dsrl_v7_qclip0_sim_baseline_s42`：
- best SO = 0.94
- best SAE = 0.66
- final SAE = 0.66

这说明：

> DSRL 的当前 tuned regime 完全可以在 sim reward 下学出稳定末态保持；但 ACP reward 仍然没有提供让 DSRL 保住末态成功的价值差异。

换句话说：
- DSRL 本体不是问题；
- qclip 也不是问题了；
- 剩下来的解释就更集中到 ACP 自身。

---

## 5. ACP 现在最像什么问题

在当前证据下，ACP 的问题可以收敛到下面这几条：

### A. reward semantics 仍然偏向 progress，不偏向 retention
当前现象非常典型：
- SO 仍然不算低
- SAE 极低
- final SAE 回零

这和之前的判断一致：

> ACP TD reward 更擅长鼓励“状态变好”，不擅长鼓励“已经好时继续维持”。

### B. grasp bonus 只能鼓励“抓住”，不能鼓励“稳住”
报告里仍然能看到：
- `acp_grasp_bonus_mean` 很高
- `is_grasping_rate` 也不低

但末态仍然不行。

这说明 grasp bonus 的信息内容不够：
- 它能告诉策略“你还在抓”
- 但不能告诉策略“你是否在朝着 episode end 的稳定 upright 末态逼近”

### C. critic 虽然被 qclip 放开了，但 reward 本身仍然不能把 hold 轨迹排序清楚
`qclip0` 之后：
- sim baseline 恢复
- acp mirror 还是不恢复

这意味着 critic 不是不能学，而是：

> 在 ACP reward 下，critic 没有拿到足够区分 `brief success` vs `stable end-state success` 的目标信号。

### D. AWSC 不是反例，而是“强 anchor 掩盖 reward 缺口”
这也是为什么“grasp bonus 只能鼓励抓住、不能鼓励稳住”并不与 `AWSC + ACP` 的成功矛盾。

`docs/vlaw/figures/rlpd_acp_v7_diag/diagnosis_report.md` 已经表明：
- AWSC sim: final SAE = 0.74
- AWSC acp: final SAE = 0.52
- 同时 AWSC 的 ACP reward / advantage 侧并不健康：
  - `acp_step_mean ≈ 1e-4`
  - online/offline reward gap 很大
  - `advantage_mean ≈ 1.03`

这说明：

> **AWSC + ACP 能成功，不是因为 ACP 已经学会奖励 stable hold，而是因为 AWSC actor 本身带有很强的 BC / flow / shortcut anchor，不需要靠 ACP 从零“发明 hold behavior”。**

更具体地说：
- AWSC 的 reward 更像是在已有好策略流形上做重加权；
- PLD / DSRL 的 reward 则更直接决定 actor/critic 会把哪类行为保留下来；
- 因此当 reward 只能表达 progress + grasp、却不能表达 hold quality 时：
  - AWSC 还能靠先验保住大量 hold 行为；
  - PLD / DSRL 则会完整暴露出“能抓但留不住”的问题。

所以当前更准确的项目共识应是：

> **AWSC + ACP 的成功不是 ACP hold-signal 充分的证据；它恰好说明，当算法自带强 hold-preserving anchor 时，ACP 的 reward semantics 缺口可以被部分掩盖；当算法必须靠 reward 自己学 hold 时，这个缺口就会完全暴露。**

---

## 6. 现在最合理的下一步

### Priority 1
把当前结论固化为项目共识：
- sim baseline 默认走 `qclip0`
- ACP 问题不再和 sim baseline drift 混在一起

### Priority 2
下一轮 ACP 专项诊断应只围绕 reward semantics 做，不再纠缠 qclip：

建议只测 2–4 组最小实验：
1. `td` vs `potential`
2. `grasp_bonus=1.0` vs 更高/更低
3. 如果还要再加一组，只加一个 `reward_scale` 改动

### Priority 3
如果目标是最终让 PLD/DSRL + ACP work，最值得怀疑的方向已经不是“critic 稳定化”，而是：
- ACP value 是否能区分 hold quality
- shaping 是否在成功平台区间给出足够非零信号
- 是否需要一个真正与 `success_at_end` 对齐的 reward design

---

## 7. 最终一句话结论

> **在去掉 `q_target_clip=20` 这个共享 drift 之后，PLD/DSRL 的 sim baseline 已恢复，但 ACP mirror 仍然明显失败，因此 ACP 的 reward semantics / hold-signal 缺失现在已经被单独且清晰地证实。**
