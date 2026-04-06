# CARM 当前状态、计划与执行进度总览

日期：2026-04-05  
范围：CARM 真机 teleop / inference / train 主链，timeline 分析，inference 数据离线回流

---

## 1. 当前状态概览

### 1.1 已闭环完成的主线修复
以下问题已经完成修复并经过样本或 smoke 验证：

1. **teleop / inference 时间戳语义对齐**
   - `observations/timestamps` 统一到 `obs_stamp_ros`
   - inference recorder 已写入 `timestamp_semantics=obs_stamp_ros`

2. **teleop / inference camera 契约对齐**
   - `primary_camera`
   - `images_by_camera`
   - `camera_names`
   - `camera_topics`
   - `observations/images` 继续作为 primary-camera 兼容字段

3. **teleop backend 假超时问题修复**
   - 根因不是 backend 变慢，而是代理污染
   - 代码层已在 `env_ros.py` 中显式禁用代理
   - shell 层已补 `no_proxy/NO_PROXY`

4. **主训练链默认启用 inactive teleop 过滤**
   - `train_carm` 主路径现在默认过滤 `teleop_scale == 0` 样本
   - `CARMDataset` 会输出过滤统计信息

5. **inference recorder -> 训练读取契约守卫补齐**
   - inference recorder 新样本已包含 `observations/qpos_joint`
   - `load_carm_episode()` 已能直接读取新的 inference recorder 样本
   - loader 会更早报出旧样本缺字段问题

6. **离线 inference 回流最小链路已打通**
   已完成如下闭环：
   ```text
   inference_episode_*.hdf5
   → staging conversion
   → load_carm_episode / CARMDataset
   → train_carm second-training smoke
   ```

### 1.2 当前已经可用的工具/入口

1. **timeline baseline 包装入口**
   - `scripts/run_carm_timeline_baseline.py`
   - 作用：把 `timeline_*.jsonl` + `run_info_*.json` 组合成稳定 baseline 摘要

2. **teleop vs inference gap 分析脚本**
   - `scripts/analyze_carm_gap.py`
   - 作用：直接对比 teleop/inference 的时间链路和行为差异

3. **inference 离线 staging 转换脚本**
   - `scripts/convert_inference_to_training_staging.py`
   - 作用：把 `inference_episode_*.hdf5` 转换成训练 staging 目录里的 `episode_*.hdf5`

4. **中文分析报告**
   - `docs/carm_timeline_analysis_report.md`
   - 已整合 timeline 分析计划、当前进展、teleop vs inference gap 结论、离线回流状态

---

## 2. 已完成的关键验证

### 2.1 Teleop 样本验证
样本：
- `recorded_data/test_fps/episode_0001_20260405_131754.hdf5`
- `recorded_data/test_fps/timeline_record_20260405_131722.jsonl`

结论：
- 采样频率恢复正常（约 47Hz）
- `teleop_scale` 不再全 0
- `action != qpos_end`
- teleop 主路径可继续作为训练基准数据

### 2.2 Inference recorder 样本验证
样本：
- `inference_logs/inference_episode_0001_20260405_132709.hdf5`
- `inference_logs/run_info_20260405_132637.json`
- `inference_logs/timeline_20260405_132633.jsonl`

结论：
- schema 满足新版训练契约要求
- `qpos_joint` 已存在
- `load_carm_episode()` 可以直接读取
- `images_by_camera` / `primary_camera` / `timestamp_semantics` 均正常

### 2.3 离线二次训练 smoke
已在 staging 目录上成功跑通：
- `train_carm` second-training smoke
- 说明当前 blocker 已不在数据格式，而是已经成功打通到训练入口并完成最小 smoke

---

## 3. 当前最重要的分析结论

基于：
- teleop 样本
- inference 样本
- timeline baseline
- `scripts/analyze_carm_gap.py`

当前的核心结论是：

### 3.1 teleop vs inference 的最大 gap 不在 observation freshness
- teleop `delta_obs ≈ 0.0324s`
- inference `delta_obs ≈ 0.0370s`

结论：
- 观测链路差异不大
- 不是当前主矛盾

### 3.2 inference 的 timing gap 主要来自模型推理 + chunk staging
- `inference_time ≈ 0.0657s`
- `delta_chunk_obs ≈ 0.1066s`
- `control_send_minus_query ≈ 0.00015s`

结论：
- control 下发本身不是问题
- 真正的 timing burden 在 inference / chunk 形成阶段

### 3.3 更大的 gap 在行为语义，而不只是 timing
比较 `action` 相对当前 `qpos_end` 的差分：

#### Teleop
- `position-gap mean ≈ 0.0142`
- `position-gap p95 ≈ 0.0342`
- `static_ratio ≈ 0.121`

#### Inference
- `position-gap mean ≈ 0.3930`
- `position-gap p95 ≈ 0.4446`
- `static_ratio = 0.0`

结论：
- inference 比 teleop 激进得多
- 当前 gap 不只是“慢了一点”，而是 action-amplitude semantics mismatch

这意味着：
> 只优化几毫秒 latency，并不能自动消除 teleop / inference 的真实差异。

---

## 4. 当前执行计划（最新）

### 4.1 主线 A：timeline 分析与优化
当前目标不是继续堆日志，而是回答：
1. teleop / inference 的最大 gap 在哪一层？
2. timing gap 和 behavior gap 各占多大？
3. 下一轮优先该调哪个参数 / 哪段逻辑？

当前建议顺序：
1. 先基于现有 timeline / run_info 稳定做 baseline
2. 再补最小高价值 observability：
   - `cycle_id`
   - chunk first-use timing
   - loop overrun
3. 然后才决定是否要调：
   - `act_horizon`
   - `chunk_time_base`
   - `crossfade_steps`
   - warmup / 首帧行为

### 4.2 主线 B：inference 数据离线回流
当前已完成最小闭环：
```text
部署推理
→ recorder 保存 inference_episode_*.hdf5
→ staging conversion
→ 二次训练 smoke
```

下一步应该继续推进：
1. 明确 inference 数据准入策略
   - 是否过滤 intervention
   - 是否筛 success/high-quality episode
   - 是否过滤高频 safety clip 片段
2. 跑更真实一点的二次训练（不只 2 iter smoke）
3. 再做一次部署 smoke

长期方向：
- 等离线版本稳定后，再考虑内存 / 显存内回流
- 落盘只作为异步分析副产物

---

## 5. 当前暂缓项
以下内容已经明确暂缓，不纳入当前执行主线：

1. `pi05 bridge/export` 的 inactive 过滤对齐
2. inference 数据自动混入训练目录扫描
3. 多视角进入 policy 输入
4. 大规模 logger / recorder / timeline 全架构重构

---

## 6. 当前执行进度

### 已完成
- [x] timestamp 对齐
- [x] camera 契约对齐
- [x] teleop backend 代理问题修复
- [x] inactive teleop 默认过滤
- [x] inference recorder 训练契约守卫
- [x] inference -> staging conversion
- [x] train_carm second-training smoke
- [x] timeline baseline wrapper
- [x] teleop vs inference gap 分析脚本
- [x] 中文综合分析报告
- [x] `gripper_max` 对齐到真实硬件上限后，最新 4 组 inference 对照实验已不再出现 `gripper_clip` 假阳性，`safety_clips=0`
- [x] `scripts/analyze_carm_gap.py` 已升级为同时输出 same-step gap 与 motion-centric 指标（realized motion / target motion / realized-vs-target ratio / future-k improvement / early-middle-late window）
- [x] 已确认单看 same-step target-vs-state gap 会误判“策略体感表现”，新的 motion 指标更接近人工观察排序
- [x] inference 离线回流链路已从“最小可读”推进到“带准入元数据的可训练 staging”
- [x] staging conversion 已支持 episode-level admission、run_info 摘要挂载、per-episode sidecar metadata、目录级 conversion 汇总
- [x] `scripts/verify_hdf5_format.py` 已区分 `teleop_v2` 与 `inference_staging` schema，并能对历史样本给出兼容性 warning 而非直接误报失败
- [x] 基于 admission 后的 `/tmp/carm_staging_audit` 已成功跑通一轮更真实的 `train_carm` second-training smoke（10 iter）

### 正在进行
- [ ] 基于现有 baseline，继续完善 timeline 指标与归因能力
- [ ] 把 inference 数据离线回流从 smoke 推进到更真实的二次训练 / 二次部署流程
- [ ] 明确 inference 数据进入下一轮训练的准入标准

### 下一步建议
1. 把当前 admission 规则从“最小工程可用”收敛成正式准入标准（如 `min_steps`、`max_intervention_ratio`、是否纳入 high-quality / success 筛选）
2. 基于当前 staging + admission 结果，设计第一轮真正的 second-training baseline（inference-only 或 teleop+inference mixed）
3. 在重新推进部署前，补一轮 deployment smoke，确认 second-training 模型在真实 receding-horizon 下的行为变化

---

## 7. 当前最简结论
当前 CARM 项目已经从“修数据契约问题”进入“优化行为与闭环”的阶段：
- 数据契约主线已基本稳定
- 离线 inference 回流已具备可执行最小闭环
- 当前最大的 teleop / inference gap 不是观测链，而是：
  1. **模型推理 + chunk staging 的约 100ms 量级时延**
  2. **inference action 相比 teleop 明显更激进的行为语义差异**

因此，下一阶段不应只盯着继续补字段，而应该围绕：
- timeline 的真实因果归因
- inference 数据离线回流的准入与二次训练策略

来继续推进。