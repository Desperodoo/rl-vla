# CARM 时间线分析与离线回流进展报告

## 目标
本报告用于统一整理当前 CARM 下一阶段两条主线的计划、进展与结论：
1. **时间线分析与优化**：围绕 `obs -> infer -> chunk -> control` 的延迟链路，判断 teleop 与 inference 的真正 gap 在哪里，以及下一轮应该先调什么。
2. **inference 数据离线回流**：先打通“部署推理 → 本地保存 inference 数据 → 离线转换/筛选 → 二次训练 → 二次部署”的最小闭环。

本报告强调两点：
- 不只是继续加日志，而是要从现有数据中真正得出“gap 在哪里、先改什么”的结论。
- 先做离线回流版本，稳定后再考虑把数据处理优化到内存/显存闭环。

---

## 一、当前前提（已闭环）
当前已经完成并验证的基础修复包括：
- teleop / inference 的 `observations/timestamps` 已统一到 `obs_stamp_ros`
- teleop / inference 的 `primary_camera` / `images_by_camera` 契约已对齐
- teleop backend 代理污染导致的假超时已修复
- `train_carm` 主训练链默认启用 inactive teleop 过滤
- 新版 inference recorder 样本已包含 `observations/qpos_joint`，可被 `load_carm_episode()` 直接读取

这些修复意味着：
- 当前真正的主线问题已经从“数据契约错误”切换到了“时间线差异分析”和“离线回流闭环”。

---

## 二、teleop vs inference gap：当前结论

### 2.1 数据来源
#### Teleop
- `recorded_data/test_fps/episode_0001_20260405_131754.hdf5`
- `recorded_data/test_fps/timeline_record_20260405_131722.jsonl`

#### Inference
- `inference_logs/inference_episode_0001_20260405_132709.hdf5`
- `inference_logs/timeline_20260405_132633.jsonl`
- `inference_logs/run_info_20260405_132637.json`

#### 当前分析入口
- `scripts/analyze_carm_gap.py`
- `scripts/run_carm_timeline_baseline.py`
- `carm_ros_deploy/src/carm_deploy/tools/analyze_timeline.py`

---

### 2.2 观测侧 gap
#### Teleop
- `delta_obs ≈ 0.0324s`

#### Inference
- `delta_obs ≈ 0.0370s`

#### 结论
- teleop 和 inference 在 **观测新鲜度** 上差距不大
- 传感器/观测链不是当前最大的瓶颈

---

### 2.3 推理 / 调度侧 gap
#### Inference
- `inference_time ≈ 0.0657s`
- `delta_chunk_obs ≈ 0.1066s`
- `control_send_minus_query ≈ 0.00015s`

#### 结论
- 控制发送本身几乎不是问题
- 真正的时延主要发生在：
  1. 模型推理
  2. chunk 生成 / 调度形成
- 也就是说，当前 inference 相比 teleop 的时延差异，主要不是控制口，而是 **model + chunk staging**

---

### 2.4 行为层 gap（最关键）
这里直接比较 `action` 与当前 `qpos_end` 的差异分布。

#### Teleop
- `position-gap mean ≈ 0.0142`
- `position-gap p95 ≈ 0.0342`
- `static_ratio ≈ 0.121`

#### Inference
- `position-gap mean ≈ 0.3930`
- `position-gap p95 ≈ 0.4446`
- `static_ratio = 0.0`

#### 结论
这是当前最重要的发现：

> inference 不只是“比 teleop 慢一点”，而是它输出的目标相对当前状态**远比 teleop 激进**。

这意味着：
- teleop 更接近“细粒度、近状态、可 hold 的控制”
- inference 当前更接近“每次给出离当前较远的目标”
- 因此如果只去优化几毫秒延迟，并不能自动消除 teleop / inference 的真实差异

---

### 2.5 当前综合判断
当前 teleop vs inference gap 主要分成两层：

#### 层 A：时间差异
- 主要来自 `inference_time + delta_chunk_obs`
- 不是来自观测链本身
- 也不是来自 control send 的最终发送成本

#### 层 B：行为差异
- inference 的 action amplitude 比 teleop 大很多
- 这会影响执行体感与回流训练分布
- 因此接下来的优化不能只盯时间，还必须盯行为语义

---

## 三、时间线分析：当前计划

### 3.1 我们真正要回答的问题
不是单纯“延迟大不大”，而是：
1. teleop 和 inference 的最大 gap 在哪一层？
2. 是 timing 问题、调度问题，还是 action 语义问题？
3. 下一轮应该先调哪个参数 / 哪段逻辑？

### 3.2 当前最值得继续补的量
在当前 baseline 已够用的前提下，下一轮最有价值的 observability 增量是：
1. `cycle_id`：把 `obs / inference / chunk` 串起来
2. 每个 chunk 的首次 control use 时间
3. inference / control loop 的 overrun 指标

原则：
- 只补最小高价值字段
- 不做 logger / recorder 全系统重构
- 让新的字段直接服务于 gap 归因，而不是“为了有更多日志而加日志”

### 3.3 当前建议的优化顺序
1. 先稳定跑多次 baseline，对比 p50 / p90 / p95 / p99
2. 再补最小 observability（如 chunk first-use）
3. 然后再决定是否去调：
   - `chunk_time_base`
   - `act_horizon`
   - `crossfade_steps`
   - 推理 warmup / 首帧优化

即：

> 先测清楚，再优化；避免在没有归因结论的情况下乱调参数。

---

## 四、inference 数据离线回流：当前进展

### 4.1 目标闭环
当前离线版本的目标是：

```text
train_carm
→ 部署 inference
→ 本地保存 inference_episode_*.hdf5
→ 离线转换到 staging 目录
→ 二次 train_carm
→ 新 checkpoint 再部署
```

### 4.2 为什么当前保留 staging 转换层
短期内保留转换层的原因是：
- recorder 原始产物仍承担诊断职责
- training-ready artifact 与诊断 artifact 分离，更安全
- 当前还没有完全冻结“训练到底吃哪个 action”的最终策略

所以现阶段采用：
- raw inference artifact
- staging training artifact

两层结构更稳妥。

---

### 4.3 已完成
#### 已新增入口
- `scripts/convert_inference_to_training_staging.py`
- `scripts/run_carm_timeline_baseline.py`
- `scripts/analyze_carm_gap.py`

#### 已补强
- `InferenceDatasetConverter` 现在支持：
  - 批量转换 `inference_episode_*.hdf5`
  - 复制 grouped obs（如 `images_by_camera`）
  - 保留关键 attrs（如 `timestamp_semantics` / camera metadata）
  - 生成 `conversion_metadata.json`

#### 已验证
离线回流已打通到：

```text
inference_episode
→ staging conversion
→ load_carm_episode() 可直接读取
```

这意味着：
- 数据格式已经不是 blocker
- 主阻塞点已经转移到训练环境本身

---

### 4.4 当前 blocker
当前二次 `train_carm` smoke 仍未跑通，但原因已经明确：

#### 不是数据问题
- staging 输出已能被 `load_carm_episode()` 直接读取
- inference recorder 新样本也已满足关键字段契约

#### 是训练环境问题
当前 `rlft_ms3` 环境里：
- `torch == 2.6.0`
- `torchvision == 0.21.0`
- `transformers == 5.1.0`
- `diffusers == 0.37.1`

当前最明确的错误是 `torchvision` 导入失败：

```text
RuntimeError: operator torchvision::nms does not exist
```

这说明：
- `torch` / `torchvision` 的二进制安装或依赖状态不兼容
- 后续 `transformers` / `diffusers` 图像处理导入链也会因此失败

所以当前离线闭环的 blocker 是：

> `rlft_ms3` 的训练环境栈，而不是 inference 数据回流格式。

---

## 五、下一步建议

### 5.1 A 线：继续优先打通二次训练 smoke
下一步优先级最高的是：
- 修复 `rlft_ms3` 的 `torchvision` / `transformers` / `diffusers` 导入栈
- 目标是把：

```text
staging 目录 -> train_carm smoke
```

真正跑通

### 5.2 B 线：继续用现有 gap 结论指导优化
当前已经足够支持下一轮优化判断：
- observation freshness 不是主矛盾
- 主要 timing gap 在 inference + chunk staging
- 更大的 gap 在行为幅度语义

所以下一轮不该直接先调 `chunk_time_base`，而应先补：
- chunk first-use latency
- cycle 级归因
- 然后再看是否需要调：
  - `act_horizon`
  - chunk replace/use 策略
  - warmup / 首帧行为

---

## 六、当前一句话结论
当前 CARM 的 teleop vs inference gap，核心不是观测链，而是：
1. **模型推理 + chunk staging 带来的约 100ms 量级延迟**
2. **inference action 相比 teleop 明显更激进的行为语义差异**

同时，inference 数据离线回流已经打通到 staging + loader 层，下一步真正需要解决的是 `rlft_ms3` 训练环境兼容性，而不是数据格式本身。
