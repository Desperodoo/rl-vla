# CARM Second-Stage Baseline Matrix v1

日期：2026-04-07  
范围：CARM second-stage training 第一轮 baseline 实验矩阵  
依赖前提：
- admission v1 已落地
- `train_carm_stage2.py` 已支持 `gold / gold+silver / policy_version / mixed`
- inference staging 可用：`/tmp/carm_staging_admission_v1`
- 当前确认可用的 teleop datasets：
  - `recorded_data/fixed_dual_light`
  - `recorded_data/fixed_left_light`
  - `recorded_data/fixed_no_light`
  - `recorded_data/random_no_light`

---

## 1. 目标

第一轮 baseline 的目标不是立刻找最优超参，而是建立一组**可解释、可复现、可比较**的 second-stage 对照组。

要回答的问题是：

1. second-stage 只用 teleop 数据时，训练入口是否保持稳定？
2. second-stage 只用 inference gold 数据时，是否可以形成最小高质量回流基线？
3. teleop + inference gold mixed 时，是否能形成最基本的混合回流训练基线？

---

## 2. 当前固定输入

### 2.1 inference staging
- 路径：`/tmp/carm_staging_admission_v1`
- 当前 admission 汇总：
  - gold = 4
  - silver = 3
  - reject = 1
- 当前 policy version：`carm_inference_admission_v1`

### 2.2 teleop datasets
- `/home/amax/rl-vla/recorded_data/fixed_dual_light`
- `/home/amax/rl-vla/recorded_data/fixed_left_light`
- `/home/amax/rl-vla/recorded_data/fixed_no_light`
- `/home/amax/rl-vla/recorded_data/random_no_light`

当前统计：
- teleop 总 episode 数 = 127

---

## 3. baseline 矩阵

### Baseline A：teleop-only

#### 目的
验证 second-stage 入口在“不使用 inference 数据”的情况下，是否与 baseline training 主线保持兼容。

#### 数据来源
- teleop only
- inference 不接入

#### 运行模式
- `mix_mode=mixed` 或等效 teleop-only 路径
- `inference_staging_paths=[]`

#### 推荐命令（dry-run）
```bash
conda run -n rlft_ms3 python -m rlft.offline.train_carm_stage2 \
  --teleop_demo_paths \
    /home/amax/rl-vla/recorded_data/fixed_dual_light \
    /home/amax/rl-vla/recorded_data/fixed_left_light \
    /home/amax/rl-vla/recorded_data/fixed_no_light \
    /home/amax/rl-vla/recorded_data/random_no_light \
  --mix_mode mixed \
  --dry_run_selection \
  --no-track
```

#### 推荐命令（smoke）
```bash
conda run -n rlft_ms3 python -m rlft.offline.train_carm_stage2 \
  --teleop_demo_paths \
    /home/amax/rl-vla/recorded_data/fixed_dual_light \
    /home/amax/rl-vla/recorded_data/fixed_left_light \
    /home/amax/rl-vla/recorded_data/fixed_no_light \
    /home/amax/rl-vla/recorded_data/random_no_light \
  --mix_mode mixed \
  --no-track \
  --total_iters 1 \
  --batch_size 2 \
  --save_freq 1000 \
  --eval_freq 1000
```

#### 期望结果
- `source_counts.teleop > 0`
- `source_counts.inference == 0`
- `used_episodes.jsonl` 中全部为 `source_type=teleop`

---

### Baseline B：inference-only gold

#### 目的
建立最保守的 inference 回流基线，只使用 admission v1 的 `gold` 桶。

#### 数据来源
- inference staging only
- bucket = `gold`
- policy version = `carm_inference_admission_v1`（建议显式固定）

#### 推荐命令（dry-run）
```bash
conda run -n rlft_ms3 python -m rlft.offline.train_carm_stage2 \
  --inference_staging_paths /tmp/carm_staging_admission_v1 \
  --admission_buckets gold \
  --policy_version carm_inference_admission_v1 \
  --dry_run_selection \
  --no-track
```

#### 推荐命令（smoke）
```bash
conda run -n rlft_ms3 python -m rlft.offline.train_carm_stage2 \
  --inference_staging_paths /tmp/carm_staging_admission_v1 \
  --admission_buckets gold \
  --policy_version carm_inference_admission_v1 \
  --no-track \
  --total_iters 1 \
  --batch_size 2 \
  --save_freq 1000 \
  --eval_freq 1000
```

#### 期望结果
- `source_counts.teleop == 0`
- `source_counts.inference == 4`
- `bucket_counts.gold == 4`
- `used_episodes.jsonl` 中 inference episode 的 `policy_version` 全为 `carm_inference_admission_v1`

---

### Baseline C：teleop + inference gold mixed

#### 目的
建立最小 mixed 回流基线，用高质量 inference 数据给 teleop 增量补充。

#### 数据来源
- teleop datasets
- inference staging gold only

#### 推荐命令（dry-run）
```bash
conda run -n rlft_ms3 python -m rlft.offline.train_carm_stage2 \
  --teleop_demo_paths \
    /home/amax/rl-vla/recorded_data/fixed_dual_light \
    /home/amax/rl-vla/recorded_data/fixed_left_light \
    /home/amax/rl-vla/recorded_data/fixed_no_light \
    /home/amax/rl-vla/recorded_data/random_no_light \
  --inference_staging_paths /tmp/carm_staging_admission_v1 \
  --admission_buckets gold \
  --policy_version carm_inference_admission_v1 \
  --mix_mode mixed \
  --dry_run_selection \
  --no-track
```

#### 推荐命令（smoke）
```bash
conda run -n rlft_ms3 python -m rlft.offline.train_carm_stage2 \
  --teleop_demo_paths \
    /home/amax/rl-vla/recorded_data/fixed_dual_light \
    /home/amax/rl-vla/recorded_data/fixed_left_light \
    /home/amax/rl-vla/recorded_data/fixed_no_light \
    /home/amax/rl-vla/recorded_data/random_no_light \
  --inference_staging_paths /tmp/carm_staging_admission_v1 \
  --admission_buckets gold \
  --policy_version carm_inference_admission_v1 \
  --mix_mode mixed \
  --no-track \
  --total_iters 1 \
  --batch_size 2 \
  --save_freq 1000 \
  --eval_freq 1000
```

#### 期望结果
- `source_counts.teleop == 127`
- `source_counts.inference == 4`
- `bucket_counts.teleop == 127`
- `bucket_counts.gold == 4`
- `total_selected_episodes == 131`

---

## 4. 当前阶段暂不纳入 baseline 的组合

以下组合建议暂不进入第一轮 baseline：

1. `gold + silver` mixed
2. `reject` 桶参与训练
3. teleop / inference 不同比例采样
4. reward-based ranking
5. robometer reward 介入

原因：
- 第一轮 baseline 的目标是建立最稳定、最容易解释的对照组
- `silver` 会引入额外分布复杂度，适合放到第二轮 ablation

---

## 5. 每组实验必须检查的产物

每次 dry-run / smoke 后，至少检查：

1. `runs/<run_name>/training_manifest.json`
2. `runs/<run_name>/used_episodes.jsonl`

重点核对：
- `source_counts`
- `bucket_counts`
- `admission_buckets`
- `policy_version`
- `selected_teleop_roots`
- `selected_inference_roots`

---

## 6. 推荐执行顺序

### 第一步：只做 dry-run
顺序：
1. Baseline A teleop-only
2. Baseline B inference-only gold
3. Baseline C mixed gold

目的：
- 先确认 episode selection / manifest 完全符合预期

### 第二步：再做 smoke
每组只跑：
- `total_iters=1`
- `batch_size=2`

目的：
- 验证 dataset -> dataloader -> model -> training loop 全链路无误

### 第三步：再决定正式训练配置
等 smoke 全部通过后，再收敛：
- 正式 `total_iters`
- 是否 `resume_from` baseline checkpoint
- 是否加入 val set / eval

---

## 7. 最简结论

第一轮 second-stage baseline 建议只固定三组：

1. **teleop-only**
2. **inference-only gold**
3. **teleop + inference gold mixed**

这样就能先把：
- baseline 主链兼容性
- inference 回流最小高质量基线
- teleop + inference 混合基线

这三件事情稳定下来，再进入第二轮 `gold+silver` 和更复杂的数据配比实验。
