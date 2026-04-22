# PI05 报告接入 consistency_flow_resnet18 横向对比计划（2026-04-22）

## 1. 当前已确认事实

- 待纳入模型：
  - [`runs/consistency_flow_resnet18_seed1__1774269110/checkpoint/final.pt`](/home/wjz/rl-vla/runs/consistency_flow_resnet18_seed1__1774269110/checkpoint/final.pt)
- 对应训练配置：
  - [`runs/consistency_flow_resnet18_seed1__1774269110/checkpoint/args.json`](/home/wjz/rl-vla/runs/consistency_flow_resnet18_seed1__1774269110/checkpoint/args.json)
- 关键配置：
  - `algorithm=consistency_flow`
  - `visual_encoder_type=resnet18`
  - `action_mode=ee_only`
  - `state_mode=joint_only`
  - `batch_size=256`
  - `total_iters=100000`
- 现有评估脚本：
  - [`eval_carm.py`](/home/wjz/rl-vla/rlft/offline/eval_carm.py)
- 当前 PI05 报告脚本：
  - [`generate_pi05_batch_report.py`](/home/wjz/rl-vla/rlft/offline/generate_pi05_batch_report.py)
- 当前本机可直接使用的数据切分：
  - [`recorded_data_splits/val`](/home/wjz/rl-vla/recorded_data_splits/val)
  - [`recorded_data_splits/test`](/home/wjz/rl-vla/recorded_data_splits/test)
- 当前可直接复用的 split 统计：
  - [`split_summary.json`](/home/wjz/rl-vla/recorded_data_splits/split_summary.json)
  - `val=13 episodes`
  - `test=13 episodes`

## 2. 目前不能直接塞进现有 PI05 总表的原因

- [`eval_pi05.py`](/home/wjz/rl-vla/rlft/offline/eval_pi05.py) 输出的是：
  - `mean_action_mse`
  - `mean_action_mae`
  - `per_dim_mae`
  - `per_episode_mean_mae`
- [`eval_carm.py`](/home/wjz/rl-vla/rlft/offline/eval_carm.py) 输出的是：
  - `joint_mae`
  - `pose_mae`
  - `ee_mae`
  - `total_mae`
  - `window8_total_mae`
- 两边指标族不同，且动作表示也不完全一样。
- 此外该 run 的原始 `val_demo_path` 指向 `/home/liangjh/...`，在当前机器上不存在，不能直接沿用原训练时的验证集路径。

## 3. 建议采用的两层方案

### 3.1 第一层：先把 CARM 模型纳入“同一份报告中的独立章节”

目标：

- 不阻塞当前 `official / batch2 / batch64` 主线报告
- 先把 `consistency_flow_resnet18` 的离线结果、误差图、代表 episode 图做出来
- 形成一个“同报告、不同指标家族”的横向展示

做法：

1. 先在 `recorded_data_splits/val` 上跑一次小规模 `EMA vs non-EMA` 探针。
2. 选定最终展示口径后，在 `val` 和 `test` 上各跑一遍完整离线评估。
3. 将 `metrics.json`、`error_distribution.png`、代表 episode 可视化图接入报告的新章节。

推荐先验口径：

- 主结果优先用 `--use_ema`
- `non-EMA` 作为附录或补充表

原因：

- `consistency_flow` 通常 EMA 更接近部署口径
- 先用 5 episode 小样本确认 EMA 是否确实更优，再决定最终主表口径

### 3.2 第二层：再做严格 apples-to-apples 对比

目标：

- 让 `PI05 official / batch2 / batch64 / consistency_flow_resnet18` 真正共享同一数据切分
- 尽量共享同一动作语义

做法：

1. 用 [`export_carm_to_lerobot.py`](/home/wjz/rl-vla/rlft/offline/export_carm_to_lerobot.py) 把 [`recorded_data_splits/val`](/home/wjz/rl-vla/recorded_data_splits/val) 和 [`recorded_data_splits/test`](/home/wjz/rl-vla/recorded_data_splits/test) 导出为 LeRobot/PI05 可评估数据集。
2. 用 [`eval_pi05.py`](/home/wjz/rl-vla/rlft/offline/eval_pi05.py) 在这两份导出数据上重跑：
   - official
   - batch2
   - batch64
3. 用 [`eval_carm.py`](/home/wjz/rl-vla/rlft/offline/eval_carm.py) 在原始 split 上重跑 `consistency_flow_resnet18`。
4. 报告中保留两类结果：
   - 原生指标表：PI05 一张，CARM 一张
   - 协议化对比图：只比较双方都能解释的指标子集

建议的协议化指标：

- `ee_mae`
- `total_mae`
- 每 episode 排序后的误差曲线

不建议强行混成单一 `mean_action_mse` 表，除非先明确定义动作维度映射。

## 4. 推荐执行顺序

### 阶段 A：当天可完成的快速接入

1. 先做 5 episode `EMA vs non-EMA` 小探针。
2. 如果 EMA 更好，就在 `val/test` 上各跑一次 `--use_ema` 全量评估。
3. 新增报告章节：
   - CARM 模型配置卡片
   - `val/test` 指标表
   - `error_distribution.png`
   - 2 到 3 张代表 episode 对比图

### 阶段 B：次级任务，做严格统一协议

1. 导出 `recorded_data_splits/val` 和 `test` 到 LeRobot 格式。
2. 在统一 split 上重跑 PI05 三组评估。
3. 新增“统一 split 横向对比”章节。

## 5. 具体命令建议

### 5.1 先跑 EMA 探针

```bash
python -m rlft.offline.eval_carm \
  --model_path /home/wjz/rl-vla/runs/consistency_flow_resnet18_seed1__1774269110/checkpoint/final.pt \
  --data_dir /home/wjz/rl-vla/recorded_data_splits/val \
  --output_dir /home/wjz/rl-vla/runs/carm_eval_consistency_flow_resnet18_val_probe \
  --compare_ema \
  --num_episodes 5 \
  --quiet
```

### 5.2 跑最终 val

```bash
python -m rlft.offline.eval_carm \
  --model_path /home/wjz/rl-vla/runs/consistency_flow_resnet18_seed1__1774269110/checkpoint/final.pt \
  --data_dir /home/wjz/rl-vla/recorded_data_splits/val \
  --output_dir /home/wjz/rl-vla/runs/carm_eval_consistency_flow_resnet18_val \
  --use_ema
```

### 5.3 跑最终 test

```bash
python -m rlft.offline.eval_carm \
  --model_path /home/wjz/rl-vla/runs/consistency_flow_resnet18_seed1__1774269110/checkpoint/final.pt \
  --data_dir /home/wjz/rl-vla/recorded_data_splits/test \
  --output_dir /home/wjz/rl-vla/runs/carm_eval_consistency_flow_resnet18_test \
  --use_ema
```

### 5.4 如需做统一 split 的 PI05 导出

```bash
python -m rlft.offline.export_carm_to_lerobot \
  --demo_path /home/wjz/rl-vla/recorded_data_splits/val \
  --output_dir /home/wjz/rl-vla/runs/pi05_eval_export_val \
  --state_mode ee_only \
  --action_representation ee_delta_pose_gripper
```

```bash
python -m rlft.offline.export_carm_to_lerobot \
  --demo_path /home/wjz/rl-vla/recorded_data_splits/test \
  --output_dir /home/wjz/rl-vla/runs/pi05_eval_export_test \
  --state_mode ee_only \
  --action_representation ee_delta_pose_gripper
```

## 6. 预期产物

阶段 A 结束后应有：

- `val/test` 两份 `metrics.json`
  - 路径示例：`runs/carm_eval_consistency_flow_resnet18_val/metrics.json`
- `error_distribution.png`
- 若干 `comparison_epXXX.png`
- 一份更新后的图文报告

阶段 B 结束后应再补充：

- 统一 split 的 PI05 eval json
- 一张真正统一协议下的横向对比表

## 7. 我建议的默认推进方式

- 先不打断当前 PI05 `batch64` 主线。
- 先做阶段 A，把 `consistency_flow_resnet18` 作为“同报告独立章节”接入。
- 等远端 `official` 和 `batch64` eval 都齐了，再决定是否进入阶段 B 做严格统一协议对比。

这样做的好处是：

- 今天就能把 CARM 模型纳入可视化报告
- 不会因为统一协议设计把当前主线整体卡住
- 后续如果要做正式结论，再升级到严格 apples-to-apples 版本即可
