# PI05 Dense Full-FT DeepSpeed 方案（2026-04-22）

## 1. 已知约束

- 当前已经验证通过的路线是：
  - `bf16 + DeepSpeed ZeRO-2 + 6 x RTX 4090`
  - dense full-ft 可以稳定完成 `2 -> resume -> 4` step 闭环
  - strict resume 已可用
- 当前没有验证通过的路线：
  - `float32 + ZeRO-2`
  - 它在首个 backward 的梯度规约阶段仍然 OOM
- 当前最现实的长期训练路线仍然是：
  - **`bf16 + ZeRO-2`**
  - 暂时不要把正式主实验押到 `ZeRO-3` 或 FSDP 上
- 需要注意的工程成本：
  - DeepSpeed 原生 checkpoint 很大
  - 6 卡 `fix9` 的 `accelerator_state` 单个 checkpoint 量级约 `43G ~ 46G`

## 2. 推荐资源分配

### 2.1 现在就能开的首选方案

- 机器：
  - **本地 6 卡**
- GPU：
  - `0,1,2,3,4,5`
- 原因：
  - 这条路已经做过 smoke 和 resume 验证
  - 风险最小
  - 远端当前还在跑 `batch64` LoRA，对 dense full-ft 再叠加新变量不划算

### 2.2 远端何时介入

- 等远端 `batch64` 两条 LoRA 训练结束、post-eval 补完后
- 如果要做第二条 dense 对照，再考虑把远端切成：
  - `8 x 4090`
  - 同样 `bf16 + ZeRO-2`
- 远端更适合做：
  - 第二条 dense 对照
  - 或更长步数的 continuation run

## 3. 推荐训练配置

### 3.1 阶段 A：pilot run

目的：

- 验证长于 smoke 的稳定性
- 验证 `5k` 级别下 loss 曲线是否平滑
- 验证大 checkpoint 的保存 / resume / offline eval 流程

建议配置：

| 项目 | 推荐值 |
| --- | --- |
| GPUs | `6` |
| launcher | `rlft/offline/launch_pi05_zero2_full_ft.py` |
| precision | `bf16` |
| policy dtype | `bfloat16` |
| zero stage | `2` |
| per-GPU micro-batch | `1` |
| gradient accumulation | `1` |
| effective global batch | `6` |
| steps | `5000` |
| optimizer lr | `2.5e-5` |
| gradient checkpointing | `true` |
| freeze vision encoder | `false` |
| train expert only | `false` |
| save freq | `2500` |
| eval 策略 | 训练外单独跑 final val/test |

理由：

- `micro-batch=1 / GPU` 是已经验证通过的保守点
- `grad_acc=1` 先把变量压到最少，优先看 dense 本体是否稳定
- `5000` step 足够看出是否存在长程漂移、loss 爆炸或 checkpoint 问题

### 3.2 阶段 B：正式 baseline run

目的：

- 拿到第一条真正可和 LoRA 做离线比较的 dense baseline

建议配置：

| 项目 | 推荐值 |
| --- | --- |
| GPUs | `6` |
| precision | `bf16` |
| zero stage | `2` |
| per-GPU micro-batch | `1` |
| gradient accumulation | `1` |
| effective global batch | `6` |
| steps | `20000` |
| optimizer lr | `2.5e-5` |
| save freq | `10000` |
| resume | 开启 |
| offline eval | `10000` / `20000` 两个 checkpoint 都做 |

理由：

- `global batch=6` 虽然比 LoRA `batch2 x 4 => global batch=8` 略小，但样本预算已在同一量级
- `20000` step 是当前 LoRA 主线的对照尺度，便于横向比较
- `save_freq=10000` 可以把磁盘开销控制在较合理范围
  - 大致是两个大 checkpoint
  - 同时保留 mid-point 和 final-point

### 3.3 阶段 C：可选放大版

如果阶段 B 稳定，再考虑：

| 项目 | 推荐值 |
| --- | --- |
| 机器 | 远端 8 卡 |
| GPUs | `0-7` 或空闲 8 卡 |
| per-GPU micro-batch | `1` |
| gradient accumulation | `1` |
| effective global batch | `8` |
| steps | `20000` 或 `30000` |
| save freq | `10000` |

目的：

- 把 global batch 对齐或超过 LoRA `batch2`
- 看 dense 在更高总吞吐下是否能拿到更稳的离线误差

## 4. 不推荐的配置

- 不推荐现在就上 `micro-batch=2 / GPU`
  - 当前虽然从 smoke 显存看似乎还有余量，但还没做 dedicated probe
  - dense full-ft 进入长程训练后还有 checkpoint、碎片化、偶发峰值等变量
- 不推荐现在就上 `gradient_accumulation > 1`
  - 这会直接拉长 wall-clock
  - 在 baseline 还没拿到前，收益不如先把最小稳定 dense run 跑出来
- 不推荐把第一条正式 dense baseline 切到 `ZeRO-3`
  - 理论上能继续降显存
  - 但当前仓库里真正验证过的是 `ZeRO-2`
  - 工程风险会明显上升

## 5. 建议执行顺序

1. 先等远端 `batch64` LoRA 完成，补齐最终离线 eval 和 batch2-vs-batch64 对比。
2. 本地启动 dense full-ft `pilot 5k`。
3. `pilot 5k` 若稳定，直接 resume 成 `20k` 正式 baseline。
4. 再决定是否在远端补一条 `8 卡 dense` continuation / 对照。

## 6. 推荐命令模板

### 6.1 pilot 5k

```bash
python -m rlft.offline.launch_pi05_zero2_full_ft \
  --dataset-root /mnt/disk_2/wjz/datasets/pi05_ee_delta_ee_only/train \
  --output-dir /mnt/disk_2/wjz/pi05_runs/pi05_fullft_zero2_6gpu_pilot_5k \
  --policy-repo-id carm/pi05-full-ft-zero2-6gpu-pilot-5k \
  --job-name pi05-full-ft-zero2-6gpu-pilot-5k \
  --gpus 0,1,2,3,4,5 \
  --num-processes 6 \
  --main-process-port 29710 \
  --batch-size 1 \
  --steps 5000 \
  --learning-rate 5e-5 \
  --gradient-accumulation-steps 1 \
  --zero-stage 2 \
  --mixed-precision bf16 \
  --policy-dtype bfloat16
```

注意：

- 当前 launcher 里 policy preset 最终会把有效学习率落到 `2.5e-5`
- 如果要完全显式控制，可以后续再单独补 launcher 参数映射

### 6.2 从 5k resume 到 20k

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
PYTHONUNBUFFERED=1 \
accelerate launch \
  --use_deepspeed \
  --zero_stage 2 \
  --gradient_accumulation_steps 1 \
  --mixed_precision bf16 \
  --main_process_port 29711 \
  --num_processes 6 \
  lerobot-train \
  --config_path=/mnt/disk_2/wjz/pi05_runs/pi05_fullft_zero2_6gpu_pilot_5k/checkpoints/005000/pretrained_model/train_config.json \
  --resume=true \
  --steps=20000
```

## 7. 当前结论

最合理的 dense full-ft 主方案是：

- **本地 6 卡**
- **`bf16 + ZeRO-2`**
- **per-GPU batch = 1**
- **先 5k，再 resume 到 20k**
- **`save_freq=2500`（pilot）/ `10000`（正式）**

这条路线的优点是：

- 已被 smoke 和 strict resume 验证
- 与当前 LoRA 主线最容易形成可解释对照
- 对代码和 launcher 的新增工程风险最小
