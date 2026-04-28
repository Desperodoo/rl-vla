# PI05 微调踩坑与当前护栏（2026-04-28）

这份文档专门记录 CARM 上 PI0.5/PI05 微调过程中已经踩过的坑，以及当前最低测试门槛。时间线型记录继续看 `PI05_PROGRESS_2026-04-22.md`；这里更偏“以后不要再犯同一种错”的操作备忘。

## 当前基线

- 分支：`maniskill`
- 最新 checkpoint/resume 修复 commit：`9a80593 fix pi05 zero checkpoint resume`
- 主训练入口：`accelerate launch -m rlft.offline.patched_lerobot_train`
- 当前推荐初始化：`pi05_base`
- 当前推荐数据/action contract：
  - dataset：`pi05_ee_delta_ee_only`
  - observation state：`ee_only`
  - action representation：`ee_delta_pose_gripper`
- 当前 dense full-ft 训练形态：
  - `10 x RTX 4090`
  - DeepSpeed ZeRO-2
  - `bf16`
  - per-GPU micro-batch `1`
  - `gradient_accumulation_steps=4`
  - effective global batch `40`

## 硬性测试门槛

PI05 训练修复不能再只靠接口 smoke 或 dummy policy 就宣称完成。dense ZeRO PI05 的最低 checkpoint 测试必须覆盖正式训练同构路径：

1. 使用真实转换后的 `/home/wjz/openpi/pi05_base_pytorch`。
2. 使用真实 CARM LeRobot 数据集，不用 dummy module。
3. 按正式训练形态启动：multi-GPU、ZeRO-2、bf16、目标 gradient accumulation。
4. 跑到 checkpoint step，并检查 checkpoint 子目录完整存在。
5. 从该 checkpoint resume，且 resume 发生在 `accelerator.prepare(...)` 之后。
6. 日志必须明确加载 model weights、optimizer states、scheduler states、dataloader sampler states、random states。
7. resume 后继续训练若干 step，并再次保存 checkpoint。

2026-04-28 已通过上述门槛的严格 smoke：

- 机器：`10.20.64.119`
- GPU：`10 x RTX 4090`
- 环境：`/home/wjz/miniconda3/envs/rlft_ms3_lerobot`
- 模型：`/home/wjz/openpi/pi05_base_pytorch`
- 数据：`/home/wjz/datasets/pi05_ee_delta_ee_only/train`
- 配置：`10 GPU + ZeRO-2 + bf16 + gradient_accumulation_steps=4`
- 保存 smoke：`steps=2, save_freq=2`
- resume smoke：从 `000002` resume 到 `000004`

低于这个标准的测试只能叫 partial smoke，不能叫“训练链路已经验证完毕”。

## 已踩过的坑

### 1. OpenPI checkpoint 准备

- 官方 OpenPI checkpoint 起点是 JAX/OpenPI 资产；LeRobot PI05 训练需要转换后的 PyTorch 目录，并带兼容的 `config.json`、processor config、tokenizer config、action dimension。
- `pi05_base` 在 converter 里映射到 `pi05_droid` config family，这是预期行为，但容易误读。
- 公开 GCS 下载必须有稳健 fallback；只看本地 cache 路径存在，可能复用到不完整 artifact。

护栏：

- 把转换后的 PyTorch 目录当成一等 artifact 管理。
- 长训练前必须先测试 `make_policy(...)` 能从转换目录真实加载。

### 2. Tokenizer / processor 接线

- PI05 需要 complementary data 里有语言/task 文本。
- 当前 repo 生成的 preprocessor 包含：
  - `pi05_prepare_state_tokenizer_processor_step`
  - `tokenizer_processor`
- tokenizer 是 PaliGemma 兼容的 `google/paligemma-3b-pt-224`。
- PI05 processor 不是直接 tokenize 原始 task 文本，而是拼成：
  - `Task: {task}, State: {discretized_state};\nAction: `

护栏：

- 只要 task 字段、state normalization、processor 顺序有变化，训练前必须让真实 batch 过一遍 policy preprocessor，检查 keys、dtypes、shapes 和实际 prompt。

### 3. 当前 task 语义过于隐式

- 当前 export 对每一帧写死：
  - `task = "carm_fixed_dual_light"`
- LeRobot 数据文件里主要保存 `task_index`；`LeRobotDataset.__getitem__` 再从 `meta/tasks.parquet` 映射回 task string。
- 当前 dataset metadata 显示 `total_tasks = 1`。
- 所以 PI05 不是完全没收到 task，而是收到了一个固定 identifier，不是用户显式提供的自然语言任务描述。
- PI05 processor 会把下划线替换为空格，所以实际 prompt 近似：
  - `Task: carm fixed dual light, State: ...; Action: `

含义：

- 对单任务 BC 来说，这能跑通：所有 demo 属于同一任务分布，视觉/state 上下文和恒定任务字符串共同约束动作。
- 对多任务切换、长程 phase 控制、以及部署时 prompt 与训练 prompt 不一致的场景，这个做法偏弱，必须显式设计 task/subtask 语义。

护栏：

- 未来多任务或长程数据导出前，必须先定义 task/subtask 语义写法，并验证 PI05 真实 forward/select-action 路径消费这些语义。

### 4. 当前 PI05 行为没有真正用上 subtask

- LeRobot batch 如果带 `subtask` 字段，processor/converter 可以把它放进 complementary data。
- `TokenizerProcessorStep` 可以把 `subtask` tokenize 成：
  - `OBS_LANGUAGE_SUBTASK_TOKENS`
  - `OBS_LANGUAGE_SUBTASK_ATTENTION_MASK`
- 但当前 PI05 forward/select-action 路径只读取：
  - `OBS_LANGUAGE_TOKENS`
  - `OBS_LANGUAGE_ATTENTION_MASK`
- 当前 CARM export 也没有写 `subtask`。

含义：

- 只加一个 `subtask` column 不等于 PI05 行为真的受 subtask 控制。
- 当前最可靠的 phase 条件注入方式，是把 phase/subtask 文本编码到主 `task` 字符串里，按帧或按片段变化。

### 5. Action representation 不可混用

- PI05 action dim 必须和转换后的 policy config、dataset bridge contract 对齐。
- 当前已转向 `ee_delta_pose_gripper`，导出 action dim 为 `7`。
- 混用 `absolute_pose_gripper`、raw CARM action、delta EE action，会让训练 target 和 offline metric 都失真。

护栏：

- 训练/eval 前检查 `pi05_bridge_metadata.json`、dataset `meta/info.json`、policy `config.json` 的 action dim 与 representation 是否一致。

### 6. Observation state contract 不可混用

- `joint_only` 和 `ee_only` 是不同 contract。
- 当前 PI05 数据集是 `ee_only`，state dim 为 `8`；processor 会 pad 到 PI05 `max_state_dim=32`。
- PI05 state-token prompt 构造前，state 预期已经被 normalizer 归一化。

护栏：

- 做横向比较前，必须确认 dataset `observation.state` shape 和 normalization stats。

### 7. `pi05_base` vs `pi05_libero`

- 在当前 CARM setting 下，未微调的 `pi05_libero` 某些 MAE 略低，但 MSE 更高。
- 进入 LoRA 微调后，`pi05_base` 在现有 val/test 比较中稳定优于 `pi05_libero`。

护栏：

- 除非新的受控实验推翻现有结论，否则 dense/full fine-tuning 继续以 `pi05_base` 为起点。

### 8. LoRA batch 早期过于保守

- LoRA `batch2` 能跑，但没有充分利用硬件。
- `batch64` LoRA 改善了对比结果，并推动后续 larger effective batch dense 实验。

护栏：

- 报告 batch 时必须报告 effective batch，而不只是 per-device batch。
- 横向比较必须使用同一套 val/test offline eval 脚本和数据 split。

### 9. 普通 DDP dense full-ft 会 OOM

- PI05 dense full-ft 在普通 DDP 下复制模型和 optimizer state，显存压力过大。
- 目标训练形态必须使用 ZeRO。

护栏：

- dense full-ft 使用 DeepSpeed ZeRO-2。
- 保持 gradient checkpointing。
- 非 ZeRO dense 结果不能和 ZeRO dense 训练稳定性混为一谈。

### 10. Gradient accumulation 早期只是“参数传进去了”

- 旧 LeRobot loop 里只传 `--gradient_accumulation_steps` 不够。
- 训练循环没有使用 `accelerator.accumulate(...)`，每个 micro-batch 都直接 `optimizer.step()`。
- 日志已经暴露问题：effective batch 和 sample counter 都没有体现 accumulation。

修复：

- `rlft/offline/patched_lerobot_train.py` 在 update 中使用 `accelerator.accumulate(policy)`。
- 同时修正 accumulated update 下的 progress/sample 统计。

护栏：

- 检查日志打印的 effective batch 是否等于预期。
- step time 应该体现多个 micro-step 聚合成一次 optimizer update。

### 11. PI05 bf16 dtype 问题

多个 bf16 问题只有在真实 GPU 分布式训练时才暴露：

- `action_in_proj`：float action tensor 进入 bf16 projection。
- `time_mlp`：time embedding 路径需要 dtype 对齐。
- `action_out_proj`：suffix output 路径需要 dtype 对齐。
- MSE/loss 路径也需要避免 mixed precision 意外。

修复：

- `rlft/offline/patched_lerobot_train.py` 里 runtime patch PI05 action/time projection 路径，把输入 cast 到相关 module weight dtype。

护栏：

- CPU 或 fp32 smoke 不能验证这个问题。
- 必须用真实 PI05 + GPU + bf16 测。

### 12. Checkpoint saver 函数签名漂移

- 不同 LeRobot 环境的 `save_checkpoint(...)` 签名不一致。
- 有的接受 `accelerator=...`，有的不接受。

修复：

- `_save_checkpoint_compat(...)` inspect 函数签名，只传当前环境支持的 kwargs。

护栏：

- 不假设本机和远端 site-packages 完全一致。
- 优先使用 repo-local patched launcher，不直接走环境里的 CLI 入口。

### 13. shared tensor / safetensors 保存失败

- 真实 PI05 模型保存时遇到 shared/incomplete storage safetensors 错误。
- dummy policy 暴露不出这个问题。
- 失败会留下不完整 checkpoint 目录。

修复：

- runtime patch `PreTrainedPolicy._save_pretrained`：
  - 先尝试原 safetensors 保存。
  - 命中 shared-storage 问题时，clone incomplete-storage tensors。
  - 再走 `save_torch_state_dict(..., safe_serialization=True)` fallback。

护栏：

- 要验证真实 `policy.save_pretrained(...)` 和真实 reload，不能只看 checkpoint 目录被创建。

### 14. DeepSpeed ZeRO optimizer state 不是普通 optimizer state dict

- 旧 LeRobot optimizer saver 期待 `state_dict()["param_groups"]`。
- DeepSpeed ZeRO optimizer state 不是普通 PyTorch optimizer 的 `{"state", "param_groups"}` 结构。
- 该问题在长训练 checkpoint step 才出现：
  - `KeyError: 'param_groups'`

修复：

- DeepSpeed 场景下不再调用旧 LeRobot optimizer-state saver。
- main rank 保存 policy/config/processors。
- 全部 rank 共同调用 `accelerator.save_state(..., safe_serialization=False)` 保存原生 ZeRO 分片训练态。
- resume 在 `accelerator.prepare(...)` 后调用 `accelerator.load_state(...)`。

护栏：

- ZeRO 训练的 checkpoint 验证必须包含 resume，并确认 optimizer/scheduler/random/dataloader state 都加载成功。

### 15. 远端 launcher 漂移

- 远端 run 可能悄悄使用 `/home/wjz/miniconda3/envs/.../bin/lerobot-train`，绕过 repo-local patch。
- 这之前已经导致过 gradient accumulation 和 bf16 修复没有生效。

护栏：

- 通过 repo-local module 启动：
  - `python -m rlft.offline.launch_pi05_zero2_full_ft`
  - 内部应走 `accelerate launch -m rlft.offline.patched_lerobot_train`
- 确认 `PYTHONPATH` 包含 repo root。
- 启动前确认远端 git commit。

### 16. Home/root 磁盘紧张时的 checkpoint/cache 落盘

- 当前 `/` 和 `/home` 在同一块盘上，空间很紧张；大模型 checkpoint、HF cache、导出的 LeRobot 数据集不能默认落在 home/root。
- `/home/wjz/.cache` 当前是软链接到 `/mnt/disk_2/wjz/.cache`，但长任务不能只依赖这个隐式事实。
- 下载 VLM/PI05 checkpoint 前显式设置：
  - `HF_HOME=/mnt/disk_2/wjz/.cache/huggingface`
  - `HF_HUB_CACHE=/mnt/disk_2/wjz/.cache/huggingface/hub`
  - `XDG_CACHE_HOME=/mnt/disk_2/wjz/.cache`
  - `TMPDIR=/mnt/disk_2/wjz/tmp`
- 不建议设置 `TRANSFORMERS_CACHE`，否则 transformers 可能在 `.../huggingface/transformers` 下另建一套模型缓存，和 hub cache 重复占空间。
- Qwen2.5-VL 下载时遇到过 Hugging Face Xet `416 Range Not Satisfiable`。当前更稳的下载方式是加：
  - `HF_HUB_DISABLE_XET=1`

护栏：

- 大下载启动后立刻检查：
  - `df -h / /mnt/disk_2`
  - `readlink -f /home/wjz/.cache/huggingface`
  - `lsof -p <pid> | rg 'huggingface|incomplete|tmp'`
- 如果出现重复 cache 或 Xet 残片异常，停掉进程，只清理目标模型的 `.incomplete` 和 `.locks`，不要删除已完整落盘的 blobs/snapshots。

## 长程任务 / subtask 语义建议

PI05 长程任务的条件注入可以分三层：

1. 单一固定 task string
   - 这是当前做法。
   - 只适合所有 demo 都来自同一行为分布的单任务 BC。
   - 不足以做显式 phase 控制。

2. 按帧或按片段变化的主 `task` string
   - 这是当前 PI05 wiring 下最推荐的第一步升级。
   - 例子：
     - `pick up the object`
     - `move the object above the target`
     - `place the object and release gripper`
   - 这些文本会进入 PI05 当前实际消费的主 language tokens。
   - 推理时必须提供同样的 phase 调度，可以来自 script policy、task tree、heuristic detector 或 learned stage estimator。

3. 单独 `subtask` 字段
   - LeRobot 可以存储并 tokenize。
   - 但当前 PI05 model path 看起来没有直接消费 `OBS_LANGUAGE_SUBTASK_*`。
   - 只有在新增/确认模型支持后，或有其它 policy head 显式消费 subtask tokens 时，才应该依赖这个路径。

对 CARM 的建议：

- 当前单任务 dense full-ft 可以继续使用固定 task string，但必须把它记为限制。
- 采集或导出多任务/长程数据前，先定义小而稳定的 task vocabulary 和 phase 标注策略。
- 优先使用自然语言 task string，不要继续依赖 opaque identifier。
- 增加数据验证脚本：训练前打印 task counts、样例 task strings、可选 subtask counts、以及一个真实 PI05 processed prompt。

## 2026-04-28 subtask 自动标注与重导出入口

当前已为 `pick_and_place_tape_into_cup` 增加本地优先的 subtask 标注/重导出工具链：

- 任务语义配置：
  - `configs/pi05_task_semantics/pick_and_place_tape_into_cup.json`
- 标注入口：
  - `python -m rlft.offline.annotate_pi05_subtasks`
- 带 subtask prompt 的 LeRobot 导出：
  - `python -m rlft.offline.export_carm_to_lerobot --task-semantics-path ... --subtask-annotations-path ...`
- 导出后验证：
  - `python -m rlft.offline.validate_pi05_subtask_dataset`

推荐执行顺序：

1. 先跑 pilot，每个条件目录 2 条：
   - `python -m rlft.offline.annotate_pi05_subtasks --pilot-episodes-per-subset 2 --run-vlm`
2. 人工检查 `runs/pi05_subtask_annotations/pick_and_place_tape_into_cup/review.html`。
3. pilot 通过后跑全量：
   - `python -m rlft.offline.annotate_pi05_subtasks --run-vlm`
4. 基于 `recorded_data_splits/{train,val,test}_manifest.json` 分别重导出 train/val/test。
5. 用 validator 确认：
   - `meta/tasks.parquet` 有两个自然语言 task prompt。
   - 每个 episode 的 `task_index` 只切换一次。
   - 边界前后 prompt 与 sidecar annotation 一致。

注意：

- 原始 HDF5 不改；标注结果落到 sidecar `annotations.json`。
- 当前 PI05 行为仍依赖主 `task` 文本，因此导出时把当前 subtask instruction 拼进 `task` prompt。
- 独立 `subtask` 字段暂不作为训练行为依赖，除非后续确认 PI05 forward/select-action 真实消费 `OBS_LANGUAGE_SUBTASK_*`。

### 2026-04-28 VLM 标注 pilot 结果

本地 VLM 路径已经跑通，但当前结果不能直接全自动信任：

- `Qwen/Qwen3-VL-30B-A3B-Instruct` 不能直接用于当前训练 env：`transformers==4.53.2` 不识别 `qwen3_vl`。
- 改用 `Qwen/Qwen2.5-VL-7B-Instruct`，权重缓存必须放在 `/mnt/disk_2/wjz/.cache/huggingface/hub`。
- Hugging Face Xet 下载遇到过 `416 Range Not Satisfiable`，已改为 `HF_HUB_DISABLE_XET=1`。
- 小文件 HEAD 请求遇到代理 SSL/connection reset；当前可用 workaround 是：
  - 大权重使用 HF hub cache。
  - 本地隔离目录 `/mnt/disk_2/wjz/models_local/Qwen2.5-VL-7B-Instruct-local` symlink 权重。
  - tokenizer/processor 小文件来自已缓存 VL 模型并修正为 `Qwen2_5_VLProcessor`。
  - 标注时加 `--local-files-only`、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`。
- `qwen_vl_utils` 默认先尝试 torchcodec，会因当前 torch/ffmpeg/torchcodec 组合输出大量 warning；当前脚本默认 `FORCE_QWENVL_VIDEO_READER=torchvision`。
- `fixed_no_light` 的早期 HDF5 没有 `observations/images_by_camera`，只有 `observations/images`；review video 已支持单视角 fallback。

pilot 输出：

- probe：`/mnt/disk_2/wjz/runs/pi05_subtask_annotations/pick_and_place_tape_into_cup_probe1_qwen25vl7b_local`
- prompt v1 pilot：`/mnt/disk_2/wjz/runs/pi05_subtask_annotations/pick_and_place_tape_into_cup_pilot_qwen25vl7b_local`
  - 8 条中 2 条 `needs_review`。
  - 但 raw VLM 几乎全部输出 `00:02` 边界，疑似受模板诱导。
- prompt v2 pilot：`/mnt/disk_2/wjz/runs/pi05_subtask_annotations/pick_and_place_tape_into_cup_pilot_qwen25vl7b_local_prompt2`
  - 8 条中 5 条 `needs_review_boundary_signal_disagreement`。
  - VLM 边界不再全是 2 秒，但和 robot signal 冲突比例较高。

结论：

- 当前本地 Qwen2.5-VL 可以作为候选边界生成器，但全量标注必须经过 `review_status` gate。
- 不允许把 `needs_review` episode 默认导出进正式 PI05 subtask prompt 数据集。
- 正式全量跑完后先执行：
  - `python -m rlft.offline.validate_pi05_subtask_annotations --subtask-annotations-path ... --review-queue-path ...`
- 只有 review queue 清空或人工修正 sidecar 后，才能重导出 train/val/test。

全量候选标注已完成：

- 输出目录：`/mnt/disk_2/wjz/runs/pi05_subtask_annotations/pick_and_place_tape_into_cup_full_qwen25vl7b_local_prompt2`
- sidecar：`annotations.json`
- review 页面：`review.html`
- review queue：`review_queue.txt`
- 统计：
  - 总数：127/127
  - `auto`：53
  - `needs_review`：74
  - `needs_review_boundary_signal_disagreement`：73
  - `needs_review_vlm_parse_error`：3
- subset review 数：
  - `fixed_dual_light`：16/26
  - `fixed_left_light`：17/25
  - `fixed_no_light`：27/50
  - `random_no_light`：14/26

这份全量 sidecar 是候选标注，不是可直接训练的 gold annotation。当前 export 默认会拒绝 `needs_review`，这是正确行为。

2026-04-28 review app 状态：

- 交互式 review app 可用：
  - `python -m rlft.offline.review_pi05_subtask_annotations --annotations-path ... --port 8766`
- 浏览器入口：
  - `http://127.0.0.1:8766/`
  - 远程机器当前可用 `http://10.20.14.169:8766/`
- 由于 OpenCV 写出的 `mp4v` review video 和普通 HTTP server 的 Range 行为在浏览器里不稳定，review app 已改为默认逐帧 JPEG 审核：
  - 拖动 frame slider 查看双视角帧。
  - 找到边界帧后点击 `Set Boundary To Shown Frame`。
  - 点击 `Accept / Save` 写入 `annotations_reviewed.json`。
- app 不覆盖原始 `annotations.json`。

2026-04-28 人工 review 后更新 subtask 边界标准：

- 旧 VLM 标准偏弱：容易把“刚抓住 tape roll”当成 pick 完成。
- 新标准必须写入 VLM prompt：
  - `pick_tape` 持续到机械臂夹爪抓住 black tape roll 并往上提。
  - 不能在首次接触或刚稳定抓住时结束 pick。
  - 只有当夹爪已经抓住并抬起 black tape roll，且 wrist/第一视角中首次出现 blue cup 时，才算完成 pick。
  - 从这个时刻开始进入 `place_tape_in_cup`，即 lifted tape roll 开始朝可见 blue cup 移动。
- 下一轮全量标注应使用新 prompt，输出到新目录，不覆盖：
  - `/mnt/disk_2/wjz/runs/pi05_subtask_annotations/pick_and_place_tape_into_cup_full_qwen25vl7b_local_cup_visible_boundary`

2026-04-28 晚间，新标准全量候选标注已完成：

- 输出目录：`/mnt/disk_2/wjz/runs/pi05_subtask_annotations/pick_and_place_tape_into_cup_full_qwen25vl7b_local_cup_visible_boundary`
- sidecar：`annotations.json`
- review queue：`review_queue.txt`
- 统计：
  - 总数：127/127
  - `auto`：47
  - `needs_review`：80
  - `needs_review_boundary_signal_disagreement`：80
- subset review 数：
  - `fixed_dual_light`：16/26
  - `fixed_left_light`：15/25
  - `fixed_no_light`：37/50
  - `random_no_light`：12/26
- validator 已确认没有 missing/extra episode；退出码为 1 是因为仍有 `needs_review`，这是预期 gate。
- 这份新 sidecar 仍是候选标注，必须人工 review 后才允许导出正式 PI05 subtask prompt 数据集。

## 当前正式 run 快照

截至 2026-04-28 下午 CST，严格 save/resume smoke 通过后：

- 正式训练 screen：`pi05_fullft_remote10_accum4_5k_20260428_fix4`
- post-eval watcher：`pi05_fullft_remote10_accum4_5k_posteval_20260428_fix4`
- run dir：`/home/wjz/pi05_runs/pi05_fullft_zero2_10gpu_accum4_5k_20260428_fix4`
- 配置：`10 GPU / ZeRO-2 / bf16 / accum4 / effective batch 40 / steps 5000 / save_freq 1000`
- 早期日志检查显示已进入训练循环，并跑到早期 step，暂未看到 traceback。
- 这还不能算完整成功；至少要跨过第一个正式 checkpoint，并完成或可 clean resume 后，才可以说当前长训练链路真正闭环。
