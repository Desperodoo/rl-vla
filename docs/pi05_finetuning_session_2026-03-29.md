# pi0.5 / pi05 微调集成会话总结

> 日期：2026-03-29
> 仓库：`/home/wjz/rl-vla`
> 主题：在当前项目中补齐基于 LeRobot / OpenPI 的 pi0.5（LeRobot 中命名为 `pi05`）微调链路

---

## 1. 本次工作的目标

本轮工作的目标不是训练出最终可用的大规模模型，而是把当前仓库中的 **CARM 数据 → LeRobot 数据集 → pi05 训练入口** 这一整条链路补齐并验证打通。

具体目标包括：

1. 确认当前环境里是否有 LeRobot / `lerobot-train`
2. 如果主环境不适合，建立可隔离的 fallback 环境
3. 给 `rlft/offline/train_pi05.py` 补上结构化的：
   - env probe
   - dataset / bridge validation
   - auto export
   - command generation
4. 把 CARM HDF5 数据导出成 **LeRobot 原生本地数据集格式**
5. 验证 `pi05` 在本机上的：
   - 单卡最省显存 smoke
   - 双卡 direct `lerobot-train` smoke
   - 双卡 LoRA smoke
6. 记录上游依赖、已踩坑、可行路线和当前结论

---

## 2. 最终结论

### 2.1 当前已经验证打通的路线

当前已经在这台机器上实际打通了三条训练路线：

1. **direct 双卡 full / 半冻结训练**
2. **direct 双卡 LoRA 训练（基于本地 warmup checkpoint 继续训练）**
3. **direct 双卡 LoRA 训练（基于 official OpenPI `pi05_droid` checkpoint 转换产物）**

### 2.2 当前推荐的工程路线

推荐把整体链路固定为：

- `train_pi05.py` 负责：
  - environment probe
  - bridge validation
  - auto export
  - command generation
- 真正的分布式训练统一走：
  - `accelerate launch ... lerobot-train ...`

也就是：

- **wrapper 负责准备工作**
- **真正训练不再通过 wrapper dispatch，而是 direct `lerobot-train`**

### 2.3 LoRA 路线的关键结论

LeRobot **支持 PEFT / LoRA**，但：

- **不支持从零初始化直接做 PEFT**
- 必须提供：
  - `policy.pretrained_path`

本次已经验证两条可行来源：

- 可以直接拿我们刚跑出的 **full 10-step checkpoint** 作为 `policy_pretrained_path`
- 也可以使用 **official OpenPI `pi05_droid` checkpoint** 的本地转换产物：
  - `/mnt/disk_2/wjz/openpi/pi05_droid_pytorch`
- 这两条路线都已成功完成 **双卡 LoRA 10-step smoke**

---

## 3. 主要代码改动

### 3.1 新增文件

#### `rlft/offline/pi05_bridge/env_probe.py`

作用：

- 结构化探测当前 Python / conda / CUDA / LeRobot / CLI 状态
- 输出可写入 JSON 的 probe 结果
- 供 `train_pi05.py` 在 dispatch 前做 gate

实现重点：

- probe `sys.executable` / `CONDA_PREFIX`
- probe `torch` / `numpy` / `h5py` / `tyro` / `lerobot`
- probe `torch.cuda.is_available()` / GPU 数量
- probe `shutil.which("lerobot-train")`
- probe `lerobot-train --help`
- 返回 `summary.ok + checks[]`

后续修复：

- 把 `lerobot-train --help` timeout 从致命错误改为非致命 warning
- 允许 help 探测超时但保留 CLI 可用判定

---

#### `rlft/offline/pi05_bridge/validate.py`

作用：

- 做 bridge dataset 结构校验
- 做 LeRobot 本地数据集路径校验
- 做 `lerobot-train` 命令校验

实现重点：

- `validate_bridge_dataset(...)`
- `validate_lerobot_dataset_path(...)`
- `validate_lerobot_train_command(...)`

后续增强：

- 从“自定义 bridge 格式”切换到 **LeRobot 原生本地 layout** 校验
- 增加 PEFT 约束：
  - `use_peft=True` 时必须有 `policy_pretrained_path`

---

#### `rlft/offline/pi05_bridge/export.py`

作用：

- 将 CARM HDF5 导出成 **LeRobot 原生本地数据集**

最终实现方式：

- 使用 `LeRobotDataset.create(...)`
- 使用 `dataset.add_frame(...)`
- 使用 `dataset.save_episode()`

导出结果：

- `meta/info.json`
- `meta/tasks.parquet`
- `meta/episodes/...`
- `data/.../*.parquet`
- `pi05_bridge_metadata.json`

这一步很关键，因为早期版本只是自定义 HDF5 bridge 产物，**不能被 LeRobot 原生回读**。

---

#### `rlft/offline/export_carm_to_lerobot.py`

作用：

- 作为独立 CLI，单独跑数据导出 + 回读验证

用途：

- 数据侧单独验证
- 不依赖训练链路即可确认导出有效

---

### 3.2 主要修改文件

#### `rlft/offline/train_pi05.py`

这是本轮集成的中心入口。

新增 / 强化的能力包括：

- bridge smoke
- bridge validate
- strict mode
- auto export LeRobot dataset
- env probe artifact 写出
- bridge validation artifact 写出
- dataset validation artifact 写出
- lerobot command artifact 写出
- 分布式场景下 rank-aware export / validation
- dispatch 前严格校验

新增的核心参数包括：

- `bridge_smoke_only`
- `bridge_validate_only`
- `strict`
- `auto_export_lerobot_dataset`
- `export_output_dir`
- `policy_repo_id`
- `policy_pretrained_path`
- `policy_push_to_hub`
- `use_peft`
- `peft_method_type`
- `peft_r`
- `peft_target_modules`
- `peft_full_training_modules`
- `lerobot_dataset_repo_id`
- `lerobot_dataset_path`
- `dispatch_to_lerobot`

本轮还额外加了一个关键约束：

- 当 `use_peft=True` 且没有 `policy_pretrained_path` 时
- 在本地入口直接报错
- 不再等到 LeRobot runtime 阶段才失败

相关位置：

- `rlft/offline/train_pi05.py`
- `rlft/offline/pi05_bridge/config_bridge.py`
- `rlft/offline/pi05_bridge/validate.py`

---

#### `rlft/offline/pi05_bridge/config_bridge.py`

作用：

- 生成稳定的 run config
- 生成 LeRobot 训练命令

本轮关键改动：

- 修正策略命名映射：
  - `pi0.5 -> pi05`
- 增加：
  - `policy.repo_id`
  - `policy.push_to_hub=false`
  - `policy.pretrained_path`
- 增加默认低显存训练参数：
  - `--policy.gradient_checkpointing=true`
  - `--policy.freeze_vision_encoder=true`
  - `--policy.train_expert_only=true`
  - `--policy.dtype=bfloat16`
- 暴露 LoRA 参数：
  - `--peft.method_type`
  - `--peft.r`
  - `--peft.target_modules`
  - `--peft.full_training_modules`
- 增加 PEFT 早报错：
  - 没有 `policy_pretrained_path` 时直接拒绝生成 LoRA 命令

---

#### `rlft/offline/__init__.py`

做了简化。

原因：

- 原先存在和 `train_maniskill` / diffusers / transformers 等重依赖链的耦合
- 导致只是运行 `python -m rlft.offline.pi05_bridge.env_probe` 也会被无关依赖拖死

调整后效果：

- `env_probe` 可以独立运行
- 不再被无关模块 import 链阻塞

---

## 4. 数据链路最终形态

### 4.1 输入数据

本次用于验证的 CARM demo 路径：

- `recorded_data/fixed_dual_light`

该数据量不大，但足够做链路验证。

### 4.2 CARM → LeRobot 导出结果

导出后得到 **LeRobot 原生本地数据集**，本次验证用的数据目录为：

- `/mnt/disk_2/wjz/runs/pi05_train_smoke_export_mgpu4`

训练日志中可见：

- `dataset.num_frames=897`
- `dataset.num_episodes=1`

这说明：

- 导出结果可被 LeRobot 正常识别
- 数据读取链路是通的

---

## 5. 环境与依赖处理

### 5.1 主环境 `rlft_ms3` 的结论

一开始确认发现：

- `rlft_ms3` 中没有 `lerobot`
- 也没有 `lerobot-train`

用户同意后，曾尝试直接在 `rlft_ms3` 安装 LeRobot。

### 5.2 为什么后来不继续污染 `rlft_ms3`

直接在 `rlft_ms3` 安装后，引入了多组依赖漂移：

- `huggingface_hub`
- `wandb`
- `gymnasium`
- `av`
- `transformers`

而且和现有项目依赖、ManiSkill/其它链路混在一起，风险较高。

所以后来采取的做法是：

1. 把 `rlft_ms3` 尽量恢复到被污染前状态
2. 切换到独立 fallback 环境：
   - `rlft_ms3_lerobot`

这一步是对的。后续的 pi05 / LeRobot / OpenPI 相关工作都主要在这个隔离环境中推进。

---

## 6. 本轮踩过的主要坑与解决方式

下面只记录对后续最有价值的坑。

### 6.1 坑：wrapper + accelerate + tyro 在 distributed 模式下参数解析不稳定

现象：

- `accelerate launch -m rlft.offline.train_pi05 ... --dispatch_to_lerobot`
- 或者 `--dispatch-to-lerobot`
- 都在多次尝试中表现不稳定 / 被判为 unrecognized option

结论：

- `train_pi05.py` 不适合作为 distributed training 的外层 entrypoint
- 尤其不适合在 `accelerate launch -m ...` 下再做 wrapper dispatch

解决：

- 不再用 wrapper 直接承载 distributed dispatch
- 固定架构为：
  - wrapper 做 prepare / export / validate
  - 真正训练直接 `accelerate launch lerobot-train ...`

这是本轮最重要的架构调整之一。

---

### 6.2 坑：初版 exporter 导出的不是 LeRobot 原生格式

现象：

- 初版 `export.py` 只是写了自定义 bridge 数据
- 后续 `LeRobotDataset(...)` 无法像原生数据集那样回读

解决：

- 重写 exporter，改用 LeRobot 官方 API：
  - `LeRobotDataset.create(...)`
  - `add_frame(...)`
  - `save_episode()`

结果：

- 生成了标准 `meta/` + `data/` parquet 布局
- 训练链路成功读取

---

### 6.3 坑：分布式导出时 rank 竞争导致非主进程过早校验

现象：

- rank 0 还没写完数据
- 其他 rank 已经开始校验 dataset path
- 导致误报缺文件 / 不完整

解决：

- rank0-only export
- rank0-only strict dataset validation
- 非主进程等待 `.export_done` 或完整 parquet 结构

这是后来 distributed smoke 能稳定跑通的关键之一。

---

### 6.4 坑：LeRobot / OpenPI 对 transformers 有专门 patch 依赖

现象：

- 标准 `transformers` 版本下，会出现：
  - `transformers.models.siglip.check` 缺失
  - `PaliGemmaForConditionalGeneration` API 不匹配
  - `GemmaForCausalLM` 某些字段 / 接口不匹配

结论：

- `pi05` 不是“标准 transformers + 标准 LeRobot”就一定能直接跑
- 它依赖 OpenPI 兼容层 / patch 过的 transformer 实现

解决：

- 查 OpenPI 上游 README 和替换文件
- 应用了 OpenPI 对 transformers 的替换补丁
- 并在隔离环境中完成适配

这是本轮依赖适配里最关键的一层。

---

### 6.5 坑：Hugging Face gated 资源访问

现象：

- 部分底层模型 / 资源需要鉴权
- 无 token 时会 401

解决：

- 使用用户提供的 HF token
- 在训练命令中通过环境变量暴露：
  - `HF_TOKEN`
  - `HUGGINGFACE_HUB_TOKEN`

---

### 6.6 坑：单卡 24GB 显存不够稳定支撑完整 pi05 训练

现象：

即便使用：

- `batch_size=1`
- `gradient_checkpointing=true`
- `freeze_vision_encoder=true`
- `train_expert_only=true`
- `dtype=bfloat16`

单卡仍然非常吃紧，不适合作为稳态方案。

结论：

- 当前机器上更现实的方案是 **双卡**

---

### 6.7 坑：LoRA 不能从零开始

现象：

LoRA 直启时，LeRobot 明确报错：

- `Training from scratch using PEFT is unlikely to yield good results. Supply a policy.pretrained_path to fine-tune an existing model.`

解决：

- 补 `policy_pretrained_path`
- 并在本地 wrapper / command builder / validation 层提前报错
- 同时验证本地 warmup checkpoint 可作为 LoRA 初始化源

---

### 6.9 坑：OpenPI 官方 checkpoint 下载不是单纯路径问题，而是依赖 / Python 版本 / 运行时假设叠加

现象：

官方 `gs://openpi-assets/checkpoints/pi05_droid` 路线在真正打通之前，连续遇到了多层问题：

- `tqdm_loggable` 缺失
- `gcsfs` 缺失（本机又没有 `gsutil`）
- OpenPI 下载脚本使用 `datetime.UTC`，但当前 Python 3.10 没这个属性
- 下载阶段曾留下 stale `.lock` 文件，导致后续重试行为混乱

解决：

- 安装：
  - `tqdm-loggable`
  - `gcsfs`
- 给下载 / 转换子进程统一注入 Python 3.10 兼容补丁（`datetime.UTC -> datetime.timezone.utc`）
- 清理 stale lock 后再重试下载

结论：

- 官方 checkpoint 路径本身可用
- 主要问题是 OpenPI 运行时依赖和 Python 版本假设没有满足

---

### 6.10 坑：OpenPI JAX→PyTorch 转换脚本依赖闭包很深

现象：

`examples/convert_jax_model_to_pytorch.py` 不是一个“只依赖 JAX/Orbax”的小脚本，而是会沿着 OpenPI 的完整 import 链进入：

- model config
- tokenizer
- transforms
- normalize
- openpi-client

因此会依次暴露一串缺依赖：

- `beartype`
- `jaxtyping`
- `augmax`
- `ml_collections`
- `sentencepiece`
- `chex`
- `numpydantic`
- 本地 `openpi-client`

解决：

- 不再把它当成“单个转换脚本依赖少”的场景处理
- 直接按 OpenPI 上游 `pyproject.toml` 对齐关键版本：
  - `flax==0.10.2`
  - `orbax-checkpoint==0.11.13`
  - `jaxtyping==0.2.36`
  - `beartype==0.19.0`
- 然后继续把缺失的外围依赖逐个补齐

结论：

- 官方 checkpoint 的转换是可行的
- 但更适合放进一个专门的 OpenPI conversion 环境，而不是长期混在训练环境里

---

### 6.11 坑：官方转换产物默认 `config.json` 不是 LeRobot 可加载格式

现象：

OpenPI 转换脚本默认写出的 `config.json` 只是一个简化摘要，例如：

- `action_dim`
- `action_horizon`
- `paligemma_variant`
- `action_expert_variant`

但 LeRobot 需要的是带有：

- `type: pi05`
- policy-level fields
- normalization mapping
- scheduler / optimizer preset fields

的完整 `PreTrainedConfig` JSON。

解决：

- 在 helper 中统一重写 `config.json`
- 改写成 LeRobot 可直接读取的 `pi05` policy config

结论：

- 官方转换脚本产出的权重是可用的
- 但必须额外补一个 LeRobot-compatible config 层

---

### 6.12 坑：official checkpoint 进入 LeRobot 训练前还缺 processor 配置

现象：

即便权重和 config 都已经可被 LeRobot policy loader 读取，训练阶段依然会失败，因为还缺：

- `policy_preprocessor.json`
- `policy_postprocessor.json`

LeRobot 官方迁移脚本 `migrate_policy_normalization.py` 对这个 official checkpoint 没能自动产出有效结果：

- 它没有识别出有效 features
- 也没有成功生成可用 processor 文件

解决：

- 从本地已验证可跑的 pi05 checkpoint 中提取：
  - `policy_preprocessor.json`
  - `policy_postprocessor.json`
  - 对应的 safetensors state files
- 复制到 official checkpoint 目录
- 训练时再依赖 LeRobot 的 preprocessor overrides 用当前 dataset stats 覆盖统计信息

结论：

- 官方 checkpoint 的最后一层不是模型权重问题
- 而是 processor metadata / state 的兼容性问题

---

### 6.13 坑：tokenizer 已本地缓存，但 processor 仍可能走远程 repo id

现象：

即便本地 HF cache 已经有：

- `google/paligemma-3b-pt-224`

如果 `policy_preprocessor.json` 里仍写的是远程 repo id，`AutoTokenizer.from_pretrained(...)` 仍可能触发联网 HEAD / GET 请求，并在当前网络环境里报：

- `Connection reset by peer`

解决：

- 将 `policy_preprocessor.json` 里的：
  - `tokenizer_name: "google/paligemma-3b-pt-224"`
- 改为本地 snapshot 绝对路径：
  - `/home/wjz/.cache/huggingface/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c`

结论：

- tokenizer 不能只依赖“本地缓存应该命中”这种隐式行为
- 在不稳定网络环境下，最好直接把 processor 里的 tokenizer 指向本地 snapshot 路径

---

### 6.14 坑：为了跑 OpenPI conversion 安装本地 `openpi-client`，会进一步污染训练环境

现象：

安装本地 `openpi-client` 时，引入了额外依赖，并把：

- `numpy` 从 `2.2.6` 降到 `1.26.4`

造成更多环境漂移提示。

解决：

- 当前会话里为了继续收敛路线，接受了这次污染
- 但结论上应把 **OpenPI download/convert** 和 **LeRobot training** 分离到两个环境

结论：

- 这次路线虽然已经打通
- 但长期看，官方 checkpoint conversion 最好单独放进 conversion env 中执行

---

## 7. 训练验证结果

### 7.1 direct 双卡 full / 半冻结 smoke

命令思路：

- direct `accelerate launch ... lerobot-train`
- 2 GPU
- `freeze_vision_encoder=true`
- `train_expert_only=true`
- `gradient_checkpointing=true`
- `dtype=bfloat16`

结果：

- 成功完成 10/10 steps
- 成功保存 checkpoint
- 正常退出

关键指标：

- `dataset.num_frames=897`
- `dataset.num_episodes=1`
- `Effective batch size: 1 x 2 = 2`
- `num_learnable_params=693422112`
- `num_total_params=3616757520`

输出目录：

- `/mnt/disk_2/wjz/runs/pi05-full-smoke-direct`

生成的 checkpoint 可见：

- `checkpoints/000010/pretrained_model/`
- `checkpoints/last -> 000010`

---

### 7.2 direct 双卡 LoRA（无 pretrained） smoke

结果：

- 启动链路本身是通的
- 但运行时被 LeRobot 拒绝

失败原因：

- 从零初始化做 PEFT 不被允许

这一步的价值在于：

- 证明 PEFT / LoRA 功能在框架内是存在的
- 但必须基于已有 checkpoint 做 finetune

---

### 7.3 direct 双卡 LoRA（基于本地 full checkpoint） smoke

本次最终验证成功的关键路线。

使用的 pretrained 路径：

- `/mnt/disk_2/wjz/runs/pi05-full-smoke-direct/checkpoints/000010/pretrained_model`

结果：

- 成功完成 10/10 steps
- 成功 checkpoint
- 正常退出

关键日志：

- `Using PEFT! Wrapping model.`
- `Wrapped pi05 with PEFT (LoraConfig)`
- `Checkpoint policy after step 10`
- `End of training`

关键指标：

- `dataset.num_frames=897`
- `dataset.num_episodes=1`
- `Effective batch size: 1 x 2 = 2`
- `num_learnable_params=1287168`
- `num_total_params=3618044688`

输出目录：

- `/mnt/disk_2/wjz/runs/pi05-lora-from-local-smoke`

与 full / 半冻结相比：

- full / 半冻结可训练参数约 `6.93e8`
- LoRA 可训练参数约 `1.29e6`

说明 LoRA 路线已经实实在在地大幅降低了训练参数规模。

---

## 8. 官方 pi05 pretrained checkpoint 的确认情况

### 8.1 当前已确认的“官方来源”

从 OpenPI README 可确认，官方 checkpoint 来源是 **OpenPI 自己的 GCS bucket**，不是 README 里明确给出的 Hugging Face 模型 repo。

已确认的官方 checkpoint 路径包括：

**base checkpoints**
- `gs://openpi-assets/checkpoints/pi05_base`

**fine-tuned checkpoints**
- `gs://openpi-assets/checkpoints/pi05_droid`
- `gs://openpi-assets/checkpoints/pi05_libero`

### 8.2 下载 / 拉取方式

OpenPI README 的表述是：

- 这些 checkpoint 会在需要时从 `gs://openpi-assets` 自动下载
- 缓存在：
  - `~/.cache/openpi`
- 可通过环境变量改缓存位置：
  - `OPENPI_DATA_HOME`

README 中给出的代码使用方式类似：

```python
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")
```

然后再把下载到本地的目录传给 policy 构造 / 加载逻辑。

### 8.5 官方 `pi05_droid` 路线最终打通结果

本轮最终已经把下面这条路线真实打通：

- **official OpenPI `pi05_droid` checkpoint**
- 下载到本地 cache
- 转换为 PyTorch `model.safetensors`
- 改写成 LeRobot-compatible `config.json`
- 补齐 processor 配置
- 将 tokenizer 指向本地 HF snapshot
- 最终成功完成：
  - **direct 双卡 LoRA 10-step smoke**

最终可直接用作 `policy.pretrained_path` 的路径：

- `/mnt/disk_2/wjz/openpi/pi05_droid_pytorch`

官方 checkpoint 对应的双卡 LoRA smoke 成功输出目录：

- `/mnt/disk_2/wjz/runs/pi05-lora-openpi-droid-smoke-v3`

训练完成后产生的 LoRA adapter checkpoint 位于：

- `/mnt/disk_2/wjz/runs/pi05-lora-openpi-droid-smoke-v3/checkpoints/000010/pretrained_model`

其中包含：

- `adapter_config.json`
- `adapter_model.safetensors`
- `config.json`
- `policy_preprocessor.json`
- `policy_postprocessor.json`
- `train_config.json`

### 8.6 当前推荐的 official checkpoint preparation / training 命令

#### 8.6.1 准备 official checkpoint

当前脚本默认已经对齐到本机会话中最终跑通的路径：

```bash
conda run -n rlft_ms3_lerobot python -m rlft.offline.prepare_openpi_pi05_checkpoint
```

等价于显式写法：

```bash
conda run -n rlft_ms3_lerobot python -m rlft.offline.prepare_openpi_pi05_checkpoint \
  --checkpoint-name pi05_droid \
  --cache-dir /mnt/disk_2/wjz/.cache/openpi \
  --output-dir /mnt/disk_2/wjz/openpi/pi05_droid_pytorch
```

#### 8.6.2 使用 official checkpoint 跑双卡 LoRA smoke

```bash
HF_TOKEN=<your_token> HUGGINGFACE_HUB_TOKEN=<your_token> \
PYTORCH_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0,1 \
conda run -n rlft_ms3_lerobot accelerate launch \
  --main_process_port 29664 \
  --num_processes 2 \
  /home/wjz/miniconda3/envs/rlft_ms3_lerobot/bin/lerobot-train \
  --policy.type=pi05 \
  --dataset.repo_id=carm/pi05_local \
  --dataset.root=/mnt/disk_2/wjz/runs/pi05_train_smoke_export_mgpu4 \
  --policy.repo_id=zhili0818/pi05-smoke-lora-openpi-droid-v3 \
  --policy.pretrained_path=/mnt/disk_2/wjz/openpi/pi05_droid_pytorch \
  --policy.push_to_hub=false \
  --job_name=pi05-lora-openpi-droid-smoke-v3 \
  --output_dir=/mnt/disk_2/wjz/runs/pi05-lora-openpi-droid-smoke-v3 \
  --seed=1 \
  --batch_size=1 \
  --steps=10 \
  --optimizer.lr=0.0001 \
  --policy.gradient_checkpointing=true \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.dtype=bfloat16 \
  --peft.method_type=LORA \
  --peft.r=16
```

#### 8.6.3 当前工程入口中的便捷开关

当前代码已经额外支持一个便捷模式：

- 在 `train_pi05.py` 中设置：
  - `use_official_openpi_checkpoint=True`
- 如果此时没有显式传 `policy_pretrained_path`
- 命令生成默认会落到：
  - `/mnt/disk_2/wjz/openpi/pi05_droid_pytorch`

这使得在当前机器上复用 official checkpoint 时，不需要每次手写完整路径。


新增入口：

- [rlft/offline/prepare_openpi_pi05_checkpoint.py](rlft/offline/prepare_openpi_pi05_checkpoint.py)
- 底层实现： [rlft/offline/pi05_bridge/openpi_checkpoint.py](rlft/offline/pi05_bridge/openpi_checkpoint.py)

它的工作流程是：

1. 通过 OpenPI 自己的 `download.maybe_download(...)` 下载 / 缓存官方 checkpoint
2. 定位到本地缓存目录（默认在 `~/.cache/openpi/openpi-assets/checkpoints/...`）
3. 如果是 JAX Orbax checkpoint，则调用 OpenPI 自带的转换脚本转成 PyTorch safetensors 目录
4. 对产物做最小 LeRobot 兼容性验证
5. 输出 `lerobot_pretrained_path`，可直接传给：
   - `--policy.pretrained_path=...`

当前支持的 checkpoint 名称：

- `pi05_base`
- `pi05_droid`
- `pi05_libero`

示例命令：

```bash
conda run -n rlft_ms3_lerobot python -m rlft.offline.prepare_openpi_pi05_checkpoint \
  --checkpoint-name pi05_droid
```

如果需要自定义缓存目录或输出目录：

```bash
conda run -n rlft_ms3_lerobot python -m rlft.offline.prepare_openpi_pi05_checkpoint \
  --checkpoint-name pi05_base \
  --cache-dir /mnt/disk_2/wjz/.cache/openpi \
  --output-dir /mnt/disk_2/wjz/openpi/pi05_base_pytorch
```

成功后，输出 JSON 中的：

- `lerobot_pretrained_path`

即可直接用于：

```bash
--policy.pretrained_path=/path/to/converted/openpi/pi05_checkpoint
```


截至本次会话结束，项目里已经具备：

1. **pi05 环境结构化探测**
2. **bridge dataset 严格校验**
3. **CARM → LeRobot 原生本地数据集导出**
4. **导出后 LeRobotDataset 回读校验**
5. **LeRobot 训练命令结构化生成**
6. **strict 模式 gate**
7. **分布式场景下 rank0-only export / validation**
8. **双卡 direct full / 半冻结训练 smoke**
9. **双卡 direct LoRA 训练 smoke（基于本地 checkpoint）**
10. **PEFT 缺少 pretrained_path 时的本地早报错**

---

## 10. 当前推荐的后续动作

### 优先级 A

1. 尝试官方 OpenPI checkpoint：
   - `gs://openpi-assets/checkpoints/pi05_base`
   - `gs://openpi-assets/checkpoints/pi05_droid`
2. 验证这些 checkpoint 是否能直接作为：
   - `policy_pretrained_path`
3. 对比：
   - 官方 checkpoint → LoRA
   - 本地 warmup checkpoint → LoRA

### 优先级 B

1. 检查本地 warmup checkpoint 作为 LoRA 初始化时的 key remap warning
2. 确认是否只影响非关键权重映射
3. 若需要，补一个更严格的 checkpoint compatibility probe

### 优先级 C

1. 将当前 direct 双卡 full / LoRA 命令固化成脚本
2. 减少手工命令拼接
3. 让 `train_pi05.py` 只承担 prepare / validate / export / emit command 的职责

---

## 11. 一句话总结

本轮工作的核心成果不是“已经训练好最终 pi05 模型”，而是：

**已经把 CARM 数据导出、LeRobot 原生数据集回读、pi05 双卡 full / 半冻结训练、以及基于本地 checkpoint 的双卡 LoRA 训练这整条工程链路，在当前机器上真实打通了。**
