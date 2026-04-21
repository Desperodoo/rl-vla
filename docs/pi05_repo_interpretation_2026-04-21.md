# 当前仓库中 pi05 相关数据准备、训练与测试链路解读

> 日期：2026-04-21  
> 仓库：`/home/amax/rl-vla`  
> 目标：基于当前仓库里的源码和已有 `docs/` 记录，梳理 `pi0.5 / pi05` 在本项目中的数据准备、训练、测试/评估链路，并给出工程层面的理解。

## 1. 先给结论

当前仓库里的 `pi05` 不是一套“从数据到训练到部署全部内聚在仓库内部”的独立框架，而是一条以 **CARM 数据桥接到 LeRobot / OpenPI 生态** 为核心的工程链路。

更准确地说，它由四层组成：

1. **原始数据层**：CARM HDF5 episode 数据，由 `rlft.datasets.data_utils` 读取。
2. **bridge 层**：`rlft/offline/pi05_bridge/*`，负责把 CARM 数据解释成 `pi05` 可接受的观测/动作约定，并导出为 LeRobot 原生本地数据集。
3. **训练调度层**：`rlft/offline/train_pi05.py` 和几个 launcher 脚本，负责做探测、校验、导出、拼训练命令；真正训练由 `lerobot-train` 执行。
4. **测试/评估层**：一部分是离线误差评估 `rlft/offline/eval_pi05.py`，另一部分是 `ctrl_world` 里的 rollout / interact 脚本，它们使用 OpenPI policy 做世界模型中的推理和交互。

所以，如果一句话概括当前仓库里的 `pi05`：

**它本质上是“CARM -> LeRobot/OpenPI”的桥接与工程集成方案，训练本体并不在本仓库里重写，而是借用上游 `lerobot-train` / OpenPI policy。**

---

## 2. 仓库里和 pi05 直接相关的关键文件

### 2.1 bridge 与训练入口

| 文件 | 作用 |
| --- | --- |
| [../rlft/offline/train_pi05.py](../rlft/offline/train_pi05.py) | `pi05` 主入口；负责 probe、校验、可选导出、命令生成、可选 dispatch |
| [../rlft/offline/pi05_bridge/contract.py](../rlft/offline/pi05_bridge/contract.py) | 定义 `pi05` bridge 的观测/动作契约 |
| [../rlft/offline/pi05_bridge/dataset_bridge.py](../rlft/offline/pi05_bridge/dataset_bridge.py) | 把 CARM episode 转成 bridge dataset，主要用于 smoke / validation |
| [../rlft/offline/pi05_bridge/export.py](../rlft/offline/pi05_bridge/export.py) | 把 CARM 数据导出成 LeRobot 原生本地数据集 |
| [../rlft/offline/pi05_bridge/validate.py](../rlft/offline/pi05_bridge/validate.py) | 校验 bridge dataset、LeRobot dataset、训练命令 |
| [../rlft/offline/pi05_bridge/config_bridge.py](../rlft/offline/pi05_bridge/config_bridge.py) | 生成稳定 run config 和 `lerobot-train` 命令 |
| [../rlft/offline/pi05_bridge/env_probe.py](../rlft/offline/pi05_bridge/env_probe.py) | 检测 `lerobot`、CUDA、CLI、Python 运行环境 |
| [../rlft/offline/pi05_bridge/openpi_checkpoint.py](../rlft/offline/pi05_bridge/openpi_checkpoint.py) | 下载并转换 OpenPI 官方 `pi05` checkpoint 成 LeRobot 可读目录 |

### 2.2 数据准备与实验辅助

| 文件 | 作用 |
| --- | --- |
| [../rlft/offline/export_carm_to_lerobot.py](../rlft/offline/export_carm_to_lerobot.py) | 独立 CLI：导出 CARM -> LeRobot，并立即回读校验 |
| [../rlft/offline/prepare_pi05_splits.py](../rlft/offline/prepare_pi05_splits.py) | 将原始 episode 划分成 train / val / test |
| [../rlft/offline/prepare_openpi_pi05_checkpoint.py](../rlft/offline/prepare_openpi_pi05_checkpoint.py) | CLI 包装：转换 OpenPI 官方 checkpoint |
| [../rlft/offline/launch_pi05_full_train.py](../rlft/offline/launch_pi05_full_train.py) | 多卡训练 launcher，附带日志与显存监控 |
| [../rlft/offline/probe_pi05_batch_scaling.py](../rlft/offline/probe_pi05_batch_scaling.py) | 扫 batch size 的显存 smoke 脚本 |
| [../rlft/offline/eval_pi05.py](../rlft/offline/eval_pi05.py) | 离线评估脚本，计算 action MSE / MAE |

### 2.3 Ctrl-World 里的 pi05 运行入口

| 文件 | 作用 |
| --- | --- |
| [../ctrl_world/config.py](../ctrl_world/config.py) | Ctrl-World 训练/rollout 配置，默认 `policy_type='pi05'` |
| [../ctrl_world/config_eval.py](../ctrl_world/config_eval.py) | Ctrl-World eval 配置，默认 `policy_type='pi05'` |
| [../ctrl_world/scripts/rollout_interact_pi.py](../ctrl_world/scripts/rollout_interact_pi.py) | 使用 OpenPI policy 与世界模型交互 rollout |
| [../ctrl_world/scripts/rollout_interact_pi_eval.py](../ctrl_world/scripts/rollout_interact_pi_eval.py) | rollout 的 eval 版本 |

### 2.4 已有历史文档

| 文档 | 作用 |
| --- | --- |
| [pi05_finetuning_session_2026-03-29.md](./pi05_finetuning_session_2026-03-29.md) | 记录 bridge 打通、OpenPI checkpoint 转换、smoke 训练过程 |
| [pi05_full_finetune_report_2026-03-30.md](./pi05_full_finetune_report_2026-03-30.md) | 记录一次全量数据 LoRA 微调与离线评估结果 |

这两份文档不是“当前状态说明书”，但对理解为什么现在的代码结构会长成这样很有帮助。

---

## 3. 数据准备链路怎么走

## 3.1 原始数据格式是什么

底层读取在 [../rlft/datasets/data_utils.py](../rlft/datasets/data_utils.py)。

仓库里的 CARM 数据默认组织成一个目录，下面是若干 `episode_*.hdf5` 文件。单个 episode 至少要求这些键：

- `observations/images`
- `observations/qpos_joint`
- `observations/qpos_end`
- `observations/gripper`
- `observations/timestamps`

如果文件里还有：

- `action`
- `teleop_scale`

也会一起读入。

从代码看，加载器同时兼容两种 action 版本：

- `v1`：旧格式，动作维度更大
- `v2`：当前更关键的格式，动作是 **8 维**，即目标位姿 `7D` + gripper `1D`

`pi05 bridge` 明显更偏向 `v2` 语义。

## 3.2 观测是怎么构造的

`create_carm_obs_process_fn(...)` 定义了 bridge 里看到的观测构造逻辑：

- 图像来自 `observations/images`
- 可以 resize，默认目标尺寸跟视觉编码器类型相关
- 可输出 `NCHW` 或 `NHWC`
- `state_mode` 支持三种：
  - `joint_only`：只用 joint state
  - `ee_only`：只用末端位姿
  - `both`：joint + ee pose

另外还会单独保留：

- `ee_pose = qpos_end[:, :7]`

这意味着 `pi05` bridge 的观测不是单一 state，而是明确拆成：

- `observation.image`
- `observation.state`
- `observation.ee_pose`

## 3.3 pi05 bridge 的动作语义是什么

这里是整个仓库里最值得注意的一点。

在 [../rlft/offline/pi05_bridge/contract.py](../rlft/offline/pi05_bridge/contract.py) 里，动作契约被写死成：

- `target_dim = 8`
- `representation = absolute_pose_gripper`

也就是：

- 绝对目标位姿 `7D`
- gripper 标量 `1D`

而仓库主线的 CARM imitation 数据集类 [../rlft/datasets/carm_dataset.py](../rlft/datasets/carm_dataset.py) 则仍然保留了另一套语义：

- 训练时通常把 raw target pose 变成**相对位姿动作**
- 并把 gripper 单独离散化成 label

所以，当前仓库里有两条动作语义并存：

1. `train_carm.py` 一侧：偏向相对动作
2. `pi05 bridge` 一侧：明确保留 CARM v2 的 **8D 绝对动作**

这不是 bug，而是工程选择。因为 `pi05` 这一条线的目标不是复用 `train_carm.py` 训练范式，而是把当前 CARM 记录数据接到 LeRobot / OpenPI 的 `pi05` policy 上。

## 3.4 bridge dataset 本身做什么

[../rlft/offline/pi05_bridge/dataset_bridge.py](../rlft/offline/pi05_bridge/dataset_bridge.py) 构造的是一个 **轻量 bridge dataset**，主要用于：

- smoke 测试
- 样本结构校验
- action normalizer 拟合

它的特点：

- 先把整个 CARM 数据加载进内存
- 每个 episode 处理成：
  - `rgb`
  - `state`
  - `ee_pose`
  - `action`
- 再按照 `action_horizon` 和 `window_stride` 切 window
- 如果窗口不足 horizon，会用最后一帧重复 pad

返回的 sample key 是：

- `observation.image`
- `observation.state`
- `observation.ee_pose`
- `action`
- `action_unnormalized`
- `episode_index`
- `start_index`

需要注意：

- 这个 dataset 主要服务于 bridge 自检，不是 LeRobot 真正训练时直接读取的数据格式。
- 真正给 `lerobot-train` 的训练数据，是导出的 LeRobot 原生本地数据集。

## 3.5 CARM -> LeRobot 的导出逻辑

导出核心在 [../rlft/offline/pi05_bridge/export.py](../rlft/offline/pi05_bridge/export.py)。

它会：

1. 读取原始 CARM episode
2. 用 `create_carm_obs_process_fn(..., output_format="NHWC")` 生成可存储图像和状态
3. 调用 `LeRobotDataset.create(...)`
4. 对每一帧调用 `dataset.add_frame(...)`
5. 每个 episode 结束后调用 `dataset.save_episode()`

导出的特征固定包含：

- `observation.image`
- `observation.state`
- `observation.ee_pose`
- `action`

导出目录会生成 LeRobot 原生 layout：

- `meta/info.json`
- `meta/tasks.parquet`
- `meta/episodes/...`
- `data/.../*.parquet`

同时额外写一份：

- `pi05_bridge_metadata.json`

这个 metadata 会记录：

- 原始 `demo_path`
- episode 数
- `get_carm_data_info(...)` 的结果
- bridge contract
- repo id

### 这里有几个很具体的实现特征

1. `repo_id` 被硬编码成 `carm/pi05_local`
2. `use_videos=False`，即用 parquet + image data 的本地数据集方式，不走视频封装
3. 每帧的 `task` 被硬编码成 `carm_fixed_dual_light`

第 3 点尤其值得注意，因为当前仓库实际上已经出现多子集数据的使用痕迹。也就是说，**导出脚本目前并没有根据 source subset 动态生成 task 名称**，而是统一塞成一个固定 task string。

这不一定会立刻导致训练错误，但会让 task 语义比真实数据来源更粗糙。

## 3.6 train / val / test 划分怎么做

[../rlft/offline/prepare_pi05_splits.py](../rlft/offline/prepare_pi05_splits.py) 是 `pi05` 这条线专门的数据划分脚本。

它的逻辑不是简单随机打散，而是：

1. 扫描 `source_root` 下每个子目录里的 `episode_*.hdf5`
2. 读取每个文件的：
   - `source_subset`
   - `data_version`
3. 以 `(source_subset, data_version)` 作为 bucket
4. 在 bucket 内按 seed 打乱，再切 train / val / test
5. 输出：
   - `train_manifest.json`
   - `val_manifest.json`
   - `test_manifest.json`
   - `split_summary.json`

物理落盘时支持两种模式：

- `symlink`
- `copy`

这意味着当前仓库对 `pi05` 的数据准备已经不只是“把一个目录喂进去”，而是明确支持：

- 先做可复现划分
- 再分别导出 train / val / test 的 LeRobot dataset

---

## 4. 训练链路怎么走

## 4.1 `train_pi05.py` 的真实角色

[../rlft/offline/train_pi05.py](../rlft/offline/train_pi05.py) 名字叫 `train`，但它更像：

**训练前桥接准备器 + 训练命令调度器**

它自己做的事情包括：

1. 解析参数并设置默认实验名
2. 校验 `use_peft` 与 `policy_pretrained_path` 的约束
3. 固化随机种子
4. 生成 `run_config`
5. 写出运行元数据到 `runs/<run_name>/checkpoints`
6. 做 `lerobot` 环境探测
7. 做 bridge dataset 校验
8. 建一个 DataLoader，拿第一批样本做 smoke
9. 可选自动导出 LeRobot dataset
10. 可选校验导出的 LeRobot dataset
11. 生成 `lerobot-train` 命令
12. 可选真正 dispatch 到 `lerobot-train`

也就是说，`train_pi05.py` 有明显的“wrapper”属性。

## 4.2 它会产出哪些中间 artifact

`train_pi05.py` 会往 `runs/<run_name>/checkpoints` 写很多 JSON，这一点对调试非常有价值：

- `args.json`
- `pi05_bridge_config.json`
- `env_probe.json`
- `bridge_validation.json`
- `action_normalizer.json`
- `lerobot_export.json`
- `lerobot_dataset_validation.json`
- `lerobot_train_command.json`
- `lerobot_train_command_validation.json`

所以它不是黑盒式训练入口，而是偏向“可审计的工程前处理器”。

## 4.3 bridge smoke / validate 两个快捷模式

这个入口还支持两个很实用的早停模式：

- `bridge_smoke_only`
- `bridge_validate_only`

前者的重点是：

- 能不能把 CARM 数据读起来
- 能不能做 window 化
- 第一批样本 shape 对不对

后者则更进一步：

- 校验 bridge dataset
- 校验导出后的 LeRobot dataset
- 校验训练命令的基本合法性

这说明当前仓库对 `pi05` 的“测试”思路，不是先上大训练，而是先把桥接链路拆成多个结构化 gate。

## 4.4 真正的训练命令是谁拼的

命令构造逻辑在 [../rlft/offline/pi05_bridge/config_bridge.py](../rlft/offline/pi05_bridge/config_bridge.py)。

它会把外部参数统一翻译成：

- `--policy.type=pi05`
- `--dataset.repo_id=...`
- `--dataset.root=...`
- `--policy.repo_id=...`
- `--policy.pretrained_path=...`
- `--job_name=...`
- `--output_dir=...`
- `--seed=...`
- `--batch_size=...`
- `--steps=...`
- `--optimizer.lr=...`

还会默认加几项明显是为了控显存和降低训练难度的设置：

- `--policy.gradient_checkpointing=true`
- `--policy.freeze_vision_encoder=true`
- `--policy.train_expert_only=true`
- `--policy.dtype=bfloat16`

如果启用 PEFT，还会补：

- `--peft.method_type`
- `--peft.r`
- `--peft.target_modules`
- `--peft.full_training_modules`

这里也把 `pi0.5 -> pi05` 做了映射，说明当前仓库里虽然文案上会说 `pi0.5`，但落实到 LeRobot CLI 时统一用 `pi05`。

## 4.5 真正执行训练的是谁

不是 `train_pi05.py` 自己。

真实训练仍然是：

- `lerobot-train`

`train_pi05.py` 只有在：

- `dispatch_to_lerobot=True`

时才会 `subprocess.run(command, ...)` 去调用它。

从已有历史文档 [pi05_finetuning_session_2026-03-29.md](./pi05_finetuning_session_2026-03-29.md) 的结论看，当前更推荐的路线实际上是：

- `train_pi05.py` 用来做 prepare / validate / export / emit command
- 真正的分布式训练直接走 `accelerate launch ... lerobot-train ...`

这个建议和当前代码结构是吻合的。

## 4.6 OpenPI 官方 checkpoint 是怎么接进来的

这一层由：

- [../rlft/offline/prepare_openpi_pi05_checkpoint.py](../rlft/offline/prepare_openpi_pi05_checkpoint.py)
- [../rlft/offline/pi05_bridge/openpi_checkpoint.py](../rlft/offline/pi05_bridge/openpi_checkpoint.py)

共同完成。

它的流程是：

1. 假设本地已经有 OpenPI 仓库克隆在 `/tmp/openpi_for_patch`
2. 通过 OpenPI 的 `download.maybe_download(...)` 拉取
   - `gs://openpi-assets/checkpoints/pi05_base`
   - `gs://openpi-assets/checkpoints/pi05_droid`
   - `gs://openpi-assets/checkpoints/pi05_libero`
3. 调用 OpenPI 仓库里的 `examples/convert_jax_model_to_pytorch.py`
4. 再写一份 LeRobot 侧可读的 `config.json`
5. 用 LeRobot 的 policy factory 做一次 load 校验

最后会得到一个本地目录，例如：

- `/mnt/disk_2/wjz/openpi/pi05_droid_pytorch`

然后把这个路径作为：

- `policy.pretrained_path`

喂给 `lerobot-train`。

### 这里的关键理解

当前仓库并没有把 OpenPI 官方 checkpoint 原样直接拿来给 LeRobot 用，而是专门做了一次**桥接转换**。

也就是说：

- OpenPI checkpoint 来源是上游
- LeRobot 训练消费的是本地转换产物

## 4.7 专门的 launcher 与显存探测脚本

围绕正式训练，仓库里还有两个工程脚本：

### [../rlft/offline/launch_pi05_full_train.py](../rlft/offline/launch_pi05_full_train.py)

这个脚本本质上是：

- `accelerate launch lerobot-train ...`

的封装器，并额外做：

- `train.log`
- `resource_monitor.jsonl`
- `launch_command.sh`
- `launch_config.json`

它会定期抓 `nvidia-smi`，把显存/利用率记下来，适合跑长训练。

### [../rlft/offline/probe_pi05_batch_scaling.py](../rlft/offline/probe_pi05_batch_scaling.py)

这个脚本更偏实验诊断：

- 用多个 batch size 反复做 warmup smoke
- 定时抓 GPU 显存
- 超过 warmup 时间后主动结束
- 汇总最大显存占用

它不追求收敛，只是为了回答“这个 batch size 站不站得住”。

## 4.8 当前训练链路里的几个隐含约束

### 约束 1：PEFT 不能从零开始

`train_pi05.py` 和 `config_bridge.py` 都明确限制：

- `use_peft=True` 时必须提供 `policy_pretrained_path`

也就是说，仓库当前认可的 LoRA 路线是：

- 基于已有 `pi05` pretrained checkpoint 做适配

而不是：

- 从随机初始化直接 LoRA

### 约束 2：很多路径是硬编码的

当前代码里明显存在环境相关硬编码，例如：

- `/mnt/disk_2/wjz/openpi/pi05_droid_pytorch`
- `/home/wjz/miniconda3/envs/rlft_ms3_lerobot/bin/accelerate`
- `/home/wjz/miniconda3/envs/rlft_ms3_lerobot/bin/lerobot-train`
- `/tmp/openpi_for_patch`

这说明：

- 代码链路是通的
- 但复用到另一台机器前，需要先去硬编码或补配置层

### 约束 3：bridge 的 horizon 与 OpenPI config 并不是完全同一层

bridge contract 默认：

- `action_horizon = 16`

而 OpenPI checkpoint 转换后写入的 LeRobot config 里：

- `chunk_size = 15`
- `n_action_steps = 15`

这两者并不直接冲突，因为：

- bridge horizon 主要用于本地 smoke/validation 的窗口化
- LeRobot 真正训练时读的是导出的 dataset + policy config

但这确实意味着：

- **bridge 的“序列窗口设置”和最终 `pi05` policy 的训练 chunk 设置不是同一份配置源。**

后续如果要继续统一工程语义，这会是一个值得收口的地方。

---

## 5. 测试 / 评估链路怎么理解

当前仓库里的 `pi05` “测试”大致分三类。

## 5.1 第一类：结构完整性测试

这类测试的目标不是看模型效果，而是看链路是否可运行。

### 环境探测

[../rlft/offline/pi05_bridge/env_probe.py](../rlft/offline/pi05_bridge/env_probe.py) 会检查：

- Python runtime
- `torch / numpy / h5py / tyro / lerobot` 能否 import
- CUDA 是否可用
- GPU 数量
- `lerobot-train` 是否在 PATH
- `lerobot-train --help` 是否可运行

### bridge dataset 校验

[../rlft/offline/pi05_bridge/validate.py](../rlft/offline/pi05_bridge/validate.py) 会检查：

- dataset 是否非空
- sample keys 是否齐
- image/state/ee_pose/action 的 rank 和 shape 是否符合约定

### LeRobot dataset 校验

同一个文件还会检查导出目录是否具备：

- `meta/info.json`
- `meta/tasks.parquet`
- `meta/episodes/...`
- `data/.../*.parquet`

并尝试真实构造：

- `LeRobotDataset(repo_id=..., root=...)`

### 训练命令校验

还会检查：

- `lerobot_dataset_repo_id` 和 `lerobot_dataset_path` 是否只设置了一个
- CLI 是否可用
- `use_peft=True` 时是否给了 pretrained path
- 关键 flag 是否都出现在命令里

这部分其实就是当前仓库里的“工程测试框架”。

## 5.2 第二类：离线误差评估

[../rlft/offline/eval_pi05.py](../rlft/offline/eval_pi05.py) 是比较标准的离线 eval 脚本。

它会：

1. 加载 LeRobot dataset
2. 从 `policy_pretrained_path` 或 `peft_adapter_path` 加载 policy
3. 构造 preprocessor / postprocessor
4. 对 dataset 的每个 frame 跑 action 预测
5. 计算：
   - overall `mean_action_mse`
   - overall `mean_action_mae`
   - `per_dim_mse`
   - `per_dim_mae`
   - `per_episode_mean_mae`

这是一种**监督式离线指标**，它衡量的是：

- 模型对离线数据中 action 的拟合程度

它不等价于：

- 真机成功率
- rollout 任务完成率

所以当前仓库里的 `eval_pi05.py` 更接近“behavior cloning / imitation quality”评估，而不是最终 deployment 指标。

### 这份脚本也有环境依赖

默认参数里可以看到一些比较强的本机依赖：

- `tokenizer_path_override` 指向本地 HuggingFace snapshot
- `device='cuda'`

所以它也是“已打通的工程脚本”，但不是完全去环境耦合的通用工具。

## 5.3 第三类：Ctrl-World 中的 rollout / interact 测试

这一层在：

- [../ctrl_world/scripts/rollout_interact_pi.py](../ctrl_world/scripts/rollout_interact_pi.py)
- [../ctrl_world/scripts/rollout_interact_pi_eval.py](../ctrl_world/scripts/rollout_interact_pi_eval.py)
- [../ctrl_world/config.py](../ctrl_world/config.py)
- [../ctrl_world/config_eval.py](../ctrl_world/config_eval.py)

这里的 `pi05` 和离线微调链路有一个重要差别：

- 它不是走 LeRobot 训练路径
- 而是直接通过 OpenPI policy config 创建 policy

代码里写得很直接：

- 如果 `policy_type` 包含 `pi05`
- 就取 `config_pi.get_config("pi05_droid")`
- 然后 `policy_config.create_trained_policy(config, args.pi_ckpt)`

默认 checkpoint 甚至还是：

- `/cephfs/shared/llm/openpi/openpi-assets-preview/checkpoints/pi05_droid`

也就是说，`ctrl_world` 这套运行时更像：

- “拿 OpenPI `pi05_droid` policy 去和世界模型耦合推理”

而不是：

- “直接消费 `train_pi05.py` 那条 LeRobot 微调产物”

### 为什么还要加 action adapter

`rollout_interact_pi.py` 里有一段很关键的注释：

- 官方 Pi-Droid 输出 joint velocity
- Ctrl-World 在 cartesian space 上训练
- 因此需要一个轻量 adapter 把 policy action 变成 cartesian pose action

这说明 `ctrl_world` 的 `pi05` 测试更偏向：

- 世界模型交互验证
- 动作空间适配后的 rollout 验证

它和 `eval_pi05.py` 的离线 MAE/MSE，不是同一种测试范式。

## 5.4 当前仓库里没有专门的 pi05 pytest

仓库整体是有 pytest 体系的，例如 `rlft/tests/*` 下有 AWSC、DSRL、VLAW 等测试。

但我检索下来，当前并没有专门面向 `pi05` 的：

- `test_pi05.py`
- `rlft/tests/pi05/*`

所以，当前 `pi05` 这条线的验证方式主要还是：

1. 结构化 probe / validate
2. smoke 训练
3. 离线 eval
4. rollout 脚本
5. 历史实验报告

而不是单元测试优先的风格。

---

## 6. 如何理解当前仓库里 pi05 的“成熟度”

我会把当前状态分成三档。

## 6.1 已经比较成熟的部分

### CARM -> LeRobot dataset 导出与回读

这一块代码相对完整，且职责清晰：

- CARM HDF5 读取逻辑明确
- 导出脚本独立
- 校验逻辑独立
- train/val/test split 脚本已经补上

### 训练前的工程 gate

`train_pi05.py` + `env_probe.py` + `validate.py` 这一组已经形成了较稳定的训练前检查流。

这部分的优点是：

- 失败点前移
- artifact 留痕充分
- 分布式导出时考虑了 rank-aware 行为

## 6.2 已经打通，但仍偏工程脚本的部分

### 正式多卡训练

多卡训练是打通过的，但强依赖外部环境：

- `lerobot-train`
- `accelerate`
- OpenPI 转换后的 pretrained
- 本机 conda 环境路径

所以它是：

- “当前机器上能跑”

而不是：

- “任何环境即插即用”

### 离线评估

`eval_pi05.py` 可用，但它衡量的是 action reconstruction 误差，不是最终任务成功率。

它更适合用作：

- 模型比较
- adapter 对比
- smoke 验证

不适合单独充当全部效果结论。

## 6.3 仍然值得继续收口的部分

### 动作语义在仓库里并不统一

`pi05 bridge` 与 `train_carm.py` 系列的动作表示不一样，这在当前阶段是合理的，但长期会增加理解成本。

### 配置分散

现在至少存在三套需要同时记住的配置来源：

1. bridge contract
2. LeRobot / training command flags
3. OpenPI-converted `config.json`

它们没有完全统一成单一配置真源。

### 路径与 task 名称硬编码较多

这使得“本机可复现”和“团队可移植”之间还差一层工程整理。

---

## 7. 我对当前仓库中 pi05 的整体理解

如果把现在仓库里的 `pi05` 用一句更工程化的话来描述，我会这样说：

**这个仓库已经把 `pi05` 做成了一条可操作的集成链路，而不是一个完全自研的策略训练系统。**

其中：

- 数据侧：负责把 CARM episode 变成 LeRobot 可以吃的数据
- 训练侧：负责把上游 `lerobot-train` 和 OpenPI checkpoint 接进来
- 测试侧：既有离线 action 误差评估，也有 Ctrl-World 里的 rollout 运行入口

这条线现在最强的地方是：

- 工程链路已经通
- 调试 artifact 比较全
- 能支持 split、导出、LoRA、batch 探测、离线评估

它现在最不强的地方是：

- 配置和路径依赖还比较本机化
- `pi05` 与仓库主线 imitation 语义没有完全统一
- 缺少专门的自动化测试套件

---

## 8. 建议的阅读顺序

如果后续要继续接手这条线，我建议按下面顺序读：

1. [../rlft/offline/train_pi05.py](../rlft/offline/train_pi05.py)
2. [../rlft/offline/pi05_bridge/contract.py](../rlft/offline/pi05_bridge/contract.py)
3. [../rlft/offline/pi05_bridge/dataset_bridge.py](../rlft/offline/pi05_bridge/dataset_bridge.py)
4. [../rlft/offline/pi05_bridge/export.py](../rlft/offline/pi05_bridge/export.py)
5. [../rlft/offline/pi05_bridge/validate.py](../rlft/offline/pi05_bridge/validate.py)
6. [../rlft/offline/pi05_bridge/config_bridge.py](../rlft/offline/pi05_bridge/config_bridge.py)
7. [../rlft/offline/prepare_openpi_pi05_checkpoint.py](../rlft/offline/prepare_openpi_pi05_checkpoint.py)
8. [../rlft/offline/eval_pi05.py](../rlft/offline/eval_pi05.py)
9. [../ctrl_world/scripts/rollout_interact_pi.py](../ctrl_world/scripts/rollout_interact_pi.py)
10. [pi05_finetuning_session_2026-03-29.md](./pi05_finetuning_session_2026-03-29.md)
11. [pi05_full_finetune_report_2026-03-30.md](./pi05_full_finetune_report_2026-03-30.md)

这样会先把“现在代码怎么跑”看清楚，再回头理解“为什么会这样设计”。

---

## 9. 最后一句总结

当前仓库里的 `pi05` 已经不是“概念验证”了，而是一个**以 bridge 为核心、以 LeRobot/OpenPI 为训练后端、以离线评估和 Ctrl-World rollout 为验证手段**的可运行工程分支。

如果下一步要继续推进，我认为最值得做的不是再重复打通一次链路，而是：

- 统一配置来源
- 去掉环境硬编码
- 补充 `pi05` 专项自动化测试
- 明确离线 MAE/MSE 与 rollout 成功率之间的评价边界
