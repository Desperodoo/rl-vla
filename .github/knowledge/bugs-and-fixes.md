# Bug 数据库

> 记录已发现并修复的 Bug，供后续 Agent 参考，避免重蹈覆辙。

---

## BUG-001: validate_video_generation 硬编码 DROID latent shape

- **发现**: 2026-02-25 代码审查
- **文件**: `ctrl_world/scripts/train_wm.py` (validate_video_generation 函数)
- **症状**: 训练启动 validation_steps 时崩溃，`AssertionError: (4, 48, 24) != (4, 72, 40)`
- **根因**: DROID 3-cam latent 是 72×40，ManiSkill 2-cam latent 是 48×24，代码未适配
- **修复**: `(4, 72, 40)` → `(4, args.height//8, args.width//8)`（动态计算）
- **预防**: 所有涉及 latent shape 的地方都应从 args 动态推导，不要硬编码

---

## BUG-002: validate_video_generation 中 height 乘以错误倍数

- **发现**: 2026-02-25 代码审查
- **文件**: `ctrl_world/scripts/train_wm.py`
- **症状**: pipeline 调用 height 参数错误，生成视频尺寸不匹配
- **根因**: DROID args.height=192（单相机），需 ×3。ManiSkill args.height=384（已含拼接），不能再×3
- **修复**: `int(3*args.height)` → `args.height if task_type=='maniskill' else int(3*args.height)`

---

## BUG-003: validate_video_generation 中 rearrange m=3 硬编码

- **发现**: 2026-02-25 代码审查
- **文件**: `ctrl_world/scripts/train_wm.py`（两处）
- **症状**: ManiSkill 只有 2 相机，m=3 会导致 rearrange 维度不整除运行时崩溃
- **修复**: `m=3,n=1` → `n_cams=2 if maniskill else 3; m=n_cams, n=1`

---

## BUG-004: vae_local_path 硬编码绝对用户路径

- **发现**: 2026-02-25 代码审查
- **文件**: `rlft/vlaw/data_pipeline.py`，`PipelineConfig.vae_local_path` 默认值
- **症状**: 在其他机器或其他用户账号下直接失效，静默 fallback 到 HF 下载
- **修复**: 默认值改为 `""`，`load_vae()` 中用 `huggingface_hub.try_to_load_from_cache()` 自动查找
- **预防**: 路径默认值不得包含用户名或绝对机器路径，一律用相对路径或环境变量

---

## BUG-005: stat.json 缺失时静默跳过动作归一化

- **发现**: 2026-02-25 代码审查
- **文件**: `ctrl_world/dataset/dataset_maniskill.py`
- **症状**: train 模式下 stat.json 不存在时只打 warning，动作未归一化，loss 异常但不崩溃，极难排查
- **修复**: `mode=="train"` 时改为 `raise FileNotFoundError` 并给出生成命令提示
- **预防**: 所有"缺少必要文件"的情况，train 模式下必须 fail fast

---

## BUG-006: demo_prep.py CUDA_VISIBLE_DEVICES 重复赋值

- **发现**: 2026-02-25 P1.3 执行时
- **文件**: `rlft/vlaw/demo_prep.py`，`replay_to_rgb()` 函数
- **症状**: 外部已设置 `CUDA_VISIBLE_DEVICES` 时，`dict(**os.environ, CUDA_VISIBLE_DEVICES=...)` 报 `TypeError: got multiple values for keyword argument`
- **修复**: `env_vars = dict(os.environ); env_vars["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)`
- **预防**: 构建环境变量字典时，先复制再覆盖，不要用 `**os.environ` 解包后直接添加同名参数

---

## BUG-007: swanlab 需要 API key，训练崩溃

- **发现**: 2026-02-25 WM 训练启动时
- **文件**: `ctrl_world/scripts/train_wm.py`
- **症状**: `swanlab.error.KeyFileError: api key not configured (no-tty)` 导致训练启动即崩溃
- **修复**: `swanlab.sync_wandb()` → `swanlab.sync_wandb(mode="local", logdir=...)`
  - 额外安装: `conda run -n ctrl_world pip install swanlab[dashboard]`
  - 启动命令增加: `WANDB_MODE=offline`
- **预防**: 无外网时所有日志工具都需切换到 local/offline 模式，不要假设 API key 已配置

---

## BUG-008: SwanLab DDP 多进程并发写入 SQLite 崩溃

- **发现**: 2026-02-25 WM Phase-A 重启后 step 100 崩溃
- **文件**: `ctrl_world/scripts/train_wm.py`
- **症状**: 4 个 DDP worker 同时初始化 SwanLab → SQLite `cannot commit - no transaction is active`
- **修复**: `swanlab.sync_wandb()` 调用加 `if LOCAL_RANK == 0:` 保护 + try-except
- **预防**: 日志工具初始化必须只在主进程执行（`LOCAL_RANK == 0` 或 `dist.get_rank() == 0`）

---

## BUG-009: train_wm.py 含未知参数 --num_workers 导致崩溃

- **发现**: 2026-02-25 WM 训练第 2 次重启
- **文件**: `ctrl_world/scripts/train_wm.py`（argparse 定义）
- **症状**: 启动命令中包含 `--num_workers 4`，但 argparse 未定义此参数 → 立即报错退出
- **修复**: 启动命令中移除 `--num_workers` 参数
- **预防**: 新增参数时检查 argparse 是否已声明；DataLoader num_workers 应通过代码内部 `getattr(args, 'num_workers', 4)` 设置

---

## BUG-010: state_dim 设计值 29 与实测值 25 不符

- **发现**: 2026-02-25 D_real 收集后验证 HDF5 结构
- **文件**: `rlft/vlaw/state_predictor.py`（StatePredictorConfig.state_dim），`.github/knowledge/interfaces.md`
- **症状**: 设计文档和代码默认值均为 29，但实际 ManiSkill LiftPegUpright-v1 obs_agent shape=(T, 25)
- **根因**: 初始设计假设 qpos(7)+qvel(7)+ee_pose(7)+gripper(7)+extra(1)=29，实测为 qpos+qvel=25D
- **修复**: `state_predictor.py` 两处 `state_dim=29` → `state_dim=25`；`interfaces.md` 更新注释
- **注意**: `imagination_env.py` 动态推断（不硬编码），不受影响；`PickCube-v1`/`StackCube-v1` 可能 state_dim 不同，待验证
- **预防**: 接口文档中的 shape 必须从实际数据验证，不要依赖理论推算

---

## BUG-011: 零样本 VLM 奖励在 Iter-1 D_real 上极低 (p_yes ≈ 0.001)

- **发现**: 2026-02-25 P3.2 VLM 标注完成后分析
- **症状**: 所有 D_real 轨迹 p_yes < 0.003，即使 env_success=True 轨迹也是如此；threshold=0.8 → vlm_success=0%
- **根因**: 
  1. Qwen3-VL-4B 零样本对 ManiSkill 机器人操作任务理解不足
  2. 成功轨迹视觉特征在 192×192、T=10 帧下不明显
  3. 原 reward_model.py 仅使用小写 "yes"/"no" token，遗漏 "Yes"/"No" 变体
  4. 原代码未使用 `process_vision_info` 正确解耦图像
- **修复**: 
  1. `reward_model.py`: `_yes_token_ids`/`_no_token_ids` 包含所有大小写变体
  2. `_forward_p_yes`: 使用 `process_vision_info` (qwen_vl_utils)
  3. p_yes 改为 softmax 聚合所有变体 logits
- **修复后效果**: success=0.000911 vs failure=0.000335（3x ratio，方向正确），demo=0.037（明显更高）
- **对算法影响**: Iter-1 D_real 全部 vlm_reward=0，策略更新主要靠 D_demo；Iter-2 须用 env_success 标注数据微调 VLM (P3.2) 以提升奖励质量
- **预防**: VLM 奖励在使用前要用 env_success 做相关性校验；p_yes 绝对值不应作为信号强度指标

---

## BUG-012: 零样本 VLM 不适用于 D_syn 标注（必须先 fine-tune）

- **发现**: 2026-02-25 查阅 VLAW 论文 Section 4.1 + Appendix C
- **文件**: `rlft/vlaw/reward/reward_model.py`, `rlft/vlaw/reward/train_reward_model.py`
- **症状**: iter1_v2 标注后 p_yes_max=0.148，alpha=0.8 → vlm_success=0/150，LiftPeg 8条真实成功也未被识别
- **根因分析 (论文原文)**:
  > "we find that the zero-shot VLM is not accurate enough, so in the first iteration, we fine-tune the VLM with the success labels r_τ in D_real"
  - 论文明确指出零样本不够好，必须先 fine-tune
  - alpha=0.8 阈值仅适用于 fine-tuned 模型，zero-shot 下 p_yes 从不超过 0.2
  - 论文 Table 3 (Appendix C)：fine-tuned+threshold=0.8 → FP=2/40=5%
- **根因技术层**:
  - VLM 未经机器人操作领域 fine-tuning，对 ManiSkill 渲染图像理解很弱
  - 192×192 低分辨率、俯视角、关节臂的外观与 VLM 预训练分布差距大
- **正确的执行顺序（已更新 reward-agent.agent.md）**:
  1. P3.1 (已完成): 实现 reward_model.py
  2. **P3.2-new**: 用 D_real + env_success_at_end 标签 fine-tune VLM (200 steps, batch=128)
  3. 保存 fine-tuned checkpoint → `checkpoints/vlaw/reward_model/lora_iter1/`
  4. **P5 (D_syn 标注)**: 用 fine-tuned VLM 标注合成轨迹（而非 zero-shot）
- **Iter-1 过渡方案**:
  - D_real 策略训练：使用 `env_success_at_end` 直接作为过滤标准（不需 VLM）
  - D_syn 标注：等待 P3.2 fine-tuning 完成后再运行 Imagination 和标注
- **数据量需求**: 50条/任务 × 3任务 = 150条，其中成功约 8条（LiftPeg）
  - 注意：150条中仅 8 条为正样本，类别严重不平衡，fine-tuning 时需加权
- **参考**: VLAW arXiv:2602.12063, Appendix C; RoboReward arXiv:2601.00675
