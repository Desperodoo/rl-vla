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

---

## BUG-013: config.py SVD/CLIP 模型路径缺少 `../` 前缀

- **发现**: 2026-02-28 V1.1 WM 验证视频测试
- **文件**: `ctrl_world/config.py` (L189-190, `wm_args_maniskill` dataclass)
- **症状**: `OSError: checkpoints/vlaw/world_model/pretrained/svd is not a local folder and is not a valid model identifier`
- **根因**: `wm_args_maniskill` 中 `svd_model_path` 和 `clip_model_path` 默认值为 `checkpoints/...`（相对于项目根），但 `train_wm.py` 从 `ctrl_world/` 目录运行，实际需要 `../checkpoints/...`
- **修复**:
  ```python
  # 修复前:
  svd_model_path: str = "checkpoints/vlaw/world_model/pretrained/svd"
  clip_model_path: str = "checkpoints/vlaw/world_model/pretrained/clip"
  # 修复后:
  svd_model_path: str = "../checkpoints/vlaw/world_model/pretrained/svd"
  clip_model_path: str = "../checkpoints/vlaw/world_model/pretrained/clip"
  ```
- **预防**: Ctrl-World 的训练脚本统一从 `ctrl_world/` 子目录运行，所有 config 路径默认值需添加 `../` 前缀

---

## BUG-014: tmux 会话中 ffmpeg 不在 PATH

- **发现**: 2026-02-28 V1.1 初次运行
- **文件**: `ctrl_world/scripts/train_wm.py` (validate_video_generation 通过 mediapy 调用 ffmpeg)
- **症状**: `RuntimeError: Program ffmpeg is not found` (mediapy 内部调用)
- **根因**: tmux 新建会话不自动加载 conda 环境，ffmpeg (安装在 conda env ctrl_world) 不在 PATH 中
- **修复**:
  1. tmux 命令中显式: `eval "$(conda shell.bash hook 2>/dev/null)" && conda activate ctrl_world`
  2. `train_wm.py` L30-34 已有硬编码 PATH hack（作为双保险）
- **预防**: 所有 tmux 启动的训练命令，必须在开头显式激活 conda 环境

---

## BUG-015: train_wm.py 训练循环不在 max_train_steps 处中断

- **发现**: 2026-02-28 V1.1 测试期间
- **文件**: `ctrl_world/scripts/train_wm.py` (L194-197)
- **症状**: `--max_train_steps=6` 时训练跑完整个 dataloader epoch (~4378 步) 而非在 6 步停止
- **根因**: 内层和外层循环均缺少 `if global_step >= max_train_steps: break` 条件
- **修复**: 在 inner loop 和 outer loop 各添加一处 break 判断:
  ```python
  # inner loop (梯度累积完成后):
  if global_step >= args.max_train_steps:
      break
  # outer loop (epoch 循环):
  if global_step >= args.max_train_steps:
      break
  ```
- **预防**: 任何训练循环必须在 `global_step >= max_steps` 时有显式 break，不能只靠 tqdm 显示

---

## BUG-016: Pretrained Policy 评估中的零动作 padding

- **发现**: 2026-03-01 Pretrained Policy 评估 success_once 仅 2%
- **文件**: `scripts/vlaw/eval/eval_pretrained_policy.py`
- **症状**: 基线策略 success_once 仅 ~2%，远低于预期 (ManiSkill demo replay ~85%)
- **根因**: `pred_horizon=8 + obs_horizon=2` → ShortCutFlowWrapper 从 index 1 开始切片 → 仅产出 7 个有效 action。评估脚本强制 pad 第 8 个为零动作 `[0,0,0,0,0,0,0]`，导致每 8 步都有一步完全无操作，破坏轨迹连续性
- **修复**: 移除零动作 padding，直接返回 7 个有效 action
- **修复后**: success_once = 74-88% (avg ~80%)
- **预防**: ShortCut Flow policy 的 `act_steps` 不等于 `pred_horizon`，代码中不应假设二者相等

---

## BUG-017: Imagination 引擎三合一 bug (PlainConv + API + obs 格式)

- **发现**: 2026-03-01 real policy Imagination 测试全部生成零轨迹
- **文件**: `scripts/vlaw/run/run_imagination_iter1.py`, `rlft/vlaw/world_model/imagination_env.py`
- **症状**: real policy in-the-loop rollout 只产出零动作，合成轨迹全失败
- **Bug 1 — load_policy 缺少 PlainConv 参数**: `load_policy()` 未传递 PlainConv 视觉编码器参数 → `visual_encoder=None` → 策略无法处理图像观测
- **Bug 2 — get_actions API 不匹配**: PolicyAdapter 调用 `self.policy.get_actions()`（不存在），FlowWrapper 的实际方法是 `self.policy.flow_wrapper.get_action()` → 静默 fallback 到零动作
- **Bug 3 — obs 格式不匹配**: obs_tensor 是 flat VAE latent (9216-dim)，但策略需要结构化 obs_cond (562-dim = 256-dim visual + 25-dim state × 2 + 6-dim extra)
- **修复**: 
  1. `load_policy()` 传入 `in_channels=3, out_dim=256, pool_feature_map=True`
  2. PolicyAdapter 完全重写：维护 obs_history buffer，正确调用 `flow_wrapper.get_action()`
  3. PolicyAdapter 接收 `decoded_rgb` + `agent_state` kwargs 构建 obs_cond
- **验证**: mock 5/5 OK, real policy 5/5 OK (~178s/traj)
- **预防**: Imagination 中 policy 接口必须与策略训练时的 obs 格式完全一致；接口对齐需通过端到端测试验证，不能只看 API 签名

---

## BUG-018: EMA Checkpoint 保存格式缺少 ema_agent 键

- **发现**: 2026-03-01 T-EVAL-ITER1-001 评估时，iter-1 策略 success_once 仅 10.9%
- **文件**: `rlft/vlaw/policy/policy_updater.py` (`_save_checkpoint()` 方法)
- **症状**: 评估脚本加载 `policy_iter1.pt` 后回退到 online 权重，EMA 权重未被保存
- **根因**: `_save_checkpoint()` 保存了完整的 `agent.state_dict()`（包含 `velocity_net_ema.*` 前缀的键），但未提取出独立的 `ema_agent` 顶级键。评估脚本 `load_pretrained_policy()` 优先查找 `ckpt["ema_agent"]`，找不到时回退到 `ckpt["agent"]` (online 权重)
- **修复**:
  ```python
  # _save_checkpoint() 中新增：
  ema_agent = {
      k.replace("velocity_net_ema.", "velocity_net."): v
      for k, v in agent_sd.items()
      if k.startswith("velocity_net_ema.")
  }
  if ema_agent:
      ckpt["ema_agent"] = ema_agent
  ```
- **修复效果**: 10.9% → 17.2% (+6.3%)，但核心退化问题仍存在
- **已修复的 ckpt**: `checkpoints/vlaw/policy/iter1/policy_iter1.pt` (重新保存)
- **预防**: 所有保存 checkpoint 的代码必须保证 EMA 权重以评估脚本期望的格式存储；新增 checkpoint 保存后应立即做 load-and-verify 测试

---

## BUG-019: Imagination 初始 latent 使用随机噪声而非真实帧 VAE 编码

- **发现**: 2026-03-02 WM/VAE pipeline 深入调查
- **文件**: `scripts/vlaw/run/run_imagination_iter1.py` (L242-345, `load_initial_frames()` 函数)
- **症状**: 200 条合成轨迹视觉质量极差（VLM p_yes max=0.27），解码帧为纯噪声乱码，非 ManiSkill 场景
- **根因**: `load_initial_frames()` 使用 `torch.randn(1, 4, 48, 24)` 作为初始 latent 输入给 WM，而非从真实帧的 VAE 编码结果加载。WM autoregressive rollout 从随机噪声起步，所有后续帧质量均无法恢复
- **修复**:
  ```python
  # 修复前 (L280 附近):
  initial_latent = torch.randn(1, 4, 48, 24)  # 随机噪声!
  
  # 修复后:
  encoded_path = os.path.join(encoded_dir, task, file)
  data = torch.load(encoded_path)
  initial_latent = data["latent_concat"][0:1]  # 真实首帧 VAE 编码
  ```
- **影响范围**: 
  - `data/vlaw/synthetic/iter1_wm_real/` 中全部 200 条旧合成轨迹 **无效，需重新生成**
  - `data/vlaw/labeled/synthetic_iter1_wm_real/` VLM 标注结果 **无效，需重新标注**
  - 之前"WM 合成质量极差"的诊断结论 (T-DIAG-SYN-001) 实际根因是此 bug，而非 WM 训练不足
- **验证**: 
  - 修复后 3/3 条轨迹成功生成 (`data/vlaw/synthetic/iter1_fixtest3/`)
  - 解码帧清晰显示 ManiSkill 机械臂+peg 场景 (`results/vlaw/dsyn_diagnosis_frames/fixtest/`)
  - 用户目视确认 strip 图片质量正常
  - 图像随时间推移仍有模糊趋势 (WM autoregressive 误差累积，属预期行为)
- **预防**: 
  1. Imagination pipeline 必须有"初始帧来源"的显式配置和日志打印，禁止使用 `torch.randn` 生成初始 latent
  2. 新增合成轨迹后，必须先解码若干帧做 sanity check，再大批量生成
  3. 所有 pipeline 关键输入（初始帧、动作序列）应有 shape/range 断言和日志记录

---

## BUG-020: demo_prep.py 将 rgb_base 复制为 rgb_render — 全数据链污染

- **发现**: 2026-03-04 WM 可视化对比时发现 timing_test base/render 同一视角
- **文件**: `rlft/vlaw/data/demo_prep.py` (L296, `replay_to_rgb()` 函数)
- **症状**: 所有 demo 轨迹的 `rgb_base` 与 `rgb_render` 完全相同 (pixel diff = 0.00)，而 rollout 数据双相机差异正常 (diff = 56.68)
- **根因**: ManiSkill 官方 demo replay 只产出 `base_camera` 一个相机视角（无法 `env.render()` 获取 render_camera）。`demo_prep.py` 第 296 行 `rgb_render = rgb_base.copy()` 直接复制，docstring L10 明确写着 "由于标准 replay 只有 single base_camera，复用 rgb_base 作为 rgb_render"
- **影响范围（全链路污染）**:
  1. `data/vlaw/demos/` — 所有 25 条 demo 的 rgb_render ≡ rgb_base
  2. `data/vlaw/encoded/demos/` — VAE 编码后 latent_concat top-bot diff ≈ 0.057（正常应 ~0.85）
  3. `data/vlaw/encoded/eval_fixed/` traj 0-4 — demo 来源，同样污染
  4. **全部 WM 训练** (iter1, ablation_*, optimal_*) — 只用 demos 训练 → 学到的是 "两个视角相同" 的分布
  5. **全部 Imagination 合成数据** — 基于污染 WM 生成
  6. **全部 VLM 标注** — 基于污染 synthetic 标注
  7. **全部策略训练** — 基于污染 labeled + combined 数据
- **修复**: 彻底放弃官方 demo，用预训练策略直接收集新数据（双相机正确）
  - 新数据收集后立即校验: `rgb_base` vs `rgb_render` pixel mean diff > 30
  - `demo_prep.py` 标记为 DEPRECATED
- **归档**: 全部旧数据移至 `data/vlaw/_archive/v1_contaminated/`，旧 ckpt 移至 `checkpoints/vlaw/_archive/v1/`
- **预防**:
  1. 任何两相机数据管线，收集完成后必须做 `assert (rgb_base - rgb_render).mean() > MIN_DIFF` 校验
  2. 不信任 demo replay 输出的多相机数据，除非显式验证每个相机视角独立
  3. 新建 `encoded/` 数据后，检查 latent_concat 的 top/bottom 半区 diff > 0.5
- **详细计划**: [VLAW_FRESH_START_PLAN.md](../VLAW_FRESH_START_PLAN.md)

---

## BUG-024: Collector success_at_end=100% — Selection Bias (非代码 bug)

- **发现**: 2026-03-05 v3 pilot 50 条数据全部 success_at_end=True
- **文件**: `rlft/vlaw/data/collector.py` (collect 逻辑)
- **症状**: pilot 50 条轨迹 success_at_end=100%，预期 AWSC ~46% (eval 验证值)
- **根因链**:
  1. **ManiSkill3 success → terminated**: `BaseEnv.step()` L1054: `terminated = info["success"].clone()`，成功瞬间 episode 立即结束
  2. **成功 episode 远快于失败 episode**: 成功轨迹 10-120 步完成 (T=5-35)，失败轨迹固定 200 步 (T=51)
  3. **64 并行 env + 采 50 条就停**: ~51 个 env 会成功 (80% success_once) 并快速完成，先于 ~13 个失败 env。采 50 条时全是先完成的成功轨迹
  4. **蒙特卡洛验证**: 1000 次模拟 P(100% success in first 50) = 71.7%
- **独立评估验证 (eval 脚本, 无 early termination)**:
  - success_once = 80.0%
  - success_at_end (200步) = 46.0%
  - episode_len = 200 (全部跑满)
- **视觉验证**: 用户确认 GIF/strip 图中 peg 确实被竖直扶正，数据本身是真实成功
- **影响**:
  - pilot 数据全部是成功轨迹，0 条失败 → 不适合训练 (VLM/WM 需要正负样本)
  - 非代码 bug，是采集策略问题 (selection bias)
- **解决方案**: 方案 A — 大量采集 (num_episodes=1200+)，让失败轨迹也有时间跑完。64 env × 多轮后失败的 ~13 env/round 也会贡献失败数据
- **预防**:
  1. 大量采集时 selection bias 自然消失 (后续批次包含失败轨迹)
  2. 采集后必须检查 success_at_end 比率，不应为 100% 或 0%
  3. 注意 ManiSkill3 所有任务都有 success → terminated 行为，成功 episode 时长 << 失败 episode
  4. 小批量 pilot 数据只能验证数据格式/质量，不能代表真实成功率分布
