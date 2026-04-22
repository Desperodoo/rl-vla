# PI0.5 真机调试计划（CARM）

## 0. 目标与边界
- 目标：在不牺牲安全性的前提下，完成 pi0.5（pi05）真机推理链路调通、稳定性验证与首版归因。
- 控制边界：当前真机运行使用 **joint 控制链路**（`joint_control_nostep`），避免 ee contract 混用。
- 终止条件：任何异常抖动、持续 safety clipping、动作突变、通信异常，立即停机并进入复盘。

## 1. 环境与资产检查（上机前）
1. 权重与元数据
   - `--pretrain` 指向有效 checkpoint 目录。
   - `--dataset_root` 与 `--repo_id` 能正确构建 LeRobot policy config。
2. ROS/相机
   - `camera_topics` 有稳定图像流，`sync_slop` 合理（建议 0.02~0.05 起步）。
3. 机械臂
   - 急停可用，工作空间清空。
   - `robot_mode != 1`（Position 模式禁用）。
4. 启动参数
   - 初期使用较保守频率：`desire_inference_freq=10~15`，`control_freq=30~50`。
   - 小步数上限：`max_steps=100~300`。

## 2. 分阶段调试流程

### Phase A：离线/半离线干跑（不下发控制）
目的：验证 observation → preprocess → policy → postprocess 输出稳定。

检查项：
- 输出 `a_hat` shape 稳定（batch, horizon, action_dim）。
- 无 NaN/Inf。
- 推理时延稳定（p50/p95）。
- action 连续性：相邻 step 不出现异常尖峰。

通过标准：连续 10~20 分钟无异常。

### Phase B：低风险真机联调（小范围动作）
目的：在真实控制链验证 joint contract 与执行节奏。

策略：
- 先短 episode（10~30 秒），单任务重复。
- 限制初始动作幅度与夹爪变化速率。
- 保持人工随时介入（键盘 intervention 开启）。

检查项：
- 控制频率 EMA 稳定。
- safety clipping 比例低且可解释。
- 末端与关节无明显振荡。

通过标准：20 个短 episode 中无危险行为，且失败可归因。

### Phase C：任务级验证（可重复）
目的：验证完整 pick-and-place（或目标任务）闭环质量。

策略：
- 固定初始位姿与场景。
- 每轮记录成功/失败与关键事件。
- 保持录制（record_inference）便于回放。

检查项：
- 成功率、失败模式分布。
- 失败是否集中在感知、抓取时序、还是控制执行。

### Phase D：稳定性 Soak Test
目的：验证连续运行稳定性。

策略：
- 1~2 小时持续运行。
- 定期检查线程状态、日志写入、动作队列行为。

通过标准：无内存泄漏迹象、无频率明显漂移、无累计异常。

## 3. 安全策略（必须执行）
1. 双重急停：物理急停 + 软件中断。
2. 每次启动前确认工作空间清空。
3. 设定 `max_steps` 防止长时间失控。
4. 发现连续异常（如连续 N 次 safety 警告）自动停当前 episode。
5. 结束时执行标准 shutdown，确保机械臂回位策略一致。

## 4. 数据记录与归因模板
每个 episode 建议记录：
- 基础：时间戳、task、checkpoint、参数快照。
- 性能：inference time、control hz、chunk 队列状态。
- 行为：raw action / executed action / intervention mask。
- 安全：safety clipping 次数与原因。
- 结果：成功/失败标签、失败阶段。

首轮归因建议标签：
- 感知问题（图像时序/视角）
- 动作语义问题（输出维度/归一化/夹爪语义）
- 控制执行问题（频率、抖动、时延）
- 安全约束问题（workspace/joint limit）

## 5. 推荐首轮执行命令（示例）
```bash
python carm_ros_deploy/src/carm_deploy/inference/inference_pi05_ros.py \
  --pretrain <CKPT_DIR> \
  --dataset_root /mnt/disk_2/wjz/runs/pi05_full_export/train \
  --repo_id carm/pi05_local \
  --control_mode joint \
  --action_representation joint_absolute_gripper \
  --desire_inference_freq 12 \
  --control_freq 40 \
  --execution_mode receding_horizon \
  --act_horizon 8 \
  --max_steps 200 \
  --record_inference \
  --intervention
```

## 6. 交付物
- 真机调试日志（按 episode）。
- 失败案例回放（含动作与观测）。
- 首版归因报告（按失败标签统计）。
- 下一轮参数调整建议（频率、horizon、安全阈值）。
