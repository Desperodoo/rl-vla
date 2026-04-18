# CARM Script Policy Platform Plan

## 目标

在现有仓库上新增 `script_runtime/`，把仓库从“训练与部署并存的研究仓库”升级为“学习模块可插拔的 skill-based execution platform”。

当前实现默认决策：
- 执行栈：Python + `arm_control_sdk` + 行为树
- 首个验收任务线：Pick-and-Place
- 复用策略：架构参考优先，轻依赖，不重度 vendoring

## 已施工的首版范围

### 1. 新增平级执行层子系统

已新增：

```text
script_runtime/
  core/
  skills/
  tasks/
  executors/
  adapters/
  safety/
  configs/
  tests/
```

它与 `rlft/` 平级，明确区分：
- `rlft/` 负责学习算法、训练、评测、learned module 导出
- `script_runtime/` 负责任务执行、技能组合、恢复、安全、trace、world state

### 2. 统一运行时接口

已建立统一核心抽象：
- `WorldState`
- `TaskBlackboard`
- `Skill`
- `SkillResult`
- `RecoveryAction`
- `FailureCode`
- `SkillRegistry`

World state 默认覆盖：
- 机器人状态
- 场景状态
- 感知质量
- 执行上下文
- learned hints

### 3. 行为树执行中枢

已实现轻量行为树执行器：
- `SequenceNode`
- `SelectorNode`
- `RetryNode`
- `TimeoutNode`
- `SkillNode`
- `TreeExecutor`

实现风格向 `py_trees` 对齐，但当前不强依赖外部包，也能独立运行与测试。

### 4. 第一批 primitive skills

已铺设的技能域：
- motion
- gripper
- perception
- checks
- recovery
- learned

当前包含的代表性技能：
- `MoveJ`, `MoveL`, `ServoDelta`, `Stop`
- `GoHome`, `GoPregrasp`, `Lift`, `PlaceApproach`, `Retreat`, `ResetArm`
- `OpenGripper`, `CloseGripper`, `GuardedClose`, `ResetGripper`
- `CheckSceneReady`, `WaitForObjectStable`, `CheckGrasp`, `CheckContact`
- `GetObjectPose`, `GetGraspCandidates`, `ReacquirePerception`
- `SafeRetreat`, `RetryWithNextCandidate`, `HumanTakeover`

### 5. 标准 failure code

当前统一 failure code 为：
- `PERCEPTION_LOST`
- `NO_OBJECT_DETECTED`
- `NO_GRASP_CANDIDATE`
- `NO_IK`
- `COLLISION_RISK`
- `WORKSPACE_VIOLATION`
- `GRASP_FAIL`
- `CONTACT_ANOMALY`
- `TIMEOUT`
- `ROBOT_FAULT`
- `SDK_ERROR`
- `HUMAN_ABORT`

要求是：
- 技能返回标准 failure code
- trace / 恢复 / 统计全部基于标准码

### 6. learned module 适配层

已新增 `rlft_policy_adapter` 统一接口，支持以下角色：
- proposal
- residual correction
- success estimator
- risk estimator
- recovery ranking

接口为：
- `predict(obs, goal, context)`
- `score(candidates, obs, context)`
- `success_prob(obs, context)`
- `risk_prob(obs, context)`

### 7. SDK-first Pick-and-Place 主线

已新增 `PickPlaceTask` 骨架，默认任务序列为：
1. `CheckSceneReady`
2. `GetObjectPose`
3. `GetGraspCandidates`
4. `GoPregrasp`
5. `GuardedClose`
6. `CheckGrasp`
7. `Lift`
8. `PlaceApproach`
9. `OpenGripper`
10. `Retreat`
11. `GoHome`

这是首个 milestone 的 nominal 主线，后续继续往 guarded approach、刷新 place target、human takeover 分支扩展。

当前阶段的关键落点已经调整为：
- 先直接对接 `arm_control_sdk`
- 使用静态配置或手工给定抓取候选，先打通真机 nominal runtime
- 不要求与 `carm_ros_deploy/` 建立深耦合
- `carm_ros_deploy/` 只作为未来感知、HITL、timeline 融合时的可选来源

同时新增一条更稳妥的前置验证路线：
- 在 ManiSkill `PickCube-v1` 中先跑 `script_runtime`
- 第一阶段默认使用 `oracle_state` 验证模式，而不是视觉感知模式
- 优先验证任务树、skill 接口、恢复分支、标准 failure code
- 等仿真逻辑稳定后，再把视觉感知和真机链路逐步接回

建议的验证顺序固定为：
1. `MockSDKBridge` dry-run
2. ManiSkill 单 episode `oracle_state` 验证
3. ManiSkill 多 episode rollout + trace summary
4. 通过后再切真机 `arm_control_sdk`

仓库内可复用的 ManiSkill 经验主要来自：
- `rlft/envs/make_env.py`
- `rlft/envs/evaluate.py`
- `rlft/datasets/data_utils.py`
- `scripts/convert_demos_to_ee_pose.py`

第一阶段不直接复用 `rlft` 的 vectorized eval runner 来执行 script policy，本轮只复用其：
- 环境创建经验
- ManiSkill 状态字段理解
- `pd_ee_pose / pd_ee_delta_pose` 控制接口认知
- rollout / success 统计思路

## 与现有系统的边界

### 优先复用

- `arm_control_sdk/python`

### 可选复用

- `carm_ros_deploy/src/carm_deploy/core/env_ros.py`
- `carm_ros_deploy/src/carm_deploy/core/safety_controller.py`
- `carm_ros_deploy/src/carm_deploy/inference/*`
- `carm_ros_deploy/src/carm_deploy/data/teleop_bridge.py`

### 当前不做

- 不把任务执行逻辑塞进 `rlft/algorithms`
- 不直接让 learned policy 接管整条真机执行流
- 不在首版做 ROS2 / MoveIt2 全量迁移
- 不在首版引入 PDDL 级 symbolic planner

## 下一步建议

第一阶段代码已把运行时骨架立起来，后续建议按以下顺序继续：

1. 把 `sdk_bridge` 接到真实 `arm_control_sdk` 与 `RealEnvironment`
2. 用配置驱动的 `sdk_pick_place` runner 先打通独立真机执行
3. 在 pick-place 上补 guarded approach、next candidate、human takeover 分支
4. 增加 trace 落盘与 replay CLI
5. 等 SDK-first runtime 稳定后，再评估是否把 `ros_bridge` 接到 owner/source / timeline / HITL
6. 在稳定 5-10 个技能后，再评估引入 MoveIt Task Constructor 做几何层规划升级

## 当前进展更新（2026-04-14）

### 1. 运行时主线已从“骨架”推进到“可验证执行层”

当前 `script_runtime/` 已不只是目录和接口骨架，而是已经具备：
- 可执行的 `PickPlaceTask`
- 统一 `session` 装配
- 轻量行为树执行器
- `arm_control_sdk` 直连桥接
- ManiSkill 仿真验证桥接
- trace / summary / markdown report / 可视化产物链

与最初计划相比，当前已经额外落地：
- `ExecuteGraspPhase` 统一抓取阶段技能
- `CheckTaskSuccess` 环境成功验收节点
- `PlaceRelease` 放置释放阶段技能
- 基于 candidate 的 grasp retry 结构
- episode 级 GIF / grounding JSON / top-down grounding PNG

### 2. ManiSkill 验证已经成为当前主验证面

当前阶段明确采用：
- 先在 ManiSkill 验证 task tree / skill contract / recovery / success semantics
- 再把成熟路径迁到真机 `arm_control_sdk`

现有验证任务线：

1. `PickCube-v1`
   - 目标：验证 pick / lift / move / goal-hold 主线
   - 当前状态：稳定跑通
   - 当前结果：`runtime_successes=3/3`, `sim_successes=3/3`

2. `StackCube-v1`
   - 目标：验证 release / stack / retreat / settle / env success 对齐
   - 当前状态：已接入并能跑通主线
   - 当前结果：`runtime_successes=2/3`, `sim_successes=2/3`
   - 剩余主要波动点：`ExecuteGraspPhase` 的抓取鲁棒性，而不是放置逻辑

### 3. runtime success 与 sim success 已经对齐

前一阶段曾出现：
- 任务树判定成功
- 但 ManiSkill 环境 `success=False`

当前已通过以下方式修正：
- 新增 `CheckTaskSuccess`
- 在任务树末尾统一做环境级成功验收
- 在 `PickCube-v1` 中根据目标高度决定是否 release
- 在 `StackCube-v1` 中检测 support object 后强制走 release 分支

这意味着：
- 当前不再接受“树成功但环境不认”的假成功
- runtime 成功标准已经开始与环境任务定义显式对齐

### 4. 可视化链路已建立

当前验证会自动生成：
- episode trace JSONL
- rollout GIF
- grounding JSON
- grounding top-down PNG
- 汇总 markdown 报告
- success / failure / timeline 图

GIF 当前已经不是纯录像，而是会叠加：
- `tcp`
- `object`
- `grasp`
- `goal`
- `support`（如存在）

grounding JSON 则会记录：
- 当前 step
- 当前 skill
- command 类型
- success / is_grasped
- `tcp_pose`
- `object_pose`
- `goal_pose`
- `grasp_pose`
- `support_pose`

### 5. 当前最重要的工程结论

1. `script_runtime` 已经证明可以作为独立于 `carm_ros_deploy/` 的 script policy 执行层存在。
2. ManiSkill 验证链已经足够支撑“先仿真打磨 task semantics，再迁真机”的路线。
3. `PickCube-v1` 已经验证了“高空目标下保持抓持”的 success 语义。
4. `StackCube-v1` 已经验证了“必须 release 才能成功”的 success 语义。
5. 当前最主要的剩余瓶颈是抓取鲁棒性，而不是任务树结构本身。

### 6. 当前推荐的直接使用方式

单次 `PickCube-v1`：

```bash
conda run -n carm python -m script_runtime.runners.maniskill_pick_cube \
  --config script_runtime/configs/tasks/pick_cube_maniskill.yaml
```

批量 `PickCube-v1`：

```bash
conda run -n carm python -m script_runtime.runners.maniskill_validate \
  --config script_runtime/configs/tasks/pick_cube_maniskill.yaml
```

单次 `StackCube-v1`：

```bash
conda run -n carm python -m script_runtime.runners.maniskill_pick_cube \
  --config script_runtime/configs/tasks/stack_cube_maniskill.yaml
```

批量 `StackCube-v1`：

```bash
conda run -n carm python -m script_runtime.runners.maniskill_validate \
  --config script_runtime/configs/tasks/stack_cube_maniskill.yaml
```

### 7. 下一阶段最值得推进的点

当前建议把重点放在：

1. 提升 `StackCube-v1` 抓取鲁棒性，把 `2/3` 继续往上推
2. 把 grasp candidate 从固定偏移升级成更自适应的 candidate generation
3. 给失败 episode 自动导出关键帧与 candidate 诊断
4. 在抓取稳定后，再把相同 runtime 逐步迁到真实 `arm_control_sdk`
