# Script Policy Migration Handoff

日期：2026-04-15

目标仓库：
- `git@github.com:Desperodoo/script_policy.git`

迁移目标：
- 将当前 `rl-vla` 中已施工的 `script_runtime/` 与 `arm_control_sdk/` 抽离到独立仓库
- 在新仓库中以 `robotwin2.0` 作为主要仿真验证环境继续演进
- 保留未来迁回真机 `arm_control_sdk` 的路径，但不再以当前 ManiSkill 场景作为主验证面

## 1. 为什么迁移

当前 `rl-vla` 中的 `script_runtime/` 已经证明：
- skill-based runtime 的任务树、恢复、trace、成功验收机制是可行的
- `arm_control_sdk` 可以作为执行侧主接口，不需要强耦合 `carm_ros_deploy/`
- 先在仿真里打磨任务语义，再迁移到实机，这条路线合理

但当前主验证环境 ManiSkill 的局限也很明确：
- 场景风格与真实桌面操作差异较大
- 任务类型以教学型 benchmark 为主，不足以覆盖真实脚本任务
- 物体类型过于简单，尚不足以支撑盘子、杯子、不规则物体等复杂对象

因此迁移到独立仓库后，建议明确新的主线：
- `script policy runtime + arm_control_sdk + robotwin2.0`

## 2. 当前已具备的可迁移资产

### 2.1 `script_runtime/`

当前可直接迁移的主要内容：
- `core/`
  - `WorldState`
  - `TaskBlackboard`
  - `FailureCode`
  - `Skill`
  - `SkillResult`
- `executors/`
  - 轻量行为树执行器
  - trace recorder / trace replayer
- `skills/`
  - motion / gripper / perception / checks / recovery / learned
- `tasks/`
  - `PickPlaceTask`
  - `peg_insert.py`
  - articulated 骨架
- `adapters/`
  - `sdk_bridge.py`
  - `camera_bridge.py`
  - `rlft_policy_adapter.py`
  - `maniskill_bridge.py`（可保留为历史验证参考，不再作为主线）
- `session.py`
  - SDK-first runtime 装配入口
- `factory.py`
  - 默认 skill registry 组装入口
- `runners/`
  - `sdk_pick_place.py`
  - ManiSkill runners 可作为“验证 runner 模板”
- `tests/`
  - 基础执行器 / task / session / ManiSkill 验证测试

### 2.2 `arm_control_sdk/`

当前目录中既包含：
- C++/头文件接口
- Python pybind 构建入口
- 已编译的动态库与 Poco 依赖

建议迁移时保留整个目录，原因：
- `script_runtime/adapters/sdk_bridge.py` 默认依赖 `arm_control_sdk/python/lib`
- 新仓库后续做真机联调时需要同样的 pybind 入口
- 即使 `robotwin2.0` 阶段不直接使用真机，保留 SDK 边界能避免后续再次拆分

## 3. 当前系统的真实状态

这部分非常重要，避免迁移后对现状有误判。

### 3.1 已经成熟的部分

- runtime 的主干结构成立
- `PickPlaceTask` 的 nominal 流程、retry、recovery、trace 已打通
- `FailureCode`、`CheckTaskSuccess`、`session` 组装方式已经稳定
- `sdk_bridge.py` 已经可以直接对接 `arm_control_sdk`
- 整体设计已经以 `arm_control_sdk` 为主，而不是 `carm_ros_deploy`

### 3.2 仍然只是阶段性验证的部分

- ManiSkill 验证仍然大量依赖 `oracle_state`
- `GetObjectPose` / `GetGraspCandidates` 在仿真中大量使用环境真值
- 真机模式下默认 `object_pose` / `grasp_candidates` 仍然主要来自静态配置或上游注入
- 目前还没有真正完备的开放场景视觉 grounding、6D pose、grasp synthesis 链

### 3.3 当前最不该误解的一点

当前 runtime 已经解决的是：
- “当上游告诉我抓哪个物体、候选抓取位姿是什么之后，我如何安全执行、恢复、记录 trace”

当前 runtime 还没有彻底解决的是：
- “我如何从开放世界图像里先找对目标物体，再生成高质量 grasp pose”

这意味着迁移到新仓库后，最重要的新施工面不是执行树本身，而是：
- target grounding
- object state estimation
- grasp candidate generation
- candidate ranking / risk estimation
- sim-to-real perception contract

## 4. 建议在新仓库保留的目录结构

建议新仓库第一版使用：

```text
script_policy/
  README.md
  docs/
    HANDOFF_FROM_RL_VLA.md
    MIGRATION_NOTES.md
    ROBOTWIN2_RUNTIME_PLAN.md
  script_runtime/
  arm_control_sdk/
```

等 `robotwin2.0` 主线接入后，再逐步扩展：

```text
script_policy/
  docs/
  script_runtime/
  arm_control_sdk/
  sim_runtime/
    robotwin2_bridge/
    tasks/
    assets/
  tests/
```

这里不建议一开始就大拆目录。先保持 `script_runtime/` 内部结构稳定，避免迁移和重构同时发生。

## 5. 迁移后的第一批任务

### 第一阶段：完成仓库脱钩

目标：
- 让新仓库里的 `script_runtime/` 能独立 import / 运行
- 让 `sdk_pick_place --dry-run` 能跑
- 让 `arm_control_sdk` 的 Python 入口位置保持兼容

重点：
- 去掉对 `rl-vla` 仓库路径的隐式假设
- 修正 README、配置默认路径、输出目录
- 保留 ManiSkill runner 作为历史验证参考，但标注“非主线”

### 第二阶段：接入 `robotwin2.0`

目标：
- 新增 `robotwin2` 仿真桥接
- 复用当前 `session + task + skill + blackboard + trace` 主干
- 让 `GetObjectPose / GetGraspCandidates / CheckTaskSuccess` 对接新的仿真环境语义

建议新增：
- `script_runtime/adapters/robotwin_bridge.py`
- `script_runtime/runners/robotwin_pick_place.py`
- `script_runtime/validation/robotwin_rollout.py`

### 第三阶段：为真实复杂物体做感知抽象升级

建议把当前 perception 链路升级为：

1. `ResolveTargetObject`
2. `EstimateObjectState`
3. `GenerateGraspCandidates`
4. `RankGraspCandidates`
5. `RefineActiveGrasp`

这样后续不管你接的是：
- 传统几何/点云方案
- GraspNet 类模型
- foundation model 的 target grounding / proposal
- task-conditioned ranking

都能挂在清晰的接口上，不需要改动主行为树结构。

## 6. 当前代码中的关键入口

### 执行层主入口

- `script_runtime/session.py`
- `script_runtime/factory.py`
- `script_runtime/tasks/pick_place.py`

### 真机执行主边界

- `script_runtime/adapters/sdk_bridge.py`

### 当前仿真参考实现

- `script_runtime/adapters/maniskill_bridge.py`
- `script_runtime/validation/maniskill_rollout.py`
- `script_runtime/runners/maniskill_pick_cube.py`

### 当前感知相关技能

- `script_runtime/skills/perception/primitives.py`

### 当前动作与恢复技能

- `script_runtime/skills/motion/primitives.py`
- `script_runtime/skills/gripper/primitives.py`
- `script_runtime/skills/recovery/primitives.py`

## 7. 迁移时的注意事项

### 7.1 不建议一起迁移的内容

以下内容不建议作为新仓库第一版的核心依赖：
- `.codex/` 的全部历史文档
- `script_runtime/artifacts/` 下的历史产物
- `__pycache__/`
- `rlft/` 相关训练链
- `carm_ros_deploy/`

这些可以保留在 `rl-vla`，避免新仓库一开始就背上过多历史包袱。

### 7.2 但建议保留的参考材料

建议至少带过去：
- 本交接文档
- `script_runtime/README.md`
- `.codex/SCRIPT_POLICY_PLATFORM_PLAN.md` 的关键内容，或其浓缩版

### 7.3 关于 `arm_control_sdk`

当前 `arm_control_sdk/` 内包含已编译库与第三方依赖，迁移后要注意：
- 新仓库是否接受直接存放二进制
- 是否需要后续改成 release asset / 单独 submodule / 私有依赖包

短期建议：
- 先原样迁移，保证功能不断
- 等 `robotwin2.0` 和真机双链都稳定后，再决定是否做更干净的 SDK packaging

## 8. 迁移完成后的最小验收

在新仓库中至少完成以下验收：

1. `python -m script_runtime.runners.sdk_pick_place --dry-run`
2. 基础 tests 可运行
3. `script_runtime` 可以脱离 `rl-vla` 独立 import
4. `arm_control_sdk` 的 pybind 查找路径仍然有效
5. 文档中明确 ManiSkill 是历史验证链，不再是唯一主线

## 9. 迁移后的直接建议

进入新仓库后，最值得优先做的不是继续堆任务，而是：

1. 建立 `robotwin2` 桥接
2. 重构 perception contract
3. 明确“目标语义 grounding -> 几何状态 -> grasp 候选 -> 排序 -> 闭环修正”的链路
4. 给盘子、杯子这类对象定义独立 grasp family，而不是把它们硬塞进 cube 风格抓取

一句话总结：

当前这套代码最值得迁走的是“执行层骨架与 SDK 边界”，不是 ManiSkill 里的具体启发式。新仓库的核心任务应该是把这套执行层落到 `robotwin2.0` 与复杂物体感知链路上，再逐步回到真机。
