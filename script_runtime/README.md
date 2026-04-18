# script_runtime

`script_runtime/` 是 CARM 真机 script policy 平台的执行层子系统。

定位：
- 不承载学习算法本体
- 负责世界状态、技能抽象、行为树执行、恢复策略、安全门控、trace 与 learned module 适配
- 当前阶段以 `arm_control_sdk` 直连为主，不把 `carm_ros_deploy/` 放在关键路径上

设计原则：
- 与 `rlft/` 平级，避免把任务执行逻辑塞进学习框架
- 通过 adapter 复用现有 `arm_control_sdk/` 与 `rlft/`
- `carm_ros_deploy/` 仅保留为未来可选融合层，而不是首阶段依赖
- 首个验收任务线固定为 `pick-place`

当前首版已包含：
- `core/`：blackboard、skill base、registry、failure code、result type
- `executors/`：轻量行为树执行器与 trace recorder
- `skills/`：motion / gripper / perception / checks / recovery / learned 基础技能
- `tasks/`：pick-place、peg-insert、drawer-open-pick 骨架
- `adapters/`：以 sdk 为主的统一接口，保留 ros / camera / learned 可选适配
- `safety/`：workspace、speed、watchdog、e-stop 包装
- `factory.py`：默认 skill registry 组装入口
- `session.py`：独立于 ROS 的 SDK-first runtime session 装配
- `runners/sdk_pick_place.py`：可直接从配置运行的 pick-place 入口

说明：
- 执行器优先兼容 `py_trees` 风格，但当前实现不强依赖外部包即可运行和测试
- 中期可继续对接 `py_trees_ros`、MoveIt Task Constructor、ROS2 行为树栈
- 如果本机尚未编译 `arm_control_sdk/python` 的 pybind 扩展，可先使用 `--dry-run` 走 `MockSDKBridge`

快速起步：

```python
from script_runtime import build_default_skill_registry
from script_runtime.tasks.pick_place import PickPlaceTask

registry = build_default_skill_registry()
task = PickPlaceTask()
root = task.build(registry)
```

直接运行 SDK-first pick-place：

```bash
python -m script_runtime.runners.sdk_pick_place --dry-run
```

真机运行前提：
- 已按仓库 README 编译并安装 `arm_control_sdk/python`
- `carm` pybind 模块可导入，或通过 `CARM_PYTHON_LIB` 指向 `arm_control_sdk/python/lib`
- 按实际机器人修改 [pick_place_sdk.yaml](/home/amax/rl-vla/script_runtime/configs/tasks/pick_place_sdk.yaml)

先在 ManiSkill 做 state-oracle 验证：

```bash
python -m script_runtime.runners.maniskill_pick_cube
```

切到更强调 `release / stack / recovery` 的 `StackCube-v1`：

```bash
python -m script_runtime.runners.maniskill_pick_cube --config script_runtime/configs/tasks/stack_cube_maniskill.yaml
```

批量做 rollout 验证并汇总失败分布：

```bash
python -m script_runtime.runners.maniskill_validate
```

批量验证 `StackCube-v1`：

```bash
python -m script_runtime.runners.maniskill_validate --config script_runtime/configs/tasks/stack_cube_maniskill.yaml
```

把验证结果渲染成图表和 markdown 报告：

```bash
python -m script_runtime.runners.render_validation_report
```

扫描 ManiSkill 抓取参数并生成热力图：

```bash
python -m script_runtime.runners.maniskill_grasp_sweep
```

说明：
- 这条 runner 默认对接 `PickCube-v1`
- 优先验证任务树、技能流转、failure code 与 grasp 检查
- 当前是 `oracle_state` 模式，`GetObjectPose / GetGraspCandidates / CheckGrasp` 会尽量读取仿真真值
- 这比直接验证视觉感知更适合当前阶段，因为现实里感知逻辑还没有完全收敛
- `maniskill_validate` 会批量跑多个 episode，并把 trace 与 summary 落到 `script_runtime/artifacts/`
- `render_validation_report` 会把当前 summary 渲染成柱状图、timeline 图和 markdown 报告
- 当前默认会额外生成 episode 级 `rollout.gif`、grounding JSON 和 top-down grounding PNG，便于快速检查 script policy 是否真的沿着目标/抓取/放置语义在执行
- `maniskill_grasp_sweep` 会把抓取高度/夹爪保持步数的扫参结果渲染成热力图，方便阶段性看趋势
- `PickCube-v1` 的 goal 可能是空中目标，因此 runtime 会根据目标高度自适应决定是否执行 release；低位目标走 `open -> retreat -> settle`，高位目标保持抓持并直接做环境成功验收

推荐顺序：
1. 先跑 `python -m script_runtime.runners.sdk_pick_place --dry-run`
2. 再跑 `python -m script_runtime.runners.maniskill_pick_cube`
3. 然后跑 `python -m script_runtime.runners.maniskill_validate`
4. 最后才把相同任务树切到真实 `arm_control_sdk`
