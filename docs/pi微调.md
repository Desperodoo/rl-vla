按你的要求，我把 `ctrl_world` 完全排除了，只聚焦“当前仓库的 `pi` 微调主线”。

**先给结论**
- 我找到的**最强、最干净、最可信的开源成功先例**，不是 `pi05_droid`，而是官方 `pi05_libero` 这条线。
- 这条线公开地证明了：**`pi0.5 / pi05` 在 end-effector 类型 action space 上可以成功微调并取得很强结果**。
- 但我**没有找到一个同等质量、公开可复现、而且明确是“自采真实机器人数据 + absolute ee_pose”成功闭环**的案例。
- 相反，我找到的公开自定义 `ee_pose` 讨论里，有两类典型风险：
  - `Cartesian eef-pose` 数据如果直接套官方的 delta/absolute transform，旋转部分很容易处理错。
  - 自定义 real robot 平台上，用 pose-based action 时，常见失败点就在 grasp 阶段。

所以，如果你让我基于资料和你仓库现状给一个“最适合你自采数据集”的方案，我的建议很明确：

**不要继续把当前 8D absolute ee_pose 当主线。主线应改成“LIBERO-like 的 7D ee-delta + gripper”方案，并从 `pi05_base` 启动，而不是从 `pi05_droid` 启动。**

**1. 我找到的开源资料里，哪些真正支持 `pi0.5` + `ee_pose` 这条路**

**最强正例：官方 `pi05_libero`**
- `openpi` README 直接把 `pi05_libero` 列成公开 checkpoint，并且说它是 `pi0.5` 在 LIBERO benchmark 上微调得到的官方模型。  
  来源：<https://github.com/Physical-Intelligence/openpi>
- README 还明确说，**官方用 LIBERO 作为“如何在你自己的数据上微调 pi0.5”的 running example**，并给了完整流程：
  1. 转成 LeRobot dataset
  2. 定义 data config / train config
  3. 启动 policy server 做 inference  
  来源：<https://github.com/Physical-Intelligence/openpi>
- `convert_libero_data_to_lerobot.py` 里，官方示例数据格式就是：
  - 两张图：`image`、`wrist_image`
  - 状态：`state`，8 维
  - 动作：`actions`，7 维  
  来源：<https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/convert_libero_data_to_lerobot.py>
- `LeRobotLiberoDataConfig` 明确说明：
  - 这是给你“复制修改成自己数据集”的模板
  - LIBERO 这条线会先做键名 remap
  - 再用 `LiberoInputs / LiberoOutputs`
  - 对 `pi05_libero`，`extra_delta_transform=False`，因为 LIBERO 原始动作已经是 delta  
  来源：<https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py>
- `libero_policy.py` 明确说明：
  - 它就是给训练和推理同时使用的自定义 IO 适配层
  - 会把 `observation/image`、`observation/wrist_image`、`observation/state` 映射到模型内部格式
  - 输出时只取正确的前 `N` 个 action 维度  
  来源：<https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/policies/libero_policy.py>

这条链路已经足够说明一件事：

**官方公开成功、公开可复现、而且和 `ee_pose` 最接近的 `pi0.5` 微调范式，就是 LIBERO 这条线。**

**第二层证据：后续公开研究都把 `pi0.5-LIBERO` 当作 end-effector 版本**
- `DR-VLA` 这篇公开论文在分析 `pi0.5-LIBERO` 和 `pi0.5-DROID` 时，明确写到：
  - LIBERO：7 维动作，`6-DoF end-effector + gripper`
  - DROID：8 维动作，`7 joint positions + gripper`
- 它还特别指出：两者依赖的是**完全不同的 action representation**。  
  来源：<https://drvla.github.io/drvla.pdf>

这个来源不是训练教程，但它很有价值，因为它独立验证了：
- `pi0.5-LIBERO` 的确是 end-effector action family
- `pi0.5-DROID` 的确是 joint family

**第三层证据：后续工作继续在 `pi0.5 + LIBERO` 上拿到强结果**
- `SnapFlow` 最近的公开论文，在四个 LIBERO suite 上验证了 `pi0.5` 系列模型的加速蒸馏方案，并报告很高成功率。
- 这不是“你的自定义真实机器人数据”案例，但它说明：**`pi0.5 + LIBERO-style end-effector action` 这条公开路线是稳定、成熟、可继续叠加研究的。**  
  来源：<https://arxiv.org/abs/2604.05656>

**反例 / 风险证据**
- 官方 issue #416：用户明确说自己是基于 `Cartesian eef-pose` action fine-tune，自查后发现 LIBERO 的 7D/8D 是 eef pose，但直接套 `pi0` 的 delta/absolute action transform 效果不好，特别是旋转不能简单相减。  
  来源：<https://github.com/Physical-Intelligence/openpi/issues/416>
- 官方 issue #912：用户在自研平台上用 `pi0.5` + end-effector pose `(xyz + Euler)`，现象是 approach 很好，但 grasp 不发生，只输出很小动作、持续推物体。  
  来源：<https://github.com/Physical-Intelligence/openpi/issues/912>

这两个反例非常像你现在可能会踩的坑：
- `ee_pose` 不是不能训
- 但 **Cartesian absolute pose 的变换、旋转表示、gripper 表示、归一化和控制接口**，都很容易出错

**2. 结合这些资料，我对你当前仓库主线的判断**

你当前仓库里 `pi05 bridge` 主线还是：

- 在 [contract.py](/home/amax/rl-vla/rlft/offline/pi05_bridge/contract.py#L20) 里定义成 `absolute_pose_gripper`
- 8 维
- 来自 CARM v2 的 raw teleop 语义
- 在 [export.py](/home/amax/rl-vla/rlft/offline/pi05_bridge/export.py#L101) 里直接写进 LeRobot dataset

也就是说，当前主线是：

- `absolute Cartesian target pose (7D) + gripper (1D)`

而公开资料里最强的成功先例是：

- `pi05_libero`
- `7D end-effector delta + gripper`
- 配套有明确的 `Inputs / Outputs / DataConfig / remap / extra_delta_transform=False`

所以两者不是“差一点点”，而是**训练契约不一样**。

你仓库历史实验里，[pi05_full_finetune_report_2026-03-30.md](/home/amax/rl-vla/docs/pi05_full_finetune_report_2026-03-30.md) 的确已经证明：
- 当前 absolute 8D 方案可以把 offline MAE/MSE 做下来

但我对这件事的解释是：

- 它证明了这条链路**在离线监督拟合上能学**
- 还不能证明它是**最稳的动作语义选择**
- 更不能说明它已经对齐到了官方公开成功路线

如果你现在问我：

“在开源证据支持下，哪条线更适合继续做主线？”

我的答案是：

- **主线应该向 `pi05_libero` 靠**
- **当前 absolute 8D 方案更适合降级成 baseline / ablation**

**3. 最适合你自采数据集的改造方案**

我建议你走一个非常明确的方案：

## 方案核心
**把当前数据契约从 `8D absolute ee_pose + gripper` 改成 `7D LIBERO-like ee-delta + gripper`，并以 `pi05_base` 为主初始化。**

不是 `pi05_droid`，也不是继续把当前 absolute 8D 当主线。

---

## 3.1 为什么我推荐 `pi05_base`，而不是 `pi05_libero` 或 `pi05_droid`

**不推荐 `pi05_droid`**
- 它的公开成功主线是 joint-domain
- 你现在想走的是 ee-domain
- action 先验偏差太大

**不把 `pi05_libero` 当唯一主线**
- 它的 action semantics 和你目标更接近，这是优点
- 但它是 simulator-specific、视觉域和任务分布都更偏 LIBERO
- 对真实自采图像域来说，不一定比 base 更稳

**推荐 `pi05_base`**
- `pi0.5` 论文明确说，预训练阶段同时用了 joint 和 end-effector control，并通过 control-mode token 区分。  
  来源：<https://www.physicalintelligence.company/download/pi05.pdf>
- 所以 `pi05_base` 比 `pi05_droid` 更不容易把你锁死在 joint 语义里
- 同时又没有 `pi05_libero` 那么强的模拟环境 specialization

所以我的主推荐是：

- **主线初始化：`pi05_base`**
- **对照初始化：`pi05_libero`**
- **不建议再把 `pi05_droid` 当主线**

---

## 3.2 目标动作定义怎么改

当前：
- `absolute pose (7D) + gripper (1D)`，共 8 维

建议改成：
- `delta position (3)`
- `delta rotation (3)`
- `gripper (1)`
- 共 **7 维**

这就是 LIBERO-style 的方向。

### 这里最重要的一点
**不要用“直接减 Euler / 直接减 quaternion 分量”的方式做 delta。**

这是我最想强调的工程点，因为 issue #416 基本已经把这个坑公开点出来了。

正确做法应该是：

- 位置：`delta_xyz = target_xyz - ref_xyz`
- 姿态：
  - 用 `R_delta = R_ref^{-1} * R_target`
  - 再把 `R_delta` 转成一个 3 维旋转增量表示
- gripper：
  - 保持一个清晰稳定的语义
  - 最好是和控制器一致的 absolute open/close 标量，或二值 open/close

### 旋转用什么表示
我建议优先级如下：

1. **和你真实控制器最一致的 3 维旋转 delta 表示**
2. 如果没有既定控制器强约束，尽量靠近 LIBERO / OSC_POSE 习惯的 3 维旋转增量表示
3. 不要继续保留 quaternion absolute orientation 进 action

也就是说：
- `state` 可以继续保留 quaternion / pose
- `action` 不建议继续用 absolute quaternion target

---

## 3.3 状态定义怎么改

我建议把当前训练主线状态改成：

- `state_mode = ee_only`

原因很简单：

- 你当前仓库里 `ee_only` 对应 8 维状态
- 这和 LIBERO 示例里的 `state` 维度是最接近的
- 比 `joint_only` 更符合你要把 action 主线改成 end-effector delta 的方向

也就是：

- `state = current_ee_pose(7) + gripper(1)`，共 8 维

这会让你的输入输出契约尽量贴近公开成功的 `pi05_libero` 线。

---

## 3.4 图像输入怎么改

LIBERO 示例默认是两张图：
- `image`
- `wrist_image`

你当前 CARM 数据现在看起来更像单路 RGB。

我建议不要硬凑成 DROID 风格，而是做一个**自定义 LIBERO-like Inputs**：

- `observation/image`：用你的主相机
- `observation/wrist_image`：
  - 如果你没有 wrist cam，就先用零图占位
  - 或者临时复制主图做对照实验，但我更倾向零图占位

`libero_policy.py` 本身就明确说明了：
- 不存在的图像可以用零填充
- 关键是把输入 dict 的最终键组织成模型预期格式  
  来源：<https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/policies/libero_policy.py>

---

## 3.5 数据转换层怎么改

这是当前仓库最值得补的一层。

你现在主线只有：
- bridge contract
- export
- normalize

但缺了官方那层：
- `RepackTransform`
- dataset-key -> inference-key remap
- `Inputs / Outputs`
- 是否需要 `extra_delta_transform` 的显式开关

### 我建议你补一套“CARM-LIBERO-like data config”

核心思路是仿照官方 `LeRobotLiberoDataConfig`，但改成你的 CARM 版本。

你应该补四个对象：

1. `CarmEeInputs`
2. `CarmEeOutputs`
3. `LeRobotCarmEeDataConfig`
4. `TrainConfig / 训练入口` 对应这套 config

### 这套 config 要明确三件事
1. **repack**
   - 把你导出的 dataset 键 remap 到 policy 期望键
2. **data transforms**
   - 如果你导出的数据已经是 delta，就像 `pi05_libero` 一样：
   - `extra_delta_transform = False`
3. **outputs**
   - 只返回真实有效的前 7 维 action

### 最关键的设计决策
我建议你在**数据导出阶段就把 absolute action 转成 delta action**，而不是在训练时再做 transform。

理由：
- 你的原始数据是 Cartesian absolute pose
- 官方 generic `DeltaActions` 对 joint 很自然，但对 Cartesian rotation 不够安全
- 你在离线转换时可以用正确的 SE(3) 逻辑处理旋转

所以：

- **训练 config 里不要再做额外 delta transform**
- **数据集落盘时就已经是目标 7D delta 格式**

这点要尽量做成“和 `pi05_libero` 一样”：
- dataset already delta
- `extra_delta_transform=False`

---

## 3.6 训练策略怎么改

### 第一阶段
- 初始化：`pi05_base`
- 动作：7D ee-delta + gripper
- 状态：8D ee_only
- action horizon：**10**
- fresh norm stats：重新算，不复用 DROID stats
- 训练方式：
  - 优先 full finetune 或至少 expert-heavy finetune
  - 不把 LoRA 当第一优先

### 为什么我不推荐一上来就 LoRA
- 官方 README 在 DROID 线明确说过，他们试过 LoRA，但效果不太理想。  
  来源：<https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/README_train.md>
- 你这里不仅是换任务，还是换 action contract
- 这种情况下 LoRA 更容易“离线 loss 好看，但控制行为奇怪”

### 更稳的两阶段策略
1. **阶段 A**
   - 冻结 vision encoder
   - 先主要训 action expert / 后层
   - 看 loss 和 held-out rollout consistency
2. **阶段 B**
   - 如果 grasp / 接触行为仍然差
   - 短程解冻后部感知层或更大范围继续 finetune

---

## 3.7 评估方案怎么改

你现在已经有 offline MAE/MSE 评估，这是有价值的，但不够。

我建议至少补三类评估：

1. **teacher-forcing offline MAE/MSE**
   - 继续保留
2. **open-loop rollout consistency**
   - 从真实状态起点开始，连续滚动预测 action chunk
   - 积分成 ee trajectory
   - 看末端位姿和 gripper 误差
3. **小规模闭环真实执行**
   - 先只看 approach
   - 再单独看 grasp
   - 最后再看 full task

因为从公开 issue #912 来看，很多 `ee_pose` 路线的问题不是“不会靠近”，而是：
- 靠近没问题
- 但 grasp phase 崩掉

如果只盯 MAE，很可能会漏这个问题。

---

## 3.8 我建议你保留的对照实验

为了不把已有工作全推翻，我建议你保留 3 条并行线：

### A. 主线
- `pi05_base`
- `7D ee-delta + gripper`
- `state=ee_only`
- `action_horizon=10`

### B. 语义接近对照
- `pi05_libero`
- 同样的 7D ee-delta 数据
- 看它是否比 `pi05_base` 更快收敛

### C. 现有基线
- 你当前的 `8D absolute pose + gripper`
- 作为历史 baseline 保留
- 不再作为首选主线

这样你不会失去现有结果，同时能很快回答一个关键问题：

- **对你这份自采数据，action semantics 对结果的影响到底有多大？**

**4. 这件事我找到的资料里，什么是足够的，什么是不足够的**

我找到的“足够强”的资料有：
- 官方 `pi05_libero` checkpoint、训练 README、转换脚本、data config、inputs/outputs
- 官方 README 明确把 LIBERO 当成“你自己数据集微调”的模板
- 后续公开论文把 `pi0.5-LIBERO` 明确当作 end-effector action 版本来分析

我**没找到**的，是：
- 一个高质量、公开可复现、明确基于**真实自采数据**、并且使用**absolute ee_pose** action、最后成功闭环的 `pi0.5` 方案

我找到的相关公开案例更多是：
- 官方 LIBERO 成功
- 自定义真实机器人 `ee_pose` 路线的公开 issue 和踩坑讨论

所以这部分资料给出的信号非常一致：

- **`ee_pose` 不是不能做**
- **但最可靠的公开路径是 “LIBERO-like delta ee action”，不是“absolute ee_pose 直接喂进去”**

**5. 我对你当前最推荐的最终方案**

一句话版：

**把当前仓库的 `pi05` 主线改造成 `pi05_base + CARM-LIBERO-like ee-delta dataset + custom Inputs/Outputs + no extra delta transform`。**

更具体一点：

1. 把 [contract.py](/home/amax/rl-vla/rlft/offline/pi05_bridge/contract.py#L20) 的主 action contract 从 `absolute_pose_gripper` 改成新的 `ee_delta_gripper`
2. 在导出阶段把 raw absolute pose 先转换成正确的 7D delta action
3. 状态主线改为 8D `ee_only`
4. 新增一套 `CarmEeInputs / CarmEeOutputs / CarmEeDataConfig`
5. 主初始化从 `pi05_droid` 改为 `pi05_base`
6. 把 `pi05_libero` 作为次优 warm-start 对照
7. 现有 absolute-8D 路线保留为 baseline，不再当主线

如果你愿意，我下一步可以直接继续帮你做一件更具体的事：

- 我可以把这个“CARM-LIBERO-like 改造方案”落成一份**文件级实施清单**，直接列出你仓库里该改哪些文件、每个文件要新增什么类/字段/脚本，以及应该先做哪几个最小实验。

---

## 6. 本轮对话最终敲定的大方向

这一节记录这轮对话里已经明确拍板、后续不再反复摇摆的方向。

### 6.1 用户明确拍板的约束

1. **初始化主线**：从 `pi05_droid` 切到 `pi05_base` 和 `pi05_libero`，并且这两个初始化都重要，不再把 `pi05_droid` 当主线。
2. **目标动作定义**：尽量对齐 `pi05_base / pi05_libero` 官方语义；你自采数据的学习层本质更接近 `ee_delta_pose`，之前 bridge 为了贴 `pi05_droid` 才转去 `ee_pose absolute`，这是弯路。
3. **训练优先级**：优先 `LoRA / PEFT`，因为显存紧张；在这条线稳定后，再尝试 full finetune。
4. **范围约束**：`ctrl_world` 完全排除，不纳入这次主线设计和施工。

### 6.2 基于这些约束，我收敛出的主线

- `pi05` 当前仓库主线应该改成：`pi05_base / pi05_libero + ee_delta + ee_only state + LoRA first`
- `pi05_droid + absolute ee_pose` 不删除，但降级为 baseline / ablation
- delta 不放到 train-time transform 里做，而是在 **bridge/export 阶段离线算好**
- 旋转不再保留 absolute quaternion 进入 action，而是转成更接近官方 end-effector control 的 **3D rotvec delta**

### 6.3 这轮施工采用的具体动作契约

这一轮代码里，我把主线 contract 定义成了：

- `ee_delta_pose_gripper`
- 7 维
- 语义是：`[dx, dy, dz, d_rx, d_ry, d_rz, gripper]`

其中：

- `dx, dy, dz`：由当前帧 `observation.ee_pose` 到 raw `target_pose` 的相对平移
- `d_rx, d_ry, d_rz`：先做 `SE(3)` 相对变换，再把相对旋转转成 `rotvec`
- `gripper`：沿用 raw 数据里的 gripper 标量，不做额外 delta 化

这比此前的：

- `absolute_pose_gripper`
- 8 维
- `[x, y, z, qx, qy, qz, qw, gripper]`

更接近 `pi05_libero` 那条公开成功路线。

---

## 7. 本轮已经完成的施工

下面是这轮已经真正落到仓库里的改动，不是停留在分析层。

### 7.1 新增统一动作语义转换层

新增文件：

- `rlft/offline/pi05_bridge/action_transform.py`

它负责两件事：

1. 统一声明 `pi05` bridge 支持的 action representation
   - `ee_delta_pose_gripper`
   - `absolute_pose_gripper`
2. 统一完成 raw CARM action 到目标 bridge action 的转换

关键实现：

- 自动识别 CARM raw action layout
  - `8D` 数据：`target_pose(7) + gripper(1)`
  - `15D` 数据：`... + FK_pose(7) + gripper(1)`
- 对主线 `ee_delta_pose_gripper`：
  - 先从 `current ee_pose` 和 `target_pose absolute` 计算相对变换
  - 再把相对四元数变成 `rotvec`
  - 最后拼回 `[delta_xyz, delta_rotvec, gripper]`

这样做的目的很明确：

- 让 `dataset_bridge` 和 `LeRobot export` 使用同一套几何逻辑
- 避免一个地方按 absolute 导出、另一个地方按 delta 训练，导致语义漂移

### 7.2 `Pi05ActionContract` 已切成“主线 delta，保留 absolute baseline”

已修改：

- `rlft/offline/pi05_bridge/contract.py`

现在的 `Pi05ActionContract` 不再写死 `absolute_pose_gripper`，而是：

- 默认 `representation="ee_delta_pose_gripper"`
- 根据 representation 自动推导：
  - `target_dim`
  - `pose_slice`
  - `gripper_index`
  - `rotation_mode`
  - `description`

因此当前仓库已经具备：

- 主线：`ee_delta_pose_gripper`
- 对照基线：`absolute_pose_gripper`

这一步很关键，因为它把“我们到底在训什么动作语义”从口头约定变成了显式 contract。

### 7.3 bridge dataset 已不再直接 passthrough raw absolute action

已修改：

- `rlft/offline/pi05_bridge/dataset_bridge.py`

现在 `build_pi05_dataset_bridge(...)` / `Pi05LeRobotDatasetBridge` 在加载 episode 时，会：

1. 先用 `create_carm_obs_process_fn(...)` 得到 `ee_pose`
2. 再调用 `transform_carm_raw_action_sequence(...)`
3. 把 raw absolute action 转成 contract 指定的目标 action

也就是说，bridge smoke/validation 看到的 action 已经和导出/训练主线一致，不再是“烟囱式的第二套语义”。

### 7.4 LeRobot 导出已经切到“按 contract 落盘”

已修改：

- `rlft/offline/pi05_bridge/export.py`
- `rlft/offline/export_carm_to_lerobot.py`

当前导出逻辑已经不是“把 HDF5 里的 `action` 原样写进 LeRobot dataset”，而是：

1. 先根据 `contract.action.representation` 转换 action
2. 再把转换后的 action 写入 LeRobot dataset

同时 metadata 里额外保留了：

- `raw_action_dim`
- `exported_action_dim`
- `bridge_contract`

这意味着：

- raw 数据仍然可追溯
- 导出后的 dataset 语义也可追溯
- 后续比较 `absolute baseline` 和 `ee_delta mainline` 时不会混淆

### 7.5 `train_pi05.py` 默认项已经改成新的主线

已修改：

- `rlft/offline/train_pi05.py`

当前默认值已经调整为：

- `state_mode = "ee_only"`
- `action_representation = "ee_delta_pose_gripper"`
- `use_official_openpi_checkpoint = True`
- `official_openpi_checkpoint_name = "pi05_base"`
- `use_peft = True`

并且：

- 如果开启 `LoRA/PEFT` 但没有手动给 `policy_pretrained_path`
- 训练入口会自动按 `official_openpi_checkpoint_name` 填默认 checkpoint 路径

这件事直接把你的决策“LoRA first + base/libero mainline”落实到了训练入口层。

### 7.6 OpenPI checkpoint 默认选择已经从 droid 转向 base/libero

已修改：

- `rlft/offline/pi05_bridge/config_bridge.py`
- `rlft/offline/pi05_bridge/__init__.py`
- `rlft/offline/prepare_openpi_pi05_checkpoint.py`
- `rlft/offline/launch_pi05_full_train.py`
- `rlft/offline/probe_pi05_batch_scaling.py`

当前仓库已经新增：

- `DEFAULT_OPENPI_PI05_BASE_PRETRAINED_PATH`
- `DEFAULT_OPENPI_PI05_LIBERO_PRETRAINED_PATH`
- `DEFAULT_OPENPI_PI05_PRETRAINED_PATHS`
- `resolve_default_openpi_pi05_pretrained_path(...)`

同时：

- checkpoint 准备脚本默认从 `pi05_base` 开始
- launcher / batch scaling probe 默认也优先走 `pi05_base`
- 但仍保留 `pi05_libero` 可选

这保证主线不再被 `pi05_droid` 的默认路径绑住。

### 7.7 评估结果现在会记录 bridge contract

已修改：

- `rlft/offline/eval_pi05.py`

现在评估脚本会尝试读取：

- `pi05_bridge_metadata.json`

并把其中的：

- `bridge_contract`

写回评估结果 JSON。

这对于后面做下面这些对照尤其重要：

- `pi05_base + ee_delta`
- `pi05_libero + ee_delta`
- `pi05_base + absolute baseline`

因为单看 `MAE/MSE` 文件名，经常看不出动作语义是否一致。

---

## 8. 这轮还没有做，但已经明确排进下一步的内容

### 8.1 还没有补“官方风格的 Inputs / Outputs / DataConfig”

这轮先做的是：

- **离线把 action 语义改对**
- 并让 `bridge / export / train defaults / eval metadata` 先一致

还没有做的是：

- 自定义 `CarmEeInputs`
- 自定义 `CarmEeOutputs`
- 自定义 `LeRobotCarmEeDataConfig`

也就是说，这轮的策略是：

- **先把数据本身导成目标语义**
- 暂时不在 train-time 再叠一层复杂 transform

这是刻意的，因为对 real robot Cartesian pose 来说，最大的坑首先就是动作语义本身。

### 8.2 还没有补 `wrist_image` 占位通道

当前导出仍然是：

- `observation.image`
- `observation.state`
- `observation.ee_pose`
- `action`

尚未加入：

- `observation.wrist_image`

这是后续可以补的增强项，但我刻意没有在这一轮同时上，因为：

- 当前最高优先级是动作契约校正
- `wrist_image` 会把数据 schema、上游 policy 输入和缺省填零逻辑一起复杂化

### 8.3 还没有实际跑通一轮训练

原因不是代码没落，而是这次执行用的 shell 环境缺少关键依赖：

- 当前环境里没有 `torch`
- 当前环境里也没有 `lerobot`

因此这轮验证我能做的主要是：

- `python -m py_compile ...` 静态语法检查
- 对新加的 `action_transform.py` 做脱离包入口的数值 smoke test

我已经确认：

- `ee_delta_pose_gripper` 的数值转换逻辑在合成样例上是自洽的
- 但真实训练/导出仍需要在装好 `torch + lerobot` 的目标环境里跑

---

## 9. 当前最建议的最小实验矩阵

结合这轮新代码，我建议你按下面顺序做，而不是一口气开很多变量。

### 9.1 先准备两个官方 checkpoint

1. `pi05_base`
2. `pi05_libero`

建议命令：

```bash
python -m rlft.offline.prepare_openpi_pi05_checkpoint --checkpoint_name pi05_base
python -m rlft.offline.prepare_openpi_pi05_checkpoint --checkpoint_name pi05_libero
```

### 9.2 先导出一份新的 delta 主线数据集

建议命令：

```bash
python -m rlft.offline.export_carm_to_lerobot \
  --demo_path ~/rl-vla/recorded_data/mix \
  --output_dir ~/rl-vla/runs/pi05_lerobot_export_delta \
  --state_mode ee_only \
  --action_representation ee_delta_pose_gripper
```

### 9.3 主线 LoRA 先跑两条

A. `pi05_base + ee_delta`

```bash
python -m rlft.offline.train_pi05 \
  --demo_path ~/rl-vla/recorded_data/mix \
  --auto_export_lerobot_dataset true \
  --export_output_dir ~/rl-vla/runs/pi05_lerobot_export_delta \
  --state_mode ee_only \
  --action_representation ee_delta_pose_gripper \
  --official_openpi_checkpoint_name pi05_base \
  --use_peft true
```

B. `pi05_libero + ee_delta`

```bash
python -m rlft.offline.train_pi05 \
  --demo_path ~/rl-vla/recorded_data/mix \
  --auto_export_lerobot_dataset true \
  --export_output_dir ~/rl-vla/runs/pi05_lerobot_export_delta \
  --state_mode ee_only \
  --action_representation ee_delta_pose_gripper \
  --official_openpi_checkpoint_name pi05_libero \
  --use_peft true
```

### 9.4 absolute baseline 只保留一条对照

```bash
python -m rlft.offline.train_pi05 \
  --demo_path ~/rl-vla/recorded_data/mix \
  --auto_export_lerobot_dataset true \
  --export_output_dir ~/rl-vla/runs/pi05_lerobot_export_abs \
  --state_mode ee_only \
  --action_representation absolute_pose_gripper \
  --official_openpi_checkpoint_name pi05_base \
  --use_peft true
```

这三条就足够回答目前最重要的问题：

- `base` 和 `libero` 哪个初始化更适合你的数据
- 把 action 从 `absolute_pose` 拉回 `ee_delta` 之后，离线指标和闭环行为是否明显更稳

---

## 10. 我对当前仓库状态的最新判断

截至这一轮施工结束，我对当前仓库的判断更新为：

1. **主线方向已经转正**
   - 从代码默认项上，已经不再是 `pi05_droid + absolute pose`
   - 而是 `pi05_base + ee_delta + LoRA first`

2. **最危险的 mismatch 已被直接削弱**
   - 之前最危险的是 action contract mismatch
   - 现在 bridge/export 默认已经回到 delta 语义

3. **但还没有到“完全官方风格对齐”**
   - 还缺 `Inputs / Outputs / DataConfig`
   - 还缺 `wrist_image` 占位策略
   - 还缺真实训练环境下的端到端验证

也就是说，当前状态不是“全部完成”，而是：

- **最关键的第一刀已经切对**
- 后续进入的是“验证与细化”阶段，而不再是“主方向摇摆”阶段
