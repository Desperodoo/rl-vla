# carm_ros_deploy 在线测试计划

> **分支**: `carm_clean`  
> **前置**: 项目结构重构完成, catkin_make 编译通过, 129/129 离线测试通过  
> **目标**: 连接真实机械臂，完整验证 deploy 包所有功能

---

## 0. 环境准备

### 0.1 硬件检查清单

| 项目 | 检查内容 | 状态 |
|------|----------|------|
| 机械臂 | 通电, 网络 `10.42.0.101` 可 ping 通 | ☐ |
| RealSense 相机 | USB 连接, `rs-enumerate-devices` 可识别 (S/N: `218622279840`) | ☐ |
| 工作台 | 机械臂运动范围内无障碍物 | ☐ |
| 急停按钮 | 可触及, 功能正常 | ☐ |

### 0.2 软件环境

```bash
# 1. 激活 conda 环境
conda activate carm

# 2. source ROS 工作空间
source ~/rl-vla/carm_ros_deploy/devel/setup.bash

# 3. 验证包可用
rospack find carm_deploy
# 期望输出: /home/lizh/rl-vla/carm_ros_deploy/src/carm_deploy

# 4. 验证 SDK (不连机械臂也能验证 import)
python -c "import carm_py; print('SDK OK')"

# 5. 验证路径配置
python -c "from carm_deploy.utils.paths import *; print(f'ROOT={RL_VLA_ROOT}')"
# 期望输出: ROOT=/home/lizh/rl-vla

# 6. 确认 safety_config.json 存在
ls -la ~/rl-vla/carm_ros_deploy/src/carm_deploy/safety_config.json
```

### 0.3 ROS Master

```bash
# 终端 0: 启动 roscore（若无其他 launch 文件自动启动）
roscore
```

---

## 1. record_workspace — 录制安全工作空间

### 1.1 目的

通过拖拽示教录制机械臂工作空间边界，生成 `safety_config.json`。本次测试目的是**验证工具是否正常工作**，已有 `safety_config.json` 可跳过此步。

### 1.2 命令

```bash
cd ~/rl-vla/carm_ros_deploy/src/carm_deploy

python tools/record_workspace.py \
    --output ~/rl-vla/safety_config_test.json \
    --robot_ip 10.42.0.101 \
    --margin 0.05
```

### 1.3 操作步骤

| 步骤 | 操作 | 预期响应 |
|------|------|----------|
| 1 | 脚本启动 | 打印当前关节角度, 显示按键提示 |
| 2 | 按 `Enter` | 机械臂进入 DRAG 模式, 可自由拖拽 |
| 3 | 拖拽机械臂 | 终端实时打印 xyz 边界更新 |
| 4 | 按 `P` | 打印当前录制的工作空间范围 |
| 5 | 按 `R` | 重置边界记录 (可选) |
| 6 | 拖够后按 `S` | 保存 `safety_config_test.json` 并退出 |

### 1.4 验证

```bash
# 检查生成的配置文件
cat ~/rl-vla/safety_config_test.json | python -m json.tool

# 必须包含以下 key:
# - joint_limits (joint_min, joint_max, gripper_min, gripper_max)
# - workspace_limits (x_min, x_max, y_min, y_max, z_min, z_max)
# - safety_params (max_joint_delta, max_gripper_delta, ...)
# - metadata (created_at, robot_ip, sample_count, margin)
```

### 1.5 通过标准

- [ ] 机械臂成功进入 DRAG 模式
- [ ] 拖拽过程中边界实时更新
- [ ] 生成的 JSON 文件格式正确, 包含所有必要字段
- [ ] 边界值合理 (xyz 范围 > 0.1m, 样本数 > 100)

---

## 2. verify_safety_config — 验证安全配置

### 2.1 目的

验证已有的 `safety_config.json` 是否与当前机械臂状态匹配。

### 2.2 测试模式 A: 静态检查 (check)

```bash
python tools/verify_safety_config.py \
    --config ~/rl-vla/carm_ros_deploy/src/carm_deploy/safety_config.json \
    --test_mode check \
    --robot_ip 10.42.0.101
```

**预期输出**: 打印当前关节角 & 末端位姿, 判定是否在安全范围内。

### 2.3 测试模式 B: 可视化 (visual) — 可选

```bash
python tools/verify_safety_config.py \
    --config ~/rl-vla/carm_ros_deploy/src/carm_deploy/safety_config.json \
    --test_mode visual \
    --robot_ip 10.42.0.101
```

**预期输出**: 绘制工作空间范围图 (3D 框), 标注当前位置。

### 2.4 测试模式 C: 边界测试 (boundary) — ⚠️ 需谨慎

```bash
python tools/verify_safety_config.py \
    --config ~/rl-vla/carm_ros_deploy/src/carm_deploy/safety_config.json \
    --test_mode boundary \
    --robot_ip 10.42.0.101
```

**⚠️ 注意**: 此模式会**实际控制机械臂运动到各轴边界**。
- 确保工作空间内无障碍物
- 随时准备急停
- 使用 MIT 模式 (mode=2), 运动速度较慢

**操作流程**:
1. 脚本提示确认 → 按 Enter 开始
2. 机械臂依次运动到: X_min → X_max → Y_min → Y_max → Z_min → Z_max → 中心位置
3. 最后测试夹爪开合

### 2.5 通过标准

- [ ] `check` 模式正确报告当前位置与安全范围的关系
- [ ] `visual` 模式生成工作空间可视化图 (若有 display)
- [ ] `boundary` 模式机械臂安全运动到各边界并返回 (可选)

---

## 3. record_data — 遥操作数据录制

### 3.1 目的

使用拖拽示教录制训练数据，验证相机图像 + 机械臂状态的同步采集。

### 3.2 启动相机 + 录制

```bash
# 方法 A: launch 文件一键启动 (推荐)
roslaunch carm_deploy record.launch \
    output_dir:=$HOME/rl-vla/recorded_data_test \
    use_camera:=true \
    vis:=true

# 方法 B: 手动分步启动
# 终端 1: 启动相机
roslaunch carm_deploy camera.launch

# 终端 2: 启动录制 (等相机 topic 就绪)
rosrun carm_deploy record_data_ros.py \
    --output_dir ~/rl-vla/recorded_data_test \
    --vis
```

### 3.3 操作步骤

| 步骤 | 操作 | 预期响应 |
|------|------|----------|
| 1 | 脚本启动 | 连接机械臂, 等待相机图像, 进入 DRAG 模式 |
| 2 | 确认初始化 | 按 `Enter` 完成 env 初始化 |
| 3 | 按 `S` | 开始第一集录制，终端显示 "Recording started" |
| 4 | 拖拽机械臂 | 实时显示帧计数, `vis=true` 时弹出图像窗口 |
| 5 | 按 `S` | 停止录制，终端显示已录帧数 |
| 6 | 按 `Y` | 保存当前 episode，自动编号 |
| 7 | (可选) 按 `N` | 丢弃当前 episode |
| 8 | 重复 3-6 | 录 2-3 集用于后续分析 |
| 9 | 按 `Q` | 退出录制 |

### 3.4 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `robot_mode` | 3 (DRAG) | 拖拽模式, 不主动控制 |
| `record_freq` | 30 Hz | 录制频率 |
| `max_steps` | 1000 | 单集最大步数 |
| `image_width/height` | 320×240 | 保存图像尺寸 |

### 3.5 验证

```bash
# 检查输出目录
ls ~/rl-vla/recorded_data_test/
# 期望: episode_0.hdf5, episode_1.hdf5, ...

# 快速检查 HDF5 内容
python -c "
import h5py
with h5py.File('$HOME/rl-vla/recorded_data_test/episode_0.hdf5', 'r') as f:
    def print_tree(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f'  {name}: shape={obj.shape}, dtype={obj.dtype}')
    print('HDF5 结构:')
    f.visititems(print_tree)
"
# 期望:
#   observations/images: shape=(T, 240, 320, 3)
#   observations/qpos_joint: shape=(T, 7)
#   observations/qpos_end: shape=(T, 8)
#   action: shape=(T, 14)
```

### 3.6 通过标准

- [ ] 相机图像正常订阅 (无超时报错)
- [ ] `vis=true` 窗口显示实时图像
- [ ] 拖拽过程中帧计数稳定递增
- [ ] HDF5 文件包含完整数据结构
- [ ] 图像尺寸与配置一致 (320×240)
- [ ] 关节角 & 末端位姿数据非零且合理
- [ ] 多集录制编号正确递增

---

## 4. analyze_dataset — 分析采集数据

### 4.1 目的

对刚采集的数据集进行完整性检查和统计分析。

### 4.2 命令

```bash
python data/analyze_dataset.py \
    --data_dir ~/rl-vla/recorded_data_test \
    --all
```

`--all` 启用: 基础统计 + 完整性检查 + 可视化 + 图像导出 + montage

### 4.3 预期输出

1. **终端**: 每集步数、action 维度、关节范围统计
2. **文件**: `dataset_info.json` — 完整数据集报告
3. **图表** (若有 display):
   - 关节角度轨迹
   - 夹爪状态分布
   - 样本图像

### 4.4 验证

```bash
# 检查报告文件
cat ~/rl-vla/recorded_data_test/dataset_info.json | python -m json.tool
```

### 4.5 通过标准

- [ ] 无完整性错误 (缺失字段 / NaN / 维度不匹配)
- [ ] 关节角范围在安全限位内
- [ ] 图像尺寸一致
- [ ] `dataset_info.json` 正确生成

---

## 5. offline_test — 离线模型评估

### 5.1 目的

用已录制数据集验证模型推理管线，**无需连接机械臂**。

### 5.2 前置条件

- 需要训练好的 checkpoint (`.pt` 文件)
- 需要同目录下的 `args.json` (训练配置)
- 可选: `action_normalizer.json`

### 5.3 命令

```bash
# 需要切到有 torch 的环境
conda activate carm

python tools/offline_test.py \
    --model_path /path/to/checkpoint.pt \
    --data_dir ~/rl-vla/recorded_data_test \
    --output_dir ~/rl-vla/offline_eval_test \
    --num_episodes 2 \
    --use_ema \
    --device cuda:0
```

### 5.4 可选: EMA 对比评估

```bash
python tools/offline_test.py \
    --model_path /path/to/checkpoint.pt \
    --data_dir ~/rl-vla/recorded_data_test \
    --output_dir ~/rl-vla/offline_eval_test \
    --compare_ema \
    --device cuda:0
```

### 5.5 预期输出

1. **终端**: 每集 MSE/MAE, 关节误差 & 末端误差分离报告
2. **图表**: 轨迹对比图 (预测 vs 真实), 误差分布直方图
3. **保存**: 评估结果到 `output_dir`

### 5.6 通过标准

- [ ] 模型加载成功 (checkpoint + args.json + normalizer)
- [ ] 逐步推理无异常
- [ ] MSE 误差在合理范围 (与训练时验证集指标一致)
- [ ] 可视化图表正确生成

---

## 6. inference_ros — 在线推理 (含键盘干预)

### 6.1 目的

**核心测试**: 加载策略模型, 实时控制机械臂。这是最关键的功能验证。

### 6.2 ⚠️ 安全注意事项

> **强制要求**: 
> - `safety_config.json` 必须存在且已验证 (步骤 2)
> - 默认使用 PF 位力混合模式 (mode=4), **严禁使用 Position 模式**
> - 随时准备急停按钮
> - 第一次运行时 `teleop_scale` 建议设为较小值 (如 0.5)

### 6.3 方法 A: full_system.launch (推荐)

```bash
# 一键启动: 相机 + 推理
roslaunch carm_deploy full_system.launch \
    pretrain:=/path/to/checkpoint.pt \
    vis:=true
```

### 6.4 方法 B: 分步启动

```bash
# 终端 1: 启动相机
roslaunch carm_deploy camera.launch

# 终端 2: 启动推理
roslaunch carm_deploy inference.launch \
    pretrain:=/path/to/checkpoint.pt \
    vis:=true
```

### 6.5 方法 C: 直接运行 (更多控制参数)

```bash
# 终端 1: 相机
roslaunch carm_deploy camera.launch

# 终端 2: 推理 (完整参数)
rosrun carm_deploy inference_ros.py \
    --pretrain /path/to/checkpoint.pt \
    --safety_config ~/rl-vla/carm_ros_deploy/src/carm_deploy/safety_config.json \
    --robot_ip 10.42.0.101 \
    --execution_mode temporal_ensemble \
    --desire_inference_freq 30 \
    --temporal_factor_k 0.05 \
    --control_freq 50 \
    --teleop_scale 1.0 \
    --intervention \
    --record_inference \
    --use_ema \
    --vis
```

### 6.6 推理操作流程

| 步骤 | 操作 | 预期响应 |
|------|------|----------|
| 1 | 脚本启动 | 加载模型, 连接机械臂, 等待相机 |
| 2 | 按 `Enter` | 机械臂运动到初始位姿 `[0.2475, 0.0014, 0.3251, ...]` |
| 3 | 按 `I` | **开始推理**, 终端打印推理频率 |
| 4 | 观察机械臂 | 根据相机输入执行动作, 实时显示图像 (vis=true) |
| 5 | (干预) 按方向键 | `W/S`=X轴, `A/D`=Y轴, `Q/E`=Z轴, `G`=开夹爪, `H`=关夹爪 |
| 6 | 按 `O` | **停止推理**, 机械臂停在当前位置 |
| 7 | 按 `I` 重启 | 可多次开始/停止 |
| 8 | `Ctrl+C` | 安全关闭 (机械臂回零位) |

### 6.7 录制干预数据 (DAgger)

如启用 `--record_inference --intervention`:

| 步骤 | 操作 | 预期响应 |
|------|------|----------|
| 1 | 按 `R` | 开始录制当前 episode |
| 2 | 推理运行 + 键盘干预 | 同时记录模型动作和人工干预动作 |
| 3 | 按 `R` | 停止录制 |
| 4 | 按 `Y` 或 `N` | 保存或丢弃当前录制 |

### 6.8 关键参数调优

| 参数 | 默认值 | 调优建议 |
|------|--------|----------|
| `execution_mode` | `temporal_ensemble` | 若抖动严重可试 `receding_horizon` |
| `temporal_factor_k` | 0.05 | 越大越偏向最新 chunk (减少延迟, 增加不稳定) |
| `teleop_scale` | 1.0 | 首次运行建议 0.5, 确认安全后逐步提高 |
| `pos_lookahead_step` | 1 | 增大可提高跟踪速度, 但可能超调 |
| `num_inference_steps` | (模型默认) | 降低可提高推理频率, 但降低质量 |

### 6.9 通过标准

- [ ] 模型加载成功, 打印模型架构摘要
- [ ] 机械臂正确运动到初始位姿
- [ ] 按 `I` 后推理频率稳定在 ~30Hz
- [ ] 动作执行平滑, 安全裁剪率 < 10%
- [ ] 键盘干预立即响应 (WASD/QE/GH)
- [ ] `vis=true` 图像窗口正常刷新
- [ ] 推理录制数据正常保存
- [ ] `Ctrl+C` 安全关闭, 机械臂回零位
- [ ] 推理日志 (HDF5 + JSON) 自动生成在 `~/rl-vla/inference_logs/`

---

## 7. analyze_inference_data — 分析推理数据

### 7.1 目的

分析步骤 6 中产生的推理日志和干预录制数据。

### 7.2 分析推理日志 (InferenceLogger 输出)

```bash
# 使用 InferenceLogAnalyzer
python -c "
from carm_deploy.inference.inference_logger import InferenceLogAnalyzer
analyzer = InferenceLogAnalyzer('$HOME/rl-vla/inference_logs/inference_XXXXXXXX_XXXXXX.hdf5')
stats = analyzer.get_statistics()
print(stats)
analyzer.generate_report()
"
```

**预期输出**: 总步数、时长、平均推理时间、安全裁剪率、轨迹图、动作对比图 (raw vs executed)

### 7.3 分析干预数据

```bash
python inference/analyze_inference_data.py \
    --data_dir ~/rl-vla/inference_logs \
    --save_dir ~/rl-vla/inference_analysis \
    --no_viz  # 无显示器时使用
```

**预期输出**:
1. 基础统计: 每集步数 / 时长 / 干预率
2. 干预检测: 智能识别干预起止点 (XYZ 位移阈值 + 夹爪变点)
3. 动作对比: model action vs intervened action 差异
4. 3D 轨迹: 标注干预段
5. 多集对比: 汇总表格

### 7.4 分析 timeline 日志

```bash
# 查找 timeline 日志
find ~/rl-vla -name "timeline_*.jsonl" -mmin -60

python tools/analyze_timeline.py \
    --logs ~/rl-vla/inference_logs/timeline_*.jsonl \
    --out ~/rl-vla/inference_analysis/timeline_summary.json \
    --visualize \
    --fig_out ~/rl-vla/inference_analysis/timeline.png
```

**预期输出**: 基础统计 (FPS / 总步数), chunk 重叠分析, 甘特图

### 7.5 通过标准

- [ ] 推理日志 HDF5 可正常加载分析
- [ ] 统计数据合理 (推理时间 < 50ms, 帧率 > 25Hz)
- [ ] 安全裁剪率合理 (< 20%)
- [ ] 干预数据 (若有) 正确检测干预段
- [ ] 图表正常生成

---

## 测试执行总结模板

完成所有测试后，填写下表：

| 序号 | 功能 | 状态 | 备注 |
|------|------|------|------|
| 1 | record_workspace | ☐ Pass / ☐ Fail | |
| 2 | verify_safety_config | ☐ Pass / ☐ Fail | |
| 3 | record_data (遥操作) | ☐ Pass / ☐ Fail | |
| 4 | analyze_dataset | ☐ Pass / ☐ Fail | |
| 5 | offline_test | ☐ Pass / ☐ Fail | |
| 6 | inference_ros (推理+干预) | ☐ Pass / ☐ Fail | |
| 7 | analyze_inference_data | ☐ Pass / ☐ Fail | |

### 已知问题 & 注意事项

1. **相机序列号**: 当前硬编码 `218622279840`, 若更换相机需修改 launch 参数
2. **机械臂 IP**: 默认 `10.42.0.101`, 确保网络接口配置正确
3. **conda 环境**: `carm_py` SDK 仅在 `carm` 环境可用
4. **Position 模式**:  **严禁使用** `robot_mode=1`, 代码已有硬保护
5. **dry-run 建议**: 推理首次运行可先用 `offline_test.py` 确认模型无异常
6. **文件命名**: `analyze_inference_data.py` 默认 pattern 为 `inference_episode_*.hdf5`, 若不匹配需用 `--files` 指定具体文件
