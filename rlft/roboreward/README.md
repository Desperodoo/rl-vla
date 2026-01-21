# RoboReward: 机器人操作 Reward 标注工具

基于 [RoboReward-8B](https://huggingface.co/teetone/RoboReward-8B) (Qwen3-VL-8B-Instruct) 的机器人操作奖励标注工具，用于给 HDF5 格式的机械臂数据集自动打 Reward 标签。

## 功能特性

- 🎯 **自动评分**：基于视觉语言模型对机器人操作视频进行 1-5 分的离散评分
- 📁 **批量处理**：支持批量处理整个数据集目录
- 🔄 **帧采样**：智能关键帧采样，减少计算开销
- 💾 **无损输出**：在新目录生成带 reward 的 HDF5 文件，不修改原始数据
- 📊 **统计报告**：自动生成标注摘要和分数分布统计

## 评分标准

| 分数 | 含义 | 描述 |
|:---:|:---|:---|
| 1 | No Success | 最终状态没有任何与目标相关的变化 |
| 2 | Minimal Progress | 有微小但不充分的进展 |
| 3 | Partial Completion | 有良好进展但违反多项要求 |
| 4 | Near Completion | 区域和意图正确，但遗漏次要要求 |
| 5 | Perfect Completion | 完美完成所有要求 |

## 环境要求

- **GPU**: NVIDIA GPU (RTX 3090 24GB 或更高)
- **CUDA**: 12.1+
- **Python**: 3.10+
- **依赖**: PyTorch 2.1+, Transformers (源码版), Flash Attention (可选)

## 安装

### 方式一：使用安装脚本（推荐）

```bash
cd rlft/roboreward
chmod +x setup_env.sh
./setup_env.sh roboreward  # 创建名为 roboreward 的 conda 环境
```

### 方式二：手动安装

```bash
# 创建虚拟环境
conda create -n roboreward python=3.10 -y
conda activate roboreward

# 安装 PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装 transformers (源码版本，支持 Qwen3-VL)
pip install git+https://github.com/huggingface/transformers

# 安装其他依赖
pip install h5py numpy Pillow tqdm qwen-vl-utils accelerate

# [可选] 安装 Flash Attention (提升推理速度)
pip install flash-attn --no-build-isolation
```

## 使用方法

### 基本用法

```bash
# 激活环境
conda activate roboreward

# 批量标注（统一任务描述）
python batch_label.py \
    --input-dir ../../recorded_data/mix \
    --task "pick up the object and place it in the target area"
```

### 高级选项

```bash
# 指定输出目录和采样帧数
python batch_label.py \
    --input-dir ../../recorded_data/mix \
    --output-dir ../../recorded_data/mix_with_reward \
    --task "grasp the red cube" \
    --sample-frames 16 \
    --verbose

# 使用任务描述文件（为不同 episode 指定不同任务）
python batch_label.py \
    --input-dir ../../recorded_data/mix \
    --task-file task_descriptions.json

# Dry run 模式（仅扫描，不推理）
python batch_label.py \
    --input-dir ../../recorded_data/mix \
    --task "test task" \
    --dry-run
```

### 命令行参数

| 参数 | 短写 | 说明 | 默认值 |
|:---|:---:|:---|:---|
| `--input-dir` | `-i` | 输入数据目录 | (必需) |
| `--task` | `-t` | 统一任务描述 | (与 task-file 二选一) |
| `--task-file` | `-tf` | 任务描述 JSON 文件 | (与 task 二选一) |
| `--output-dir` | `-o` | 输出目录 | `<input>_with_reward` |
| `--model` | `-m` | 模型路径 | `teetone/RoboReward-8B` |
| `--sample-frames` | `-sf` | 采样帧数 | 8 |
| `--sample-method` | | 采样方法 (`uniform`/`keyframe`) | `keyframe` |
| `--dtype` | | 模型精度 | `bfloat16` |
| `--no-flash-attn` | | 禁用 Flash Attention | False |
| `--verbose` | `-v` | 详细输出 | False |
| `--dry-run` | | 仅扫描不推理 | False |

### 任务描述文件格式

创建 `task_descriptions.json` 文件：

```json
{
    "default": "pick up the object and place it in the target area",
    "episode_patterns": {
        "episode_00*": "grasp the red cube",
        "episode_01*": "push the blue block"
    },
    "episodes": {
        "episode_0001": "specific task for episode 0001",
        "episode_0005": "pick up the green sphere"
    }
}
```

## 输出格式

### 目录结构

```
recorded_data/
├── mix/                      # 原始数据（不修改）
│   ├── episode_0001.hdf5
│   ├── episode_0002.hdf5
│   └── ...
└── mix_with_reward/          # 带 reward 的新数据
    ├── episode_0001.hdf5     # 添加了 reward 属性
    ├── episode_0002.hdf5
    ├── ...
    └── reward_summary.json   # 标注摘要
```

### HDF5 新增属性

标注后的 HDF5 文件会新增以下属性：

```python
# 使用 h5py 读取
with h5py.File('episode_0001.hdf5', 'r') as f:
    reward = f.attrs['reward']              # int: 1-5
    model = f.attrs['reward_model']         # str: "RoboReward-8B"
    raw_output = f.attrs['reward_raw_output']  # str: 模型原始输出
```

### 摘要文件

`reward_summary.json` 包含：

```json
{
    "timestamp": "2026-01-20T10:30:00",
    "input_dir": "./recorded_data/mix",
    "output_dir": "./recorded_data/mix_with_reward",
    "model": "teetone/RoboReward-8B",
    "sample_frames": 8,
    "task_description": "pick up the object",
    "statistics": {
        "total": 49,
        "score_distribution": {"1": 5, "2": 10, "3": 15, "4": 12, "5": 7},
        "mean_score": 3.12
    },
    "episodes": [
        {"name": "episode_0001.hdf5", "reward": 4, "task": "...", ...},
        ...
    ]
}
```

## Python API

```python
from rlft.roboreward import RoboRewardLabeler, DatasetConverter, RoboRewardConfig

# 配置
config = RoboRewardConfig(
    sample_frames=8,
    use_flash_attention=True,
)

# 初始化
labeler = RoboRewardLabeler(config)
converter = DatasetConverter(config)

# 加载 episode 帧
frames, metadata = converter.load_episode_frames("episode_0001.hdf5")

# 评分
score = labeler.score_episode(frames, "pick up the red cube")
print(f"Reward: {score}")

# 保存带 reward 的文件
converter.save_episode_with_reward(
    src_hdf5_path="episode_0001.hdf5",
    dst_hdf5_path="output/episode_0001.hdf5",
    reward=score
)
```

## 常见问题

### Q: Flash Attention 安装失败

Flash Attention 需要 CUDA 编译环境，如果安装失败：

```bash
# 确保有 CUDA toolkit
nvcc --version

# 安装编译依赖
pip install ninja packaging

# 重新安装
pip install flash-attn --no-build-isolation
```

如果仍然失败，可以使用 `--no-flash-attn` 选项禁用，会自动使用 SDPA 替代。

### Q: 显存不足

RTX 3090 (24GB) 应该足够。如果出现 OOM：

1. 减少 `--sample-frames` (如从 8 改为 4)
2. 使用 `--dtype float16` 替代 bfloat16
3. 确保没有其他程序占用 GPU 显存

### Q: Qwen3-VL 导入失败

需要安装最新版 transformers (源码版)：

```bash
pip uninstall transformers
pip install git+https://github.com/huggingface/transformers
```

## 参考

- [RoboReward Paper](https://arxiv.org/abs/2601.00675)
- [RoboReward-8B Model](https://huggingface.co/teetone/RoboReward-8B)
- [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
