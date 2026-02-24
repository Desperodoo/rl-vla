---
name: 'Python General Standards'
description: '项目通用 Python 编码规范'
applyTo: '**/*.py'
---

# Python 编码规范

- Python 3.10+，所有函数签名使用 type hints
- 使用 `dataclass` + `tyro` 管理配置（不用 argparse）
- 日志使用 `wandb`，不用 tensorboard
- 训练框架: PyTorch 2.x
- 多 GPU: HuggingFace Accelerate 或 PyTorch DDP
- 数据格式: HDF5 (轨迹), safetensors (模型)
- 环境: conda `rlft_ms3`
- 使用 `pathlib.Path` 管理文件路径
- import 顺序: stdlib → third-party → local
