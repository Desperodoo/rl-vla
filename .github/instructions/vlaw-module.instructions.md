---
name: 'VLAW Module Standards'
description: 'rlft/vlaw/ 模块下的编码规范'
applyTo: 'rlft/vlaw/**/*.py'
---

# VLAW 模块编码规范

## 文件结构
- 每个 `.py` 文件必须包含模块级 docstring，说明用途和所属 VLAW 阶段 (P0-P7)
- 所有类和公开函数必须有 type hints 和 docstring

## 配置
- 所有超参数使用 `tyro` dataclass 管理：
```python
@dataclass
class SomeConfig:
    """说明"""
    param: int = 42  # 参数说明
```

## 日志
- 使用 `wandb` 记录训练指标
- 关键步骤使用 `print(f"[VLAW] ...")` 输出进度

## 路径约定
- checkpoint 保存: `checkpoints/vlaw/{module_name}/`
- 数据保存: `data/vlaw/{data_type}/`
- 使用 `pathlib.Path` 管理路径

## GPU 管理
- 10 × RTX 4090 (24GB each)
- 通过 `CUDA_VISIBLE_DEVICES` 环境变量分配
- GPU 0-3: WM 训练, GPU 4-5: 数据收集, GPU 6-7: VLM, GPU 8-9: 策略/评估
- 显存敏感操作使用 `torch.cuda.empty_cache()` 释放

## 数据格式
- 轨迹数据: HDF5 格式
- 模型权重: safetensors (优先) 或 PyTorch `.pt`
- VAE latent: float16

## 接口约定
- 所有模块通过 `rlft/vlaw/__init__.py` 暴露公开接口
- 模块间通过文件 I/O 传递数据 (HDF5, checkpoint)
- 状态更新: 完成任务后更新 `.github/vlaw-status.md`
