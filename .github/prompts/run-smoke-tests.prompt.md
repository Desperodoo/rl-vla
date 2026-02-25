---
name: run-smoke-tests
description: "快速冒烟测试 — 验证已完成模块的接口不崩溃"
agent: Test-Agent
tools: ['runCommands', 'read']
---

# 冒烟测试执行步骤

## Step 1: 读取项目状态

读取 `.github/vlaw-status.md`，找出所有标记为 ✅ 已完成的模块，跳过 ⬜/🔄/❌ 状态的模块。

```bash
cat .github/vlaw-status.md
```

## Step 2: 检查前置条件

在运行任何模块测试前，先验证：

```bash
# 2.1 检查 conda 环境存在
conda env list | grep rlft_ms3

# 2.2 检查核心依赖可导入
conda run -n rlft_ms3 python -c "
import torch, h5py, numpy, PIL, einops
print('[VLAW-Test] ✅ 核心依赖: torch=%s, h5py=%s' % (torch.__version__, h5py.__version__))
"

# 2.3 检查 rlft 包可导入
conda run -n rlft_ms3 python -c "
import sys; sys.path.insert(0, '.')
import rlft; print('[VLAW-Test] ✅ rlft 包导入 OK')
"
```

如果 `rlft_ms3` 不存在，输出：
```
[VLAW-Test] ❌ 环境: rlft_ms3 未找到 — 请先运行 conda env create
```
并跳过后续所有测试。

## Step 3: 对每个已完成模块运行冒烟测试

对 `.github/vlaw-status.md` 中每个 ✅ 模块，依次运行：

### data_pipeline (P1.2)

```bash
conda run -n rlft_ms3 python -c "
import sys; sys.path.insert(0, '.')
from rlft.vlaw.data_pipeline import concat_cameras, PipelineConfig
import numpy as np
# 测试 concat_cameras 不崩溃
a = np.zeros((5, 192, 192, 3), dtype=np.uint8)
out = concat_cameras(a, a, mode='vertical')
assert out.shape == (5, 384, 192, 3), f'shape 错误: {out.shape}'
cfg = PipelineConfig()
assert cfg.vae_local_path == '', 'vae_local_path 不应有硬编码路径'
print('[VLAW-Test] ✅ data_pipeline: 冒烟测试通过')
" 2>&1
```

### reward_model (P3.1)

```bash
conda run -n rlft_ms3 python -c "
import sys; sys.path.insert(0, '.')
from rlft.vlaw.reward_model import VLAWRewardConfig, uniform_sample_frames
import numpy as np
cfg = VLAWRewardConfig()
arr = np.random.randint(0, 255, (20, 192, 192, 3), dtype=np.uint8)
frames = uniform_sample_frames(arr, num_frames=8)
assert len(frames) == 8, f'采样帧数错误: {len(frames)}'
print('[VLAW-Test] ✅ reward_model: 冒烟测试通过')
" 2>&1
```

### ctrl_world_adapter (P2/P4)

```bash
conda run -n rlft_ms3 python -c "
import sys; sys.path.insert(0, '.')
import ast
code = open('rlft/vlaw/ctrl_world_adapter.py').read()
ast.parse(code)
print('[VLAW-Test] ✅ ctrl_world_adapter: 语法检查通过')
" 2>&1
```

### data_collector (P1.1)

```bash
conda run -n rlft_ms3 python -c "
import sys; sys.path.insert(0, '.')
from rlft.vlaw.data_collector import DataCollectorConfig
cfg = DataCollectorConfig()
print('[VLAW-Test] ✅ data_collector: 冒烟测试通过')
" 2>&1
```

## Step 4: 检查关键目录和文件结构

```bash
# 数据目录
for d in data/vlaw/rollouts data/vlaw/encoded data/vlaw/synthetic checkpoints/vlaw; do
    if [ -d "$d" ]; then
        echo "[VLAW-Test] ✅ 目录存在: $d"
    else
        echo "[VLAW-Test] ⚠️  目录缺失: $d"
    fi
done

# 测试文件
for f in rlft/tests/vlaw/conftest.py rlft/tests/vlaw/test_data_pipeline.py rlft/tests/vlaw/test_shapes.py; do
    if [ -f "$f" ]; then
        echo "[VLAW-Test] ✅ 测试文件存在: $f"
    else
        echo "[VLAW-Test] ⚠️  测试文件缺失: $f"
    fi
done
```

## Step 5: 输出汇总

统计上述每个测试的 pass/fail/skip 数量，按格式输出：

```
[VLAW-Test] 汇总: X passed / Y failed / Z skipped
```

将结果更新到 `.github/vlaw-status.md` 的测试状态表格中。
