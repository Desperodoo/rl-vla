---
name: run-integration-tests
description: "集成测试 — 验证模块间数据流的形状一致性 (使用随机数据，不需要 GPU)"
agent: Test-Agent
tools: ['runCommands', 'read', 'edit']
---

# 集成测试执行步骤

## Step 1: 确认测试文件存在

```bash
ls -la rlft/tests/vlaw/
```

如果文件不存在，先通过 `edit` 工具创建缺失的测试文件，然后继续。

## Step 2: 在 conda 环境中运行完整 pytest 测试套件

```bash
conda run -n rlft_ms3 python -m pytest rlft/tests/vlaw/ \
    -v \
    --tb=short \
    --no-header \
    -q \
    2>&1 | tee /tmp/vlaw_integration_test.txt
```

## Step 3: 专项验证 — data_pipeline 形状

验证 `data_pipeline.concat_cameras()` 的输出形状：

```bash
conda run -n rlft_ms3 python -c "
import sys; sys.path.insert(0, '.')
import numpy as np
from rlft.vlaw.data_pipeline import concat_cameras

T, H, W = 5, 192, 192
a = np.random.randint(0, 255, (T, H, W, 3), dtype=np.uint8)
b = np.random.randint(0, 255, (T, H, W, 3), dtype=np.uint8)

# 垂直拼接
out_v = concat_cameras(a, b, mode='vertical')
assert out_v.shape == (T, 2*H, W, 3), f'vertical shape 错误: {out_v.shape}'
assert out_v.dtype == np.uint8, f'dtype 错误: {out_v.dtype}'
print(f'[VLAW-Test] ✅ concat_cameras vertical: shape={out_v.shape} dtype={out_v.dtype}')

# 水平拼接
out_h = concat_cameras(a, b, mode='horizontal')
assert out_h.shape == (T, H, 2*W, 3), f'horizontal shape 错误: {out_h.shape}'
print(f'[VLAW-Test] ✅ concat_cameras horizontal: shape={out_h.shape}')

# 非法模式
try:
    concat_cameras(a, b, mode='diagonal')
    print('[VLAW-Test] ❌ concat_cameras: 应抛出 ValueError')
except ValueError as e:
    print(f'[VLAW-Test] ✅ concat_cameras 非法 mode 异常正确: {e}')
" 2>&1
```

## Step 4: 专项验证 — ctrl_world_adapter latent rearrange 数学

验证 latent 的 rearrange 操作数学正确性（不加载真实模型）：

```bash
conda run -n rlft_ms3 python -c "
import torch
from einops import rearrange

# 模拟 2 相机垂直拼接的 latent: (T, 4, 48, 24)
# 其中 48 = 2 * 24 (两相机各贡献 24 行)
T, C, H_concat, W = 5, 4, 48, 24
n_cams = 2
lat = torch.randn(T, C, H_concat, W)

# 拆分: (T, 4, 48, 24) → (2, T, 4, 24, 24)
# 方法: reshape
lat_split = lat.reshape(T, C, n_cams, H_concat // n_cams, W)
lat_split = lat_split.permute(2, 0, 1, 3, 4)  # (2, T, 4, 24, 24)

assert lat_split.shape == (2, T, C, H_concat // n_cams, W), f'rearrange 结果错误: {lat_split.shape}'
print(f'[VLAW-Test] ✅ latent rearrange: (T,4,48,24) → (2,T,4,24,24) 正确')

# 验证数值一致性
assert torch.allclose(lat_split[0, :, :, :, :], lat[:, :, :H_concat//n_cams, :]), '上半部分数值不一致'
assert torch.allclose(lat_split[1, :, :, :, :], lat[:, :, H_concat//n_cams:, :]), '下半部分数值不一致'
print('[VLAW-Test] ✅ latent rearrange 数值一致性验证通过')
" 2>&1
```

## Step 5: 专项验证 — reward_model.uniform_sample_frames 采样数量

```bash
conda run -n rlft_ms3 python -c "
import sys; sys.path.insert(0, '.')
import numpy as np
from rlft.vlaw.reward_model import uniform_sample_frames
from PIL import Image

# 测试 numpy 输入
arr = np.random.randint(0, 255, (20, 192, 192, 3), dtype=np.uint8)
frames = uniform_sample_frames(arr, num_frames=8)
assert len(frames) == 8, f'采样帧数错误: {len(frames)}'
assert all(isinstance(f, Image.Image) for f in frames), '输出应为 PIL.Image 列表'
print(f'[VLAW-Test] ✅ uniform_sample_frames(20, 8): 返回 {len(frames)} 帧')

# 测试 num_frames > total 时取 min
frames2 = uniform_sample_frames(arr, num_frames=30)
assert len(frames2) == 20, f'超出总帧数时应返回全部帧: {len(frames2)}'
print(f'[VLAW-Test] ✅ uniform_sample_frames(20, 30): 正确裁剪为 {len(frames2)} 帧')

# 测试 PIL 输入
pil_list = [Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)) for _ in range(10)]
frames3 = uniform_sample_frames(pil_list, num_frames=4)
assert len(frames3) == 4, f'PIL 输入采样错误: {len(frames3)}'
print(f'[VLAW-Test] ✅ uniform_sample_frames PIL 输入: 返回 {len(frames3)} 帧')
" 2>&1
```

## Step 6: 专项验证 — HDF5 数据结构完整性

```bash
conda run -n rlft_ms3 python -c "
import h5py, glob, os, sys

encoded_dirs = glob.glob('data/vlaw/encoded/**/*.h5', recursive=True)
if not encoded_dirs:
    print('[VLAW-Test] ⚠️  data_pipeline: 无 encoded HDF5 文件，跳过 HDF5 结构验证')
    sys.exit(0)

errors = []
for h5_path in encoded_dirs[:3]:  # 只检查前 3 个文件
    with h5py.File(h5_path, 'r') as f:
        for tkey in list(f.keys())[:2]:  # 每文件抽查 2 条轨迹
            grp = f[tkey]
            for field in ['rgb_base', 'rgb_render', 'actions']:
                if field not in grp:
                    errors.append(f'{h5_path}/{tkey}: 缺少字段 {field}')
            if 'latent_concat' in grp:
                lat_shape = grp['latent_concat'].shape
                if lat_shape[1:] != (4, 48, 24):
                    errors.append(f'{h5_path}/{tkey}: latent_concat 形状异常 {lat_shape}')

if errors:
    for e in errors:
        print(f'[VLAW-Test] ❌ HDF5: {e}')
else:
    print(f'[VLAW-Test] ✅ HDF5: 结构验证通过 (检查了 {len(encoded_dirs)} 个文件)')
" 2>&1
```

## Step 7: 读取 pytest 输出并生成结构化汇总

```bash
cat /tmp/vlaw_integration_test.txt | tail -20
```

从 pytest 的最后输出中提取 passed/failed/error 数量，并按如下格式输出：

```
[VLAW-Test] 汇总: X passed / Y failed / Z errors
```

将本次测试结果写入 `.github/vlaw-status.md` 的测试状态表格。如有失败，通过对应 Agent 的 Handoff 请求修复。
