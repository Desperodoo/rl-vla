---
name: run-integration-tests
description: "集成测试 — 验证模块间数据流的形状一致性 (随机数据，无 GPU)"
agent: Eval-Agent
tools: ['runCommands', 'read', 'edit']
---

# 集成测试

1. 运行 pytest: `conda run -n rlft_ms3 python -m pytest rlft/tests/vlaw/ -v --tb=short -q`
2. 专项验证（在 pytest 外逐一确认）:
   - `concat_cameras`: vertical (T,384,192,3) + horizontal (T,192,384,3) + 非法 mode ValueError
   - `latent rearrange`: (T,4,48,24) → (2,T,4,24,24) 数学正确性 + 数值一致
   - `uniform_sample_frames`: 采样数正确 + num_frames>total 时取 min + PIL 输入
   - `HDF5 结构`: encoded/ 下前 3 文件 x 2 条轨迹，检查字段存在 + latent shape (4,48,24)
3. 输出: `[VLAW-Test] 汇总: X passed / Y failed / Z errors`

> 具体测试代码在 `rlft/tests/vlaw/test_*.py`，不在此 prompt 中内联。
