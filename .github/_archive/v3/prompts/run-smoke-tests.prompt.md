---
name: run-smoke-tests
description: "快速冒烟测试 — 验证已完成模块的接口不崩溃"
agent: Eval-Agent
tools: ['runCommands', 'read']
---

# 冒烟测试

1. 读取 `.github/vlaw-status.md`，找出 ✅ 模块
2. 对每个模块运行轻量导入+基本断言（无 GPU 依赖）:
   - `data_pipeline`: `concat_cameras` shape 验证, `PipelineConfig` 无硬编码路径
   - `reward_model`: `VLAWRewardConfig` + `uniform_sample_frames` 采样数正确
   - `ctrl_world_adapter`: 语法 `ast.parse` 通过
   - `data_collector`: `DataCollectorConfig` 可实例化
3. 检查目录: `data/vlaw/{rollouts,encoded,synthetic}`, `checkpoints/vlaw/`
4. 检查测试文件: `rlft/tests/vlaw/{conftest,test_data_pipeline,test_shapes}.py`
5. 输出: `[VLAW-Test] 汇总: X passed / Y failed / Z skipped`

> 测试代码示例见 `rlft/tests/vlaw/`，不在此 prompt 中内联。
