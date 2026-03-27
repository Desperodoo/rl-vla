# archive/ — 旧版扁平模块归档

此目录存放从 `rlft/vlaw/` 根目录迁移的原始扁平模块文件。

## 迁移原因
代码重构后，功能已迁移到对应子包（权威来源）：
- `data/` — 数据收集与管线
- `world_model/` — Ctrl-World 适配与 Imagination
- `reward/` — VLM 奖励模型
- `policy/` — 策略更新与状态预测
- `utils/` — 工具脚本

## 归档文件列表
| 文件 | 权威路径 | 归档日期 |
|------|---------|---------|
| ctrl_world_adapter.py | rlft/vlaw/world_model/ctrl_world_adapter.py | 2026-02-25 |
| demo_prep.py | rlft/vlaw/data/demo_prep.py | 2026-02-25 |
| train_reward_model.py | rlft/vlaw/reward/train_reward_model.py | 2026-02-25 |
| validate_rgb_data.py | rlft/vlaw/utils/validate_rgb_data.py | 2026-02-25 |
| imagination.py | rlft/vlaw/utils/imagination.py | 2026-02-25 |
| imagination_env.py | rlft/vlaw/world_model/imagination_env.py | 2026-02-25 |
| test_reward_model.py | rlft/tests/vlaw/test_reward_model.py | 2026-02-25 |

## 注意
- **请勿直接导入 archive 目录中的文件**，使用子包路径
- 向后兼容旧路径由 `rlft/vlaw/*.py` shim 文件处理
- archive 目录仅供历史参考，不参与任何 import 路径
