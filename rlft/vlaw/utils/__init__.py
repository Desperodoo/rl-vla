"""rlft.vlaw.utils — 工具脚本模块

包含:
    validate_rgb_data ← validate_rgb_data.py (数据验证脚本)
    imagination       ← imagination.py       (旧版 Imagination 引擎，仅供参考)

注意: imagination 模块依赖 Ctrl-World 重型依赖，按需显式导入:
    from rlft.vlaw.utils.imagination import ImaginationEngine
"""

# 工具模块不做自动 star-import，避免加载重型依赖
# 使用方按需显式导入：
#   from rlft.vlaw.utils.validate_rgb_data import ...
#   from rlft.vlaw.utils.imagination import ImaginationConfig, ImaginationEngine
