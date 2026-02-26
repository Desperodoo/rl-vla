"""rlft.vlaw.world_model — 世界模型相关模块

包含:
    ctrl_world_adapter ← ctrl_world_adapter.py (Ctrl-World 推理封装 P2.1)
    imagination_env    ← imagination_env.py    (env.step() Imagination P4.3)

导入路径:
    from rlft.vlaw.world_model import CtrlWorldAdapter
"""

# 惰性导入：Ctrl-World 依赖 einops / ctrl_world 环境，不在 rlft_ms3 中强制安装。
# 直接使用时请从子模块显式导入：
#   from rlft.vlaw.world_model.ctrl_world_adapter import CtrlWorldAdapter
#   from rlft.vlaw.world_model.imagination_env import ImaginationEnvConfig

def __getattr__(name: str):  # noqa: ANN001
    if name in ("CtrlWorldAdapter",):
        from .ctrl_world_adapter import CtrlWorldAdapter
        return CtrlWorldAdapter
    if name in ("ImaginationEnvConfig", "ImaginationEnvEngine"):
        from .imagination_env import ImaginationEnvConfig, ImaginationEnvEngine  # noqa: PLC0415
        return {"ImaginationEnvConfig": ImaginationEnvConfig,
                "ImaginationEnvEngine": ImaginationEnvEngine}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "CtrlWorldAdapter",
    "ImaginationEnvConfig",
    "ImaginationEnvEngine",
]
