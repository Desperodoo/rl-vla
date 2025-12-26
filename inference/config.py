"""
rl-vla 路径配置模块

自动检测项目根目录并配置所有子模块路径。
支持环境变量覆盖。

使用方法:
    from config import RL_VLA_CONFIG
    
    # 导入 arx5 接口
    RL_VLA_CONFIG.setup_arx5()
    import arx5_interface as arx5
    
    # 导入 rlft 模块
    RL_VLA_CONFIG.setup_rlft()
    from diffusion_policy.algorithms import FlowMatchingAgent
"""

import os
import sys
from pathlib import Path
from typing import Optional


class RLVLAConfig:
    """rl-vla 项目配置类"""
    
    _instance: Optional['RLVLAConfig'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._arx5_setup_done = False
        self._rlft_setup_done = False
        
        # 自动检测项目根目录
        self.root = self._find_root()
        
        # 子项目路径
        self.arx5_sdk_path = Path(os.environ.get(
            'ARX5_SDK_PATH', 
            self.root / 'arx5-sdk'
        ))
        self.rlft_path = Path(os.environ.get(
            'RLFT_PATH',
            self.root / 'rlft'
        ))
        self.inference_path = Path(os.environ.get(
            'INFERENCE_PATH',
            self.root / 'inference'
        ))
        self.lerobot_path = self.root / 'LeRobot-Anything-U-Arm'
    
    def _find_root(self) -> Path:
        """自动检测 rl-vla 根目录"""
        # 优先使用环境变量
        if 'RL_VLA_ROOT' in os.environ:
            return Path(os.environ['RL_VLA_ROOT'])
        
        # 从当前文件位置向上查找
        current = Path(__file__).resolve().parent
        
        # 如果在 inference/ 目录下
        if current.name == 'inference':
            return current.parent
        
        # 如果在 scripts/ 目录下
        if current.name == 'scripts':
            return current.parent
        
        # 向上查找包含 arx5-sdk 的目录
        for parent in [current] + list(current.parents):
            if (parent / 'arx5-sdk').exists():
                return parent
        
        # 兜底: 使用家目录下的 rl-vla
        fallback = Path.home() / 'rl-vla'
        if fallback.exists():
            return fallback
        
        raise RuntimeError(
            "无法找到 rl-vla 项目根目录!\n"
            "请设置环境变量: export RL_VLA_ROOT=/path/to/rl-vla\n"
            "或者运行: source scripts/setup_env.sh"
        )
    
    def setup_arx5(self):
        """配置 arx5-sdk 路径"""
        if self._arx5_setup_done:
            return
        
        python_path = self.arx5_sdk_path / 'python'
        
        if not python_path.exists():
            raise RuntimeError(
                f"arx5-sdk Python 目录不存在: {python_path}\n"
                f"请先运行编译脚本: ./scripts/build_all.sh"
            )
        
        # 添加到 sys.path
        path_str = str(python_path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
        
        # 设置工作目录 (arx5 需要从其目录运行)
        os.chdir(python_path)
        
        # 设置库路径
        import platform
        arch = 'aarch64' if platform.machine() == 'aarch64' else 'x86_64'
        lib_path = self.arx5_sdk_path / 'lib' / arch
        
        current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        if str(lib_path) not in current_ld_path:
            os.environ['LD_LIBRARY_PATH'] = f"{lib_path}:{current_ld_path}"
        
        self._arx5_setup_done = True
    
    def setup_rlft(self):
        """配置 rlft 路径"""
        if self._rlft_setup_done:
            return
        
        diffusion_path = self.rlft_path / 'diffusion_policy'
        
        if not diffusion_path.exists():
            raise RuntimeError(
                f"rlft diffusion_policy 目录不存在: {diffusion_path}"
            )
        
        # 添加到 sys.path
        path_str = str(diffusion_path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
        
        self._rlft_setup_done = True
    
    def setup_all(self):
        """配置所有路径"""
        self.setup_arx5()
        self.setup_rlft()
    
    def get_model_path(self, urdf_name: str) -> Path:
        """获取 URDF 模型路径"""
        return self.arx5_sdk_path / 'models' / urdf_name
    
    def __repr__(self):
        return (
            f"RLVLAConfig(\n"
            f"  root={self.root}\n"
            f"  arx5_sdk={self.arx5_sdk_path}\n"
            f"  rlft={self.rlft_path}\n"
            f"  inference={self.inference_path}\n"
            f")"
        )


# 全局单例
RL_VLA_CONFIG = RLVLAConfig()


def setup_arx5():
    """便捷函数: 配置 arx5-sdk"""
    RL_VLA_CONFIG.setup_arx5()


def setup_rlft():
    """便捷函数: 配置 rlft"""
    RL_VLA_CONFIG.setup_rlft()


def setup_all():
    """便捷函数: 配置所有模块"""
    RL_VLA_CONFIG.setup_all()


if __name__ == '__main__':
    print(RL_VLA_CONFIG)
    print()
    print("测试 arx5-sdk 导入...")
    setup_arx5()
    import arx5_interface as arx5
    print("  ✓ arx5_interface 导入成功")
    
    print()
    print("测试 rlft 导入...")
    setup_rlft()
    from diffusion_policy.algorithms import FlowMatchingAgent
    print("  ✓ FlowMatchingAgent 导入成功")
