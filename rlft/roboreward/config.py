"""
RoboReward 配置管理

提供模型路径、设备、采样参数等配置项。
"""

from dataclasses import dataclass, field
from typing import Optional, List
import os


@dataclass
class RoboRewardConfig:
    """RoboReward 配置类"""
    
    # 模型配置
    model_name_or_path: str = "teetone/RoboReward-8B"
    torch_dtype: str = "bfloat16"  # "float16", "bfloat16", "float32"
    device_map: str = "auto"
    use_flash_attention: bool = True
    
    # 推理配置
    max_new_tokens: int = 128
    do_sample: bool = False  # greedy decoding for deterministic results
    
    # 视频采样配置
    sample_frames: int = -1  # 从每个episode采样的帧数，-1 表示使用所有帧
    min_frames: int = 4      # 最小帧数（仅当采样时生效）
    max_frames: int = 1024    # 最大帧数限制（防止OOM）
    
    # 数据路径配置
    input_data_dir: str = ""   # 输入数据目录
    output_data_dir: str = ""  # 输出数据目录（带reward的HDF5）
    
    # 任务描述配置
    default_task_description: str = "complete the manipulation task"
    task_descriptions_file: Optional[str] = None  # 可选的任务描述文件路径
    
    # 批处理配置
    batch_size: int = 1  # VLM一般只支持batch_size=1
    num_workers: int = 4
    
    # 日志配置
    verbose: bool = True
    save_visualization: bool = False
    
    def __post_init__(self):
        """后处理配置"""
        # 自动设置输出目录
        if self.input_data_dir and not self.output_data_dir:
            parent_dir = os.path.dirname(self.input_data_dir.rstrip('/'))
            input_name = os.path.basename(self.input_data_dir.rstrip('/'))
            self.output_data_dir = os.path.join(parent_dir, f"{input_name}_with_reward")
    
    @classmethod
    def from_args(cls, args) -> "RoboRewardConfig":
        """从 argparse 参数创建配置"""
        return cls(
            model_name_or_path=getattr(args, 'model', cls.model_name_or_path),
            torch_dtype=getattr(args, 'dtype', cls.torch_dtype),
            use_flash_attention=getattr(args, 'flash_attn', cls.use_flash_attention),
            sample_frames=getattr(args, 'sample_frames', cls.sample_frames),
            input_data_dir=getattr(args, 'input_dir', cls.input_data_dir),
            output_data_dir=getattr(args, 'output_dir', cls.output_data_dir),
            default_task_description=getattr(args, 'task', cls.default_task_description),
            task_descriptions_file=getattr(args, 'task_file', cls.task_descriptions_file),
            verbose=getattr(args, 'verbose', cls.verbose),
        )


# RoboReward 评分的 Prompt 模板
REWARD_PROMPT_TEMPLATE = """Given the task, assign a discrete progress score reward (1,2,3,4,5) for the robot in the video in the format: ANSWER: <score>
Rubric for end-of-episode progress (judge only the final state without time limits):
1 - No Success: Final state shows no goal-relevant change for the command.
2 - Minimal Progress: Final state shows a small but insufficient change toward the goal.
3 - Partial Completion: The final state shows good progress toward the goal but violates more than one requirement or a major requirement.
4 - Near Completion: Final state is correct in region and intent but misses a single minor requirement.
5 - Perfect Completion: Final state satisfies all requirements.

Task: {task}"""


# 评分含义说明
SCORE_DESCRIPTIONS = {
    1: "No Success - 无任务相关变化",
    2: "Minimal Progress - 微小但不充分的进展",
    3: "Partial Completion - 有进展但违反多项要求",
    4: "Near Completion - 区域和意图正确，但遗漏次要要求",
    5: "Perfect Completion - 完美完成所有要求",
}
