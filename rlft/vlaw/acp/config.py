"""ACP 配置模块（Phase P6.A1）

所有 ACP（Advantage-Conditioned Policy）相关配置，tyro dataclass 格式。
源自 Evo-RL Pistar06Config，适配 VLAW ManiSkill3 仿真环境。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Value Model 配置
# ---------------------------------------------------------------------------


@dataclass
class ValueModelConfig:
    """Pistar06 value model 架构与预训练权重配置。

    SigLIP 视觉编码器 + Gemma 语言模型 + distributional value head (201 bins)。
    双相机分别编码后 mean-pool，不做竖拼。
    """

    # Backbone 预训练路径（本地目录或 HuggingFace repo_id）
    vision_repo_id: str = "checkpoints/vlaw/acp/pretrained/siglip"
    """SigLIP 预训练权重路径（google/siglip-so400m-patch14-384）"""

    language_repo_id: str = "checkpoints/vlaw/acp/pretrained/gemma"
    """Gemma 语言模型预训练路径（google/gemma-3-270m）"""

    # 输入规格
    camera_keys: list[str] = field(default_factory=lambda: ["rgb_base", "rgb_render"])
    """HDF5 中的相机 key 列表，各自独立输入 SigLIP"""

    image_size: tuple[int, int] = (128, 128)
    """HDF5 中原始图像分辨率 (H, W)；SigLIP 内部会 resize 到 384x384"""

    task_instruction: str = "Pick up the peg and lift it upright."
    """任务文本描述，输入 Gemma tokenizer"""

    # 架构超参
    num_bins: int = 201
    """Distributional value head 离散 bin 数量"""

    bin_min: float = -1.0
    """Value 范围下界"""

    bin_max: float = 0.0
    """Value 范围上界"""

    fusion_hidden_dim: int = 512
    """Vision-Language fusion 隐层维度"""

    fusion_num_layers: int = 2
    """Fusion transformer 层数"""

    fusion_num_heads: int = 8
    """Fusion multi-head attention 头数"""

    dropout: float = 0.1
    """Projection / value head dropout"""

    dtype: str = "bfloat16"
    """模型运算精度 ('float32' 或 'bfloat16')\n    bfloat16 推荐：frozen backbone 推理安全且 ~2x 加速"""

    # Backbone 冻结策略
    freeze_vision_encoder: bool = True
    """冻结 SigLIP backbone（只训练 projector + value head）"""

    freeze_language_model: bool = True
    """冻结 Gemma backbone"""

    unfreeze_vision_top_n: int = 0
    """解冻 SigLIP 顶部 N 层 transformer（0 = 全冻结）。
    Evo-RL 默认全部可训练。部分解冻（如 4-8 层）是单卡训练的折中方案。
    仅在 freeze_vision_encoder=True 时生效。"""

    use_gradient_checkpointing: bool = False
    """启用 gradient checkpointing 节省显存"""

    tokenizer_max_length: int = 200
    """Gemma tokenizer 最大 token 长度"""


# ---------------------------------------------------------------------------
# Value Target 配置
# ---------------------------------------------------------------------------


@dataclass
class ValueTargetConfig:
    """Per-frame value target 计算参数。

    target = clip((-remaining_steps - c_fail * (1-success)) / (max_len + c_fail), -1, 0)
    """

    c_fail_coef: float = 1.0
    """失败轨迹惩罚系数。乘以 max_episode_len 得到 c_fail"""

    clip_min: float = -1.0
    """Value target 下界"""

    clip_max: float = 0.0
    """Value target 上界"""

    success_key: str = "env_success"
    """HDF5 中 success 信号的来源。
    - 'env_success': 仿真环境 GT，per-frame (T,) bool dataset
    - 'vlm_success': VLM reward model 标注，per-trajectory scalar attribute（int 0/1）
    读取逻辑自动适配两种格式。"""


# ---------------------------------------------------------------------------
# Advantage 配置
# ---------------------------------------------------------------------------


@dataclass
class AdvantageConfig:
    """N-step advantage 计算与二值化参数。"""

    n_step: int = 4
    """N-step return 步数"""

    positive_ratio: float = 0.3
    """Per-task quantile 阈值：top-k% 帧标记为 positive"""

    use_continuous_weights: bool = True
    """True: advantage 归一化为 [0,1] 连续权重；False: 二值 indicator"""

    weight_clip_min: float = 0.0
    """连续权重下界"""

    weight_clip_max: float = 5.0
    """连续权重上界（防止极端值）"""


# ---------------------------------------------------------------------------
# 训练配置
# ---------------------------------------------------------------------------


@dataclass
class ACPTrainConfig:
    """ACP value model 训练配置。"""

    # 路径
    data_dirs: list[str] = field(
        default_factory=lambda: ["data/vlaw/rollouts/mixed"]
    )
    """HDF5 训练数据目录列表"""

    output_dir: str = "checkpoints/vlaw/acp/iter1"
    """输出 checkpoint 目录"""

    # Value model
    value_model: ValueModelConfig = field(default_factory=ValueModelConfig)
    """Value model 架构配置"""

    value_target: ValueTargetConfig = field(default_factory=ValueTargetConfig)
    """Value target 计算参数"""

    # 训练超参
    num_steps: int = 8000
    """总训练步数"""

    batch_size: int = 32
    """Mini-batch 大小"""

    learning_rate: float = 5e-5
    """AdamW 峰值学习率"""

    weight_decay: float = 1e-5
    """AdamW weight decay"""

    grad_clip_norm: float = 10.0
    """梯度裁剪 max norm"""

    warmup_steps: int = 500
    """线性 warmup 步数"""

    lr_min: float = 0.0
    """最小学习率（cosine 衰减下限）。Evo-RL 使用 1e-6。"""

    eval_interval: int = 200
    """每 N 步做一次验证"""

    save_interval: int = 1000
    """每 N 步保存 checkpoint"""

    val_split: float = 0.1
    """验证集比例"""

    # 运行时
    num_workers: int = 4
    """DataLoader worker 数"""

    seed: int = 42
    """随机种子"""

    use_wandb: bool = True
    """是否启用 wandb"""

    wandb_run_name: str = "acp_value_iter1"
    """wandb run 名称"""


# ---------------------------------------------------------------------------
# 推理配置
# ---------------------------------------------------------------------------


@dataclass
class ACPInferConfig:
    """ACP advantage 推理与标注配置。"""

    # 路径
    checkpoint_path: str = "checkpoints/vlaw/acp/iter1/best.safetensors"
    """Value model checkpoint 路径"""

    data_dirs: list[str] = field(
        default_factory=lambda: ["data/vlaw/rollouts/mixed"]
    )
    """待标注的 HDF5 数据目录列表"""

    # Value model（推理用，需与训练一致）
    value_model: ValueModelConfig = field(default_factory=ValueModelConfig)
    """Value model 架构配置"""

    # Target & advantage
    value_target: ValueTargetConfig = field(default_factory=ValueTargetConfig)
    """Value target 计算参数"""

    advantage: AdvantageConfig = field(default_factory=AdvantageConfig)
    """Advantage 计算参数"""

    # 运行时
    batch_size: int = 64
    """推理 batch 大小"""

    num_workers: int = 4
    """DataLoader worker 数"""

    seed: int = 42
    """随机种子"""

    write_back: bool = True
    """是否将标注结果写回 HDF5"""
