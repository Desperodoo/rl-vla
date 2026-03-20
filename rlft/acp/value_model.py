"""ACP value model（Phase P6.B1）

Pistar06 value model 封装：SigLIP + Gemma + distributional value head。
移植自 Evo-RL modeling_pistar06.py，去除 LeRobot 依赖，适配 VLAW HDF5 schema。
双相机各自独立编码后 mean-pool，不做竖拼。
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from rlft.acp.config import ValueModelConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Distributional value utilities（从 modeling_pistar06.py 移植）
# ---------------------------------------------------------------------------


def build_bin_centers(
    num_bins: int,
    bin_min: float,
    bin_max: float,
    device: torch.device | None = None,
) -> Tensor:
    """构建等间距 bin 中心点。"""
    return torch.linspace(bin_min, bin_max, num_bins, dtype=torch.float32, device=device)


def project_values_to_bins(values: Tensor, bin_centers: Tensor) -> Tensor:
    """将连续 value 投影到 bin 上的 soft target 分布。

    Args:
        values: (B,) float — 连续 value
        bin_centers: (num_bins,) float — bin 中心

    Returns:
        (B, num_bins) float — soft target 分布
    """
    values = values.clamp(min=bin_centers[0], max=bin_centers[-1])
    step = bin_centers[1] - bin_centers[0]
    scaled = (values - bin_centers[0]) / step
    low = torch.floor(scaled).long()
    high = torch.clamp(low + 1, max=bin_centers.shape[0] - 1)
    high_weight = (scaled - low.float()).clamp(0.0, 1.0)
    low_weight = 1.0 - high_weight

    target = torch.zeros(values.shape[0], bin_centers.shape[0], device=values.device, dtype=torch.float32)
    target.scatter_add_(1, low.unsqueeze(1), low_weight.unsqueeze(1))
    target.scatter_add_(1, high.unsqueeze(1), high_weight.unsqueeze(1))
    return target


def expected_value_from_logits(logits: Tensor, bin_centers: Tensor) -> Tensor:
    """从 logits 计算期望 value。"""
    probs = F.softmax(logits, dim=-1)
    return (probs * bin_centers).sum(dim=-1)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _resolve_load_dtype(dtype_name: str) -> torch.dtype:
    requested = torch.bfloat16 if dtype_name == "bfloat16" else torch.float32
    if requested == torch.bfloat16 and not torch.cuda.is_available():
        return torch.float32
    return requested


def _freeze_module(module: nn.Module) -> None:
    module.eval()
    for p in module.parameters():
        p.requires_grad = False


def _maybe_enable_gradient_checkpointing(module: nn.Module) -> None:
    if hasattr(module, "gradient_checkpointing_enable"):
        module.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )


def _unfreeze_vision_top_layers(vision_encoder: nn.Module, top_n: int) -> None:
    """解冻 SigLIP 视觉编码器顶部 N 层 transformer + attention pooler head。

    SigLIP-so400m 有 27 层（index 0-26）。例如 top_n=4 解冻 layers 23-26 + head。

    Args:
        vision_encoder: 已冻结的 SigLIP 视觉编码器
        top_n: 要解冻的顶部 transformer 层数
    """
    # 计算总层数
    total_layers = 0
    for name, _ in vision_encoder.named_parameters():
        if "encoder.layers." in name:
            idx = int(name.split("encoder.layers.")[1].split(".")[0])
            total_layers = max(total_layers, idx + 1)

    if total_layers == 0:
        logger.warning("未检测到 encoder.layers，跳过部分解冻")
        return

    start_layer = total_layers - top_n
    unfrozen_count = 0
    for name, param in vision_encoder.named_parameters():
        should_unfreeze = False
        # 解冻 encoder.layers.{start_layer..total_layers-1}
        if "encoder.layers." in name:
            layer_idx = int(name.split("encoder.layers.")[1].split(".")[0])
            if layer_idx >= start_layer:
                should_unfreeze = True
        # 解冻 attention pooler head
        if "head." in name:
            should_unfreeze = True
        if should_unfreeze:
            param.requires_grad = True
            unfrozen_count += 1

    logger.info(
        f"部分解冻 SigLIP: top {top_n}/{total_layers} layers + head, "
        f"解冻 {unfrozen_count} 个参数张量"
    )


# ---------------------------------------------------------------------------
# Pistar06 Model（核心 nn.Module）
# ---------------------------------------------------------------------------


class Pistar06Model(nn.Module):
    """SigLIP + Gemma + distributional value head。

    输入：多相机 RGB 图像 + 任务文本
    输出：value logits (B, num_bins)

    多相机通过 mean-pool 融合（不做竖拼）。
    """

    def __init__(self, cfg: ValueModelConfig) -> None:
        super().__init__()
        from transformers import AutoConfig, AutoImageProcessor, AutoModel, AutoModelForCausalLM

        self.cfg = cfg
        self.model_dtype = _resolve_load_dtype(cfg.dtype)

        # ---- Vision encoder (SigLIP) ----
        # 只加载 vision tower，不需要 text encoder
        vision_config = AutoConfig.from_pretrained(cfg.vision_repo_id)
        if hasattr(vision_config, "vision_config"):
            # 完整 SigLIP config → 用 SiglipVisionModel 只加载 vision tower
            from transformers import SiglipVisionModel

            self.vision_encoder = SiglipVisionModel.from_pretrained(
                cfg.vision_repo_id,
                dtype=self.model_dtype,
            )
        else:
            # 已经是 vision-only config
            self.vision_encoder = AutoModel.from_pretrained(
                cfg.vision_repo_id,
                dtype=self.model_dtype,
            )
        image_processor = AutoImageProcessor.from_pretrained(
            cfg.vision_repo_id, use_fast=True,
        )

        # SigLIP 期望的图像分辨率和归一化参数
        size = getattr(image_processor, "size", None)
        if isinstance(size, dict) and "height" in size:
            self.image_resolution = (int(size["height"]), int(size["width"]))
        elif isinstance(size, int):
            self.image_resolution = (size, size)
        else:
            self.image_resolution = (384, 384)

        mean_raw = getattr(image_processor, "image_mean", [0.5, 0.5, 0.5])
        std_raw = getattr(image_processor, "image_std", [0.5, 0.5, 0.5])
        self.register_buffer(
            "image_mean",
            torch.tensor(mean_raw, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(std_raw, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        # ---- Language model (Gemma) ----
        model_config = AutoConfig.from_pretrained(cfg.language_repo_id)
        architectures = getattr(model_config, "architectures", None) or []
        prefer_causal = any(
            isinstance(a, str) and a.endswith("ForCausalLM") for a in architectures
        )
        if prefer_causal:
            lm_with_head = AutoModelForCausalLM.from_pretrained(
                cfg.language_repo_id, dtype=self.model_dtype,
            )
            self.language_model = lm_with_head.model
        else:
            self.language_model = AutoModel.from_pretrained(
                cfg.language_repo_id, dtype=self.model_dtype,
            )

        # ---- 推断 hidden sizes ----
        vision_feature_size = self._infer_hidden_size(self.vision_encoder, "vision")
        language_hidden_size = self._infer_hidden_size(self.language_model, "language")

        # ---- Projection + value head ----
        self.image_projector = nn.Sequential(
            nn.Linear(vision_feature_size, cfg.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.language_projector = nn.Sequential(
            nn.Linear(language_hidden_size, cfg.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.final_norm = nn.LayerNorm(cfg.fusion_hidden_dim * 2)
        self.value_head = nn.Sequential(
            nn.Linear(cfg.fusion_hidden_dim * 2, cfg.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_hidden_dim, cfg.num_bins),
        )

        # ---- Freeze / gradient checkpointing ----
        if cfg.use_gradient_checkpointing:
            _maybe_enable_gradient_checkpointing(self.language_model)
            _maybe_enable_gradient_checkpointing(self.vision_encoder)
        if cfg.freeze_vision_encoder:
            _freeze_module(self.vision_encoder)
            # 部分解冻：顶部 N 层 + attention pooler head
            if cfg.unfreeze_vision_top_n > 0:
                _unfreeze_vision_top_layers(
                    self.vision_encoder, cfg.unfreeze_vision_top_n,
                )
        if cfg.freeze_language_model:
            _freeze_module(self.language_model)

    @staticmethod
    def _infer_hidden_size(model: nn.Module, label: str) -> int:
        config = getattr(model, "config", None)
        if config is None:
            raise ValueError(f"Cannot infer hidden size for {label}: missing .config")
        if hasattr(config, "projection_dim"):
            return int(config.projection_dim)
        if hasattr(config, "vision_config") and hasattr(config.vision_config, "projection_dim"):
            return int(config.vision_config.projection_dim)
        if hasattr(config, "hidden_size"):
            return int(config.hidden_size)
        if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
            return int(config.text_config.hidden_size)
        raise ValueError(f"Cannot infer hidden size for {label}")

    def _encode_images(self, flat_images: Tensor) -> Tensor:
        if hasattr(self.vision_encoder, "get_image_features"):
            return self.vision_encoder.get_image_features(pixel_values=flat_images)
        outputs = self.vision_encoder(pixel_values=flat_images, return_dict=True)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state.mean(dim=1)
        raise ValueError("Unsupported vision encoder output")

    def _encode_language(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        token_mask = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)
        denom = token_mask.sum(dim=1).clamp_min(1.0)
        return (hidden * token_mask).sum(dim=1) / denom

    def forward(
        self,
        images: Tensor,
        image_mask: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """前向传播。

        Args:
            images: (B, N_cam, C, H, W) float — 多相机 RGB（0-255 uint8 或 0-1 float）
            image_mask: (B, N_cam) bool — 有效相机 mask
            input_ids: (B, T_tok) long — Gemma tokenizer 输出
            attention_mask: (B, T_tok) long — token attention mask

        Returns:
            (B, num_bins) float — value logits
        """
        B, N_cam = images.shape[:2]

        # ---- Image preprocessing: resize + normalize ----
        images_f = images.to(dtype=torch.float32)
        if images_f.max() > 1.0:
            images_f = images_f / 255.0

        flat = images_f.reshape(B * N_cam, *images_f.shape[2:])
        if flat.shape[-2:] != self.image_resolution:
            flat = F.interpolate(flat, size=self.image_resolution, mode="bilinear", align_corners=False)

        mean = self.image_mean.to(device=flat.device, dtype=flat.dtype)
        std = self.image_std.to(device=flat.device, dtype=flat.dtype)
        flat = (flat - mean) / std
        flat = flat.to(dtype=self.model_dtype)

        # ---- Encode images ----
        ctx_v = torch.no_grad() if self.cfg.freeze_vision_encoder else nullcontext()
        with ctx_v:
            img_feat = self._encode_images(flat)  # (B*N_cam, D_v)

        # ---- Encode language ----
        ctx_l = torch.no_grad() if self.cfg.freeze_language_model else nullcontext()
        with ctx_l:
            lang_feat = self._encode_language(input_ids, attention_mask)  # (B, D_l)

        # ---- Project + mean-pool cameras ----
        # Projector 和 value head 始终在 float32 运算（trainable 部分）
        img_feat = img_feat.float()
        lang_feat = lang_feat.float()

        img_tokens = self.image_projector(img_feat).view(B, N_cam, -1)  # (B, N, D_f)
        cam_mask = image_mask.unsqueeze(-1).to(dtype=img_tokens.dtype)  # (B, N, 1)
        img_tokens = img_tokens * cam_mask
        denom = cam_mask.sum(dim=1).clamp_min(1.0)
        img_pooled = img_tokens.sum(dim=1) / denom  # (B, D_f)

        lang_token = self.language_projector(lang_feat)  # (B, D_f)

        # ---- Fuse + value head ----
        joint = torch.cat([img_pooled, lang_token], dim=-1)  # (B, 2*D_f)
        return self.value_head(self.final_norm(joint))  # (B, num_bins)


# ---------------------------------------------------------------------------
# Value Model 高层封装
# ---------------------------------------------------------------------------


class ManiSkillValueModel:
    """面向 VLAW pipeline 的 value model 高层 API。

    封装 Pistar06Model + tokenizer + bin_centers，提供简洁的
    predict_values / compute_loss 接口。

    Args:
        cfg: ValueModelConfig
        device: 'cuda:0' 等
    """

    def __init__(self, cfg: ValueModelConfig, device: str = "cuda:0") -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.model = Pistar06Model(cfg).to(self.device)

        # Tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.language_repo_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Bin centers
        self.bin_centers = build_bin_centers(
            cfg.num_bins, cfg.bin_min, cfg.bin_max, device=self.device
        )

        # 预编码 task instruction
        self._task_tokens = self._tokenize(cfg.task_instruction)

    def _tokenize(self, text: str) -> dict[str, Tensor]:
        tokens = self.tokenizer(
            text,
            max_length=self.cfg.tokenizer_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in tokens.items()}

    def predict_values(self, images: Tensor, image_mask: Tensor) -> Tensor:
        """批量预测 value。

        Args:
            images: (B, N_cam, C, H, W) — RGB 图像
            image_mask: (B, N_cam) bool — 有效相机

        Returns:
            (B,) float32 — predicted values
        """
        B = images.shape[0]
        input_ids = self._task_tokens["input_ids"].expand(B, -1)
        attn_mask = self._task_tokens["attention_mask"].expand(B, -1)

        self.model.eval()
        amp_dtype = torch.bfloat16 if self.cfg.dtype == "bfloat16" else None
        ctx = torch.cuda.amp.autocast(dtype=amp_dtype) if amp_dtype else nullcontext()
        with torch.no_grad(), ctx:
            logits = self.model(
                images=images.to(self.device),
                image_mask=image_mask.to(self.device),
                input_ids=input_ids,
                attention_mask=attn_mask,
            )
        return expected_value_from_logits(logits, self.bin_centers)

    def compute_loss(
        self,
        images: Tensor,
        image_mask: Tensor,
        value_targets: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        """计算 distributional cross-entropy loss。

        Args:
            images: (B, N_cam, C, H, W)
            image_mask: (B, N_cam) bool
            value_targets: (B,) float32 — GT value targets

        Returns:
            (loss, metrics_dict)
        """
        B = images.shape[0]
        input_ids = self._task_tokens["input_ids"].expand(B, -1)
        attn_mask = self._task_tokens["attention_mask"].expand(B, -1)

        logits = self.model(
            images=images.to(self.device),
            image_mask=image_mask.to(self.device),
            input_ids=input_ids,
            attention_mask=attn_mask,
        )

        targets = value_targets.to(self.device, dtype=torch.float32)
        soft_target = project_values_to_bins(targets, self.bin_centers)
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(soft_target * log_probs).sum(dim=-1).mean()

        pred_value = expected_value_from_logits(logits, self.bin_centers)
        value_mae = (pred_value - targets).abs().mean()

        metrics = {
            "loss": float(loss.detach().item()),
            "value_mae": float(value_mae.detach().item()),
        }
        return loss, metrics

    def trainable_parameters(self) -> list[nn.Parameter]:
        """返回可训练参数列表（冻结 backbone 后）."""
        return [p for p in self.model.parameters() if p.requires_grad]

    def save(self, path: str | Path) -> None:
        """保存可训练参数（不含冻结 backbone）."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # 按 requires_grad 判断（支持部分解冻场景）
        trainable_keys = {
            name for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        state = {k: v for k, v in self.model.state_dict().items() if k in trainable_keys}
        from safetensors.torch import save_file
        save_file(state, str(path))
        logger.info(f"Saved value model ({len(state)} tensors) to {path}")

    def _is_frozen(self, key: str) -> bool:
        """检查参数是否冻结（基于 requires_grad，支持部分解冻）。"""
        for name, param in self.model.named_parameters():
            if name == key:
                return not param.requires_grad
        # state_dict 可能包含 buffer（非 parameter），视为冻结
        return True

    def load(self, path: str | Path) -> None:
        """加载 checkpoint（strict=False 以兼容 partial save）."""
        path = Path(path)
        from safetensors.torch import load_file
        state = load_file(str(path), device=str(self.device))
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        logger.info(
            f"Loaded value model from {path} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
