"""
ResNet-based Visual Encoder for Robot Imitation Learning

Supports pretrained ResNet18/34/50 with configurable freezing options.
Automatically handles ImageNet normalization internally.

Usage:
    encoder = ResNetEncoder(
        backbone_name='resnet18',
        out_dim=256,
        pretrained=True,
        freeze_backbone=False,
        freeze_bn=True,
    )
    features = encoder(images)  # [B, 3, H, W] -> [B, out_dim]
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Literal, Optional, Tuple


# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ImageNetNormalize(nn.Module):
    """
    Normalize images with ImageNet mean and std.
    Expects input in [0, 1] range.
    """
    def __init__(self):
        super().__init__()
        self.register_buffer('mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
    
    def forward(self, x):
        # x: [B, 3, H, W] in [0, 1] range
        return (x - self.mean) / self.std


class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and affine parameters are fixed.
    
    Copy from torchvision.ops.misc.FrozenBatchNorm2d with minor modifications.
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x):
        # Cast all parameters to the same dtype as input
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

    def __repr__(self):
        return f"{self.__class__.__name__}({self.weight.shape[0]}, eps={self.eps})"


def _convert_bn_to_frozen_bn(module: nn.Module) -> nn.Module:
    """
    Recursively convert all BatchNorm2d layers to FrozenBatchNorm2d.
    """
    module_output = module
    if isinstance(module, nn.BatchNorm2d):
        module_output = FrozenBatchNorm2d(module.num_features, module.eps)
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
    for name, child in module.named_children():
        module_output.add_module(name, _convert_bn_to_frozen_bn(child))
    del module
    return module_output


class ResNetEncoder(nn.Module):
    """
    ResNet-based visual encoder with pretrained weights.
    
    Features:
        - Supports ResNet18, ResNet34, ResNet50
        - Optional backbone freezing for few-shot learning
        - Optional BatchNorm freezing (recommended for small batch sizes)
        - Automatic ImageNet normalization (expects input in [0, 1])
        - Configurable output dimension via projection layer
    
    Args:
        backbone_name: One of 'resnet18', 'resnet34', 'resnet50'
        out_dim: Output feature dimension (default: 256)
        pretrained: Whether to use ImageNet pretrained weights
        freeze_backbone: Whether to freeze all backbone parameters
        freeze_bn: Whether to convert BatchNorm to FrozenBatchNorm
        pool_type: Pooling type ('avg' or 'max')
    
    Input:
        x: [B, 3, H, W] RGB images in [0, 1] range (or [0, 255] if normalize_input=False)
        Recommended size: 224x224 for best performance, but works with any size >= 32
    
    Output:
        features: [B, out_dim] feature vectors
    """
    
    # Recommended input sizes for different backbones
    RECOMMENDED_INPUT_SIZE = {
        'resnet18': (224, 224),
        'resnet34': (224, 224),
        'resnet50': (224, 224),
    }
    
    # Output channels before pooling for each backbone
    BACKBONE_CHANNELS = {
        'resnet18': 512,
        'resnet34': 512,
        'resnet50': 2048,
    }
    
    def __init__(
        self,
        backbone_name: Literal['resnet18', 'resnet34', 'resnet50'] = 'resnet18',
        out_dim: int = 256,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        freeze_bn: bool = True,
        pool_type: Literal['avg', 'max'] = 'avg',
    ):
        super().__init__()
        
        if backbone_name not in self.BACKBONE_CHANNELS:
            raise ValueError(f"Unsupported backbone: {backbone_name}. "
                           f"Choose from {list(self.BACKBONE_CHANNELS.keys())}")
        
        self.backbone_name = backbone_name
        self.out_dim = out_dim
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.freeze_bn = freeze_bn
        
        # ImageNet normalization layer
        self.normalize = ImageNetNormalize()
        
        # Load pretrained backbone
        weights = 'IMAGENET1K_V1' if pretrained else None
        backbone = getattr(models, backbone_name)(weights=weights)
        
        # Get backbone channel count
        self.backbone_channels = self.BACKBONE_CHANNELS[backbone_name]
        
        # Remove classification head (avgpool and fc)
        # Keep: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        
        # Freeze BatchNorm layers if requested
        if freeze_bn:
            self.features = _convert_bn_to_frozen_bn(self.features)
        
        # Freeze backbone parameters if requested
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False
        
        # Pooling layer
        if pool_type == 'avg':
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
        
        # Projection layer to target dimension
        self.projection = nn.Sequential(
            nn.Linear(self.backbone_channels, out_dim),
            nn.ReLU(inplace=True),
        )
        
        # Initialize projection layer
        self._init_projection()
    
    def _init_projection(self):
        """Initialize projection layer weights."""
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images [B, 3, H, W] in [0, 1] range
               (If in [0, 255] range, divide by 255 first)
        
        Returns:
            features: [B, out_dim] feature vectors
        """
        # Normalize with ImageNet statistics
        x = self.normalize(x)
        
        # Extract features
        features = self.features(x)  # [B, C, H', W']
        
        # Pool to fixed size
        pooled = self.pool(features)  # [B, C, 1, 1]
        
        # Flatten and project
        flat = pooled.flatten(1)  # [B, C]
        out = self.projection(flat)  # [B, out_dim]
        
        return out
    
    def get_backbone_params(self):
        """Get backbone parameters (for separate learning rate)."""
        return self.features.parameters()
    
    def get_head_params(self):
        """Get projection head parameters (for separate learning rate)."""
        return self.projection.parameters()
    
    def get_param_groups(self, lr_backbone: float, lr_head: float):
        """
        Get parameter groups with different learning rates.
        
        Args:
            lr_backbone: Learning rate for backbone (typically 1e-5)
            lr_head: Learning rate for projection head (typically 1e-4)
        
        Returns:
            List of parameter groups for optimizer
        """
        return [
            {'params': self.get_backbone_params(), 'lr': lr_backbone},
            {'params': self.get_head_params(), 'lr': lr_head},
        ]
    
    @classmethod
    def get_recommended_input_size(cls, backbone_name: str) -> Tuple[int, int]:
        """Get recommended input image size for a backbone."""
        return cls.RECOMMENDED_INPUT_SIZE.get(backbone_name, (224, 224))
    
    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"backbone={self.backbone_name}, "
                f"out_dim={self.out_dim}, "
                f"pretrained={self.pretrained}, "
                f"freeze_backbone={self.freeze_backbone}, "
                f"freeze_bn={self.freeze_bn})")


# =============================================================================
# Factory function for easy creation
# =============================================================================

def create_visual_encoder(
    encoder_type: str,
    out_dim: int = 256,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    freeze_bn: bool = True,
    in_channels: int = 3,
    pool_feature_map: bool = True,
) -> nn.Module:
    """
    Factory function to create visual encoders.
    
    Args:
        encoder_type: One of 'plain_conv', 'resnet18', 'resnet34', 'resnet50'
        out_dim: Output feature dimension
        pretrained: Whether to use pretrained weights (ResNet only)
        freeze_backbone: Whether to freeze backbone (ResNet only)
        freeze_bn: Whether to freeze BatchNorm (ResNet only)
        in_channels: Input channels (PlainConv only)
        pool_feature_map: Whether to pool feature map (PlainConv only)
    
    Returns:
        Visual encoder module
    """
    if encoder_type == 'plain_conv':
        from diffusion_policy.plain_conv import PlainConv
        return PlainConv(
            in_channels=in_channels,
            out_dim=out_dim,
            pool_feature_map=pool_feature_map,
        )
    elif encoder_type in ['resnet18', 'resnet34', 'resnet50']:
        return ResNetEncoder(
            backbone_name=encoder_type,
            out_dim=out_dim,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            freeze_bn=freeze_bn,
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}. "
                        f"Choose from: plain_conv, resnet18, resnet34, resnet50")


def get_encoder_input_size(encoder_type: str, default_size: Tuple[int, int] = (128, 128)) -> Tuple[int, int]:
    """
    Get recommended input image size for an encoder type.
    
    Args:
        encoder_type: Encoder type string
        default_size: Default size for PlainConv
    
    Returns:
        (height, width) tuple
    """
    if encoder_type == 'plain_conv':
        return default_size
    elif encoder_type in ResNetEncoder.RECOMMENDED_INPUT_SIZE:
        return ResNetEncoder.RECOMMENDED_INPUT_SIZE[encoder_type]
    else:
        return (224, 224)  # Default for unknown types


# =============================================================================
# Test code
# =============================================================================

if __name__ == '__main__':
    import time
    
    print("Testing ResNetEncoder...")
    
    # Test different configurations
    configs = [
        {'backbone_name': 'resnet18', 'pretrained': True, 'freeze_backbone': False, 'freeze_bn': True},
        {'backbone_name': 'resnet18', 'pretrained': True, 'freeze_backbone': True, 'freeze_bn': True},
        {'backbone_name': 'resnet34', 'pretrained': True, 'freeze_backbone': False, 'freeze_bn': True},
        {'backbone_name': 'resnet50', 'pretrained': True, 'freeze_backbone': False, 'freeze_bn': True},
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"Config: {config}")
        
        encoder = ResNetEncoder(**config, out_dim=256).to(device)
        print(f"Encoder: {encoder}")
        
        # Count parameters
        total_params = sum(p.numel() for p in encoder.parameters())
        trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        print(f"Total params: {total_params/1e6:.2f}M")
        print(f"Trainable params: {trainable_params/1e6:.2f}M")
        
        # Test forward pass
        x = torch.randn(4, 3, 224, 224).to(device)
        
        # Warmup
        with torch.no_grad():
            _ = encoder(x)
        
        # Benchmark
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        
        with torch.no_grad():
            for _ in range(10):
                out = encoder(x)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.time() - start
        
        print(f"Output shape: {out.shape}")
        print(f"Forward time: {elapsed/10*1000:.2f}ms per batch")
    
    print("\n" + "="*60)
    print("Testing factory function...")
    
    for enc_type in ['plain_conv', 'resnet18', 'resnet34', 'resnet50']:
        encoder = create_visual_encoder(enc_type, out_dim=256)
        input_size = get_encoder_input_size(enc_type)
        print(f"{enc_type}: input_size={input_size}, out_dim=256")
    
    print("\nAll tests passed!")
