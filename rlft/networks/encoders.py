"""
Visual and State Encoders for Observations.

- PlainConv: Simple CNN for 128x128 images
- ResNetEncoder: Pretrained ResNet10/18/34/50 encoders
- StateEncoder: MLP encoder for state observations
"""

import torch
import torch.nn as nn
import torchvision.models as models
import warnings
from typing import Literal, Tuple


# =============================================================================
# ImageNet Normalization
# =============================================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ImageNetNormalize(nn.Module):
    """Normalize images with ImageNet mean and std."""
    
    def __init__(self):
        super().__init__()
        self.register_buffer('mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class FrozenBatchNorm2d(nn.Module):
    """BatchNorm2d with frozen statistics and affine parameters."""
    
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


def _convert_bn_to_frozen_bn(module: nn.Module) -> nn.Module:
    """Recursively convert BatchNorm2d to FrozenBatchNorm2d."""
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


# =============================================================================
# Plain Conv Encoder
# =============================================================================

def make_mlp(in_channels, mlp_channels, act_builder=nn.ReLU, last_act=True):
    """Build MLP from channel dimensions."""
    c_in = in_channels
    module_list = []
    for idx, c_out in enumerate(mlp_channels):
        module_list.append(nn.Linear(c_in, c_out))
        if last_act or idx < len(mlp_channels) - 1:
            module_list.append(act_builder())
        c_in = c_out
    return nn.Sequential(*module_list)


class PlainConv(nn.Module):
    """Simple CNN encoder for 128x128 images.
    
    A lightweight visual encoder suitable for smaller images and faster training.
    
    Args:
        in_channels: Number of input channels (default: 3 for RGB)
        out_dim: Output feature dimension (default: 256)
        pool_feature_map: Whether to use adaptive pooling (default: False)
        last_act: Whether to include final activation (default: True)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_dim: int = 256,
        pool_feature_map: bool = False,
        last_act: bool = True,
        image_size: int = 128,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.image_size = image_size
        
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64x64
            nn.Conv2d(16, 32, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32x32
            nn.Conv2d(32, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16x16
            nn.Conv2d(64, 128, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 8x8
            nn.Conv2d(128, 128, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
        )

        if pool_feature_map:
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
            self.fc = make_mlp(128, [out_dim], last_act=last_act)
        else:
            self.pool = None
            # For 128x128 input: after 4 max pools -> 8x8
            # For 256x256 input: after 4 max pools -> 16x16
            feature_size = (image_size // 16) ** 2 * 128
            self.fc = make_mlp(feature_size, [out_dim], last_act=last_act)

        self.reset_parameters()

    def reset_parameters(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (B, C, H, W) input images
            
        Returns:
            features: (B, out_dim) feature vectors
        """
        x = self.cnn(image)
        if self.pool is not None:
            x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


# =============================================================================
# ResNet Encoder
# =============================================================================

class HuggingFaceResNet10Wrapper(nn.Module):
    """Wrapper for HuggingFace ResNet10 model to extract tensor features."""
    
    def __init__(self, embedder: nn.Module, encoder: nn.Module):
        super().__init__()
        self.embedder = embedder
        self.encoder = encoder
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedder(x)
        encoder_output = self.encoder(x)
        return encoder_output.last_hidden_state


class ResNetEncoder(nn.Module):
    """ResNet-based visual encoder with pretrained weights.
    
    Supports ResNet10 (from HuggingFace), ResNet18, ResNet34, ResNet50.
    
    Args:
        backbone_name: One of 'resnet10', 'resnet18', 'resnet34', 'resnet50'
        out_dim: Output feature dimension (default: 256)
        pretrained: Whether to use ImageNet pretrained weights
        freeze_backbone: Whether to freeze all backbone parameters
        freeze_bn: Whether to convert BatchNorm to FrozenBatchNorm
        pool_type: Pooling type ('avg' or 'max')
    """
    
    RECOMMENDED_INPUT_SIZE = {
        'resnet10': (128, 128),
        'resnet18': (224, 224),
        'resnet34': (224, 224),
        'resnet50': (224, 224),
    }
    
    BACKBONE_CHANNELS = {
        'resnet10': 512,
        'resnet18': 512,
        'resnet34': 512,
        'resnet50': 2048,
    }
    
    def __init__(
        self,
        backbone_name: Literal['resnet10', 'resnet18', 'resnet34', 'resnet50'] = 'resnet18',
        out_dim: int = 256,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        freeze_bn: bool = True,
        pool_type: Literal['avg', 'max'] = 'avg',
        # Alias for compatibility
        depth: int = None,
    ):
        super().__init__()
        
        # Handle depth alias
        if depth is not None:
            backbone_name = f'resnet{depth}'
        
        if backbone_name not in self.BACKBONE_CHANNELS:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        self.backbone_name = backbone_name
        self.out_dim = out_dim
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.freeze_bn = freeze_bn
        
        self.normalize = ImageNetNormalize()
        self.backbone_channels = self.BACKBONE_CHANNELS[backbone_name]
        
        # Load backbone
        if backbone_name == 'resnet10':
            self._load_resnet10_backbone(pretrained)
        else:
            self._load_torchvision_backbone(backbone_name, pretrained)
        
        if freeze_bn:
            self.features = _convert_bn_to_frozen_bn(self.features)
        
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False
        
        if pool_type == 'avg':
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
        
        self.projection = nn.Sequential(
            nn.Linear(self.backbone_channels, out_dim),
            nn.ReLU(inplace=True),
        )
        
        self._init_projection()
    
    def _load_resnet10_backbone(self, pretrained: bool):
        """Load ResNet10 from HuggingFace."""
        if pretrained:
            try:
                from transformers import AutoModel
                print("Loading ResNet10 from HuggingFace: helper2424/resnet10")
                hf_model = AutoModel.from_pretrained("helper2424/resnet10", trust_remote_code=True)
                self.features = HuggingFaceResNet10Wrapper(
                    embedder=hf_model.embedder,
                    encoder=hf_model.encoder,
                )
                print("ResNet10 loaded successfully!")
            except Exception as e:
                warnings.warn(f"Failed to load ResNet10: {e}. Using random init.")
                self._create_resnet10_from_scratch()
        else:
            self._create_resnet10_from_scratch()
    
    def _create_resnet10_from_scratch(self):
        """Create ResNet10 from scratch."""
        from torchvision.models.resnet import BasicBlock
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(BasicBlock, 64, 64, 1, stride=1),
            self._make_layer(BasicBlock, 64, 128, 1, stride=2),
            self._make_layer(BasicBlock, 128, 256, 1, stride=2),
            self._make_layer(BasicBlock, 256, 512, 1, stride=2),
        )
    
    def _make_layer(self, block, in_channels, out_channels, num_blocks, stride):
        """Create a ResNet layer."""
        from torchvision.models.resnet import BasicBlock
        
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        
        layers = [block(in_channels, out_channels, stride, downsample)]
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _load_torchvision_backbone(self, backbone_name: str, pretrained: bool):
        """Load ResNet18/34/50 from torchvision."""
        weights = 'IMAGENET1K_V1' if pretrained else None
        backbone = getattr(models, backbone_name)(weights=weights)
        
        self.features = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
        )
    
    def _init_projection(self):
        """Initialize projection layer."""
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) images in [0, 1] range
            
        Returns:
            features: (B, out_dim) feature vectors
        """
        x = self.normalize(x)
        features = self.features(x)
        pooled = self.pool(features)
        flat = pooled.flatten(1)
        return self.projection(flat)
    
    def get_param_groups(self, lr_backbone: float, lr_head: float):
        """Get parameter groups with different learning rates."""
        return [
            {'params': self.features.parameters(), 'lr': lr_backbone},
            {'params': self.projection.parameters(), 'lr': lr_head},
        ]


# =============================================================================
# State Encoder
# =============================================================================

class StateEncoder(nn.Module):
    """MLP encoder for state observations.
    
    Projects state features to align with visual features.
    
    Args:
        state_dim: Input state dimension
        hidden_dim: Hidden layer dimension (default: 128)
        out_dim: Output feature dimension (default: 256)
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 256,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.out_dim = out_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Encode state observations."""
        return self.mlp(state)


# =============================================================================
# Factory Functions
# =============================================================================

def create_visual_encoder(
    encoder_type: str,
    out_dim: int = 256,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    freeze_bn: bool = True,
    in_channels: int = 3,
    pool_feature_map: bool = True,
    image_size: int = 128,
) -> nn.Module:
    """Factory function to create visual encoders.
    
    Args:
        encoder_type: 'plain_conv', 'resnet10', 'resnet18', 'resnet34', 'resnet50'
        out_dim: Output feature dimension
        pretrained: Use pretrained weights (ResNet only)
        freeze_backbone: Freeze backbone (ResNet only)
        freeze_bn: Freeze BatchNorm (ResNet only)
        in_channels: Input channels (PlainConv only)
        pool_feature_map: Use pooling (PlainConv only)
        image_size: Input image size (PlainConv only)
    
    Returns:
        Visual encoder module
    """
    if encoder_type == 'plain_conv':
        return PlainConv(
            in_channels=in_channels,
            out_dim=out_dim,
            pool_feature_map=pool_feature_map,
            image_size=image_size,
        )
    elif encoder_type in ['resnet10', 'resnet18', 'resnet34', 'resnet50']:
        return ResNetEncoder(
            backbone_name=encoder_type,
            out_dim=out_dim,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            freeze_bn=freeze_bn,
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


def get_encoder_input_size(
    encoder_type: str, 
    default_size: Tuple[int, int] = (128, 128)
) -> Tuple[int, int]:
    """Get recommended input size for encoder type."""
    if encoder_type == 'plain_conv':
        return default_size
    elif encoder_type in ResNetEncoder.RECOMMENDED_INPUT_SIZE:
        return ResNetEncoder.RECOMMENDED_INPUT_SIZE[encoder_type]
    else:
        return (224, 224)
