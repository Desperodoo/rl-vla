"""
Critic Network for Value Estimation

V(s) 网络，基于视觉和状态输入预测状态价值。
- 复用 ResNetEncoder（冻结 backbone）提取视觉特征
- 复用 StateEncoder 提取状态特征
- 输出 V(s) ∈ [0, 1]（Sigmoid）

核心特点：
1. 不输入 action（纯状态价值函数）
2. 支持 obs_horizon 帧的观测堆叠
3. 使用冻结的预训练 ResNet 作为视觉编码器
"""

import torch
import torch.nn as nn
from typing import Optional, Literal, Tuple

from diffusion_policy.resnet_encoder import ResNetEncoder
from diffusion_policy.carm_utils import StateEncoder


class CriticNetwork(nn.Module):
    """Critic V(s) 网络
    
    输入：(rgb_images, states) 或 flattened features
    输出：V(s) ∈ [0, 1]
    
    网络结构：
        rgb_images [B, obs_horizon, 3, H, W]
            -> ResNetEncoder (frozen) -> [B, obs_horizon, visual_dim]
        states [B, obs_horizon, state_dim]  
            -> StateEncoder -> [B, obs_horizon, state_dim]
        concat -> [B, obs_horizon * (visual_dim + state_dim)]
            -> MLP -> Sigmoid -> [B, 1]
    
    Args:
        visual_encoder: 预训练的视觉编码器（会被冻结）
        state_encoder: 状态编码器（可训练）
        obs_horizon: 观测堆叠帧数
        hidden_dims: MLP 隐藏层维度
        freeze_visual: 是否冻结视觉编码器
    """
    
    def __init__(
        self,
        visual_encoder: nn.Module,
        state_encoder: nn.Module,
        obs_horizon: int = 2,
        hidden_dims: Tuple[int, ...] = (256, 256),
        freeze_visual: bool = True,
    ):
        super().__init__()
        
        self.obs_horizon = obs_horizon
        self.freeze_visual = freeze_visual
        
        # 复用编码器
        self.visual_encoder = visual_encoder
        self.state_encoder = state_encoder
        
        # 冻结视觉编码器
        if freeze_visual:
            for param in self.visual_encoder.parameters():
                param.requires_grad = False
            self.visual_encoder.eval()
        
        # 获取特征维度
        self.visual_dim = getattr(visual_encoder, 'out_dim', 256)
        self.state_dim = getattr(state_encoder, 'out_dim', state_encoder.state_dim)
        
        # 计算输入维度
        feature_dim = obs_horizon * (self.visual_dim + self.state_dim)
        
        # 构建 MLP value head
        layers = []
        in_dim = feature_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(inplace=True),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())  # 输出 [0, 1]
        
        self.value_head = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化 MLP 权重"""
        for m in self.value_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def train(self, mode: bool = True):
        """Override train() to keep visual encoder frozen"""
        super().train(mode)
        if self.freeze_visual:
            self.visual_encoder.eval()
        return self
    
    def encode_observations(
        self,
        rgb: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """编码观测，返回 flattened 特征
        
        Args:
            rgb: [B, obs_horizon, 3, H, W] RGB 图像，范围 [0, 1]
            state: [B, obs_horizon, state_dim] 状态
        
        Returns:
            features: [B, obs_horizon * (visual_dim + state_dim)]
        """
        B, T = rgb.shape[:2]
        
        # 编码视觉特征
        # [B, T, 3, H, W] -> [B*T, 3, H, W] -> [B*T, visual_dim] -> [B, T, visual_dim]
        with torch.set_grad_enabled(not self.freeze_visual):
            rgb_flat = rgb.flatten(0, 1)  # [B*T, 3, H, W]
            visual_feat = self.visual_encoder(rgb_flat)  # [B*T, visual_dim]
            visual_feat = visual_feat.view(B, T, -1)  # [B, T, visual_dim]
        
        # 编码状态特征
        # [B, T, state_dim] -> [B*T, state_dim] -> [B*T, state_dim] -> [B, T, state_dim]
        state_flat = state.flatten(0, 1)  # [B*T, state_dim]
        state_feat = self.state_encoder(state_flat)  # [B*T, out_dim]
        state_feat = state_feat.view(B, T, -1)  # [B, T, out_dim]
        
        # 拼接并展平
        # [B, T, visual_dim + state_dim] -> [B, T * (visual_dim + state_dim)]
        combined = torch.cat([visual_feat, state_feat], dim=-1)
        features = combined.flatten(1)  # [B, T * (visual_dim + state_dim)]
        
        return features
    
    def forward(
        self,
        rgb: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播
        
        Args:
            rgb: [B, obs_horizon, 3, H, W] RGB 图像
            state: [B, obs_horizon, state_dim] 状态
        
        Returns:
            value: [B, 1] 状态价值 V(s) ∈ [0, 1]
        """
        features = self.encode_observations(rgb, state)
        value = self.value_head(features)
        return value
    
    def forward_from_features(self, features: torch.Tensor) -> torch.Tensor:
        """从预计算的特征进行前向传播
        
        Args:
            features: [B, feature_dim] 预编码的特征
        
        Returns:
            value: [B, 1] 状态价值
        """
        return self.value_head(features)


def create_critic_network(
    state_dim: int = 7,
    visual_feature_dim: int = 256,
    obs_horizon: int = 2,
    hidden_dims: Tuple[int, ...] = (256, 256),
    backbone_name: str = "resnet18",
    pretrained: bool = True,
    freeze_visual: bool = True,
    freeze_bn: bool = True,
    device: str = "cuda",
) -> CriticNetwork:
    """创建 Critic 网络的工厂函数
    
    Args:
        state_dim: 状态维度（joint_only=7, ee_only=8, both=14）
        visual_feature_dim: 视觉特征维度
        obs_horizon: 观测堆叠帧数
        hidden_dims: MLP 隐藏层维度
        backbone_name: ResNet backbone 名称
        pretrained: 是否使用预训练权重
        freeze_visual: 是否冻结视觉编码器
        freeze_bn: 是否冻结 BatchNorm
        device: 设备
    
    Returns:
        CriticNetwork 实例
    """
    # 创建视觉编码器
    visual_encoder = ResNetEncoder(
        backbone_name=backbone_name,
        out_dim=visual_feature_dim,
        pretrained=pretrained,
        freeze_backbone=freeze_visual,
        freeze_bn=freeze_bn,
    ).to(device)
    
    # 创建状态编码器
    state_encoder = StateEncoder(
        state_dim=state_dim,
        hidden_dim=128,
        out_dim=visual_feature_dim,  # 与视觉特征维度对齐
    ).to(device)
    
    # 创建 Critic 网络
    critic = CriticNetwork(
        visual_encoder=visual_encoder,
        state_encoder=state_encoder,
        obs_horizon=obs_horizon,
        hidden_dims=hidden_dims,
        freeze_visual=freeze_visual,
    ).to(device)
    
    return critic
