#!/usr/bin/env python3
"""
策略模型加载与推理模块

从 inference_ros.py 中提取的 PolicyInterface / RealPolicy 类，
负责模型加载、权重恢复、观测编码和动作推理。

支持的算法:
    - consistency_flow: Consistency Flow Matching
    - flow_matching: Flow Matching Policy
    - diffusion_policy: DDPM-based Diffusion Policy
    - reflected_flow: Reflected Flow Matching
    - shortcut_flow: Shortcut Flow Matching
"""

import os
import json
import numpy as np
import cv2
import logging
from collections import deque
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from rlft.networks import (
    StateEncoder, GripperHead, create_visual_encoder,
)
from rlft.datasets import ActionNormalizer
from rlft.utils.model_factory import create_agent_for_inference, SUPPORTED_ALGORITHMS


# ---------------------------------------------------------------------------
# 日志工具 – 兼容 ROS (rospy) 和非 ROS 环境
# ---------------------------------------------------------------------------
try:
    import rospy
    _log_info = rospy.loginfo
    _log_warn = rospy.logwarn
    _log_err = rospy.logerr
except ImportError:
    _logger = logging.getLogger(__name__)
    _log_info = _logger.info
    _log_warn = _logger.warning
    _log_err = _logger.error


# ============================================================================
# PolicyInterface (abstract base)
# ============================================================================

class PolicyInterface:
    """
    策略模型接口（抽象基类）
    用户需要继承此类并实现 load_model 和 __call__ 方法
    """

    def __init__(self, config: dict):
        self.config = config
        self.model = None

    def load_model(self, model_path: str):
        raise NotImplementedError("Subclass must implement load_model()")

    def __call__(self, inputs: dict):
        raise NotImplementedError("Subclass must implement __call__()")


# ============================================================================
# Defaults – single source of truth for fallback values
# These are only used when args.json is missing or a key is absent.
# ============================================================================

POLICY_DEFAULTS: Dict[str, Any] = {
    'obs_horizon': 2,
    'pred_horizon': 16,
    'action_dim': 13,          # full mode continuous (no gripper)
    'action_dim_full': 15,     # full mode with gripper slots
    'state_mode': 'joint_only',
    'target_image_size': (128, 128),
    'visual_feature_dim': 256,
    'state_encoder_hidden_dim': 128,
    'state_encoder_out_dim': 256,
    'use_state_encoder': True,
    'algorithm': 'consistency_flow',
    'num_inference_steps': 10,
    'use_ema': False,
    'gripper_threshold': 0.05,
    'gripper_open_val': 0.078,
    'gripper_close_val': 0.04,
    'gripper_head_hidden_dim': 256,
    'gripper_hysteresis_window': 1,
}


# ============================================================================
# RealPolicy
# ============================================================================

class RealPolicy(PolicyInterface):
    """
    真实策略实现
    加载训练好的 checkpoint 并执行推理

    支持的算法: 见 SUPPORTED_ALGORITHMS
    """

    def __init__(self, config: dict):
        super().__init__(config)

        _d = POLICY_DEFAULTS  # shorthand

        # 默认参数（会被 load_model -> args.json 覆盖）
        self.obs_horizon = config.get('obs_horizon', _d['obs_horizon'])
        self.pred_horizon = config.get('pred_horizon', _d['pred_horizon'])
        self.action_dim = config.get('action_dim', _d['action_dim'])
        self.action_dim_full = config.get('action_dim_full', _d['action_dim_full'])

        # State mode configuration
        self.state_mode = config.get('state_mode', _d['state_mode'])
        self.state_dim = self._get_state_dim_for_mode(self.state_mode)

        self.target_image_size = config.get('target_image_size', _d['target_image_size'])
        self.visual_feature_dim = config.get('visual_feature_dim', _d['visual_feature_dim'])
        self.state_encoder_hidden_dim = config.get('state_encoder_hidden_dim', _d['state_encoder_hidden_dim'])
        self.state_encoder_out_dim = config.get('state_encoder_out_dim', _d['state_encoder_out_dim'])
        self.use_state_encoder = config.get('use_state_encoder', _d['use_state_encoder'])
        self.algorithm = config.get('algorithm', _d['algorithm'])

        # 推理参数（可配置）
        self.num_inference_steps = config.get('num_inference_steps', _d['num_inference_steps'])
        self.use_ema = config.get('use_ema', _d['use_ema'])

        # Discrete gripper parameters
        self.gripper_threshold = config.get('gripper_threshold', _d['gripper_threshold'])
        self.gripper_open_val = config.get('gripper_open_val', _d['gripper_open_val'])
        self.gripper_close_val = config.get('gripper_close_val', _d['gripper_close_val'])
        self.gripper_head_hidden_dim = config.get('gripper_head_hidden_dim', _d['gripper_head_hidden_dim'])

        # Hysteresis
        self.gripper_hysteresis_window = config.get('gripper_hysteresis_window', _d['gripper_hysteresis_window'])
        self._gripper_history: deque = deque(maxlen=self.gripper_hysteresis_window)
        self._last_gripper_state = 0  # 0=open, 1=close

        # 模型组件
        self.visual_encoder = None
        self.state_encoder = None
        self.agent = None
        self.gripper_head = None
        self.action_normalizer = None

        # 观测历史缓冲区
        self.obs_history: Dict[str, list] = {'rgb': [], 'state': []}

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loaded = False

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_state_dim_for_mode(state_mode: str) -> int:
        if state_mode == 'joint_only':
            return 7
        elif state_mode == 'ee_only':
            return 8
        elif state_mode == 'both':
            return 14
        else:
            _log_warn(f"Unknown state_mode: {state_mode}, defaulting to joint_only")
            return 7

    def build_state_from_obs(self, qpos_joint: np.ndarray, qpos_end: np.ndarray) -> np.ndarray:
        """Build state vector based on state_mode."""
        if self.state_mode == 'joint_only':
            return qpos_joint.astype(np.float32)
        elif self.state_mode == 'ee_only':
            return qpos_end.astype(np.float32)
        elif self.state_mode == 'both':
            return np.concatenate([
                qpos_joint.astype(np.float32),
                qpos_end[:7].astype(np.float32),
            ])
        else:
            _log_warn(f"Unknown state_mode: {self.state_mode}, using joint_only")
            return qpos_joint.astype(np.float32)

    # ------------------------------------------------------------------
    # load_model
    # ------------------------------------------------------------------

    def load_model(self, model_path: str):
        """加载模型 checkpoint"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")

        checkpoint_dir = os.path.dirname(model_path)

        # 1. 加载训练配置
        args_path = os.path.join(checkpoint_dir, "args.json")
        visual_encoder_type = 'plain_conv'

        if os.path.exists(args_path):
            _log_info(f"Loading config from: {args_path}")
            with open(args_path, 'r') as f:
                args = json.load(f)

            self.obs_horizon = args.get('obs_horizon', self.obs_horizon)
            self.pred_horizon = args.get('pred_horizon', self.pred_horizon)
            self.action_dim = args.get('action_dim', self.action_dim)
            self.visual_feature_dim = args.get('visual_feature_dim', self.visual_feature_dim)
            self.state_encoder_hidden_dim = args.get('state_encoder_hidden_dim', self.state_encoder_hidden_dim)
            self.state_encoder_out_dim = args.get('state_encoder_out_dim', self.state_encoder_out_dim)
            self.use_state_encoder = args.get('use_state_encoder', self.use_state_encoder)
            self.algorithm = args.get('algorithm', self.algorithm)

            # State mode
            self.state_mode = args.get('state_mode', 'joint_only')
            self.state_dim = self._get_state_dim_for_mode(self.state_mode)

            # Visual encoder
            visual_encoder_type = args.get('visual_encoder_type', 'plain_conv')

            # Image size
            if args.get('auto_image_size', True):
                if visual_encoder_type in ['resnet18', 'resnet34', 'resnet50']:
                    self.target_image_size = (224, 224)
                else:
                    self.target_image_size = (128, 128)
            else:
                target_size = args.get('target_image_size', self.target_image_size)
                if isinstance(target_size, str):
                    import ast
                    target_size = ast.literal_eval(target_size)
                elif isinstance(target_size, list):
                    target_size = tuple(target_size)
                self.target_image_size = target_size

            # action_mode
            action_mode = args.get('action_mode', 'full')
            if action_mode == 'full':
                self.action_dim = 13
                self.action_dim_full = 15
            else:
                self.action_dim = 7
                self.action_dim_full = 8

            # Gripper
            self.gripper_threshold = args.get('gripper_threshold', self.gripper_threshold)
            self.gripper_open_val = args.get('gripper_open_val', self.gripper_open_val)
            self.gripper_close_val = args.get('gripper_close_val', self.gripper_close_val)
            self.gripper_head_hidden_dim = args.get('gripper_head_hidden_dim', self.gripper_head_hidden_dim)

            # Action normalization
            self.normalize_actions = args.get('normalize_actions', False)
            self.action_norm_mode = args.get('action_norm_mode', 'standard')

            _log_info(f"Config: algorithm={self.algorithm}, action_dim={self.action_dim} (continuous), "
                      f"obs_horizon={self.obs_horizon}, pred_horizon={self.pred_horizon}")
            _log_info(f"State mode: {self.state_mode}, state_dim={self.state_dim}")
            _log_info(f"Discrete gripper: threshold={self.gripper_threshold}, open={self.gripper_open_val}, close={self.gripper_close_val}")
            _log_info(f"Inference config: num_steps={self.num_inference_steps}, use_ema={self.use_ema}")
            _log_info(f"Visual encoder: {visual_encoder_type}, image_size={self.target_image_size}")
        else:
            _log_warn("args.json not found, using POLICY_DEFAULTS — model behaviour may "
                      "differ from training. Consider placing args.json next to the checkpoint.")
            _log_warn(f"  Defaults in use: algorithm={self.algorithm}, action_dim={self.action_dim}, "
                      f"obs_horizon={self.obs_horizon}, pred_horizon={self.pred_horizon}, "
                      f"state_mode={self.state_mode}, use_ema={self.use_ema}, "
                      f"num_inference_steps={self.num_inference_steps}")

        # 2. 创建模型 -------------------------------------------------------
        _log_info("Creating models...")

        # Visual encoder (use factory to ensure architecture matches training)
        self.visual_encoder = create_visual_encoder(
            encoder_type=visual_encoder_type,
            out_dim=self.visual_feature_dim,
            pretrained=True,
            freeze_backbone=False,
            freeze_bn=False,
        ).to(self.device)
        _log_info(f"Created visual encoder: {visual_encoder_type}")

        # State encoder
        encoded_state_dim = self.state_dim
        if self.use_state_encoder:
            self.state_encoder = StateEncoder(
                state_dim=self.state_dim,
                hidden_dim=self.state_encoder_hidden_dim,
                out_dim=self.state_encoder_out_dim,
            ).to(self.device)
            encoded_state_dim = self.state_encoder_out_dim

        # global_cond_dim
        global_cond_dim = self.obs_horizon * (self.visual_feature_dim + encoded_state_dim)
        _log_info(f"global_cond_dim={global_cond_dim} = {self.obs_horizon} * ({self.visual_feature_dim} + {encoded_state_dim})")

        # Agent
        self.agent = self._create_agent(global_cond_dim)

        # 3. 加载权重 -------------------------------------------------------
        _log_info(f"Loading checkpoint from: {model_path}")
        ckpt = torch.load(model_path, map_location=self.device)

        # visual encoder
        if "visual_encoder" in ckpt:
            self.visual_encoder.load_state_dict(ckpt["visual_encoder"])
            _log_info("Loaded visual_encoder weights")
        else:
            _log_warn("visual_encoder not in checkpoint")

        # state encoder
        if self.state_encoder is not None and "state_encoder" in ckpt:
            self.state_encoder.load_state_dict(ckpt["state_encoder"])
            _log_info("Loaded state_encoder weights")

        # agent
        if self.use_ema:
            if "ema_agent" in ckpt:
                self.agent.load_state_dict(ckpt["ema_agent"])
                _log_info("Loaded EMA agent weights (better for 1-step inference)")
            elif "agent" in ckpt:
                _log_warn("EMA agent not found, falling back to regular agent")
                self.agent.load_state_dict(ckpt["agent"])
            else:
                raise ValueError("No agent weights in checkpoint")
        else:
            if "agent" in ckpt:
                self.agent.load_state_dict(ckpt["agent"])
                _log_info("Loaded Non-EMA agent weights (better for multi-step inference)")
            elif "ema_agent" in ckpt:
                _log_warn("Regular agent not found, falling back to EMA agent")
                self.agent.load_state_dict(ckpt["ema_agent"])
            else:
                raise ValueError("No agent weights in checkpoint")

        # GripperHead — detect architecture from checkpoint keys
        gripper_input_dim = self.obs_horizon * (self.visual_feature_dim + encoded_state_dim)
        gripper_use_layernorm = True  # default: new architecture with LayerNorm
        if "gripper_head" in ckpt:
            ckpt_keys = set(ckpt["gripper_head"].keys())
            # Old architecture (no LayerNorm): net has indices 0,2,4 for Linear layers
            # New architecture (LayerNorm): net has indices 0,1,3,4,6 for Linear/LN layers
            if "net.2.weight" in ckpt_keys and "net.1.weight" not in ckpt_keys:
                gripper_use_layernorm = False
                _log_info("Detected legacy GripperHead (no LayerNorm)")
            else:
                _log_info("Detected GripperHead with LayerNorm")

        self.gripper_head = GripperHead(
            obs_dim=gripper_input_dim,
            hidden_dim=self.gripper_head_hidden_dim,
            pred_horizon=self.pred_horizon,
            use_layernorm=gripper_use_layernorm,
        ).to(self.device)
        _log_info(f"Created GripperHead: obs_dim={gripper_input_dim}, hidden_dim={self.gripper_head_hidden_dim}, layernorm={gripper_use_layernorm}")

        if "gripper_head" in ckpt:
            self.gripper_head.load_state_dict(ckpt["gripper_head"])
            _log_info("Loaded gripper_head weights")
        else:
            _log_warn("gripper_head not in checkpoint! Using random initialization")

        # 5. Action normalizer
        if hasattr(self, 'normalize_actions') and self.normalize_actions:
            loaded = False

            if "action_normalizer" in ckpt:
                self.action_normalizer = ActionNormalizer.from_checkpoint(ckpt["action_normalizer"])
                _log_info("Loaded action normalizer from checkpoint")
                loaded = True
            else:
                normalizer_path = os.path.join(checkpoint_dir, "action_normalizer.json")
                if os.path.exists(normalizer_path):
                    self.action_normalizer = ActionNormalizer(mode=self.action_norm_mode)
                    self.action_normalizer.load(normalizer_path)
                    _log_info(f"Loaded action normalizer from: {normalizer_path}")
                    loaded = True

            if loaded:
                _log_info(f"  Mode: {self.action_normalizer.mode}")
                if self.action_normalizer.mode == 'standard' and self.action_normalizer.stats:
                    mean_preview = self.action_normalizer.stats['mean'][:3]
                    std_preview = self.action_normalizer.stats['std'][:3]
                    _log_info(f"  Mean: {mean_preview}...")
                    _log_info(f"  Std:  {std_preview}...")
            else:
                _log_warn("normalize_actions=True but action_normalizer not found!")
                _log_warn("Actions will NOT be denormalized – this may cause incorrect behavior!")

        # 6. eval mode
        self.visual_encoder.eval()
        if self.state_encoder is not None:
            self.state_encoder.eval()
        self.agent.eval()
        self.gripper_head.eval()

        self.loaded = True
        _log_info(f"Model loaded successfully! Algorithm: {self.algorithm}, Gripper: discrete")

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------

    def _create_agent(self, global_cond_dim: int) -> nn.Module:
        """根据算法类型创建 agent（委托给 rlft.utils.model_factory）"""
        return create_agent_for_inference(
            algorithm=self.algorithm,
            action_dim=self.action_dim,
            global_cond_dim=global_cond_dim,
            obs_horizon=self.obs_horizon,
            pred_horizon=self.pred_horizon,
            num_inference_steps=self.num_inference_steps,
            device=str(self.device),
        ).to(self.device)

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """预处理图像: resize + HWC -> CHW"""
        if self.target_image_size is not None:
            h, w = self.target_image_size
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        image = rearrange(image, 'h w c -> c h w')
        return image

    def _update_obs_history(self, rgb: np.ndarray, state: np.ndarray):
        """更新观测历史"""
        self.obs_history['rgb'].append(rgb)
        self.obs_history['state'].append(state)

        if len(self.obs_history['rgb']) > self.obs_horizon:
            self.obs_history['rgb'].pop(0)
            self.obs_history['state'].pop(0)

        while len(self.obs_history['rgb']) < self.obs_horizon:
            self.obs_history['rgb'].insert(0, self.obs_history['rgb'][0])
            self.obs_history['state'].insert(0, self.obs_history['state'][0])

    def _encode_observations(self) -> torch.Tensor:
        """编码观测历史为 obs_features [B, obs_horizon, visual_dim + state_dim]"""
        B, T = 1, self.obs_horizon

        rgb_list = [torch.from_numpy(r).float() for r in self.obs_history['rgb']]
        rgb = torch.stack(rgb_list, dim=0).unsqueeze(0).to(self.device)
        rgb_flat = rgb.view(B * T, *rgb.shape[2:]) / 255.0

        visual_feat = self.visual_encoder(rgb_flat)
        visual_feat = visual_feat.view(B, T, -1)

        state_list = [torch.from_numpy(s).float() for s in self.obs_history['state']]
        state = torch.stack(state_list, dim=0).unsqueeze(0).to(self.device)

        if self.state_encoder is not None:
            state_flat = state.view(B * T, -1)
            state_feat = self.state_encoder(state_flat)
            state_feat = state_feat.view(B, T, -1)
        else:
            state_feat = state

        obs_features = torch.cat([visual_feat, state_feat], dim=-1)
        return obs_features

    def reset(self):
        """重置观测历史和 gripper 状态"""
        self.obs_history = {'rgb': [], 'state': []}
        self._gripper_history.clear()
        self._last_gripper_state = 0

    # ------------------------------------------------------------------
    # __call__
    # ------------------------------------------------------------------

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        执行推理

        Args:
            inputs: {'qpos': Tensor, 'image': Tensor}

        Returns:
            {'a_hat': [1, pred_horizon, action_dim_full]}
        """
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        qpos = inputs['qpos'].cpu().numpy().squeeze()
        image = inputs['image'].cpu().numpy().squeeze()

        if image.ndim == 4:
            image = image[0]

        if image.shape[0] != 3:
            image = self._preprocess_image(image)

        self._update_obs_history(image, qpos)

        with torch.no_grad():
            obs_features = self._encode_observations()

            # 1. continuous actions
            actions_cont = self.agent.get_action_deterministic(obs_features)

            # 1.5 inverse normalization
            if self.action_normalizer is not None:
                actions_np = actions_cont.cpu().numpy()
                batch_size, pred_horizon, action_dim = actions_np.shape
                actions_flat = actions_np.reshape(-1, action_dim)
                actions_denorm = self.action_normalizer.inverse_transform(actions_flat)
                actions_cont = torch.from_numpy(
                    actions_denorm.reshape(batch_size, pred_horizon, action_dim)
                ).to(self.device).float()

            # 2. discrete gripper
            # GripperHead expects [B, obs_dim] (2D), flatten if 3D
            obs_flat = obs_features.reshape(obs_features.shape[0], -1) if obs_features.dim() == 3 else obs_features
            gripper_logits = self.gripper_head(obs_flat)
            gripper_cls = gripper_logits.argmax(dim=-1)

            # debug counter
            if hasattr(self, '_gripper_debug_counter'):
                self._gripper_debug_counter += 1
            else:
                self._gripper_debug_counter = 0
            should_debug = (self._gripper_debug_counter % 50 == 0)

            if should_debug:
                cls_np = gripper_cls[0].cpu().numpy()
                close_count = np.sum(cls_np == 1)
                open_count = np.sum(cls_np == 0)
                raw_logits = gripper_logits[0].cpu().numpy()
                probs = torch.softmax(gripper_logits[0], dim=-1).cpu().numpy()
                avg_close_prob = probs[:, 1].mean()
                seq_str = ''.join(['C' if c == 1 else 'O' for c in cls_np])
                _log_info(f"[Gripper Debug] step={self._gripper_debug_counter}:")
                _log_info(f"  Prediction seq: [{seq_str}] (open={open_count}, close={close_count})")
                _log_info(f"  Raw logits (open):  [{', '.join([f'{l:.3f}' for l in raw_logits[:, 0]])}]")
                _log_info(f"  Raw logits (close): [{', '.join([f'{l:.3f}' for l in raw_logits[:, 1]])}]")
                _log_info(f"  Avg close prob: {avg_close_prob:.3f}")

            # 3. hysteresis
            gripper_vals = self._apply_gripper_hysteresis(
                gripper_cls[0].cpu().numpy(), debug=should_debug
            )

            # 4. reconstruct full action
            actions_full = self._reconstruct_full_action(actions_cont[0], gripper_vals)

        return {'a_hat': actions_full.unsqueeze(0)}

    # ------------------------------------------------------------------
    # gripper helpers
    # ------------------------------------------------------------------

    def _apply_gripper_hysteresis(self, gripper_cls: np.ndarray, debug: bool = False) -> np.ndarray:
        """Apply hysteresis to gripper predictions."""
        horizon_vote_frames = min(8, len(gripper_cls))
        horizon_votes = gripper_cls[:horizon_vote_frames]
        close_in_horizon = np.sum(horizon_votes == 1)
        current_vote = 1 if close_in_horizon > horizon_vote_frames / 2 else 0

        self._gripper_history.append(current_vote)

        history_list = list(self._gripper_history)
        history_str = ''.join(['C' if h == 1 else 'O' for h in history_list])
        horizon_str = ''.join(['C' if c == 1 else 'O' for c in horizon_votes])

        if self.gripper_hysteresis_window == 1:
            new_state = current_vote
            hysteresis_mode = "horizon_vote_only"
        elif len(self._gripper_history) >= max(1, self.gripper_hysteresis_window // 2):
            close_votes = sum(self._gripper_history)
            total_votes = len(self._gripper_history)
            new_state = 1 if close_votes > total_votes / 2 else 0
            hysteresis_mode = "temporal_majority_voting"
        else:
            new_state = current_vote
            hysteresis_mode = "insufficient_history"

        if debug:
            _log_info(f"  Hysteresis: window={self.gripper_hysteresis_window}, mode={hysteresis_mode}")
            _log_info(f"  Horizon vote: [{horizon_str}] -> {'CLOSE' if current_vote else 'OPEN'}")
            _log_info(f"  History: [{history_str}] (len={len(history_list)})")
            _log_info(f"  Last state: {'CLOSE' if self._last_gripper_state else 'OPEN'} -> "
                       f"Final: {'CLOSE' if new_state else 'OPEN'}")

        if new_state != self._last_gripper_state:
            old_s = "OPEN" if self._last_gripper_state == 0 else "CLOSE"
            new_s = "OPEN" if new_state == 0 else "CLOSE"
            _log_info(f"Gripper state changed: {old_s} -> {new_s} "
                       f"(horizon=[{horizon_str}], history=[{history_str}])")
            self._last_gripper_state = new_state

        gripper_val = self.gripper_close_val if new_state == 1 else self.gripper_open_val
        return np.full(len(gripper_cls), gripper_val, dtype=np.float32)

    def _reconstruct_full_action(self, actions_cont: torch.Tensor,
                                  gripper_vals: np.ndarray) -> torch.Tensor:
        """Reconstruct full action tensor by inserting gripper values."""
        pred_horizon = actions_cont.shape[0]
        actions_full = torch.zeros(pred_horizon, self.action_dim_full, device=actions_cont.device)

        if self.action_dim_full == 15:  # full mode
            actions_full[:, :6] = actions_cont[:, :6]
            actions_full[:, 6] = torch.from_numpy(gripper_vals).to(actions_cont.device)
            actions_full[:, 7:14] = actions_cont[:, 6:13]
            actions_full[:, 14] = torch.from_numpy(gripper_vals).to(actions_cont.device)
        else:  # ee_only mode
            actions_full[:, :7] = actions_cont[:, :7]
            actions_full[:, 7] = torch.from_numpy(gripper_vals).to(actions_cont.device)

        return actions_full
