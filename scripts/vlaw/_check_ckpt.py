"""Quick check of base checkpoint structure."""
import torch
import sys
sys.path.insert(0, '.')

from rlft.algorithms.il.shortcut_flow import ShortCutFlowAgent
from rlft.networks import ShortCutVelocityUNet1D, PlainConv

ckpt = torch.load('checkpoints/il/best_eval_success_once.pt', map_location='cpu', weights_only=False)

velocity_net = ShortCutVelocityUNet1D(input_dim=7, global_cond_dim=626)
agent = ShortCutFlowAgent(
    velocity_net=velocity_net,
    action_dim=7,
    obs_horizon=2,
    pred_horizon=8,
    device='cpu',
)
missing, unexpected = agent.load_state_dict(ckpt['agent'], strict=False)
print(f"Missing: {len(missing)}")
for k in missing[:5]:
    print(f"  {k}")
print(f"Unexpected: {len(unexpected)}")
for k in unexpected[:5]:
    print(f"  {k}")

ve = PlainConv(in_channels=3, out_dim=256, pool_feature_map=True)
ve.load_state_dict(ckpt['visual_encoder'])
print("VE loaded clean")
print(f"global_cond_dim = 626, per_step = 313 = 256 + 57")
