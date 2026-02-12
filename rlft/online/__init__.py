"""
rlft.online - Online Training Scripts

Training scripts for:
- train_rlpd.py: RLPD with offline demo mixing
- train_reinflow.py: ReinFlow PPO fine-tuning

Usage:
    python -m rlft.online.train_rlpd --env_id PickCube-v1 --demo_path ~/demos/
    python -m rlft.online.train_reinflow --env_id PickCube-v1 --pretrained_path model.pt
"""

# Training scripts are standalone executables and should be run directly
try:
    from rlft.online import train_rlpd
    from rlft.online import train_reinflow
    from rlft.online import train_dsrl
except ImportError:
    pass  # Scripts have external dependencies