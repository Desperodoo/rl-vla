"""
rlft.offline - Offline Training Scripts

Training scripts for:
- train_maniskill.py: ManiSkill IL and offline RL
- train_carm.py: CARM real robot IL
- train_critic.py: Critic/reward model training

Usage:
    python -m rlft.offline.train_maniskill --env_id PushCube-v1 --algorithm flow_matching
    python -m rlft.offline.train_carm --demo_path ~/recorded_data --algorithm shortcut_flow
"""

# Training scripts are standalone executables and should be run directly
# Import for reference (optional)
try:
    from rlft.offline import train_maniskill
    from rlft.offline import train_carm
except ImportError:
    pass  # Scripts have external dependencies
# Example:
#   python -m rlft.offline.train_maniskill --env_id PickCube-v1 --algorithm flow_matching
#   python -m rlft.offline.train_carm --data_path ~/carm_data
