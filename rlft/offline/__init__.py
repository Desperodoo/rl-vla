"""
rlft.offline - Offline Training & Evaluation Scripts

Training scripts for:
- train_maniskill.py: ManiSkill IL and offline RL
- train_carm.py: CARM real robot IL
- eval_carm.py: CARM offline evaluation (model performance on recorded data)

Usage:
    python -m rlft.offline.train_maniskill --env_id PushCube-v1 --algorithm flow_matching
    python -m rlft.offline.train_carm --demo_path ~/recorded_data --algorithm shortcut_flow
    python -m rlft.offline.eval_carm --model_path /path/to/model.pt --data_dir ~/recorded_data
"""

# Training/evaluation scripts are standalone executables and should be run directly
# Import for reference (optional)
try:
    from rlft.offline import train_maniskill
    from rlft.offline import train_carm
    from rlft.offline import eval_carm
except ImportError:
    pass  # Scripts have external dependencies
# Example:
#   python -m rlft.offline.train_maniskill --env_id PickCube-v1 --algorithm flow_matching
#   python -m rlft.offline.train_carm --data_path ~/carm_data
#   python -m rlft.offline.eval_carm --model_path /path/to/model.pt --data_dir ~/recorded_data
