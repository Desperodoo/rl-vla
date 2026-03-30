"""
rlft.offline - Offline Training & Evaluation Scripts

Training scripts for:
- train_maniskill.py: ManiSkill IL and offline RL
- train_carm.py: CARM real robot IL
- train_pi05.py: LeRobot-first pi0.5 bridge smoke training entrypoint
- eval_carm.py: CARM offline evaluation (model performance on recorded data)

Usage:
    python -m rlft.offline.train_maniskill --env_id PushCube-v1 --algorithm flow_matching
    python -m rlft.offline.train_carm --demo_path ~/recorded_data --algorithm shortcut_flow
    python -m rlft.offline.train_pi05 --demo_path ~/recorded_data/mix --policy_type pi0.5
    python -m rlft.offline.eval_carm --model_path /path/to/model.pt --data_dir ~/recorded_data
"""

# Training/evaluation scripts are standalone executables and should be run directly.
# Keep this package init lightweight so `python -m rlft.offline.<module>` does not
# import unrelated heavyweight dependencies transitively.
