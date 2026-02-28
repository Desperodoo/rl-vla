"""调查 ManiSkill3 环境细节的脚本"""
import gymnasium as gym
import mani_skill.envs
import numpy as np

# ============================================================
# 1. state_dim 调查
# ============================================================
print("=" * 60)
print("1. STATE_DIM 调查")
print("=" * 60)

for task in ['LiftPegUpright-v1', 'PickCube-v1', 'StackCube-v1']:
    try:
        env = gym.make(task, obs_mode='state', num_envs=1)
        obs, _ = env.reset()
        print(f"\n=== {task} ===")
        for k, v in obs['agent'].items():
            arr = np.array(v)
            print(f"  agent[{k}]: shape={arr.shape}, dtype={arr.dtype}")
        # 计算 total state dim
        total = sum(np.array(v).shape[-1] for v in obs['agent'].values())
        print(f"  => total agent state dim: {total}")
        env.close()
    except Exception as e:
        print(f"  ERROR for {task}: {e}")

# ============================================================
# 2. rgbd 相机信息
# ============================================================
print("\n" + "=" * 60)
print("2. RGBD 相机信息")
print("=" * 60)

for task in ['LiftPegUpright-v1', 'PickCube-v1', 'StackCube-v1']:
    try:
        env = gym.make(task, obs_mode='rgbd', render_mode='rgb_array', num_envs=1)
        obs, _ = env.reset()
        print(f"\n=== {task} ===")
        if 'sensor_data' in obs:
            for cam_name, cam_data in obs['sensor_data'].items():
                print(f"  camera: {cam_name}")
                for img_key, img_val in cam_data.items():
                    arr = np.array(img_val)
                    print(f"    [{img_key}]: shape={arr.shape}, dtype={arr.dtype}")
        if 'sensor_param' in obs:
            for cam_name, param in obs['sensor_param'].items():
                print(f"  param: {cam_name}")
                for pk, pv in param.items():
                    print(f"    {pk}: {np.array(pv)}")
        env.close()
    except Exception as e:
        print(f"  ERROR for {task}: {e}")

# ============================================================
# 3. success 定义调查
# ============================================================
print("\n" + "=" * 60)
print("3. SUCCESS 定义调查")
print("=" * 60)

for task in ['LiftPegUpright-v1', 'PickCube-v1', 'StackCube-v1']:
    try:
        env = gym.make(task, obs_mode='state', num_envs=1)
        obs, _ = env.reset()
        # 随便 step 几步
        for _ in range(3):
            action = env.action_space.sample()
            obs, rew, term, trunc, info = env.step(action)
        print(f"\n=== {task} ===")
        print(f"  info keys: {list(info.keys())}")
        for k, v in info.items():
            print(f"  info[{k}]: {v}")
        print(f"  action_space: {env.action_space}")
        print(f"  observation_space: {env.observation_space}")
        env.close()
    except Exception as e:
        print(f"  ERROR for {task}: {e}")

print("\nDone!")
