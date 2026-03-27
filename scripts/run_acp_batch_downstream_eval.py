from __future__ import annotations

import json
from pathlib import Path
import subprocess
import numpy as np
import h5py

ROOT = Path('/home/wjz/rl-vla')
RAW_ROOT = ROOT / 'data/robo_dopamine/acp_data_rawfps_smoke'
OUT_ROOT = ROOT / 'results/vlaw/acp_batch_eval'
ROBOMETER_CKPT = ROOT / 'checkpoints/robometer/Robometer-4B'
TASK = 'Pick up the peg and lift it upright.'

EPISODES = [f'episode_{i:04d}' for i in range(6)]


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, check=True, env=env)


def summarize_hdf5_sources() -> dict:
    src_root = ROOT / 'data/vlaw/rollouts_acp'
    summary = {}
    for name in ['pretrained_policy_rawfps', 'teleop_sim_rawfps', 'rl_prior_rawfps', 'random_rawfps']:
        files = sorted((src_root / name).glob('*.h5'))
        trajs = 0
        lengths = []
        s_once = 0
        s_end = 0
        for p in files:
            with h5py.File(p, 'r') as f:
                for k in f.keys():
                    if not k.startswith('traj_'):
                        continue
                    es = np.asarray(f[k]['env_success'], dtype=bool)
                    trajs += 1
                    lengths.append(len(es))
                    s_once += int(es.any())
                    s_end += int(es[-1])
        summary[name] = {
            'files': len(files),
            'trajs': trajs,
            'len_mean': float(np.mean(lengths)) if lengths else 0.0,
            'len_min': min(lengths) if lengths else 0,
            'len_max': max(lengths) if lengths else 0,
            'success_once': s_once,
            'success_at_end': s_end,
            'mismatch': s_once - s_end,
        }
    return summary


def run_robometer(ep: str) -> dict:
    out = OUT_ROOT / 'robometer' / f'{ep}_rewards.npy'
    out.parent.mkdir(parents=True, exist_ok=True)
    env = {'PYTHONPATH': str(ROOT / 'rlft/robometer')}
    env.update(dict(**subprocess.os.environ))
    cmd = [
        'conda', 'run', '-n', 'robo-dopamine', 'python',
        str(ROOT / 'rlft/robometer/scripts/example_inference_local.py'),
        '--model-path', str(ROBOMETER_CKPT),
        '--video', str(RAW_ROOT / ep / 'cam_high.mp4'),
        '--task', TASK,
        '--fps', '2',
        '--max-frames', '64',
        '--out', str(out),
    ]
    run(cmd, env=env)
    rewards = np.load(out)
    succ = np.load(out.with_name(out.stem + '_success_probs.npy'))
    return {
        'episode': ep,
        'frames': int(len(rewards)),
        'reward_min': float(rewards.min()),
        'reward_max': float(rewards.max()),
        'reward_mean': float(rewards.mean()),
        'success_mean': float(succ.mean()),
    }


def run_robodopamine(ep: str) -> dict:
    script = OUT_ROOT / f'run_{ep}_robodopamine.py'
    out_root = OUT_ROOT / 'robodopamine' / ep
    out_root.mkdir(parents=True, exist_ok=True)
    script.write_text(
        f"from pathlib import Path\nfrom examples.inference import GRMInference\nmodel = GRMInference('tanhuajie2001/Robo-Dopamine-GRM-3B')\nout = model.run_pipeline(cam_high_path='{RAW_ROOT / ep / 'cam_high.mp4'}', cam_left_path='{RAW_ROOT / ep / 'cam_left_wrist.mp4'}', cam_right_path='{RAW_ROOT / ep / 'cam_right_wrist.mp4'}', out_root='{out_root}', task='{TASK}', frame_interval=10, batch_size=1, goal_image='{RAW_ROOT / 'blank_goal.png'}', eval_mode='forward', visualize=True)\nprint(out)\n",
        encoding='utf-8'
    )
    env = {'PYTHONPATH': str(ROOT / 'rlft/Robo-Dopamine')}
    env.update(dict(**subprocess.os.environ))
    cmd = ['conda', 'run', '-n', 'robo-dopamine', 'python', '-u', str(script)]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
    lines = [x.strip() for x in proc.stdout.splitlines() if x.strip()]
    out_dir = Path(lines[-1])
    with open(out_dir / 'pred_vllm.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {
        'episode': ep,
        'samples': len(data),
        'first_pred': data[0].get('pred') if data else None,
        'last_progress': data[-1].get('progress') if data else None,
        'mean_progress': float(np.mean([x.get('progress', 0.0) for x in data])) if data else None,
        'out_dir': str(out_dir),
    }


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    result = {
        'source_summary': summarize_hdf5_sources(),
        'robometer': [],
        'robodopamine': [],
    }
    for ep in EPISODES:
        result['robometer'].append(run_robometer(ep))
    for ep in EPISODES[:3]:
        result['robodopamine'].append(run_robodopamine(ep))
    out = OUT_ROOT / 'summary.json'
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
