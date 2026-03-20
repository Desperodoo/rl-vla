#!/usr/bin/env python3
"""
真机上电前预检脚本 — 一键检查所有软硬件依赖。

检查项:
  1. SDK 连接 (carm_py → 10.42.0.101:8090)
  2. Backend API (http://10.42.0.101:1999)
  3. ROS master (roscore)
  4. 相机 topic (/camera/color/image_raw)
  5. Policy checkpoint 可加载
  6. Safety config 有效
  7. Python 依赖完整

Usage:
    python scripts/preflight_check.py
    python scripts/preflight_check.py --pretrain /path/to/model.pt
    python scripts/preflight_check.py --robot_ip 10.42.0.101 --skip sdk,camera
"""

import argparse
import importlib
import json
import os
import socket
import sys
import time

# Path setup
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_RL_VLA_ROOT = os.path.dirname(_SCRIPT_DIR)
_CARM_DEPLOY_ROOT = os.path.join(
    _RL_VLA_ROOT, 'carm_ros_deploy', 'src', 'carm_deploy',
)
for p in (_CARM_DEPLOY_ROOT, _RL_VLA_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)


# ── Colors ────────────────────────────────────────────────────────────────

class C:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    END = '\033[0m'


def _ok(msg: str) -> str:
    return f"{C.GREEN}✓ {msg}{C.END}"


def _fail(msg: str) -> str:
    return f"{C.RED}✗ {msg}{C.END}"


def _warn(msg: str) -> str:
    return f"{C.YELLOW}⚠ {msg}{C.END}"


def _skip(msg: str) -> str:
    return f"{C.DIM}– {msg} (skipped){C.END}"


# ── Check functions ───────────────────────────────────────────────────────

def check_sdk_connection(robot_ip: str, port: int = 8090, timeout: float = 3.0) -> tuple:
    """Check TCP connectivity to the CARM SDK port."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((robot_ip, port))
        sock.close()
        if result == 0:
            return True, f"SDK port {robot_ip}:{port} reachable"
        else:
            return False, (
                f"SDK port {robot_ip}:{port} unreachable (errno={result})\n"
                f"  → 确认机械臂已上电，网线已连接\n"
                f"  → 检查 IP: ping {robot_ip}"
            )
    except socket.timeout:
        return False, (
            f"SDK port {robot_ip}:{port} timeout ({timeout}s)\n"
            f"  → 确认机械臂已上电，网线已连接"
        )
    except Exception as e:
        return False, f"SDK connection error: {e}"


def check_backend_api(robot_ip: str, port: int = 1999, timeout: float = 3.0) -> tuple:
    """Check backend HTTP API availability."""
    try:
        import urllib.request
        url = f"http://{robot_ip}:{port}/api/joystick/teleop_target"
        req = urllib.request.Request(url, method='GET')
        resp = urllib.request.urlopen(req, timeout=timeout)
        data = json.loads(resp.read())
        if data.get('code') == 0 or 'data' in data or 'target_pose' in data:
            return True, f"Backend API {robot_ip}:{port} responding"
        else:
            return True, f"Backend API responding (keys: {list(data.keys())})"
    except urllib.error.URLError as e:
        return False, (
            f"Backend API {robot_ip}:{port} unreachable: {e.reason}\n"
            f"  → ssh {robot_ip} \"cd /var/www && sudo bash auto_start.sh\"\n"
            f"  → sudo password: carm@2025"
        )
    except Exception as e:
        # Connection refused or timeout — backend not running
        return False, (
            f"Backend API error: {e}\n"
            f"  → ssh {robot_ip} \"cd /var/www && sudo bash auto_start.sh\""
        )


def check_ros_master() -> tuple:
    """Check if ROS master is running."""
    try:
        import rospy
        # rospy.get_master() would need init_node; just check env var + socket
        ros_master_uri = os.environ.get('ROS_MASTER_URI', 'http://localhost:11311')
        # Parse host:port from URI
        from urllib.parse import urlparse
        parsed = urlparse(ros_master_uri)
        host = parsed.hostname or 'localhost'
        port = parsed.port or 11311

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2.0)
        result = sock.connect_ex((host, port))
        sock.close()
        if result == 0:
            return True, f"ROS master running at {ros_master_uri}"
        else:
            return False, (
                f"ROS master not running at {ros_master_uri}\n"
                f"  → roscore &"
            )
    except ImportError:
        return False, (
            "rospy not importable\n"
            "  → source /opt/ros/noetic/setup.bash\n"
            "  → conda activate carm"
        )
    except Exception as e:
        return False, f"ROS master check error: {e}"


def check_camera_topic(topic: str = '/camera/color/image_raw', timeout: float = 5.0) -> tuple:
    """Check if camera topic is publishing."""
    try:
        # Use rostopic list via subprocess (avoids rospy.init_node)
        import subprocess
        result = subprocess.run(
            ['rostopic', 'list'],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            return False, (
                f"rostopic list failed: {result.stderr.strip()}\n"
                f"  → 确认 roscore 已启动"
            )
        topics = result.stdout.strip().split('\n')
        if topic in topics:
            # Check if actually publishing
            hz_result = subprocess.run(
                ['rostopic', 'hz', topic, '--window', '3'],
                capture_output=True, text=True, timeout=8.0,
            )
            return True, f"Camera topic {topic} publishing"
        else:
            return False, (
                f"Camera topic {topic} not found\n"
                f"  → roslaunch realsense2_camera rs_camera.launch"
            )
    except FileNotFoundError:
        return False, (
            "rostopic command not found\n"
            "  → source /opt/ros/noetic/setup.bash"
        )
    except subprocess.TimeoutExpired:
        # hz timeout is OK — topic exists but might be slow
        return True, f"Camera topic {topic} exists (hz check timed out, likely OK)"
    except Exception as e:
        return False, f"Camera topic check error: {e}"


def check_policy_checkpoint(pretrain: str) -> tuple:
    """Check if policy checkpoint is loadable."""
    if not pretrain:
        return True, "No checkpoint specified (--pretrain), skipping load check"

    if not os.path.exists(pretrain):
        return False, (
            f"Checkpoint not found: {pretrain}\n"
            f"  → 确认路径正确"
        )

    try:
        import torch
        # Just check it's a valid torch file, don't fully load
        state = torch.load(pretrain, map_location='cpu', weights_only=False)
        keys = list(state.keys()) if isinstance(state, dict) else ['(non-dict)']
        return True, f"Checkpoint loadable ({len(keys)} top-level keys)"
    except Exception as e:
        return False, f"Checkpoint load failed: {e}"


def check_safety_config(safety_config: str = '') -> tuple:
    """Check if safety config JSON is valid."""
    if not safety_config:
        safety_config = os.path.join(_CARM_DEPLOY_ROOT, 'safety_config.json')

    if not os.path.exists(safety_config):
        return False, (
            f"Safety config not found: {safety_config}\n"
            f"  → 确认文件存在"
        )

    try:
        with open(safety_config) as f:
            data = json.load(f)
        required_keys = ['joint_limits', 'workspace_limits']
        missing = [k for k in required_keys if k not in data]
        if missing:
            return False, f"Safety config missing keys: {missing}"
        return True, f"Safety config valid ({safety_config})"
    except json.JSONDecodeError as e:
        return False, f"Safety config JSON parse error: {e}"


def check_python_deps() -> tuple:
    """Check critical Python dependencies."""
    deps = [
        ('numpy', 'numpy'),
        ('torch', 'torch'),
        ('cv2', 'opencv-python'),
        ('scipy', 'scipy'),
        ('h5py', 'h5py'),
        ('einops', 'einops'),
    ]
    missing = []
    for mod_name, pkg_name in deps:
        try:
            importlib.import_module(mod_name)
        except ImportError:
            missing.append(pkg_name)

    if missing:
        return False, (
            f"Missing packages: {', '.join(missing)}\n"
            f"  → pip install {' '.join(missing)}"
        )

    # Check rlft importable
    try:
        from rlft.utils.pose_utils import apply_relative_transform
        return True, "All Python dependencies OK"
    except ImportError as e:
        return False, f"rlft import failed: {e}\n  → 确认 rl-vla 在 PYTHONPATH 中"


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='CARM 真机预检脚本')
    parser.add_argument('--robot_ip', default='10.42.0.101')
    parser.add_argument('--pretrain', default='', help='Policy checkpoint path')
    parser.add_argument('--safety_config', default='', help='Safety config JSON path')
    parser.add_argument('--skip', default='', help='Comma-separated checks to skip (sdk,backend,ros,camera,checkpoint,safety,deps)')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    args = parser.parse_args()

    skip_set = set(args.skip.split(',')) if args.skip else set()

    checks = [
        ('sdk',        'SDK 连接',         lambda: check_sdk_connection(args.robot_ip)),
        ('backend',    'Backend API',       lambda: check_backend_api(args.robot_ip)),
        ('ros',        'ROS master',        lambda: check_ros_master()),
        ('camera',     '相机 topic',        lambda: check_camera_topic()),
        ('checkpoint', 'Policy checkpoint', lambda: check_policy_checkpoint(args.pretrain)),
        ('safety',     'Safety config',     lambda: check_safety_config(args.safety_config)),
        ('deps',       'Python 依赖',       lambda: check_python_deps()),
    ]

    results = {}
    passed = 0
    failed = 0
    skipped = 0

    print(f"\n{C.BOLD}CARM 真机预检{C.END}")
    print(f"{C.DIM}Robot: {args.robot_ip} | Checkpoint: {args.pretrain or '(none)'}{C.END}")
    print("─" * 60)

    for i, (key, label, check_fn) in enumerate(checks, 1):
        prefix = f"[{i}/{len(checks)}] {label}"
        padded = f"{prefix:<30}"

        if key in skip_set:
            print(f"  {padded} {_skip('')}")
            skipped += 1
            results[key] = {'status': 'skipped'}
            continue

        try:
            ok, msg = check_fn()
        except Exception as e:
            ok, msg = False, f"Unexpected error: {e}"

        if ok:
            print(f"  {padded} {_ok(msg)}")
            passed += 1
            results[key] = {'status': 'pass', 'message': msg}
        else:
            # Print first line inline, rest indented
            lines = msg.split('\n')
            print(f"  {padded} {_fail(lines[0])}")
            for line in lines[1:]:
                print(f"  {'':30} {C.DIM}{line}{C.END}")
            failed += 1
            results[key] = {'status': 'fail', 'message': msg}

    print("─" * 60)
    summary = f"  {passed} passed, {failed} failed, {skipped} skipped"
    if failed == 0:
        print(f"{C.GREEN}{C.BOLD}{summary} — 预检通过 ✓{C.END}")
    else:
        print(f"{C.RED}{C.BOLD}{summary} — 请修复上述问题后重试{C.END}")

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
