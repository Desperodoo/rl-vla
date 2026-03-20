#!/usr/bin/env python3
"""
真机 Mock 动作测试 — 不需要 ROS/相机，仅需机械臂 SDK 上电。

测试项:
  1. 初始化位姿 → 验证到达
  2. 小幅 X/Y/Z 平移 → 验证方向正确
  3. 夹爪开/关 → 验证响应
  4. 工作空间边界附近 → 验证 safety clip
  5. 预定义轨迹回放 → 验证 track_pose 平滑执行

每步有人工确认提示，安全优先。
自动记录 before/after 位姿到 JSON 报告。

Usage:
    python scripts/test_mock_arm_motion.py [--robot_ip 10.42.0.101] [--speed 2.0]
    python scripts/test_mock_arm_motion.py --test 2   # 跑单个测试
    python scripts/test_mock_arm_motion.py --list      # 列出所有测试
    python scripts/test_mock_arm_motion.py --log_dir /path/to/logs  # 保存 JSON 报告
    python scripts/test_mock_arm_motion.py --auto      # 减少手动确认（仅首次+安全测试）
"""

import argparse
import json
import sys
import os
import time
import numpy as np
from datetime import datetime

# Path setup
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_RL_VLA_ROOT = os.path.dirname(_SCRIPT_DIR)
_CARM_DEPLOY_ROOT = os.path.join(
    _RL_VLA_ROOT, 'carm_ros_deploy', 'src', 'carm_deploy',
)
for p in (_CARM_DEPLOY_ROOT, _RL_VLA_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from carm import carm_py  # type: ignore
from core.safety_controller import SafetyController

# ── Constants ──────────────────────────────────────────────────────────────

# Realistic rest pose from real robot
HOME_POSE = [0.2475, 0.0014, 0.3251, 0.9996, -0.0034, 0.0255, -0.0074]
HOME_GRIPPER = 0.078
TAU = 10.0
MAX_SPEED = 2.0  # safe speed level (0-10)

# Position tolerance for arrival check (meters)
POS_TOLERANCE = 0.005  # 5mm
ARRIVAL_TIMEOUT = 5.0  # seconds

# Global flags (set by argparse)
AUTO_MODE = False
LOG_DIR = None


# ── Test Log Recorder ─────────────────────────────────────────────────────

class TestLog:
    """Records before/after poses and results for each test step."""

    def __init__(self):
        self.entries = []
        self.start_time = datetime.now().isoformat()

    def record(self, test_id: int, test_name: str, passed: bool,
               before_pose=None, after_pose=None, details=None):
        entry = {
            'test_id': test_id,
            'test_name': test_name,
            'passed': passed,
            'timestamp': datetime.now().isoformat(),
        }
        if before_pose is not None:
            entry['before_pose'] = np.asarray(before_pose).tolist()
        if after_pose is not None:
            entry['after_pose'] = np.asarray(after_pose).tolist()
        if details:
            entry['details'] = details
        self.entries.append(entry)

    def save(self, path: str):
        report = {
            'start_time': self.start_time,
            'end_time': datetime.now().isoformat(),
            'total': len(self.entries),
            'passed': sum(1 for e in self.entries if e['passed']),
            'failed': sum(1 for e in self.entries if not e['passed']),
            'tests': self.entries,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        cprint(f"  报告已保存: {path}", Colors.GREEN)


# Global test log
_test_log = TestLog()


# ── Utilities ──────────────────────────────────────────────────────────────

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'


def cprint(msg, color=Colors.END):
    print(f"{color}{msg}{Colors.END}")


def prompt_confirm(msg: str, force_ask: bool = False) -> bool:
    """Ask Y/n confirmation. In AUTO_MODE, auto-confirm unless force_ask."""
    if AUTO_MODE and not force_ask:
        cprint(f"  [auto] {msg} → Y", Colors.BLUE)
        return True
    try:
        ans = input(f"\n{Colors.YELLOW}{msg} [Y/n]: {Colors.END}").strip().lower()
        return ans in ('', 'y', 'yes')
    except (KeyboardInterrupt, EOFError):
        cprint("\nAborted by user", Colors.RED)
        return False


def wait_for_enter(msg: str = "Press ENTER to continue..."):
    try:
        input(f"\n{Colors.BLUE}{msg}{Colors.END}")
    except (KeyboardInterrupt, EOFError):
        cprint("\nAborted", Colors.RED)
        sys.exit(0)


def get_ee_pose(arm) -> np.ndarray:
    """Get current end-effector pose as [x,y,z,qx,qy,qz,qw]."""
    return np.array(arm.get_cart_pose())


def check_arrival(arm, target_xyz: list, timeout: float = ARRIVAL_TIMEOUT) -> bool:
    """Wait until arm reaches target XYZ within tolerance."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        current = get_ee_pose(arm)
        dist = np.linalg.norm(current[:3] - np.array(target_xyz))
        if dist < POS_TOLERANCE:
            return True
        time.sleep(0.05)
    return False


# ── Test functions ─────────────────────────────────────────────────────────

def test_1_init_pose(arm):
    """Test 1: 初始化位姿到达"""
    cprint("=" * 60, Colors.HEADER)
    cprint("TEST 1: 初始化位姿到达", Colors.BOLD)
    cprint(f"  目标: {HOME_POSE[:3]} (XYZ)", Colors.BLUE)
    cprint("=" * 60, Colors.HEADER)

    if not prompt_confirm("即将移动到初始位姿，工作空间是否清空？", force_ask=True):
        return False

    before = get_ee_pose(arm)
    arm.set_speed_level(MAX_SPEED)
    arm.move_pose(HOME_POSE)
    arm.set_gripper(HOME_GRIPPER, TAU)
    time.sleep(1.0)

    current = get_ee_pose(arm)
    dist = np.linalg.norm(current[:3] - np.array(HOME_POSE[:3]))
    cprint(f"  当前位姿: {current[:3].round(4).tolist()}", Colors.BLUE)
    cprint(f"  距离误差: {dist*1000:.1f} mm", Colors.BLUE)

    passed = dist < POS_TOLERANCE
    _test_log.record(1, "初始化位姿到达", passed,
                     before_pose=before, after_pose=current,
                     details={'error_mm': round(dist * 1000, 1),
                              'target': HOME_POSE[:3]})
    if passed:
        cprint("  PASS: 到达初始位姿", Colors.GREEN)
        return True
    else:
        cprint(f"  FAIL: 误差 {dist*1000:.1f}mm > {POS_TOLERANCE*1000:.1f}mm", Colors.RED)
        return False


def test_2_xyz_translation(arm):
    """Test 2: 小幅 X/Y/Z 平移验证方向"""
    cprint("=" * 60, Colors.HEADER)
    cprint("TEST 2: XYZ 平移方向验证", Colors.BOLD)
    cprint("  依次沿 +X, +Y, +Z 移动 20mm 并返回", Colors.BLUE)
    cprint("=" * 60, Colors.HEADER)

    # First go home
    arm.move_pose(HOME_POSE)
    time.sleep(1.0)

    axes = [('X', 0), ('Y', 1), ('Z', 2)]
    delta = 0.020  # 20mm

    all_pass = True
    axis_details = []
    for axis_name, axis_idx in axes:
        if not prompt_confirm(f"即将沿 +{axis_name} 移动 {delta*1000:.0f}mm"):
            return False

        base = get_ee_pose(arm)
        target = base.copy()
        target[axis_idx] += delta

        arm.move_pose(target.tolist())
        time.sleep(1.0)
        after = get_ee_pose(arm)

        actual_delta = after[axis_idx] - base[axis_idx]
        other_drift = np.linalg.norm(
            np.delete(after[:3] - base[:3], axis_idx)
        )

        cprint(f"  {axis_name}: 期望 +{delta*1000:.0f}mm, "
               f"实际 {actual_delta*1000:+.1f}mm, "
               f"其他轴漂移 {other_drift*1000:.1f}mm", Colors.BLUE)

        ok = abs(actual_delta - delta) < POS_TOLERANCE and other_drift < POS_TOLERANCE * 3
        axis_details.append({
            'axis': axis_name,
            'expected_mm': delta * 1000,
            'actual_mm': round(actual_delta * 1000, 1),
            'drift_mm': round(other_drift * 1000, 1),
            'passed': ok,
        })
        if ok:
            cprint(f"  {axis_name}: PASS", Colors.GREEN)
        else:
            cprint(f"  {axis_name}: FAIL", Colors.RED)
            all_pass = False

        # Return
        arm.move_pose(HOME_POSE)
        time.sleep(0.5)

    _test_log.record(2, "XYZ 平移方向验证", all_pass, details={'axes': axis_details})
    return all_pass


def test_3_gripper(arm):
    """Test 3: 夹爪开/关"""
    cprint("=" * 60, Colors.HEADER)
    cprint("TEST 3: 夹爪开/关", Colors.BOLD)
    cprint("  开 → 关 → 开 循环", Colors.BLUE)
    cprint("=" * 60, Colors.HEADER)

    if not prompt_confirm("即将测试夹爪，确保夹爪附近无障碍物"):
        return False

    operations = [
        ("开 (0.078m)", 0.078),
        ("关 (0.0m)", 0.0),
        ("半开 (0.04m)", 0.04),
        ("全开 (0.078m)", 0.078),
    ]

    gripper_details = []
    for desc, target in operations:
        cprint(f"  夹爪 → {desc}", Colors.BLUE)
        arm.set_gripper(target, TAU)
        time.sleep(1.0)
        actual = arm.get_gripper_pos()
        diff = abs(actual - target)
        cprint(f"    实际: {actual:.4f}m, 误差: {diff*1000:.1f}mm", Colors.BLUE)
        gripper_details.append({
            'operation': desc,
            'target': target,
            'actual': round(actual, 4),
            'error_mm': round(diff * 1000, 1),
        })

    cprint("  请目视确认夹爪运动正确", Colors.YELLOW)
    if AUTO_MODE:
        # In auto mode, pass if all errors < 5mm
        max_err = max(d['error_mm'] for d in gripper_details)
        passed = max_err < 5.0
        cprint(f"  [auto] 最大误差 {max_err:.1f}mm → {'PASS' if passed else 'FAIL'}", Colors.BLUE)
    else:
        passed = prompt_confirm("夹爪工作正常？")

    _test_log.record(3, "夹爪开/关", passed, details={'operations': gripper_details})
    if passed:
        cprint("  PASS", Colors.GREEN)
        return True
    else:
        cprint("  FAIL", Colors.RED)
        return False


def test_4_workspace_boundary(arm):
    """Test 4: 工作空间边界 safety clip 验证"""
    cprint("=" * 60, Colors.HEADER)
    cprint("TEST 4: 工作空间边界 Safety Clip", Colors.BOLD)
    cprint("  尝试发送超出工作空间的目标", Colors.BLUE)
    cprint("=" * 60, Colors.HEADER)

    # Load default safety controller
    safety_config_path = os.path.join(_CARM_DEPLOY_ROOT, 'safety_config.json')
    if os.path.exists(safety_config_path):
        safety = SafetyController.from_config(safety_config_path)
        cprint(f"  Safety config loaded: {safety_config_path}", Colors.BLUE)
    else:
        safety = SafetyController()
        cprint("  Using default safety controller (no config file)", Colors.YELLOW)

    if not prompt_confirm("即将测试工作空间边界 clip", force_ask=True):
        return False

    # Go home first
    arm.move_pose(HOME_POSE)
    time.sleep(1.0)

    # Attempt extreme X (far reach)
    extreme_pose = HOME_POSE.copy()
    extreme_pose[0] = 0.50  # 50cm X — likely outside workspace
    extreme_np = np.array(extreme_pose[:7])

    clipped_pose, warnings = safety.check_workspace(extreme_np)
    passed = len(warnings) > 0
    details = {
        'original_x': extreme_pose[0],
        'clipped_x': round(float(clipped_pose[0]), 4),
        'warnings': warnings,
        'clip_triggered': passed,
    }
    _test_log.record(4, "工作空间边界 Safety Clip", passed, details=details)

    if passed:
        cprint(f"  Safety clip triggered: {warnings}", Colors.GREEN)
        cprint(f"  Original X: {extreme_pose[0]:.3f} → Clipped X: {clipped_pose[0]:.3f}", Colors.BLUE)
        cprint("  PASS: Safety controller correctly clips", Colors.GREEN)
        return True
    else:
        cprint("  WARNING: No clip triggered for extreme pose", Colors.YELLOW)
        cprint("  (This may be expected if workspace config has wide bounds)", Colors.YELLOW)
        return prompt_confirm("结果是否符合预期？", force_ask=True)


def test_5_trajectory_replay(arm):
    """Test 5: 预定义轨迹回放 — track_pose 平滑执行"""
    cprint("=" * 60, Colors.HEADER)
    cprint("TEST 5: 预定义轨迹回放 (track_pose)", Colors.BOLD)
    cprint("  以 50Hz 执行小圆弧轨迹（XY平面，半径 15mm）", Colors.BLUE)
    cprint("=" * 60, Colors.HEADER)

    if not prompt_confirm("即将执行小圆弧轨迹（~15mm振幅），确保空间清空"):
        return False

    # Go home
    arm.move_pose(HOME_POSE)
    time.sleep(1.0)

    base = np.array(HOME_POSE)
    radius = 0.015  # 15mm
    num_steps = 100  # 2 seconds at 50Hz
    freq = 50

    cprint(f"  轨迹: XY 圆弧, 半径={radius*1000:.0f}mm, "
           f"步数={num_steps}, 频率={freq}Hz", Colors.BLUE)

    before = get_ee_pose(arm)
    dt = 1.0 / freq
    actual_positions = []
    for i in range(num_steps):
        t = 2 * np.pi * i / num_steps
        target = base.copy()
        target[0] += radius * np.cos(t)
        target[1] += radius * np.sin(t)
        arm.track_pose(target.tolist())
        time.sleep(dt)
        if i % 10 == 0:  # sample every 10 steps
            actual_positions.append(get_ee_pose(arm)[:3].tolist())

    # Return home
    arm.move_pose(HOME_POSE)
    time.sleep(1.0)
    after = get_ee_pose(arm)

    # Check return-to-home accuracy
    return_err = np.linalg.norm(after[:3] - np.array(HOME_POSE[:3]))
    cprint(f"  轨迹执行完成, 回位误差: {return_err*1000:.1f}mm", Colors.BLUE)

    if AUTO_MODE:
        passed = return_err < POS_TOLERANCE * 2  # 10mm tolerance for trajectory
        cprint(f"  [auto] 回位误差 {return_err*1000:.1f}mm → {'PASS' if passed else 'FAIL'}", Colors.BLUE)
    else:
        cprint("  请目视确认运动平滑、无抖动", Colors.YELLOW)
        passed = prompt_confirm("轨迹回放正常？")

    _test_log.record(5, "预定义轨迹回放", passed,
                     before_pose=before, after_pose=after,
                     details={
                         'radius_mm': radius * 1000,
                         'num_steps': num_steps,
                         'freq_hz': freq,
                         'return_error_mm': round(return_err * 1000, 1),
                         'sampled_positions': actual_positions,
                     })
    if passed:
        cprint("  PASS", Colors.GREEN)
        return True
    else:
        cprint("  FAIL", Colors.RED)
        return False


# ── All tests ──────────────────────────────────────────────────────────────

ALL_TESTS = [
    (1, "初始化位姿到达", test_1_init_pose),
    (2, "XYZ 平移方向验证", test_2_xyz_translation),
    (3, "夹爪开/关", test_3_gripper),
    (4, "工作空间边界 Safety Clip", test_4_workspace_boundary),
    (5, "预定义轨迹回放 (track_pose)", test_5_trajectory_replay),
]


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    global AUTO_MODE, LOG_DIR, MAX_SPEED

    parser = argparse.ArgumentParser(description='Mock arm motion tests')
    parser.add_argument('--robot_ip', default='10.42.0.101')
    parser.add_argument('--speed', type=float, default=MAX_SPEED,
                        help='Speed level for moves (0-10, default 2.0)')
    parser.add_argument('--test', type=int, default=None,
                        help='Run a single test by number')
    parser.add_argument('--list', action='store_true',
                        help='List all tests and exit')
    parser.add_argument('--auto', action='store_true',
                        help='Auto-confirm most prompts (only ask for first move + safety)')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Directory to save mock_results.json report')
    args = parser.parse_args()

    AUTO_MODE = args.auto
    LOG_DIR = args.log_dir

    if args.list:
        cprint("Available tests:", Colors.BOLD)
        for num, name, _ in ALL_TESTS:
            cprint(f"  {num}. {name}")
        return

    MAX_SPEED = min(args.speed, 3.0)  # cap at 3.0

    cprint("=" * 60, Colors.BOLD)
    cprint("CARM Mock Arm Motion Tests", Colors.BOLD)
    cprint(f"  Robot: {args.robot_ip}", Colors.BLUE)
    cprint(f"  Speed: {MAX_SPEED}", Colors.BLUE)
    cprint("=" * 60, Colors.BOLD)

    if not prompt_confirm("即将连接机械臂，确认已上电且工作空间清空？", force_ask=True):
        cprint("Aborted", Colors.RED)
        return

    # Connect
    cprint(f"\n连接机械臂 {args.robot_ip}...", Colors.BLUE)
    arm = carm_py.CArmSingleCol(args.robot_ip)
    time.sleep(1.0)
    arm.set_ready()
    arm.set_control_mode(4)  # PF mode
    arm.set_speed_level(MAX_SPEED)

    status = arm.get_status()
    cprint(f"  连接成功: {status.arm_name}, mode={status.state}", Colors.GREEN)

    # Run tests
    tests_to_run = ALL_TESTS
    if args.test is not None:
        tests_to_run = [(n, name, fn) for n, name, fn in ALL_TESTS if n == args.test]
        if not tests_to_run:
            cprint(f"Test {args.test} not found", Colors.RED)
            return

    results = []
    for num, name, fn in tests_to_run:
        try:
            ok = fn(arm)
            results.append((num, name, ok))
        except KeyboardInterrupt:
            cprint(f"\n  Test {num} interrupted", Colors.YELLOW)
            results.append((num, name, None))
            break
        except Exception as e:
            cprint(f"  ERROR in test {num}: {e}", Colors.RED)
            results.append((num, name, False))

    # Return home and summary
    try:
        arm.set_speed_level(MAX_SPEED)
        arm.move_pose(HOME_POSE)
        time.sleep(1.0)
    except Exception:
        pass

    cprint("\n" + "=" * 60, Colors.BOLD)
    cprint("RESULTS", Colors.BOLD)
    cprint("=" * 60, Colors.BOLD)
    for num, name, ok in results:
        if ok is True:
            cprint(f"  [{num}] {name}: PASS", Colors.GREEN)
        elif ok is False:
            cprint(f"  [{num}] {name}: FAIL", Colors.RED)
        else:
            cprint(f"  [{num}] {name}: SKIPPED", Colors.YELLOW)

    passed = sum(1 for _, _, ok in results if ok is True)
    total = len(results)
    cprint(f"\n  {passed}/{total} passed", Colors.BOLD)

    # Save JSON report
    if LOG_DIR:
        report_path = os.path.join(LOG_DIR, 'mock_results.json')
        _test_log.save(report_path)


if __name__ == '__main__':
    main()
