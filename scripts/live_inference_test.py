#!/usr/bin/env python3
"""
人工联合调试脚本 — 交互式清单逐步验证推理全流程。

需要: ROS、相机、机械臂上电、策略模型 checkpoint。

测试清单:
  1. 初始位姿到达（无碰撞）
  2. 单步推理 → 策略输出有效动作
  3. 多步 chunk 执行 → 平滑运动
  4. 安全裁剪触发
  5. 键盘干预 (WASD + G/H)
  6. Episode 录制 (R 启停, Y/N 保存)

Usage:
    # 先启动 ROS
    roscore &
    roslaunch realsense2_camera rs_camera.launch

    # 然后运行此脚本
    python scripts/live_inference_test.py \\
        --pretrain /path/to/model.pt \\
        --safety_config /path/to/safety_config.json
"""

import argparse
import os
import signal
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


# ── Colors ─────────────────────────────────────────────────────────────────

class C:
    H = '\033[95m'
    B = '\033[94m'
    G = '\033[92m'
    Y = '\033[93m'
    R = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'


def cprint(msg, color=C.END):
    print(f"{color}{msg}{C.END}")


def prompt(msg: str) -> str:
    try:
        return input(f"\n{C.Y}{msg}{C.END}").strip()
    except (KeyboardInterrupt, EOFError):
        return 'q'


def section(title: str, description: str):
    cprint("\n" + "=" * 60, C.BOLD)
    cprint(f"  {title}", C.BOLD)
    cprint(f"  {description}", C.B)
    cprint("=" * 60, C.BOLD)


# ── Test steps ─────────────────────────────────────────────────────────────

CHECKLIST = [
    {
        'id': 1,
        'title': '初始位姿到达',
        'description': '机械臂移动到初始位姿，确认无碰撞、位姿正确',
        'action': '观察机械臂是否安全到达初始位姿',
        'auto': False,
    },
    {
        'id': 2,
        'title': '单步策略推理',
        'description': '策略执行一步，观察输出动作是否合理',
        'action': '观察终端日志: action shape、数值范围是否正常',
        'auto': False,
    },
    {
        'id': 3,
        'title': '多步 Chunk 连续执行',
        'description': '连续执行多个推理步，观察运动平滑度',
        'action': '观察机械臂运动是否平滑、无抖动',
        'auto': False,
    },
    {
        'id': 4,
        'title': '安全裁剪验证',
        'description': '策略输出大动作时，安全控制器应 clip',
        'action': '观察终端中 "Safety" 或 "clip" 日志',
        'auto': False,
    },
    {
        'id': 5,
        'title': '键盘干预 (如已启用)',
        'description': '通过 WASD/QE 控制平移，G/H 控制夹爪',
        'action': '按键测试干预是否生效',
        'auto': False,
    },
    {
        'id': 6,
        'title': 'Episode 录制 (如已启用)',
        'description': '按 R 开始录制，再按 R 停止，Y 保存 / N 丢弃',
        'action': '确认 HDF5 文件已保存',
        'auto': False,
    },
    {
        'id': 7,
        'title': '优雅关闭',
        'description': '按 Ctrl+C，观察回零位过程',
        'action': '确认机械臂安全回位、无报错',
        'auto': False,
    },
]


def run_checklist(node_process):
    """Run through the interactive checklist."""
    results = {}

    cprint("\n" + "=" * 60, C.BOLD)
    cprint("  LIVE INFERENCE TEST CHECKLIST", C.BOLD)
    cprint("  逐步验证推理全流程", C.B)
    cprint("  输入 p=PASS, f=FAIL, s=SKIP, q=QUIT", C.B)
    cprint("=" * 60, C.BOLD)

    for item in CHECKLIST:
        section(f"[{item['id']}] {item['title']}", item['description'])
        cprint(f"\n  操作: {item['action']}", C.Y)

        ans = prompt(f"  结果? [p]ass / [f]ail / [s]kip / [q]uit: ").lower()

        if ans in ('p', 'pass', ''):
            results[item['id']] = 'PASS'
            cprint(f"  [{item['id']}] PASS", C.G)
        elif ans in ('f', 'fail'):
            results[item['id']] = 'FAIL'
            note = prompt("  失败原因 (可选): ")
            cprint(f"  [{item['id']}] FAIL: {note}", C.R)
        elif ans in ('s', 'skip'):
            results[item['id']] = 'SKIP'
            cprint(f"  [{item['id']}] SKIPPED", C.Y)
        elif ans in ('q', 'quit'):
            cprint("  退出清单", C.Y)
            break

    # Summary
    cprint("\n" + "=" * 60, C.BOLD)
    cprint("  SUMMARY", C.BOLD)
    cprint("=" * 60, C.BOLD)
    for item in CHECKLIST:
        r = results.get(item['id'], 'NOT RUN')
        color = {
            'PASS': C.G, 'FAIL': C.R, 'SKIP': C.Y,
        }.get(r, C.END)
        cprint(f"  [{item['id']}] {item['title']}: {r}", color)

    passed = sum(1 for v in results.values() if v == 'PASS')
    failed = sum(1 for v in results.values() if v == 'FAIL')
    cprint(f"\n  {passed} passed, {failed} failed, "
           f"{len(CHECKLIST) - len(results)} not run", C.BOLD)

    return results


def main():
    parser = argparse.ArgumentParser(description='Live inference test checklist')
    parser.add_argument('--pretrain', type=str, required=True,
                        help='Path to policy checkpoint')
    parser.add_argument('--safety_config', type=str, default='',
                        help='Path to safety config JSON')
    parser.add_argument('--execution_mode', type=str, default='receding_horizon',
                        choices=['temporal_ensemble', 'receding_horizon'])
    parser.add_argument('--inference_speed_scale', type=float, default=1.0)
    parser.add_argument('--control_freq', type=int, default=50)
    parser.add_argument('--intervention', action='store_true',
                        help='Enable keyboard intervention')
    parser.add_argument('--record_inference', action='store_true',
                        help='Enable inference recording')
    parser.add_argument('--record_dir', type=str, default='')
    parser.add_argument('--max_steps', type=int, default=300,
                        help='Max steps per episode (default 300 = ~10s at 30Hz)')
    args = parser.parse_args()

    # Validate checkpoint
    if not os.path.exists(args.pretrain):
        cprint(f"Checkpoint not found: {args.pretrain}", C.R)
        sys.exit(1)

    section("PRE-FLIGHT CHECK", "确认环境就绪")
    cprint(f"  Checkpoint: {args.pretrain}", C.B)
    cprint(f"  Safety config: {args.safety_config or '(default)'}", C.B)
    cprint(f"  Execution mode: {args.execution_mode}", C.B)
    cprint(f"  Speed scale: {args.inference_speed_scale}", C.B)
    cprint(f"  Intervention: {args.intervention}", C.B)
    cprint(f"  Recording: {args.record_inference}", C.B)

    checklist = [
        "ROS core 已启动 (roscore)",
        "相机节点已启动 (realsense)",
        "机械臂已上电",
        "工作空间已清空",
        "E-stop 就绪",
    ]
    cprint("\n  请确认以下条件:", C.Y)
    for i, item in enumerate(checklist, 1):
        cprint(f"    {i}. {item}", C.B)

    ans = prompt("  全部就绪? [Y/n]: ")
    if ans.lower() in ('n', 'no', 'q'):
        cprint("Aborted", C.R)
        return

    # Launch inference node in a subprocess
    cprint("\n  启动推理节点...", C.B)
    cprint("  (推理节点将在后台运行，此脚本提供交互式清单)", C.B)

    # Build the rosrun command
    cmd_parts = [
        'rosrun', 'carm_deploy', 'inference_ros.py',
        '--pretrain', args.pretrain,
        '--execution_mode', args.execution_mode,
        '--inference_speed_scale', str(args.inference_speed_scale),
        '--control_freq', str(args.control_freq),
        '--max_steps', str(args.max_steps),
    ]
    if args.safety_config:
        cmd_parts += ['--safety_config', args.safety_config]
    if args.intervention:
        cmd_parts.append('--intervention')
    if args.record_inference:
        cmd_parts.append('--record_inference')
    if args.record_dir:
        cmd_parts += ['--record_dir', args.record_dir]

    cmd_str = ' '.join(cmd_parts)
    cprint(f"\n  命令: {cmd_str}", C.B)
    cprint("  (请在另一个终端窗口运行此命令)", C.Y)
    cprint("  或者切到 tmux/screen 的另一个 pane 中运行", C.Y)

    wait_ans = prompt("  推理节点已在另一个终端启动? [Y/n]: ")
    if wait_ans.lower() in ('n', 'no', 'q'):
        cprint("Aborted", C.R)
        return

    # Run the checklist
    results = run_checklist(None)

    # Save results
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(
        _RL_VLA_ROOT, 'logs', f'live_test_report_{timestamp}.txt',
    )
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(f"Live Inference Test Report — {timestamp}\n")
        f.write(f"Checkpoint: {args.pretrain}\n")
        f.write(f"Safety config: {args.safety_config}\n\n")
        for item in CHECKLIST:
            r = results.get(item['id'], 'NOT RUN')
            f.write(f"[{item['id']}] {item['title']}: {r}\n")
    cprint(f"\n  Report saved: {report_path}", C.G)


if __name__ == '__main__':
    main()
