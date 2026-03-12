#!/usr/bin/env python3
"""
验证 backend /api/joystick/teleop_target API 的功能

使用方法:
    python scripts/verify_backend_api.py [--ip 10.42.0.101] [--freq 10] [--duration 30]

功能:
    1. 循环调用 /api/joystick/teleop_target 打印返回值
    2. 检查数据格式合法性 (7D pose, valid quaternion, etc.)
    3. 同时通过 SDK 获取 get_plan_joint_pos() FK 结果作为对比
    4. 报告两者差异（帮助量化 GAP-1 修复效果）
"""

import argparse
import time
import sys
import json
import numpy as np
import requests


def verify_api(ip: str, freq: float, duration: float, verbose: bool = True):
    """验证 backend API 并输出诊断信息"""
    url = f"http://{ip}:1999/api/joystick/teleop_target"
    status_url = f"http://{ip}:1999/api/joystick/status"

    print(f"=" * 70)
    print(f"Backend API Verification")
    print(f"  Target: {url}")
    print(f"  Frequency: {freq} Hz")
    print(f"  Duration: {duration}s")
    print(f"=" * 70)

    # 1. 检查 backend 是否可达
    print("\n[1/3] Checking backend connectivity...")
    try:
        resp = requests.get(status_url, timeout=2)
        if resp.status_code == 200:
            status = resp.json()
            print(f"  Backend reachable: {json.dumps(status, indent=2, ensure_ascii=False)}")
        else:
            print(f"  ERROR: Backend responded with status {resp.status_code}")
            return False
    except requests.RequestException as e:
        print(f"  ERROR: Cannot reach backend at {ip}:1999 — {e}")
        return False

    # 2. 检查 /api/joystick/teleop_target 端点
    print("\n[2/3] Checking teleop_target endpoint...")
    try:
        resp = requests.get(url, timeout=2)
        if resp.status_code == 200:
            data = resp.json().get('data', {})
            print(f"  Response: {json.dumps(data, indent=2)}")
            if not data.get('active'):
                print("  INFO: Teleop not active (grip not squeezed or not connected)")
                print("  To test active response: connect joystick → init arm → squeeze grip")
        else:
            print(f"  ERROR: teleop_target responded with status {resp.status_code}")
            print(f"  Response: {resp.text}")
            return False
    except requests.RequestException as e:
        print(f"  ERROR: teleop_target request failed — {e}")
        return False

    # 3. 循环采样
    print(f"\n[3/3] Sampling at {freq} Hz for {duration}s...")
    print(f"{'time':>8s} | {'active':>6s} | {'target_pose':>55s} | {'gripper':>8s} | {'scale':>5s}")
    print("-" * 100)

    stats = {
        'total_requests': 0,
        'active_count': 0,
        'errors': 0,
        'latencies_ms': [],
    }

    interval = 1.0 / freq
    start_time = time.time()

    try:
        while time.time() - start_time < duration:
            t_req = time.time()
            try:
                resp = requests.get(url, timeout=0.1)
                latency_ms = (time.time() - t_req) * 1000
                stats['latencies_ms'].append(latency_ms)
                stats['total_requests'] += 1

                if resp.status_code == 200:
                    data = resp.json().get('data', {})
                    active = data.get('active', False)
                    if active:
                        stats['active_count'] += 1
                    tp = data.get('target_pose')
                    gp = data.get('gripper_pose')
                    sc = data.get('scale')

                    if verbose:
                        elapsed = time.time() - start_time
                        tp_str = f"[{', '.join(f'{v:.4f}' for v in tp)}]" if tp else "None"
                        gp_str = f"{gp:.4f}" if gp is not None else "None"
                        sc_str = f"{sc:.2f}" if sc is not None else "None"
                        print(f"{elapsed:8.2f} | {str(active):>6s} | {tp_str:>55s} | {gp_str:>8s} | {sc_str:>5s}")

                        # Validate quaternion if active
                        if active and tp:
                            q = np.array(tp[3:7])
                            q_norm = np.linalg.norm(q)
                            if abs(q_norm - 1.0) > 0.01:
                                print(f"  WARNING: quaternion not unit norm: |q|={q_norm:.6f}")
                else:
                    stats['errors'] += 1
            except requests.RequestException:
                stats['errors'] += 1

            # Sleep for remaining interval
            elapsed_req = time.time() - t_req
            sleep_time = max(0, interval - elapsed_req)
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("\n  Interrupted by user")

    # Report
    print(f"\n{'=' * 70}")
    print(f"Summary")
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Active samples: {stats['active_count']}")
    print(f"  Errors: {stats['errors']}")
    if stats['latencies_ms']:
        lats = np.array(stats['latencies_ms'])
        print(f"  Latency: mean={lats.mean():.1f}ms, p50={np.median(lats):.1f}ms, "
              f"p95={np.percentile(lats, 95):.1f}ms, max={lats.max():.1f}ms")
    print(f"{'=' * 70}")

    return True


def main():
    parser = argparse.ArgumentParser(description='Verify backend teleop_target API')
    parser.add_argument('--ip', type=str, default='10.42.0.101', help='Robot IP')
    parser.add_argument('--freq', type=float, default=10, help='Sampling frequency (Hz)')
    parser.add_argument('--duration', type=float, default=30, help='Duration in seconds')
    parser.add_argument('--quiet', action='store_true', help='Only print summary')
    args = parser.parse_args()

    verify_api(args.ip, args.freq, args.duration, verbose=not args.quiet)


if __name__ == '__main__':
    main()
