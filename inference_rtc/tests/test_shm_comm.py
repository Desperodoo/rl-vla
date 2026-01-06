#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试共享内存通信

在两个终端分别运行:
  终端1: python -m inference_rtc.tests.test_shm_comm --writer
  终端2: python -m inference_rtc.tests.test_shm_comm --reader
"""

import os
import sys
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from inference_rtc.shared.shm_protocol import ShmKeyframeWriter
from inference_rtc.python.servo_main import ShmKeyframeReader


def run_writer():
    """写入测试"""
    print("=" * 50)
    print("共享内存写入测试 (模拟 10Hz 推理)")
    print("=" * 50)
    
    H = 8
    dof = 7
    dt_key = 1.0 / 30.0
    
    writer = ShmKeyframeWriter.create(H=H, dof=dof, dt_key=dt_key)
    
    print("\n开始写入... (Ctrl+C 退出)")
    
    try:
        count = 0
        while True:
            # 生成测试关键帧 (简单的正弦波)
            t = time.time()
            q_key = np.zeros((H, dof))
            for i in range(H):
                ti = t + i * dt_key
                q_key[i, 0] = 0.5 * np.sin(2 * np.pi * 0.5 * ti)  # 0.5Hz 正弦
                q_key[i, 1] = 0.3 * np.cos(2 * np.pi * 0.5 * ti)
                q_key[i, 6] = 0.04  # 夹爪保持
            
            version = writer.write_keyframes(q_key)
            count += 1
            
            print(f"写入 #{count}: version={version}, q[0]={q_key[0, :3]}")
            
            # 10Hz
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n退出")
    
    finally:
        writer.close()


def run_reader():
    """读取测试"""
    print("=" * 50)
    print("共享内存读取测试 (模拟 500Hz 伺服)")
    print("=" * 50)
    
    reader = ShmKeyframeReader()
    
    print("\n连接共享内存...")
    if not reader.connect(timeout=10.0):
        print("连接失败!")
        return 1
    
    print("开始读取... (Ctrl+C 退出)")
    
    try:
        loop_count = 0
        read_count = 0
        last_print = time.time()
        
        while True:
            data = reader.read()
            if data is not None:
                read_count += 1
                if read_count <= 5 or read_count % 10 == 0:
                    print(f"读取 #{read_count}: version={data['version']}, "
                          f"H={data['H']}, q[0]={data['q_key'][0, :3]}")
            
            loop_count += 1
            
            # 每秒打印统计
            if time.time() - last_print >= 1.0:
                print(f"  [统计] loops={loop_count}, reads={read_count}")
                last_print = time.time()
            
            # 500Hz
            time.sleep(0.002)
    
    except KeyboardInterrupt:
        print("\n退出")
    
    finally:
        reader.disconnect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--writer", action="store_true", help="运行写入测试")
    parser.add_argument("--reader", action="store_true", help="运行读取测试")
    
    args = parser.parse_args()
    
    if args.writer:
        run_writer()
    elif args.reader:
        run_reader()
    else:
        print("请指定 --writer 或 --reader")
        print("\n用法:")
        print("  终端1: python -m inference_rtc.tests.test_shm_comm --writer")
        print("  终端2: python -m inference_rtc.tests.test_shm_comm --reader")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
