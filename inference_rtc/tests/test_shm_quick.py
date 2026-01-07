#!/usr/bin/env python3
"""快速测试共享内存读写"""

import time
import numpy as np
import sys
import struct
from multiprocessing import shared_memory

SHM_NAME = "rtc_keyframes"

def test_read():
    """测试读取共享内存"""
    print("=" * 50)
    print("测试读取共享内存")
    print("=" * 50)
    
    try:
        shm = shared_memory.SharedMemory(name=SHM_NAME)
        buf = np.ndarray((shm.size,), dtype=np.uint8, buffer=shm.buf)
        print(f"连接成功: {SHM_NAME}, size={shm.size} bytes")
        
        # 读取 header
        header_bytes = bytes(buf[:32])
        version = struct.unpack_from('<Q', header_bytes, 0)[0]
        t_write = struct.unpack_from('<d', header_bytes, 8)[0]
        dof = struct.unpack_from('<i', header_bytes, 16)[0]
        H = struct.unpack_from('<i', header_bytes, 20)[0]
        dt_key = struct.unpack_from('<d', header_bytes, 24)[0]
        
        print(f"\nHeader:")
        print(f"  version: {version}")
        print(f"  t_write: {t_write:.6f}")
        print(f"  dof: {dof}")
        print(f"  H: {H}")
        print(f"  dt_key: {dt_key:.6f}")
        
        # 当前时间
        t_now = time.clock_gettime(time.CLOCK_MONOTONIC)
        print(f"\n时间信息:")
        print(f"  t_now: {t_now:.6f}")
        print(f"  t_now - t_write: {(t_now - t_write)*1000:.1f} ms")
        
        if H > 0 and dof > 0:
            t_end = t_write + (H - 1) * dt_key
            print(f"  t_end: {t_end:.6f}")
            print(f"  remaining: {(t_end - t_now)*1000:.1f} ms")
            
            # 读取第一帧关键帧
            payload_size = H * dof * 8
            q_key_bytes = bytes(buf[32:32 + payload_size])
            q_key = np.frombuffer(q_key_bytes, dtype=np.float64).reshape(H, dof)
            print(f"\n第一帧关键帧 (q_key[0]):")
            print(f"  {q_key[0]}")
        
        shm.close()
        print("\n✓ 测试完成")
        
    except FileNotFoundError:
        print(f"✗ 共享内存不存在: {SHM_NAME}")
        print("  请先运行 inference_main.py")
        return False
    
    return True

def test_continuous_read(duration: float = 5.0):
    """持续读取测试"""
    print("=" * 50)
    print(f"持续读取测试 ({duration}s)")
    print("=" * 50)
    
    try:
        shm = shared_memory.SharedMemory(name=SHM_NAME)
        buf = np.ndarray((shm.size,), dtype=np.uint8, buffer=shm.buf)
        print(f"连接成功: {SHM_NAME}")
        
        last_version = 0
        update_count = 0
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # 读取 version
            header_bytes = bytes(buf[:32])
            version = struct.unpack_from('<Q', header_bytes, 0)[0]
            
            if version != last_version and version != 0:
                t_write = struct.unpack_from('<d', header_bytes, 8)[0]
                t_now = time.clock_gettime(time.CLOCK_MONOTONIC)
                delay = (t_now - t_write) * 1000
                
                print(f"v={version}, delay={delay:.1f}ms")
                last_version = version
                update_count += 1
            
            time.sleep(0.01)  # 100Hz 检查
        
        print(f"\n共收到 {update_count} 次更新 ({update_count/duration:.1f} Hz)")
        shm.close()
        
    except FileNotFoundError:
        print(f"✗ 共享内存不存在: {SHM_NAME}")
        return False
    
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "continuous":
        test_continuous_read()
    else:
        test_read()
