#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相机 + 推理延迟测试

参考 Physical-Intelligence/real-time-chunking-kinetix 官方仓库的核心概念:
- inference_delay: 推理期间需要执行的前一轮动作数量
- execute_horizon: 每轮执行的总动作数量
- action_chunk_size: 策略输出的动作序列长度

本测试测量完整管道延迟:
1. 相机采集延迟 (acquire)
2. 预处理延迟 (preprocess)
3. 策略推理延迟 (inference)
4. 端到端延迟 (E2E)

目标:
- 推理频率: 10Hz (100ms 间隔)
- 推理延迟 < 100ms (留出足够的 inference_delay 裕量)
"""

import os
import sys
import time
import argparse
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import cv2

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class LatencyStats:
    """延迟统计"""
    name: str
    samples: List[float]
    
    @property
    def count(self) -> int:
        return len(self.samples)
    
    @property
    def mean(self) -> float:
        return np.mean(self.samples) * 1000 if self.samples else 0
    
    @property
    def std(self) -> float:
        return np.std(self.samples) * 1000 if self.samples else 0
    
    @property
    def p50(self) -> float:
        return np.percentile(self.samples, 50) * 1000 if self.samples else 0
    
    @property
    def p95(self) -> float:
        return np.percentile(self.samples, 95) * 1000 if self.samples else 0
    
    @property
    def p99(self) -> float:
        return np.percentile(self.samples, 99) * 1000 if self.samples else 0
    
    @property
    def min(self) -> float:
        return np.min(self.samples) * 1000 if self.samples else 0
    
    @property
    def max(self) -> float:
        return np.max(self.samples) * 1000 if self.samples else 0
    
    def summary(self) -> str:
        return (
            f"{self.name}:\n"
            f"  Mean: {self.mean:.2f} ms\n"
            f"  Std:  {self.std:.2f} ms\n"
            f"  P50:  {self.p50:.2f} ms\n"
            f"  P95:  {self.p95:.2f} ms\n"
            f"  P99:  {self.p99:.2f} ms\n"
            f"  Min:  {self.min:.2f} ms\n"
            f"  Max:  {self.max:.2f} ms"
        )


class LatencyTester:
    """延迟测试器"""
    
    def __init__(
        self,
        checkpoint_path: str = None,
        num_samples: int = 100,
        warmup_samples: int = 10,
        use_real_camera: bool = True,
        device: str = "cuda",
        verbose: bool = True,
    ):
        self.checkpoint_path = checkpoint_path
        self.num_samples = num_samples
        self.warmup_samples = warmup_samples
        self.use_real_camera = use_real_camera
        self.device = device
        self.verbose = verbose
        
        # 组件
        self.camera_manager = None
        self.policy_runner = None
        
        # 延迟记录
        self.acquire_times: List[float] = []
        self.preprocess_times: List[float] = []
        self.inference_times: List[float] = []
        self.e2e_times: List[float] = []
        
        # 配置
        self.image_size = (224, 224)  # 策略输入尺寸
    
    def setup(self):
        """初始化组件"""
        print("\n" + "=" * 60)
        print("延迟测试初始化")
        print("=" * 60)
        
        # 1. 初始化相机
        self.available_cameras = []
        if self.use_real_camera:
            print("\n[1/2] 初始化相机...")
            try:
                from inference.camera_manager import CameraManager, DEFAULT_CAMERA_CONFIGS
                self.camera_manager = CameraManager(DEFAULT_CAMERA_CONFIGS)
                if self.camera_manager.initialize(auto_assign=True):
                    self.camera_manager.start()
                    # 等待相机稳定
                    time.sleep(1.0)
                    # 检查哪些相机可用
                    self.available_cameras = list(self.camera_manager.cameras.keys())
                    print(f"  ✓ 相机初始化完成")
                    print(f"    可用相机: {self.available_cameras}")
                    if len(self.available_cameras) < 2:
                        print(f"    [注意] 缺少部分相机，将使用模拟数据填充")
                else:
                    print("  ✗ 相机初始化失败，使用模拟数据")
                    self.camera_manager = None
            except Exception as e:
                print(f"  ✗ 相机初始化错误: {e}")
                self.camera_manager = None
        else:
            print("\n[1/2] 跳过相机 (使用模拟数据)")
            self.camera_manager = None
        
        # 2. 加载策略
        if self.checkpoint_path:
            print("\n[2/2] 加载策略模型...")
            try:
                from inference_rtc.python.policy_runner import PolicyRunner
                from inference_rtc.python.config import InferenceConfig
                
                config = InferenceConfig()
                self.policy_runner = PolicyRunner(
                    checkpoint_path=self.checkpoint_path,
                    config=config,
                    device=self.device,
                    use_ema=True,
                    verbose=False,
                )
                self.policy_runner.load_model()
                self.policy_runner.warmup()
                print("  ✓ 策略加载完成")
            except Exception as e:
                print(f"  ✗ 策略加载失败: {e}")
                import traceback
                traceback.print_exc()
                self.policy_runner = None
        else:
            print("\n[2/2] 跳过策略 (仅测试相机)")
            self.policy_runner = None
        
        print("\n初始化完成")
        if self.camera_manager:
            print(f"  相机: 真实 ({', '.join(self.available_cameras)})")
            if len(self.available_cameras) < 2:
                missing = set(['wrist', 'external']) - set(self.available_cameras)
                print(f"  缺失相机: {missing} (使用模拟数据)")
        else:
            print(f"  相机: 模拟")
        print(f"  策略: {'已加载' if self.policy_runner else '无'}")
    
    def _get_camera_frame(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        获取相机帧
        
        Returns:
            (wrist_rgb, external_rgb, acquire_time)
        
        注意：如果某个相机不可用，会使用模拟数据填充
        """
        t0 = time.perf_counter()
        
        # 默认使用模拟数据
        wrist_rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        external_rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        if self.camera_manager:
            frames = self.camera_manager.get_frames()
            
            # wrist 相机
            wrist = frames.get("wrist")
            if wrist and wrist.rgb is not None:
                wrist_rgb = wrist.rgb
            
            # external 相机
            external = frames.get("external")
            if external and external.rgb is not None:
                external_rgb = external.rgb
        else:
            # 完全模拟模式，添加小延迟
            time.sleep(0.002)
        
        acquire_time = time.perf_counter() - t0
        return wrist_rgb, external_rgb, acquire_time
    
    def _preprocess(
        self, 
        wrist_rgb: np.ndarray, 
        external_rgb: np.ndarray,
        state: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        预处理观测
        
        Returns:
            (wrist_tensor, external_tensor, state_tensor, preprocess_time)
        """
        t0 = time.perf_counter()
        
        # 调整尺寸
        wrist_resized = cv2.resize(wrist_rgb, self.image_size)
        external_resized = cv2.resize(external_rgb, self.image_size)
        
        # 归一化 [0, 255] -> [-1, 1]
        wrist_normalized = (wrist_resized.astype(np.float32) / 127.5) - 1.0
        external_normalized = (external_resized.astype(np.float32) / 127.5) - 1.0
        
        # 转换为 tensor [H, W, C] -> [B, C, H, W]
        wrist_tensor = torch.from_numpy(wrist_normalized).permute(2, 0, 1).unsqueeze(0)
        external_tensor = torch.from_numpy(external_normalized).permute(2, 0, 1).unsqueeze(0)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        
        # 移动到 GPU
        wrist_tensor = wrist_tensor.to(self.device)
        external_tensor = external_tensor.to(self.device)
        state_tensor = state_tensor.to(self.device)
        
        # 同步 GPU
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        preprocess_time = time.perf_counter() - t0
        return wrist_tensor, external_tensor, state_tensor, preprocess_time
    
    def _run_inference(
        self,
        wrist_rgb: np.ndarray,
        external_rgb: np.ndarray,
        state: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        运行策略推理
        
        Args:
            wrist_rgb: [H, W, 3] uint8 wrist 相机图像
            external_rgb: [H, W, 3] uint8 external 相机图像  
            state: [state_dim] float32 机器人状态
        
        Returns:
            (keyframes, inference_time)
        """
        t0 = time.perf_counter()
        
        if self.policy_runner:
            # 真实推理 - 使用 PolicyRunner 的 add_observation + predict_keyframes 接口
            with torch.no_grad():
                # 调整图像尺寸
                wrist_resized = cv2.resize(wrist_rgb, self.image_size)
                external_resized = cv2.resize(external_rgb, self.image_size)
                
                # 拼接双相机图像 (水平拼接然后缩放回原尺寸)
                # 或者只使用 wrist 相机 (根据训练时的配置)
                # 这里简单使用 wrist 相机
                rgb = wrist_resized
                
                # 添加观测
                self.policy_runner.add_observation(rgb, state)
                
                # 执行推理
                keyframes = self.policy_runner.predict_keyframes()
                
                # 同步 GPU
                if self.device == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize()
        else:
            # 模拟推理
            time.sleep(0.050)  # 模拟 50ms 推理
            keyframes = np.zeros((8, 7))  # H=8, dof=7
        
        inference_time = time.perf_counter() - t0
        return keyframes, inference_time
    
    def run_test(self):
        """运行完整测试"""
        print("\n" + "=" * 60)
        print(f"延迟测试 (预热: {self.warmup_samples}, 采样: {self.num_samples})")
        print("=" * 60)
        
        total_samples = self.warmup_samples + self.num_samples
        
        # 模拟机器人状态 (state_dim = 13: 6 pos + 6 vel + 1 gripper)
        state = np.zeros(13, dtype=np.float32)
        
        # 清空策略缓冲区
        if self.policy_runner:
            self.policy_runner.clear_buffer()
        
        print("\n正在测试...")
        
        for i in range(total_samples):
            e2e_start = time.perf_counter()
            
            # 1. 采集 (如果相机不可用会使用模拟数据)
            wrist_rgb, external_rgb, acquire_time = self._get_camera_frame()
            
            # 2. 预处理 (测量 resize + tensor 转换时间)
            preprocess_start = time.perf_counter()
            wrist_resized = cv2.resize(wrist_rgb, self.image_size)
            external_resized = cv2.resize(external_rgb, self.image_size)
            preprocess_time = time.perf_counter() - preprocess_start
            
            # 3. 推理 (包含内部的预处理和推理)
            if self.policy_runner:
                keyframes, inference_time = self._run_inference(
                    wrist_rgb, external_rgb, state
                )
            else:
                # 模拟推理
                time.sleep(0.050)
                inference_time = 0.050
                keyframes = np.zeros((8, 7))
            
            e2e_time = time.perf_counter() - e2e_start
            
            # 跳过预热
            if i >= self.warmup_samples:
                self.acquire_times.append(acquire_time)
                self.preprocess_times.append(preprocess_time)
                self.inference_times.append(inference_time)
                self.e2e_times.append(e2e_time)
            
            # 进度显示
            if self.verbose and (i + 1) % 20 == 0:
                progress = (i + 1) / total_samples * 100
                print(f"  进度: {progress:.0f}% ({i+1}/{total_samples})")
        
        print("\n测试完成!")
    
    def report(self):
        """生成报告"""
        print("\n" + "=" * 60)
        print("延迟测试报告")
        print("=" * 60)
        
        # 创建统计对象
        stats = [
            LatencyStats("1. 相机采集 (acquire)", self.acquire_times),
            LatencyStats("2. 预处理 (preprocess)", self.preprocess_times),
            LatencyStats("3. 策略推理 (inference)", self.inference_times),
            LatencyStats("4. 端到端 (E2E)", self.e2e_times),
        ]
        
        for s in stats:
            print(f"\n{s.summary()}")
        
        # RTC 兼容性分析
        print("\n" + "=" * 60)
        print("RTC 兼容性分析")
        print("=" * 60)
        
        inference_p95 = stats[2].p95
        e2e_p95 = stats[3].p95
        
        # 计算最大 inference_delay
        # RTC 论文: inference_delay 是推理期间需要执行的动作数
        # 如果 10Hz 推理，每 100ms 一个 chunk
        # inference_delay = ceil(inference_time / action_dt)
        
        action_dt = 33.3  # ms (30Hz 动作)
        policy_dt = 100.0  # ms (10Hz 推理)
        
        inference_delay_mean = int(np.ceil(stats[2].mean / action_dt))
        inference_delay_p95 = int(np.ceil(inference_p95 / action_dt))
        
        print(f"\n配置参数:")
        print(f"  动作间隔 (dt_key): {action_dt:.1f} ms")
        print(f"  推理间隔 (policy_dt): {policy_dt:.1f} ms")
        print(f"  动作块大小 (action_chunk_size): 8")
        
        print(f"\n推荐 inference_delay:")
        print(f"  基于均值: {inference_delay_mean}")
        print(f"  基于 P95:  {inference_delay_p95}")
        
        # 可行性判断
        print(f"\n可行性评估:")
        
        # 检查推理是否能在一个策略周期内完成
        if e2e_p95 < policy_dt:
            print(f"  ✓ E2E 延迟 ({e2e_p95:.1f} ms) < 策略周期 ({policy_dt:.1f} ms)")
            print(f"    可以实现实时 10Hz 推理")
        else:
            print(f"  ✗ E2E 延迟 ({e2e_p95:.1f} ms) > 策略周期 ({policy_dt:.1f} ms)")
            print(f"    需要优化或降低推理频率")
        
        # 检查 inference_delay 是否合理
        max_delay = 4  # 最大建议 delay (留出足够的 execute_horizon)
        if inference_delay_p95 <= max_delay:
            print(f"  ✓ inference_delay ({inference_delay_p95}) <= {max_delay}")
            print(f"    execute_horizon 可设为 {8 - inference_delay_p95} ~ 8")
        else:
            print(f"  ✗ inference_delay ({inference_delay_p95}) > {max_delay}")
            print(f"    建议优化推理速度或增加 action_chunk_size")
        
        # 优化建议
        print(f"\n优化建议:")
        
        if stats[2].mean > 80:
            print(f"  - 推理延迟较高 ({stats[2].mean:.0f} ms)")
            print(f"    考虑: 减少 num_flow_steps, 使用更小的模型, 或 FP16 推理")
        
        if stats[0].mean > 10:
            print(f"  - 相机采集延迟较高 ({stats[0].mean:.0f} ms)")
            print(f"    考虑: 检查相机 FPS 设置, 或使用异步采集")
        
        if stats[1].mean > 5:
            print(f"  - 预处理延迟较高 ({stats[1].mean:.0f} ms)")
            print(f"    考虑: 使用 GPU 预处理, 或减小图像分辨率")
    
    def cleanup(self):
        """清理资源"""
        if self.camera_manager:
            self.camera_manager.stop()
        print("\n资源已清理")


def main():
    parser = argparse.ArgumentParser(description="相机+推理延迟测试")
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        default=None,
        help="策略 checkpoint 路径 (可选)"
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=100,
        help="采样数量"
    )
    parser.add_argument(
        "--warmup", "-w",
        type=int,
        default=10,
        help="预热采样数量"
    )
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="不使用真实相机 (模拟数据)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="推理设备 (cuda/cpu)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="安静模式"
    )
    
    args = parser.parse_args()
    
    # 默认 checkpoint 路径
    if args.checkpoint is None:
        default_ckpt = "/home/lizh/rl-vla/rlft/diffusion_policy/runs/consistency_flow-pick_cube-real__1__1766132160/checkpoints/latest.pt"
        if os.path.exists(default_ckpt):
            args.checkpoint = default_ckpt
            print(f"使用默认 checkpoint: {args.checkpoint}")
    
    tester = LatencyTester(
        checkpoint_path=args.checkpoint,
        num_samples=args.samples,
        warmup_samples=args.warmup,
        use_real_camera=not args.no_camera,
        device=args.device,
        verbose=not args.quiet,
    )
    
    try:
        tester.setup()
        tester.run_test()
        tester.report()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    finally:
        tester.cleanup()


if __name__ == "__main__":
    main()
