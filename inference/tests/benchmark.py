#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理性能基准测试

测试内容:
1. 推理延迟 (visual_encoder + state_encoder + agent.get_action)
2. 不同 num_flow_steps 的速度与精度权衡
3. GPU 内存占用
4. 不同 act_horizon 的影响

用法:
    python -m inference.tests.benchmark --checkpoint /path/to/checkpoint.pt
    python -m inference.tests.benchmark --checkpoint /path/to/checkpoint.pt --flow-steps 5,10,15,20
"""

import os
import sys
import time
import argparse
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

import torch
import torch.cuda

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from inference.config import setup_rlft


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    name: str
    mean_latency_ms: float
    std_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_hz: float
    gpu_memory_mb: float
    
    def __str__(self):
        return (
            f"{self.name}:\n"
            f"  延迟: {self.mean_latency_ms:.2f} ± {self.std_latency_ms:.2f} ms\n"
            f"  P50: {self.p50_latency_ms:.2f} ms, P95: {self.p95_latency_ms:.2f} ms, P99: {self.p99_latency_ms:.2f} ms\n"
            f"  吞吐量: {self.throughput_hz:.1f} Hz\n"
            f"  GPU 内存: {self.gpu_memory_mb:.1f} MB"
        )


class InferenceBenchmark:
    """推理基准测试"""
    
    ACTION_DIM = 7
    STATE_DIM = 13
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        image_size: Tuple[int, int] = (128, 128),
    ):
        self.checkpoint_path = os.path.expanduser(checkpoint_path)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.image_size = image_size
        
        # 模型组件
        self.visual_encoder = None
        self.state_encoder = None
        self.agent = None
        self.algorithm = None
        
        # 预生成的测试数据
        self._test_rgb = None
        self._test_state = None
    
    def setup(self, num_flow_steps: int = 10):
        """初始化模型"""
        print(f"\n加载模型 (num_flow_steps={num_flow_steps})...")
        
        setup_rlft()
        
        from diffusion_policy.plain_conv import PlainConv
        from diffusion_policy.utils import StateEncoder
        from diffusion_policy.algorithms import (
            DiffusionPolicyAgent,
            FlowMatchingAgent,
            ConsistencyFlowAgent,
        )
        from diffusion_policy.algorithms.networks import VelocityUNet1D
        from diffusion_policy.conditional_unet1d import ConditionalUnet1D
        
        # 加载 checkpoint
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        
        # 检测算法类型
        agent_keys = list(ckpt.get('agent', ckpt.get('ema_agent', {})).keys())
        if any('velocity_net' in k for k in agent_keys):
            if any('velocity_net_ema' in k for k in agent_keys):
                self.algorithm = "consistency_flow"
            else:
                self.algorithm = "flow_matching"
        elif any('noise_pred_net' in k for k in agent_keys):
            self.algorithm = "diffusion_policy"
        else:
            self.algorithm = "flow_matching"
        
        print(f"  算法: {self.algorithm}")
        
        # 模型参数
        visual_feature_dim = 256
        state_encoder_hidden_dim = 128
        state_encoder_out_dim = 256
        diffusion_step_embed_dim = 64
        unet_dims = (64, 128, 256)
        n_groups = 8
        
        global_cond_dim = self.obs_horizon * (visual_feature_dim + state_encoder_out_dim)
        
        # 创建模型
        self.visual_encoder = PlainConv(
            in_channels=3,
            out_dim=visual_feature_dim,
            pool_feature_map=True,
        ).to(self.device)
        
        self.state_encoder = StateEncoder(
            state_dim=self.STATE_DIM,
            hidden_dim=state_encoder_hidden_dim,
            out_dim=state_encoder_out_dim,
        ).to(self.device)
        
        if self.algorithm == "diffusion_policy":
            noise_pred_net = ConditionalUnet1D(
                input_dim=self.ACTION_DIM,
                global_cond_dim=global_cond_dim,
                diffusion_step_embed_dim=diffusion_step_embed_dim,
                down_dims=unet_dims,
                n_groups=n_groups,
            )
            self.agent = DiffusionPolicyAgent(
                noise_pred_net=noise_pred_net,
                action_dim=self.ACTION_DIM,
                obs_horizon=self.obs_horizon,
                pred_horizon=self.pred_horizon,
                num_diffusion_iters=100,
                device=str(self.device),
            )
        elif self.algorithm == "consistency_flow":
            velocity_net = VelocityUNet1D(
                input_dim=self.ACTION_DIM,
                global_cond_dim=global_cond_dim,
                diffusion_step_embed_dim=diffusion_step_embed_dim,
                down_dims=unet_dims,
                n_groups=n_groups,
            )
            self.agent = ConsistencyFlowAgent(
                velocity_net=velocity_net,
                action_dim=self.ACTION_DIM,
                obs_horizon=self.obs_horizon,
                pred_horizon=self.pred_horizon,
                num_flow_steps=num_flow_steps,
                ema_decay=0.999,
                action_bounds=None,
                device=str(self.device),
            )
        else:
            velocity_net = VelocityUNet1D(
                input_dim=self.ACTION_DIM,
                global_cond_dim=global_cond_dim,
                diffusion_step_embed_dim=diffusion_step_embed_dim,
                down_dims=unet_dims,
                n_groups=n_groups,
            )
            self.agent = FlowMatchingAgent(
                velocity_net=velocity_net,
                action_dim=self.ACTION_DIM,
                obs_horizon=self.obs_horizon,
                pred_horizon=self.pred_horizon,
                num_flow_steps=num_flow_steps,
                action_bounds=None,
                device=str(self.device),
            )
        
        self.agent = self.agent.to(self.device)
        
        # 加载权重
        agent_key = "ema_agent" if "ema_agent" in ckpt else "agent"
        self.agent.load_state_dict(ckpt[agent_key])
        self.visual_encoder.load_state_dict(ckpt["visual_encoder"])
        self.state_encoder.load_state_dict(ckpt["state_encoder"])
        
        # 设置为评估模式
        self.agent.eval()
        self.visual_encoder.eval()
        self.state_encoder.eval()
        
        # 预生成测试数据
        self._generate_test_data()
        
        print("  模型加载完成")
    
    def _generate_test_data(self):
        """生成测试数据"""
        # 随机 RGB 图像
        dummy_rgb = np.random.randint(
            0, 255, 
            (self.obs_horizon, 3, *self.image_size), 
            dtype=np.uint8
        )
        self._test_rgb = torch.from_numpy(
            dummy_rgb.astype(np.float32) / 255.0
        ).unsqueeze(0).to(self.device)
        
        # 随机状态
        dummy_state = np.random.randn(
            self.obs_horizon, self.STATE_DIM
        ).astype(np.float32)
        self._test_state = torch.from_numpy(dummy_state).unsqueeze(0).to(self.device)
    
    def _run_inference_once(self) -> np.ndarray:
        """执行一次推理"""
        with torch.no_grad():
            B, T = self._test_rgb.shape[0], self._test_rgb.shape[1]
            rgb_flat = self._test_rgb.view(B * T, *self._test_rgb.shape[2:])
            state_flat = self._test_state.view(B * T, -1)
            
            visual_feat = self.visual_encoder(rgb_flat)
            visual_feat = visual_feat.view(B, T, -1)
            
            state_feat = self.state_encoder(state_flat)
            state_feat = state_feat.view(B, T, -1)
            
            obs_features = torch.cat([visual_feat, state_feat], dim=-1)
            obs_cond = obs_features.view(B, -1)
            
            action_seq = self.agent.get_action(obs_cond)
            
            if isinstance(action_seq, tuple):
                action_seq = action_seq[0]
        
        return action_seq[0].cpu().numpy()
    
    def warmup(self, n_iterations: int = 10):
        """GPU 预热"""
        print(f"\n预热 ({n_iterations} 次)...")
        for _ in range(n_iterations):
            self._run_inference_once()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print("  预热完成")
    
    def benchmark_inference(self, n_iterations: int = 100, name: str = "inference") -> BenchmarkResult:
        """基准测试推理延迟"""
        print(f"\n测试推理延迟 ({n_iterations} 次)...")
        
        latencies = []
        
        for i in range(n_iterations):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            self._run_inference_once()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            elapsed = time.perf_counter() - start
            latencies.append(elapsed * 1000)  # 转换为毫秒
        
        latencies = np.array(latencies)
        
        # GPU 内存
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            gpu_memory = 0.0
        
        result = BenchmarkResult(
            name=name,
            mean_latency_ms=np.mean(latencies),
            std_latency_ms=np.std(latencies),
            p50_latency_ms=np.percentile(latencies, 50),
            p95_latency_ms=np.percentile(latencies, 95),
            p99_latency_ms=np.percentile(latencies, 99),
            throughput_hz=1000.0 / np.mean(latencies),
            gpu_memory_mb=gpu_memory,
        )
        
        print(result)
        return result
    
    def benchmark_components(self, n_iterations: int = 100) -> Dict[str, BenchmarkResult]:
        """分别测试各组件延迟"""
        print(f"\n测试各组件延迟...")
        
        results = {}
        
        # Visual Encoder
        print("\n[Visual Encoder]")
        latencies = []
        rgb_flat = self._test_rgb.view(-1, *self._test_rgb.shape[2:])
        
        for _ in range(n_iterations):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            
            with torch.no_grad():
                _ = self.visual_encoder(rgb_flat)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)
        
        results["visual_encoder"] = BenchmarkResult(
            name="visual_encoder",
            mean_latency_ms=np.mean(latencies),
            std_latency_ms=np.std(latencies),
            p50_latency_ms=np.percentile(latencies, 50),
            p95_latency_ms=np.percentile(latencies, 95),
            p99_latency_ms=np.percentile(latencies, 99),
            throughput_hz=1000.0 / np.mean(latencies),
            gpu_memory_mb=0,
        )
        print(f"  平均: {np.mean(latencies):.2f} ms")
        
        # State Encoder
        print("\n[State Encoder]")
        latencies = []
        state_flat = self._test_state.view(-1, self.STATE_DIM)
        
        for _ in range(n_iterations):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            
            with torch.no_grad():
                _ = self.state_encoder(state_flat)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)
        
        results["state_encoder"] = BenchmarkResult(
            name="state_encoder",
            mean_latency_ms=np.mean(latencies),
            std_latency_ms=np.std(latencies),
            p50_latency_ms=np.percentile(latencies, 50),
            p95_latency_ms=np.percentile(latencies, 95),
            p99_latency_ms=np.percentile(latencies, 99),
            throughput_hz=1000.0 / np.mean(latencies),
            gpu_memory_mb=0,
        )
        print(f"  平均: {np.mean(latencies):.2f} ms")
        
        # Agent (action generation)
        print("\n[Agent]")
        latencies = []
        
        # 预计算 obs_cond
        with torch.no_grad():
            B, T = self._test_rgb.shape[0], self._test_rgb.shape[1]
            rgb_flat = self._test_rgb.view(B * T, *self._test_rgb.shape[2:])
            state_flat = self._test_state.view(B * T, -1)
            
            visual_feat = self.visual_encoder(rgb_flat).view(B, T, -1)
            state_feat = self.state_encoder(state_flat).view(B, T, -1)
            obs_features = torch.cat([visual_feat, state_feat], dim=-1)
            obs_cond = obs_features.view(B, -1)
        
        for _ in range(n_iterations):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            
            with torch.no_grad():
                _ = self.agent.get_action(obs_cond)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)
        
        results["agent"] = BenchmarkResult(
            name="agent",
            mean_latency_ms=np.mean(latencies),
            std_latency_ms=np.std(latencies),
            p50_latency_ms=np.percentile(latencies, 50),
            p95_latency_ms=np.percentile(latencies, 95),
            p99_latency_ms=np.percentile(latencies, 99),
            throughput_hz=1000.0 / np.mean(latencies),
            gpu_memory_mb=0,
        )
        print(f"  平均: {np.mean(latencies):.2f} ms")
        
        return results


def run_flow_steps_sweep(
    checkpoint_path: str,
    flow_steps_list: List[int],
    n_iterations: int = 100,
    device: str = "cuda",
) -> List[BenchmarkResult]:
    """测试不同 num_flow_steps 的影响"""
    print("\n" + "=" * 60)
    print("Flow Steps 扫描测试")
    print("=" * 60)
    
    results = []
    
    for flow_steps in flow_steps_list:
        print(f"\n--- num_flow_steps = {flow_steps} ---")
        
        benchmark = InferenceBenchmark(
            checkpoint_path=checkpoint_path,
            device=device,
        )
        benchmark.setup(num_flow_steps=flow_steps)
        benchmark.warmup(n_iterations=10)
        
        result = benchmark.benchmark_inference(
            n_iterations=n_iterations,
            name=f"flow_steps_{flow_steps}"
        )
        results.append(result)
        
        # 清理 GPU 内存
        del benchmark
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results


def print_summary(results: List[BenchmarkResult]):
    """打印汇总表格"""
    print("\n" + "=" * 80)
    print("汇总")
    print("=" * 80)
    print(f"{'Name':<20} {'Mean (ms)':<12} {'P95 (ms)':<12} {'P99 (ms)':<12} {'Hz':<10} {'GPU (MB)':<10}")
    print("-" * 80)
    
    for r in results:
        print(f"{r.name:<20} {r.mean_latency_ms:<12.2f} {r.p95_latency_ms:<12.2f} "
              f"{r.p99_latency_ms:<12.2f} {r.throughput_hz:<10.1f} {r.gpu_memory_mb:<10.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="推理性能基准测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--checkpoint", "-c", type=str, required=True,
                       help="Checkpoint 文件路径")
    parser.add_argument("--device", type=str, default="cuda",
                       help="推理设备")
    parser.add_argument("--iterations", "-n", type=int, default=100,
                       help="测试迭代次数")
    parser.add_argument("--flow-steps", type=str, default="10",
                       help="num_flow_steps 值，逗号分隔 (例如 '5,10,15,20')")
    parser.add_argument("--components", action="store_true",
                       help="分别测试各组件")
    
    args = parser.parse_args()
    
    # 解析 flow_steps
    flow_steps_list = [int(x.strip()) for x in args.flow_steps.split(",")]
    
    print("=" * 60)
    print("ARX5 推理性能基准测试")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"设备: {args.device}")
    print(f"迭代次数: {args.iterations}")
    print(f"Flow Steps: {flow_steps_list}")
    
    # Flow steps 扫描
    results = run_flow_steps_sweep(
        checkpoint_path=args.checkpoint,
        flow_steps_list=flow_steps_list,
        n_iterations=args.iterations,
        device=args.device,
    )
    
    # 组件测试
    if args.components:
        benchmark = InferenceBenchmark(
            checkpoint_path=args.checkpoint,
            device=args.device,
        )
        benchmark.setup(num_flow_steps=10)
        benchmark.warmup()
        component_results = benchmark.benchmark_components(n_iterations=args.iterations)
        results.extend(component_results.values())
    
    # 打印汇总
    print_summary(results)


if __name__ == "__main__":
    main()
