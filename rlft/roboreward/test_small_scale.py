#!/usr/bin/env python3
"""
RoboReward 小规模测试脚本

用于测试模型加载和单个 episode 的推理功能。
"""

import os
import sys
import time
import h5py

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rlft.roboreward.config import RoboRewardConfig, SCORE_DESCRIPTIONS
from rlft.roboreward.labeler import RoboRewardLabeler
from rlft.roboreward.dataset_converter import DatasetConverter


def test_small_scale():
    """小规模测试：处理 2-3 个 episodes"""
    
    print("="*70)
    print("RoboReward 小规模测试")
    print("="*70)
    
    # 1. 配置
    print("\n[1/5] 初始化配置...")
    config = RoboRewardConfig(
        model_name_or_path="teetone/RoboReward-8B",
        torch_dtype="bfloat16",
        use_flash_attention=True,  # 自动降级到 SDPA
        sample_frames=-1,  # -1 表示使用所有帧
        max_frames=512,    # 限制最大帧数防止 OOM
        verbose=True,
    )
    print(f"✓ 模型: {config.model_name_or_path}")
    print(f"✓ 精度: {config.torch_dtype}")
    print(f"✓ 采样帧数: {'全部帧' if config.sample_frames <= 0 else config.sample_frames}")
    
    # 2. 初始化工具
    print("\n[2/5] 初始化标注器和转换器...")
    labeler = RoboRewardLabeler(config)
    converter = DatasetConverter(config)
    print("✓ 标注器和转换器初始化完成")
    
    # 3. 扫描数据
    print("\n[3/5] 扫描数据集...")
    data_dir = '../../recorded_data/mix'
    episodes = converter.scan_episodes(data_dir)
    print(f"✓ 发现 {len(episodes)} 个 episodes")
    print(f"✓ 将处理前 3 个 episodes 进行测试")
    
    # 4. 加载模型
    print("\n[4/5] 加载 RoboReward 模型...")
    print("      这可能需要 30-60 秒，请耐心等待...")
    start_time = time.time()
    labeler.load_model()
    load_time = time.time() - start_time
    print(f"✓ 模型加载完成 (耗时: {load_time:.2f}s)")
    
    # 5. 处理前 3 个 episodes
    print("\n[5/5] 开始推理...")
    print("-"*70)
    
    results = []
    output_dir = "./test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    for i, filepath in enumerate(episodes[:3]):
        episode_name = os.path.basename(filepath)
        print(f"\n[{i+1}/3] 处理: {episode_name}")
        
        try:
            # 加载帧
            print(f"       加载帧...")
            frames, metadata = converter.load_episode_frames(filepath)
            print(f"       原始帧数: {metadata['num_steps']} → 采样: {metadata['sampled_frames']} 帧")
            
            # 推理
            print(f"       推理中...")
            infer_start = time.time()
            score, raw_output = labeler.score_episode(
                frames,
                "pick up the black tape, place it in the blue cup",
                return_raw=True
            )
            infer_time = time.time() - infer_start
            
            print(f"       ✓ 完成！")
            print(f"       Reward: {score} - {SCORE_DESCRIPTIONS[score]}")
            print(f"       推理时间: {infer_time:.2f}s")
            
            # 保存带 reward 的文件
            output_path = os.path.join(output_dir, episode_name)
            converter.save_episode_with_reward(
                filepath,
                output_path,
                reward=score,
                raw_output=raw_output
            )
            print(f"       保存到: {output_path}")
            
            results.append({
                'name': episode_name,
                'score': score,
                'time': infer_time
            })
            
        except Exception as e:
            print(f"       ✗ 错误: {e}")
            results.append({
                'name': episode_name,
                'error': str(e)
            })
    
    # 统计
    print("\n" + "="*70)
    print("测试完成！统计结果")
    print("="*70)
    
    success_count = sum(1 for r in results if 'score' in r)
    print(f"\n✓ 成功处理: {success_count}/3")
    
    if success_count > 0:
        scores = [r['score'] for r in results if 'score' in r]
        times = [r['time'] for r in results if 'score' in r]
        
        print(f"\n分数分布:")
        for score in range(1, 6):
            count = sum(1 for s in scores if s == score)
            if count > 0:
                print(f"  {score} ({SCORE_DESCRIPTIONS[score][:20]:20s}): {count}")
        
        print(f"\n性能指标:")
        print(f"  平均推理时间: {sum(times)/len(times):.2f}s/episode")
        print(f"  总耗时: {sum(times):.2f}s")
    
    print(f"\n输出目录: {output_dir}")
    print(f"\n✓ 测试脚本执行完毕")
    print("="*70)


if __name__ == "__main__":
    test_small_scale()
