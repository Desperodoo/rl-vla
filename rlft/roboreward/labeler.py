"""
RoboReward 核心标注器

封装 RoboReward-8B (基于 Qwen3-VL-8B-Instruct) 模型，
提供对机器人操作视频的 Reward 评分功能。
"""

import re
import torch
from typing import List, Optional, Union, Tuple
from PIL import Image
import numpy as np

from .config import RoboRewardConfig, REWARD_PROMPT_TEMPLATE


class RoboRewardLabeler:
    """RoboReward 标注器"""
    
    def __init__(self, config: Optional[RoboRewardConfig] = None):
        """
        初始化标注器
        
        Args:
            config: 配置对象，为 None 时使用默认配置
        """
        self.config = config or RoboRewardConfig()
        self.model = None
        self.processor = None
        self._loaded = False
    
    def load_model(self):
        """加载模型和处理器"""
        if self._loaded:
            return
        
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        
        print(f"Loading RoboReward model from {self.config.model_name_or_path}...")
        
        # 确定数据类型
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.config.torch_dtype, torch.bfloat16)
        
        # 确定注意力实现（优先使用 flash_attention_2，不可用时降级到 sdpa）
        if self.config.use_flash_attention:
            try:
                import flash_attn
                attn_implementation = "flash_attention_2"
            except ImportError:
                print("Warning: Flash Attention 不可用，使用 SDPA")
                attn_implementation = "sdpa"
        else:
            attn_implementation = "sdpa"
        
        # 加载配置并修复 pad_token_id 问题
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=True
        )
        
        # 修复 transformers bug: text_config 缺少 pad_token_id
        if hasattr(config, 'text_config') and not hasattr(config.text_config, 'pad_token_id'):
            config.text_config.pad_token_id = config.pad_token_id
        
        # 加载模型
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.config.model_name_or_path,
            config=config,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            device_map=self.config.device_map,
        )
        
        # 加载处理器
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_name_or_path
        )
        
        self._loaded = True
        print(f"Model loaded successfully. Device: {self.model.device}")
    
    def _build_messages(
        self, 
        frames: List[Image.Image], 
        task_description: str
    ) -> List[dict]:
        """
        构建模型输入消息
        
        Args:
            frames: PIL.Image 列表（视频帧）
            task_description: 任务描述
            
        Returns:
            消息列表（符合 Qwen3-VL chat 格式）
        """
        prompt = REWARD_PROMPT_TEMPLATE.format(task=task_description)
        
        # 按照 Qwen3-VL 官方文档，video 输入需要使用 list of PIL Images
        # 参考: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": frames,  # PIL.Image 列表作为视频帧
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ]
            }
        ]
        
        return messages
    
    def _parse_score(self, output_text: str) -> Tuple[int, str]:
        """
        从模型输出中解析评分
        
        Args:
            output_text: 模型输出文本
            
        Returns:
            (评分, 原始输出) 元组
        """
        # 匹配 "ANSWER: <score>" 格式
        pattern = r"ANSWER:\s*(\d)"
        match = re.search(pattern, output_text, re.IGNORECASE)
        
        if match:
            score = int(match.group(1))
            # 确保分数在有效范围内
            score = max(1, min(5, score))
            return score, output_text
        
        # 如果没有匹配到标准格式，尝试找任何数字
        numbers = re.findall(r"\b([1-5])\b", output_text)
        if numbers:
            return int(numbers[-1]), output_text
        
        # 默认返回 1（无法解析时假设失败）
        print(f"Warning: Could not parse score from output: {output_text}")
        return 1, output_text
    
    @torch.inference_mode()
    def score_episode(
        self, 
        frames: List[Image.Image], 
        task_description: str,
        return_raw: bool = False
    ) -> Union[int, Tuple[int, str]]:
        """
        对单个 episode 进行评分
        
        Args:
            frames: 视频帧列表 (PIL.Image)
            task_description: 任务描述
            return_raw: 是否返回原始输出
            
        Returns:
            reward 评分 (1-5)，如果 return_raw=True 则返回 (score, raw_output) 元组
        """
        if not self._loaded:
            self.load_model()
        
        # 构建消息
        messages = self._build_messages(frames, task_description)
        
        # 按照 Qwen3-VL 官方文档的推理方式:
        # https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
        # 使用 apply_chat_template 直接返回 tokenized tensors
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        
        # 生成输出
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=self.config.do_sample,
        )
        
        # 只取新生成的 token
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # 解码输出
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True
        )[0]
        
        # 解析评分
        score, raw_output = self._parse_score(output_text)
        
        if self.config.verbose:
            print(f"Task: {task_description[:50]}... | Score: {score}")
        
        if return_raw:
            return score, raw_output
        return score
    
    def score_batch(
        self,
        episodes: List[List[Image.Image]],
        task_descriptions: Union[str, List[str]],
        return_raw: bool = False
    ) -> List[Union[int, Tuple[int, str]]]:
        """
        批量评分多个 episodes
        
        注意：由于 VLM 的特性，实际上是逐个处理的
        
        Args:
            episodes: 每个 episode 的帧列表
            task_descriptions: 任务描述（单个字符串或列表）
            return_raw: 是否返回原始输出
            
        Returns:
            评分列表
        """
        if isinstance(task_descriptions, str):
            task_descriptions = [task_descriptions] * len(episodes)
        
        results = []
        for frames, task in zip(episodes, task_descriptions):
            result = self.score_episode(frames, task, return_raw=return_raw)
            results.append(result)
        
        return results
    
    def __del__(self):
        """清理资源"""
        if self.model is not None:
            del self.model
        if self.processor is not None:
            del self.processor
        torch.cuda.empty_cache()
