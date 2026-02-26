"""
P3.1 — VLAW VLM 二分类奖励模型

实现 VLAW 论文 Eq.3 的奖励函数:
    R(τ) = 1[ P('yes' | τ, I) > α ],  α = 0.8

支持两种输入场景:
    1. 真实 rollout: PIL.Image 帧列表 (D_real)
    2. 合成数据:    uint8/float numpy 数组 [T, H, W, C] (D_syn, 来自 VAE decode)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image


# ──────────────────────────────────────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class VLAWRewardConfig:
    """VLAW 奖励模型配置"""

    # 模型路径 (本地目录或 HuggingFace ID)
    model_path: str = "checkpoints/vlaw/reward_model/qwen_vl"

    # 推理精度
    torch_dtype: str = "bfloat16"  # "float16" | "bfloat16" | "float32"

    # 设备配置 ("cuda:6", "cuda:0", "auto")
    device: str = "cuda:0"

    # Flash Attention: True 需要 flash-attn 包
    use_flash_attention: bool = False

    # 帧采样
    num_frames: int = 16       # 均匀采样帧数

    # 判定阈值 α (VLAW 论文 Section 4.1)
    threshold: float = 0.8

    # 二分类 prompt 模板
    prompt_template: str = (
        "These {n} frames show a robot manipulation trajectory. "
        "Task: '{instruction}'. "
        "Has the robot successfully completed the task? "
        "Answer only 'yes' or 'no'."
    )

    # 推理参数
    max_new_tokens: int = 8
    do_sample: bool = False

    # 批量推理时每批大小 (VLM 通常 batch_size=1)
    batch_size: int = 1


# ──────────────────────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────────────────────

def uniform_sample_frames(
    source: Union[List[Image.Image], np.ndarray],
    num_frames: int,
) -> List[Image.Image]:
    """
    均匀采样 num_frames 帧，返回 PIL.Image 列表。

    Args:
        source: PIL 帧列表 或 numpy 数组 [T, H, W, C] (uint8 或 float32 0-1)
        num_frames: 目标帧数

    Returns:
        PIL.Image 列表，长度 = min(num_frames, len(source))
    """
    if isinstance(source, np.ndarray):
        total = source.shape[0]
    else:
        total = len(source)

    if total == 0:
        raise ValueError("source 不能为空")

    n = min(num_frames, total)
    indices = np.linspace(0, total - 1, n, dtype=int)

    frames: List[Image.Image] = []
    for idx in indices:
        if isinstance(source, np.ndarray):
            arr = source[idx]
            if arr.dtype != np.uint8:
                arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
            frames.append(Image.fromarray(arr))
        else:
            frames.append(source[idx])
    return frames


# ──────────────────────────────────────────────────────────────────────────────
# 主模型类
# ──────────────────────────────────────────────────────────────────────────────

class VLAWRewardModel:
    """
    VLAW 二分类 VLM 奖励模型。

    使用 Qwen3-VL 提取 P('yes') 概率，判定轨迹是否成功。
    支持 LoRA adapter 加载。

    Example::

        model = VLAWRewardModel()
        model.load_model()
        result = model.score_trajectory(frames, "pick up the cube")
        # result = {"p_yes": 0.92, "reward": 1, "threshold": 0.8}
    """

    def __init__(self, config: Optional[VLAWRewardConfig] = None):
        self.config = config or VLAWRewardConfig()
        self.model = None
        self.processor = None
        self._yes_token_id: Optional[int] = None    # 向后兼容
        self._no_token_id: Optional[int] = None
        self._yes_token_ids: List[int] = []         # 所有 yes 变体
        self._no_token_ids: List[int] = []
        self._loaded = False

    # ── 模型加载 ──────────────────────────────────────────────────────────────

    def load_model(self, lora_path: Optional[str] = None) -> None:
        """
        加载 Qwen3-VL 模型和处理器。

        Args:
            lora_path: LoRA adapter 路径 (None = 不加载 LoRA)
        """
        if self._loaded:
            return

        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map[self.config.torch_dtype]

        attn_impl = "eager"
        if self.config.use_flash_attention:
            try:
                import flash_attn  # noqa: F401
                attn_impl = "flash_attention_2"
            except ImportError:
                print("[VLAW] flash-attn 不可用，使用 eager attention")

        print(f"[VLAW] 加载模型: {self.config.model_path}")
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_path, trust_remote_code=True
        )

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.config.model_path,
            torch_dtype=torch_dtype,
            device_map=self.config.device,
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )

        # 加载 LoRA adapter
        if lora_path is not None:
            from peft import PeftModel
            print(f"[VLAW] 加载 LoRA: {lora_path}")
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            self.model = self.model.merge_and_unload()

        self.model.eval()

        # 缓存 yes/no 所有变体的 token id（lowercase / Title / 前置空格）
        tok = self.processor.tokenizer
        self._yes_token_ids: List[int] = list({
            tok.encode(w, add_special_tokens=False)[0]
            for w in ["yes", "Yes", "YES", " yes"]
        })
        self._no_token_ids: List[int] = list({
            tok.encode(w, add_special_tokens=False)[0]
            for w in ["no", "No", "NO", " no"]
        })
        # 向后兼容旧代码
        self._yes_token_id = self._yes_token_ids[0]
        self._no_token_id  = self._no_token_ids[0]
        print(
            f"[VLAW] 模型就绪 | device={self.model.device} "
            f"| yes_ids={self._yes_token_ids} no_ids={self._no_token_ids}"
        )
        self._loaded = True

    def unload_model(self) -> None:
        """释放显存"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self._loaded = False
        torch.cuda.empty_cache()

    # ── 推理核心 ──────────────────────────────────────────────────────────────

    def _build_messages(
        self, frames: List[Image.Image], instruction: str
    ) -> list:
        """构建 Qwen3-VL 多图 chat messages"""
        n = len(frames)
        image_content = [{"type": "image", "image": f} for f in frames]
        text = self.config.prompt_template.format(
            n=n, instruction=instruction
        )
        return [
            {
                "role": "user",
                "content": image_content + [{"type": "text", "text": text}],
            }
        ]

    @torch.inference_mode()
    def _forward_p_yes(
        self, frames: List[Image.Image], instruction: str
    ) -> float:
        """
        前向传播，提取 P('yes') 概率。

        使用 qwen_vl_utils.process_vision_info（如可用）正确处理多图输入。
        P('yes') = Σ logits[yes_variants] / (Σ logits[yes_variants] + Σ logits[no_variants])
        采用 log-sum-exp 数值稳定版本。

        Returns:
            P('yes') ∈ [0, 1]
        """
        messages = self._build_messages(frames, instruction)
        # Qwen3-VL 是思维链模型，必须关闭 thinking 模式
        # 否则第一个生成 token 是 <think>，而非 yes/no
        try:
            text_input = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,  # 关闭 CoT，直接生成 yes/no
            )
        except TypeError:
            # 旧版 processor 不支持 enable_thinking 参数
            text_input = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        # 优先使用 process_vision_info 正确解耦图像
        try:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text_input],
                images=image_inputs,
                videos=video_inputs if video_inputs else None,
                return_tensors="pt",
            ).to(self.model.device)
        except (ImportError, Exception):
            # 降级：直接传入 PIL frames（Qwen3-VL processor 支持）
            inputs = self.processor(
                text=[text_input],
                images=frames,
                return_tensors="pt",
            ).to(self.model.device)

        outputs = self.model(**inputs)
        # 取最后一个位置的 logits（第一个预测 token）
        logits = outputs.logits[0, -1, :].float()  # [vocab]

        # 聚合所有 yes/no 变体的 logits（log-sum-exp 数值稳定）
        yes_logits = logits[self._yes_token_ids]  # [num_yes_variants]
        no_logits  = logits[self._no_token_ids]   # [num_no_variants]

        # softmax over {yes_variants ∪ no_variants}
        all_logits = torch.cat([yes_logits, no_logits])  # [n_yes + n_no]
        all_probs  = torch.softmax(all_logits, dim=0)
        p_yes = all_probs[:len(self._yes_token_ids)].sum().item()
        return p_yes

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    def score_trajectory(
        self,
        frames: Union[List[Image.Image], np.ndarray],
        instruction: str,
    ) -> dict:
        """
        对单条轨迹评分。

        Args:
            frames: PIL 帧列表 或 numpy [T, H, W, C]
            instruction: 任务指令字符串

        Returns:
            dict:
                p_yes    (float)  — P('yes') 概率
                reward   (int)    — 0 或 1，由阈值 α 判定
                threshold(float)  — 当前阈值 α
                num_frames(int)   — 实际使用的帧数

        Notes:
            TODO(P3.2-finetune): 当前为 zero-shot 推理，p_yes 通常 < 0.15（见 BUG-011）。
            VLAW 论文 (Sec 4.1 + Appendix C) 明确要求:
              1. 在 Iter-1 D_real (50条/任务) 上进行 LoRA fine-tune (200 steps, batch=128)
              2. 微调后 α=0.8 阈值才有实际意义（零样本 p_yes < 0.15，threshold=0.8 → reward=0 永远）
            临时方案 (D_real 标注):
              - 使用 env_success_at_end 替代 reward 字段（ManiSkill 仿真完全可信，等价论文 r_τ）
              - 或使用 p_yes 作为连续软权重用于 FM loss 加权（方向正确，~3x success/failure ratio）
            D_syn 标注必须等 fine-tuned 奖励模型就绪后才有效。
            Fine-tuning 实现: train_reward_model.py（待实现 @ P3.2）
        """
        if not self._loaded:
            self.load_model()

        # 如果传入的帧数已 ≤ num_frames，直接使用（尊重外部采样策略，如末尾帧保留）
        # 否则均匀采样到 num_frames
        if isinstance(frames, np.ndarray):
            _n = frames.shape[0]
        else:
            _n = len(frames)
        if _n <= self.config.num_frames:
            sampled = uniform_sample_frames(frames, _n)
        else:
            sampled = uniform_sample_frames(frames, self.config.num_frames)
        p_yes = self._forward_p_yes(sampled, instruction)
        reward = int(p_yes > self.config.threshold)

        return {
            "p_yes": p_yes,
            "reward": reward,
            "threshold": self.config.threshold,
            "num_frames": len(sampled),
        }

    def score_batch(
        self,
        trajectories: List[Union[List[Image.Image], np.ndarray]],
        instructions: Union[str, List[str]],
    ) -> List[dict]:
        """
        批量对多条轨迹评分。

        Args:
            trajectories: 轨迹列表，每条轨迹为 PIL 列表或 numpy [T,H,W,C]
            instructions: 单个指令（对所有轨迹相同）或指令列表

        Returns:
            dict 列表，每条对应 score_trajectory 的输出
        """
        if not self._loaded:
            self.load_model()

        if isinstance(instructions, str):
            instructions = [instructions] * len(trajectories)

        assert len(trajectories) == len(instructions), (
            f"trajectories({len(trajectories)}) 与 instructions({len(instructions)}) 数量不匹配"
        )

        results = []
        for i, (traj, instr) in enumerate(zip(trajectories, instructions)):
            result = self.score_trajectory(traj, instr)
            results.append(result)
            if (i + 1) % 10 == 0:
                print(f"[VLAW] scored {i+1}/{len(trajectories)}")
        return results

    def label_hdf5(
        self,
        hdf5_path: str,
        instruction: str,
        obs_key: str = "obs/rgb",
        camera_key: str = "base_camera",
    ) -> dict:
        """
        直接从 HDF5 rollout 文件读取帧并评分。

        Args:
            hdf5_path: HDF5 文件路径
            instruction: 任务指令
            obs_key: HDF5 中图像数据的 key
            camera_key: 相机名称 (obs_key 下的子 key)

        Returns:
            score_trajectory 的结果 dict，附加 "hdf5_path" 字段
        """
        import h5py

        with h5py.File(hdf5_path, "r") as f:
            # 兼容多种 HDF5 结构
            if obs_key in f:
                grp = f[obs_key]
                if camera_key in grp:
                    arr = grp[camera_key][:]         # [T, H, W, C]
                elif "rgb" in grp:
                    arr = grp["rgb"][:]
                else:
                    # 尝试第一个 key
                    first_key = list(grp.keys())[0]
                    arr = grp[first_key][:]
            else:
                # 向后兼容 roboreward 格式
                arr = f["observations/images"][:]

        result = self.score_trajectory(arr, instruction)
        result["hdf5_path"] = hdf5_path
        return result

    # ── 便捷属性 ──────────────────────────────────────────────────────────────

    @property
    def device(self) -> str:
        if self.model is not None:
            return str(self.model.device)
        return self.config.device

    def __repr__(self) -> str:
        loaded = "loaded" if self._loaded else "not loaded"
        return (
            f"VLAWRewardModel({self.config.model_path}, "
            f"α={self.config.threshold}, {loaded})"
        )
