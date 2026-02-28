# 环境配置备忘录

> 记录所有环境细节，避免重复踩坑。新 Agent 启动前必读。

---

## Conda 环境

| 环境名 | Python | 主要包 | 用途 | GPU |
|--------|--------|--------|------|-----|
| `ctrl_world` | 3.10 | torch 2.6.0+cu124, diffusers 0.34.0, accelerate, swanlab[dashboard] | Ctrl-World 训练/推理 | 0-3 |
| `rlft_ms3` | 3.10 | torch 2.x, mani-skill3, h5py, diffusers | ManiSkill 数据收集, VAE 编码 | 4-5 |
| `vlaw_reward` | 3.10 | torch 2.8+cu128, transformers 5.2.0, peft 0.18.1 | VLM 推理/微调 | 6-7 |

**注意**:
- `vlaw_reward` 中 flash-attn **未安装**（安装失败），推理用标准 attention，速度约慢 2×
- `ctrl_world` 中需 `swanlab[dashboard]` 而非仅 `swanlab`（否则 local mode 缺少 swanboard）

---

## 模型权重路径

| 模型 | 路径 | 大小 |
|------|------|------|
| Ctrl-World pretrained | `checkpoints/vlaw/world_model/pretrained/Ctrl-World/checkpoint-10000.pt` | 8.7GB |
| Ctrl-World Phase-A | `checkpoints/vlaw/world_model/phase_a/` | ~17GB |
| SVD (stable-video-diffusion) | `checkpoints/vlaw/world_model/pretrained/stable-video-diffusion-img2vid/` | ~7GB |
| CLIP (ViT-B/32) | `checkpoints/vlaw/world_model/pretrained/clip-vit-base-patch32/` | 581MB |
| Qwen3-VL-4B-Instruct | `checkpoints/vlaw/reward_model/qwen_vl/` | 8.3GB |
| Qwen3-VL LoRA Iter1 | `checkpoints/vlaw/reward_model/lora_iter1/final/` | 23.6MB |
| sd-vae-ft-mse | `~/.cache/huggingface/hub/models--stabilityai--sd-vae-ft-mse/` | ~335MB |
| ShortCut Flow (Base) | `checkpoints/il/best_eval_success_once.pt` | — |

---

## 网络代理

外网访问（HuggingFace、GitHub）需使用代理：
```bash
export http_proxy=http://10.20.93.149:7890
export https_proxy=http://10.20.93.149:7890
```
训练时不需要代理，仅下载时使用。

---

## GPU 分配（默认方案）

| GPU | 用途 | conda 环境 |
|-----|------|------------|
| 0-3 | Ctrl-World WM 训练 | `ctrl_world` |
| 4-5 | ManiSkill 数据收集 / VAE 编码 / Imagination | `rlft_ms3` |
| 6-7 | VLM 奖励模型推理/微调 | `vlaw_reward` |
| 8-9 | 策略训练 + 评估 | `rlft_ms3` |

> GPU 全部空闲时可按需灵活分配，上表为默认分配方案。

---

## 常用调试命令

```bash
# 检查 WM 训练进度
tail -50 /tmp/wm_train_phase_a.log
pgrep -af train_wm

# 检查 GPU 使用
nvidia-smi

# 验证 HDF5 数据结构
python -c "
import h5py, sys
with h5py.File(sys.argv[1]) as f:
    for k in list(f.keys())[:3]:
        print(k, {d: f[k][d].shape for d in f[k].keys() if hasattr(f[k][d], 'shape')})
" path/to/file.h5

# 验证 Dataset 可加载
conda run -n ctrl_world python -c "
import sys; sys.path.insert(0,'ctrl_world')
from config import wm_args_maniskill
from dataset.dataset_maniskill import Dataset_ManiSkill
args = wm_args_maniskill()
args.dataset_root_path = 'data/vlaw/encoded/demos'
args.dataset_names = 'LiftPegUpright-v1+PickCube-v1+StackCube-v1'
args.data_stat_path = 'data/vlaw/meta_info/maniskill/stat.json'
ds = Dataset_ManiSkill(args, mode='train')
print('samples:', len(ds))
"
```
