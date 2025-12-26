# rlft_ms3 环境配置总结

## ✅ 成功配置

### 核心版本
- **JAX**: 0.4.28 (CUDA 12, cuDNN 8.9)
- **Flax**: 0.8.5
- **Optax**: 0.1.9
- **Chex**: 0.1.86
- **Distrax**: 0.1.3
- **PyTorch**: 2.3.1+cu121
- **cuDNN**: 8.9.2.26

### 关键点
1. ✅ JAX和PyTorch共享cuDNN 8.9，完全兼容
2. ✅ 所有依赖版本已固定，避免自动升级
3. ✅ GPU加速正常工作（JAX和PyTorch均可用）

---

## 📝 安装步骤（从零开始）

```bash
# 1. 创建环境
conda create -n "rlft_ms3" "python==3.10"
conda activate rlft_ms3

# 2. 安装JAX（必须先安装）
pip install "jax[cuda12_pip]==0.4.28" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 3. 安装JAX生态包（使用--no-deps避免自动升级）
pip install --no-deps flax==0.8.5
pip install --no-deps optax==0.1.9
pip install --no-deps chex==0.1.86
pip install --no-deps distrax==0.1.3

# 4. 安装PyTorch
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121

# 5. 安装其他依赖
pip install gymnasium==0.29.1 gymnax==0.0.6 tensorboard wandb omegaconf \
    opencv-python transforms3d tqdm dacite matplotlib h5py moviepy scipy

# 6. 安装ManiSkill（可选）
pip install mani_skill
```

---

## ⚠️ 重要注意事项

### 1. **不要使用** `pip install -e rlpd_jax`
原因：pyproject.toml中的依赖没有版本锁定，会自动升级JAX/Flax到不兼容版本

**替代方案**：依赖已手动安装，直接使用即可

### 2. **必须使用** `XLA_PYTHON_CLIENT_PREALLOCATE=false`
```bash
# 在所有使用JAX的命令前添加
XLA_PYTHON_CLIENT_PREALLOCATE=false python train.py ...
```

**作用**：
- JAX默认会预分配所有GPU显存
- 设置为false后按需分配，避免与PyTorch冲突
- 允许多个程序共享GPU

### 3. 环境变量作用说明

#### `XLA_PYTHON_CLIENT_PREALLOCATE=false`
- **默认（true）**: JAX启动时预分配几乎所有GPU显存
- **设置为false**: 按需分配，只在需要时才分配显存
- **使用场景**:
  - ✅ 多个程序共享GPU
  - ✅ GPU显存有限
  - ✅ 同时使用JAX和PyTorch
  - ✅ 调试代码

---

## 🔧 依赖冲突历史

### 问题1: JAX API变更
- **错误**: `jax.tree_map` 在JAX 0.4.25+被废弃
- **修复**: 替换为 `jax.tree.map`
- **文件**: 
  - `rlpd_jax/rfcl/agents/sac/sac.py`
  - `rlpd_jax/rfcl/agents/base.py`
  - `rlpd_jax/rfcl/agents/sac/loss.py`

### 问题2: Flax版本不兼容
- **原因**: Flax 0.7.x/0.6.x使用了JAX 0.4.14后移除的API
- **解决**: 使用Flax 0.8.5（向后兼容JAX 0.4.28）

### 问题3: cuDNN版本冲突
- **原因**: PyTorch 2.5.1需要cuDNN 9.x，JAX 0.4.28需要cuDNN 8.9
- **解决**: 降级PyTorch到2.3.1（支持cuDNN 8.9）

### 问题4: 动作维度不匹配
- **错误**: 数据集使用`pd_ee_delta_pose`(7维)，配置使用`pd_joint_delta_pos`(8维)
- **修复**: 在baselines.sh中添加 `env.env_kwargs.control_mode="pd_ee_delta_pose"`

---

## ✅ 验证环境

```bash
conda activate rlft_ms3
python << 'EOF'
import jax, flax, optax, torch
print(f"JAX: {jax.__version__} - {jax.devices()}")
print(f"Flax: {flax.__version__}")
print(f"PyTorch: {torch.__version__} - CUDA: {torch.cuda.is_available()}")

# 测试RFCL
from rfcl.agents.sac import SAC
print("✓ RFCL/RLPD 可用")
EOF
```

---

## 📚 参考资料

- RLPD Paper: https://arxiv.org/abs/2302.02948
- JAX文档: https://jax.readthedocs.io/
- ManiSkill文档: https://maniskill.readthedocs.io/

---

最后更新: 2025-12-14
