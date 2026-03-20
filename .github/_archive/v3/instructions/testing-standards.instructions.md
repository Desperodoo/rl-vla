---
name: 'Testing Standards'
description: 'VLAW 项目测试规范'
applyTo: 'rlft/tests/**/*.py'
---

# VLAW 测试规范

## 1. 文件与命名

- 测试文件命名：`test_{module_name}.py`（与被测模块名一一对应）
- 测试函数命名：`test_{function_name}_{scenario}()`
  - 例：`test_concat_cameras_vertical()`、`test_concat_cameras_invalid_mode()`
- 每个测试函数**必须有 docstring** 说明测试目的，至少一句话

```python
def test_concat_cameras_vertical(mock_rgb_frames):
    """验证垂直拼接后输出 shape 为 (T, 2H, W, 3)。"""
    ...
```

## 2. Fixture 规范

- 公共 fixture 统一放在 `rlft/tests/vlaw/conftest.py`
- 使用 `pytest.fixture` 提供 mock 数据；**不依赖真实模型权重或网络**
- fixture 参数限于 `tmp_path`（pytest 内置）或其他已定义 fixture
- mock 数据用最小覆盖边界条件的尺寸（如 T=3、T=10，不要 T=200）
- 需要临时文件时使用 `tmp_path` fixture，不要 `os.tmpdir` 硬编码

```python
@pytest.fixture
def mock_rgb_frames():
    """提供 10 帧 192×192 RGB 随机图像，uint8。"""
    return np.random.randint(0, 255, (10, 192, 192, 3), dtype=np.uint8)
```

## 3. 形状与类型测试

- 形状测试**必须同时检查 dtype**，特别区分 float16 / float32 / uint8

```python
assert out.shape == (T, 4, 48, 24)
assert out.dtype == torch.float16  # 明确写 dtype，不要只看 shape
```

## 4. GPU 相关测试

- GPU 密集型测试用以下标记，确保 CI 无 GPU 时自动跳过：

```python
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="需要 GPU，CI 环境跳过"
)
def test_vae_encode_gpu(mock_rgb_frames):
    ...
```

## 5. 独立性原则

- **每个测试独立**：不依赖其他测试的副作用，不依赖全局状态
- 不修改 `rlft/vlaw/` 下任何文件
- 不在测试函数中调用 `os.chdir()`

## 6. 错误路径测试

- 每个核心函数**必须有至少一个异常路径测试**

```python
def test_concat_cameras_invalid_mode():
    """传入非法 concat_mode 时应抛出 ValueError。"""
    a = np.zeros((3, 192, 192, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Unknown concat_mode"):
        concat_cameras(a, a, mode="diagonal")
```

## 7. 集成测试的 Mock 原则

- 集成测试不加载真实模型，使用 `torch.randn` 或 `np.random` 替代
- 对有外部依赖的类（如 `VLAWRewardModel`），用 `unittest.mock.patch` 或 `monkeypatch` 隔离

```python
def test_reward_score_range(monkeypatch):
    """奖励分数应在 [0, 1] 范围内（使用 mock 模型）。"""
    def fake_score(*args, **kwargs):
        return 0.85
    monkeypatch.setattr("rlft.vlaw.reward_model.VLAWRewardModel.score_trajectory", fake_score)
    ...
```

## 8. 测试输出规范

- 测试中可用 `print` 输出调试信息，但**不要依赖 stdout 内容做断言**
- 失败信息要包含实际值，方便快速定位：

```python
assert out.shape == expected, f"期望 {expected}，实际 {out.shape}"
```

## 9. 禁止事项

- ❌ 不在测试中硬编码路径（如 `/home/wjz/...`）
- ❌ 不在测试中启动真实 ManiSkill 环境
- ❌ 不在测试中下载模型权重
- ❌ 不在测试中训练模型（哪怕一步）
- ❌ 不使用 `time.sleep()` 等待异步操作
