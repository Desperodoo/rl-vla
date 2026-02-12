"""
L1: DSRL 组件单元测试

测试各组件在纯 CPU 上的正确性（不需要 checkpoint 或 ManiSkill3 环境）：
  - ScaledSquashedNormal: 分布属性
  - DSRLActor: 前向传播、输出形状、输出范围
  - DSRLCritic: 前向传播、输出形状、集成 Q 网络
  - DSRLSACAgent: 损失计算、目标网络更新
  - DSRLReplayBuffer: 存储、采样、边界条件

运行:
    conda activate carm
    cd /home/lizh/rl-vla
    python -m pytest rlft/tests/dsrl/test_L1_unit.py -v
"""

import pytest
import sys
from pathlib import Path

import torch
import numpy as np

_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from rlft.algorithms.online_rl.dsrl_sac import (
    ScaledSquashedNormal,
    DSRLActor,
    DSRLCritic,
    DSRLSACAgent,
)
from rlft.buffers.dsrl_buffer import DSRLReplayBuffer


# =====================================================================
# 固定配置
# =====================================================================

OBS_DIM = 562          # 2 * (256 + 25)
ACT_STEPS = 8
ACTION_DIM = 7
NOISE_DIM = ACT_STEPS * ACTION_DIM  # 56
ACTION_MAG = 1.5
HIDDEN_DIMS = [256, 256]  # 小网络用于快速测试
BATCH = 16
DEVICE = "cpu"


# =====================================================================
# ScaledSquashedNormal 测试
# =====================================================================

class TestScaledSquashedNormal:
    """验证缩放的 TanhNormal 分布。"""

    def _make_dist(self, batch=BATCH, dim=NOISE_DIM, scale=ACTION_MAG):
        loc = torch.randn(batch, dim)
        std = torch.ones(batch, dim) * 0.5
        return ScaledSquashedNormal(loc, std, action_scale=scale)

    def test_sample_shape(self):
        dist = self._make_dist()
        action, log_prob = dist.sample_with_log_prob()
        assert action.shape == (BATCH, NOISE_DIM)
        assert log_prob.shape == (BATCH,)

    def test_sample_in_range(self):
        """采样结果必须在 [-action_magnitude, +action_magnitude] 范围内。"""
        dist = self._make_dist()
        for _ in range(10):
            action, _ = dist.sample_with_log_prob()
            assert action.abs().max() <= ACTION_MAG + 1e-5, \
                f"Action out of range: max={action.abs().max():.4f}, expected <= {ACTION_MAG}"

    def test_mean_in_range(self):
        dist = self._make_dist()
        mean = dist.mean
        assert mean.abs().max() <= ACTION_MAG + 1e-5

    def test_log_prob_finite(self):
        dist = self._make_dist()
        action, log_prob = dist.sample_with_log_prob()
        assert torch.isfinite(log_prob).all(), "Log prob contains inf/nan"

    def test_log_prob_method(self):
        """验证 log_prob() 方法与采样的 log_prob 一致。"""
        dist = self._make_dist()
        action, lp1 = dist.sample_with_log_prob()
        lp2 = dist.log_prob(action)
        diff = (lp1 - lp2).abs().max().item()
        assert diff < 1e-4, f"log_prob mismatch: {diff}"

    def test_scale_1_vs_standard(self):
        """scale=1 时应退化为标准 tanh squashed normal。"""
        loc = torch.randn(BATCH, NOISE_DIM)
        std = torch.ones(BATCH, NOISE_DIM) * 0.3
        dist = ScaledSquashedNormal(loc, std, action_scale=1.0)
        action, _ = dist.sample_with_log_prob()
        assert action.abs().max() <= 1.0 + 1e-5


# =====================================================================
# DSRLActor 测试
# =====================================================================

class TestDSRLActor:
    """测试 DSRL Actor 网络。"""

    @pytest.fixture
    def actor(self):
        return DSRLActor(
            obs_dim=OBS_DIM,
            noise_dim=NOISE_DIM,
            hidden_dims=HIDDEN_DIMS,
            action_magnitude=ACTION_MAG,
            log_std_init=-3.0,
        )

    def test_forward_returns_distribution(self, actor):
        obs = torch.randn(BATCH, OBS_DIM)
        dist = actor(obs)
        assert isinstance(dist, ScaledSquashedNormal)

    def test_output_shape(self, actor):
        obs = torch.randn(BATCH, OBS_DIM)
        action, log_prob = actor.get_action(obs, deterministic=False)
        assert action.shape == (BATCH, NOISE_DIM)
        assert log_prob.shape == (BATCH,)

    def test_deterministic_output(self, actor):
        obs = torch.randn(BATCH, OBS_DIM)
        a1, lp1 = actor.get_action(obs, deterministic=True)
        a2, lp2 = actor.get_action(obs, deterministic=True)
        assert torch.allclose(a1, a2), "Deterministic actions should be identical"
        assert lp1 is None
        assert lp2 is None

    def test_action_range(self, actor):
        obs = torch.randn(BATCH, OBS_DIM)
        for _ in range(20):
            action, _ = actor.get_action(obs, deterministic=False)
            assert action.abs().max() <= ACTION_MAG + 1e-5, \
                f"Action out of bounds: {action.abs().max():.4f}"

    def test_initial_std_small(self, actor):
        """初始 std 应约为 exp(-3) ≈ 0.05（保护预训练策略）。"""
        obs = torch.randn(BATCH, OBS_DIM)
        dist = actor(obs)
        mean_std = dist.scale_std.mean().item()
        expected_std = np.exp(-3.0)
        assert abs(mean_std - expected_std) < 0.05, \
            f"Initial std {mean_std:.4f} != expected {expected_std:.4f}"

    def test_parameter_count(self, actor):
        n_params = sum(p.numel() for p in actor.parameters())
        assert n_params > 0
        print(f"  Actor params: {n_params:,}")


# =====================================================================
# DSRLCritic 测试
# =====================================================================

class TestDSRLCritic:
    """测试 DSRL Critic 网络。"""

    @pytest.fixture
    def critic(self):
        return DSRLCritic(
            obs_dim=OBS_DIM,
            noise_dim=NOISE_DIM,
            hidden_dims=HIDDEN_DIMS,
            num_qs=2,
            use_layer_norm=False,
        )

    def test_forward_shape(self, critic):
        obs = torch.randn(BATCH, OBS_DIM)
        noise = torch.randn(BATCH, NOISE_DIM)
        q = critic(noise, obs)
        assert q.shape == (2, BATCH, 1)

    def test_min_q(self, critic):
        obs = torch.randn(BATCH, OBS_DIM)
        noise = torch.randn(BATCH, NOISE_DIM)
        min_q = critic.get_min_q(noise, obs)
        assert min_q.shape == (BATCH, 1)

    def test_mean_q(self, critic):
        obs = torch.randn(BATCH, OBS_DIM)
        noise = torch.randn(BATCH, NOISE_DIM)
        mean_q = critic.get_mean_q(noise, obs)
        assert mean_q.shape == (BATCH, 1)

    def test_min_q_leq_mean(self, critic):
        """min_q <= mean_q。"""
        obs = torch.randn(BATCH, OBS_DIM)
        noise = torch.randn(BATCH, NOISE_DIM)
        min_q = critic.get_min_q(noise, obs)
        mean_q = critic.get_mean_q(noise, obs)
        assert (min_q <= mean_q + 1e-5).all()

    def test_with_layer_norm(self):
        critic = DSRLCritic(
            obs_dim=OBS_DIM, noise_dim=NOISE_DIM,
            hidden_dims=HIDDEN_DIMS, num_qs=2, use_layer_norm=True,
        )
        obs = torch.randn(BATCH, OBS_DIM)
        noise = torch.randn(BATCH, NOISE_DIM)
        q = critic(noise, obs)
        assert q.shape == (2, BATCH, 1)


# =====================================================================
# DSRLSACAgent 测试
# =====================================================================

class TestDSRLSACAgent:
    """测试完整的 DSRL-SAC Agent。"""

    @pytest.fixture
    def agent(self):
        return DSRLSACAgent(
            obs_dim=OBS_DIM,
            act_steps=ACT_STEPS,
            action_dim=ACTION_DIM,
            action_magnitude=ACTION_MAG,
            hidden_dims=HIDDEN_DIMS,
            num_qs=2,
            gamma=0.99,
            tau=0.005,
            init_temperature=1.0,
            target_entropy=0.0,
            log_std_init=-3.0,
            use_layer_norm=False,
            device=DEVICE,
        )

    def test_select_action_shape(self, agent):
        obs = torch.randn(BATCH, OBS_DIM)
        action = agent.select_action(obs, deterministic=False)
        assert action.shape == (BATCH, NOISE_DIM)

    def test_select_action_range(self, agent):
        obs = torch.randn(BATCH, OBS_DIM)
        for _ in range(10):
            action = agent.select_action(obs, deterministic=False)
            assert action.abs().max() <= ACTION_MAG + 1e-5

    def test_select_action_single(self, agent):
        """单个观察输入（无 batch 维度）。"""
        obs = torch.randn(OBS_DIM)
        action = agent.select_action(obs, deterministic=False)
        assert action.shape == (1, NOISE_DIM)

    def test_compute_critic_loss(self, agent):
        obs = torch.randn(BATCH, OBS_DIM)
        noise = torch.randn(BATCH, NOISE_DIM)
        next_obs = torch.randn(BATCH, OBS_DIM)
        rewards = torch.randn(BATCH)
        dones = torch.zeros(BATCH)

        loss, metrics = agent.compute_critic_loss(obs, noise, next_obs, rewards, dones)
        assert loss.shape == ()
        assert torch.isfinite(loss)
        assert "critic_loss" in metrics
        assert "q_mean" in metrics
        assert "td_target_mean" in metrics

    def test_compute_actor_loss(self, agent):
        obs = torch.randn(BATCH, OBS_DIM)
        loss, metrics = agent.compute_actor_loss(obs)
        assert loss.shape == ()
        assert torch.isfinite(loss)
        assert "actor_loss" in metrics
        assert "actor_entropy" in metrics

    def test_compute_temperature_loss(self, agent):
        obs = torch.randn(BATCH, OBS_DIM)
        loss, metrics = agent.compute_temperature_loss(obs)
        assert loss.shape == ()
        assert torch.isfinite(loss)
        assert "temperature" in metrics
        assert "entropy" in metrics

    def test_target_update(self, agent):
        """验证 soft-update 后 target 参数发生变化。"""
        # 先记录 target 参数
        old_params = {
            k: v.clone() for k, v in agent.critic_target.state_dict().items()
        }
        # 随机修改 critic 参数使其不同
        with torch.no_grad():
            for p in agent.critic.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        agent.update_target()

        # target 参数应该有变化（soft update）
        changed = False
        for k, v in agent.critic_target.state_dict().items():
            if not torch.allclose(v, old_params[k]):
                changed = True
                break
        assert changed, "Target network should have changed after update"

    def test_alpha_positive(self, agent):
        """温度 α 必须为正数。"""
        assert agent.alpha.item() > 0

    def test_gradient_flow(self, agent):
        """验证梯度可以流经 actor 和 critic。"""
        obs = torch.randn(BATCH, OBS_DIM)
        noise = torch.randn(BATCH, NOISE_DIM)
        next_obs = torch.randn(BATCH, OBS_DIM)
        rewards = torch.randn(BATCH)
        dones = torch.zeros(BATCH)

        # Critic loss backward
        critic_loss, _ = agent.compute_critic_loss(obs, noise, next_obs, rewards, dones)
        critic_loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in agent.critic.parameters())
        assert has_grad, "Critic should have gradients"
        agent.zero_grad()

        # Actor loss backward
        actor_loss, _ = agent.compute_actor_loss(obs)
        actor_loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in agent.actor.parameters())
        assert has_grad, "Actor should have gradients"


# =====================================================================
# DSRLReplayBuffer 测试
# =====================================================================

class TestDSRLReplayBuffer:
    """测试 DSRL 回放缓冲区。"""

    @pytest.fixture
    def buffer(self):
        return DSRLReplayBuffer(
            capacity=100,
            obs_dim=OBS_DIM,
            noise_dim=NOISE_DIM,
            device=DEVICE,
        )

    def test_empty_buffer(self, buffer):
        assert buffer.size == 0

    def test_add_batch(self, buffer):
        n = 10
        obs = np.random.randn(n, OBS_DIM).astype(np.float32)
        action = np.random.randn(n, NOISE_DIM).astype(np.float32)
        reward = np.random.randn(n).astype(np.float32)
        next_obs = np.random.randn(n, OBS_DIM).astype(np.float32)
        done = np.zeros(n, dtype=np.float32)

        buffer.add(obs, action, reward, next_obs, done)
        assert buffer.size == n

    def test_add_single(self, buffer):
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        action = np.random.randn(NOISE_DIM).astype(np.float32)
        buffer.add_single(obs, action, 1.0, obs, 0.0)
        assert buffer.size == 1

    def test_sample_shape(self, buffer):
        # 先填充数据
        n = 50
        obs = np.random.randn(n, OBS_DIM).astype(np.float32)
        action = np.random.randn(n, NOISE_DIM).astype(np.float32)
        reward = np.random.randn(n).astype(np.float32)
        next_obs = np.random.randn(n, OBS_DIM).astype(np.float32)
        done = np.zeros(n, dtype=np.float32)
        buffer.add(obs, action, reward, next_obs, done)

        batch = buffer.sample(16)
        assert batch["obs"].shape == (16, OBS_DIM)
        assert batch["next_obs"].shape == (16, OBS_DIM)
        assert batch["actions"].shape == (16, NOISE_DIM)
        assert batch["rewards"].shape == (16,)
        assert batch["dones"].shape == (16,)

    def test_sample_returns_tensors(self, buffer):
        n = 50
        obs = np.random.randn(n, OBS_DIM).astype(np.float32)
        action = np.random.randn(n, NOISE_DIM).astype(np.float32)
        reward = np.random.randn(n).astype(np.float32)
        next_obs = np.random.randn(n, OBS_DIM).astype(np.float32)
        done = np.zeros(n, dtype=np.float32)
        buffer.add(obs, action, reward, next_obs, done)

        batch = buffer.sample(8)
        for k, v in batch.items():
            assert isinstance(v, torch.Tensor), f"{k} should be tensor"
            assert v.device.type == DEVICE

    def test_circular_overwrite(self, buffer):
        """容量满后应循环覆写。"""
        for i in range(150):
            obs = np.random.randn(OBS_DIM).astype(np.float32)
            action = np.random.randn(NOISE_DIM).astype(np.float32)
            buffer.add_single(obs, action, float(i), obs, 0.0)

        assert buffer.size == 100  # capacity=100
        assert buffer.ptr == 50   # 150 % 100

    def test_data_integrity(self, buffer):
        """验证写入数据可以正确读取。"""
        obs = np.ones((1, OBS_DIM), dtype=np.float32) * 42.0
        action = np.ones((1, NOISE_DIM), dtype=np.float32) * 7.0
        reward = np.array([3.14], dtype=np.float32)
        next_obs = np.ones((1, OBS_DIM), dtype=np.float32) * 43.0
        done = np.array([1.0], dtype=np.float32)

        buffer.add(obs, action, reward, next_obs, done)

        # 采样唯一元素，应该正是我们写入的
        batch = buffer.sample(1)
        assert torch.allclose(batch["obs"], torch.tensor(42.0))
        assert torch.allclose(batch["actions"], torch.tensor(7.0))
        assert torch.allclose(batch["rewards"], torch.tensor(3.14))
        assert torch.allclose(batch["next_obs"], torch.tensor(43.0))
        assert torch.allclose(batch["dones"], torch.tensor(1.0))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
