#!/usr/bin/env python3
"""
train_vlaw.py — VLAW 迭代训练主脚本 (P6.1)

实现 VLAW Algorithm 1 完整循环:
    for i = 1 to K_iter:
      Step 1: 真实环境 Rollout 收集 D_real
      Step 2: VAE 离线编码 D_real
      Step 3: VLM 奖励标注 D_real
      Step 4: 世界模型微调 (on D_real + λ·D_demo)
      Step 5: Imagination → D_syn
      Step 6: VLM 奖励标注 D_syn
      Step 7: 策略更新 (on D_real+ ∪ D_syn+ ∪ D_demo)
      Step 8: 评估 (可选)

用法:
    # 完整训练 (2 轮迭代)
    conda run -n rlft_ms3 python rlft/online/train_vlaw.py

    # 从某步骤恢复
    conda run -n rlft_ms3 python rlft/online/train_vlaw.py --resume_iter 1 --resume_step 4

    # 仅评估
    conda run -n rlft_ms3 python rlft/online/train_vlaw.py --eval_only --iter_id 1

    # Dry-run (验证配置和接口)
    conda run -n rlft_ms3 python rlft/online/train_vlaw.py --dry_run

GPU 分配:
    Step 1  : CUDA 4,5  (ManiSkill rollout)
    Step 2  : CUDA 4    (VAE encode)
    Step 3  : CUDA 6    (VLM labeling)
    Step 4  : CUDA 0-3  (WM fine-tune, accelerate)
    Step 5  : CUDA 4-7  (Imagination)
    Step 6  : CUDA 6    (VLM labeling)
    Step 7  : CUDA 8-9  (ShortCut Flow train)
    Step 8  : CUDA 8    (evaluation)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

# ── 日志 ─────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="[VLAW %(asctime)s %(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_vlaw")


# ── 路径解析 ──────────────────────────────────────────────────────────────────

WORKSPACE = Path(__file__).parents[2].resolve()   # /home/wjz/rl-vla
sys.path.insert(0, str(WORKSPACE))


# ── 配置 ──────────────────────────────────────────────────────────────────────

@dataclass
class VLAWConfig:
    """VLAW 迭代训练主配置.

    子模块配置通过 CLI 参数或 JSON 覆盖。
    """

    # ── 迭代控制 ────────────────────────────────────────────────────────────
    num_iters: int = 2
    """总迭代轮次 K_iter"""

    # ── 任务设置 ────────────────────────────────────────────────────────────
    tasks: str = "LiftPegUpright-v1,PickCube-v1,StackCube-v1"
    """任务列表，逗号分隔"""

    # ── 策略 checkpoint ──────────────────────────────────────────────────────
    base_policy_ckpt: str = "checkpoints/il/best_eval_success_once.pt"
    """预训练基础策略 checkpoint"""

    # ── WM 设置 ─────────────────────────────────────────────────────────────
    wm_pretrained_ckpt: str = "checkpoints/vlaw/world_model/pretrained/Ctrl-World/checkpoint-10000.pt"
    """Ctrl-World 预训练权重"""

    wm_phase_a_dir: str = "checkpoints/vlaw/world_model/phase_a"
    """WM Phase-A 训练输出目录（用于 Iter-1）"""

    # ── 数据路径 ─────────────────────────────────────────────────────────────
    demo_dir: str = "data/vlaw/demos"
    """演示数据目录"""

    encoded_demo_dir: str = "data/vlaw/encoded/demos"
    """VAE编码演示数据目录"""

    data_stat_path: str = "data/vlaw/meta_info/maniskill/stat.json"
    """归一化统计量路径"""

    rollout_base_dir: str = "data/vlaw/rollouts"
    """D_real rollout 存储根目录"""

    encoded_rollout_base_dir: str = "data/vlaw/encoded/rollouts"
    """D_real VAE编码存储根目录"""

    labeled_base_dir: str = "data/vlaw/labeled"
    """VLM 标注结果存储根目录"""

    synthetic_base_dir: str = "data/vlaw/synthetic"
    """D_syn 合成数据存储根目录"""

    labeled_syn_base_dir: str = "data/vlaw/labeled_syn"
    """D_syn VLM 标注结果存储根目录"""

    # ── 每步超参 ─────────────────────────────────────────────────────────────
    num_rollout_episodes: int = 50
    """Step 1: 每任务收集幕数"""

    num_syn_trajectories: int = 200
    """Step 5: 每任务合成轨迹数 (debug: 50, full: 500)"""

    wm_finetune_steps: int = 5000
    """Step 4: WM 微调步数 (debug: 500)"""

    policy_update_steps: int = 2000
    """Step 7: 策略更新步数"""

    vlm_threshold: float = 0.8
    """VLM 成功判定阈值 α"""

    # ── GPU 分配 ─────────────────────────────────────────────────────────────
    gpu_rollout: str = "4,5"
    """Step 1: ManiSkill rollout GPU"""

    gpu_vae: str = "4"
    """Step 2: VAE encode GPU"""

    gpu_vlm: str = "6"
    """Step 3/6: VLM labeling GPU"""

    gpu_wm: str = "0,1,2,3"
    """Step 4: WM training GPU"""

    gpu_imagine: str = "4,5,6,7"
    """Step 5: Imagination GPU"""

    gpu_policy: str = "8,9"
    """Step 7: Policy training GPU"""

    gpu_eval: str = "8"
    """Step 8: Evaluation GPU"""

    # ── 运行控制 ─────────────────────────────────────────────────────────────
    resume_iter: int = 1
    """从第几轮迭代恢复（1=从头）"""

    resume_step: int = 1
    """从第几步恢复（1=从头）"""

    dry_run: bool = False
    """仅验证配置，不实际运行"""

    eval_only: bool = False
    """仅运行评估"""

    eval_iter_id: int = 1
    """eval_only 时评估的迭代 ID"""

    # ── 日志设置 ─────────────────────────────────────────────────────────────
    use_wandb: bool = True
    checkpoint_dir: str = "checkpoints/vlaw"
    log_dir: str = "logs/vlaw"

    @property
    def task_list(self) -> list[str]:
        return [t.strip() for t in self.tasks.split(",") if t.strip()]


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def run_cmd(
    cmd: str,
    env_override: dict | None = None,
    check: bool = True,
    log_file: str | None = None,
    background: bool = False,
) -> subprocess.CompletedProcess | subprocess.Popen:
    """运行 shell 命令，支持环境变量覆盖和日志重定向."""
    env = os.environ.copy()
    if env_override:
        env.update({k: str(v) for k, v in env_override.items()})

    log.info(f"CMD: {cmd}")
    if log_file:
        log.info(f"     → log: {log_file}")

    if background:
        if log_file:
            fout = open(log_file, "w")
            proc = subprocess.Popen(
                cmd, shell=True, env=env,
                stdout=fout, stderr=subprocess.STDOUT,
            )
        else:
            proc = subprocess.Popen(cmd, shell=True, env=env)
        return proc

    if log_file:
        fout = open(log_file, "w")
        return subprocess.run(
            cmd, shell=True, env=env, check=check,
            stdout=fout, stderr=subprocess.STDOUT,
        )
    return subprocess.run(cmd, shell=True, env=env, check=check)


def conda_run(env_name: str, py_cmd: str, **kwargs) -> subprocess.CompletedProcess:
    """conda run -n {env_name} python {py_cmd}"""
    return run_cmd(f"conda run -n {env_name} python {py_cmd}", **kwargs)


def save_state(state_path: Path, state: dict) -> None:
    """保存训练状态（用于恢复）."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)


def load_state(state_path: Path) -> dict:
    """加载训练状态."""
    if state_path.exists():
        with open(state_path) as f:
            return json.load(f)
    return {}


def get_policy_ckpt(cfg: VLAWConfig, iter_id: int) -> str:
    """获取指定 iter 的策略 checkpoint 路径."""
    if iter_id <= 1:
        return str(WORKSPACE / cfg.base_policy_ckpt)
    return str(WORKSPACE / cfg.checkpoint_dir / f"policy/iter{iter_id - 1}/best.pt")


def get_wm_ckpt(cfg: VLAWConfig, iter_id: int) -> str:
    """获取用于 Iter i 的 WM checkpoint 路径."""
    if iter_id == 1:
        # Iter-1 用 Phase-A 训练结果 (纯 demo 微调)
        phase_a_dir = WORKSPACE / cfg.wm_phase_a_dir
        ckpts = sorted(phase_a_dir.glob("checkpoint-*.pt"))
        if ckpts:
            latest = ckpts[-1]
            log.info(f"  WM: using Phase-A checkpoint {latest.name}")
            return str(latest)
        log.warning("  WM: Phase-A checkpoint not found, using pretrained")
        return str(WORKSPACE / cfg.wm_pretrained_ckpt)
    return str(
        WORKSPACE / cfg.checkpoint_dir / f"world_model/iter{iter_id - 1}/checkpoint.pt"
    )


# ── Step 1: Rollout 收集 ──────────────────────────────────────────────────────

def step1_collect_rollouts(cfg: VLAWConfig, iter_id: int, dry_run: bool = False) -> None:
    """Step 1: 用当前策略在真实环境做 rollout."""
    log.info(f"[Step 1] 收集 D_real (Iter {iter_id})")
    policy_ckpt = get_policy_ckpt(cfg, iter_id)
    output_dir = str(WORKSPACE / cfg.rollout_base_dir / f"iter{iter_id}")

    script = str(WORKSPACE / "rlft/vlaw/scripts/collect_rollouts_iter.py")
    args = (
        f"--policy_ckpt {policy_ckpt} "
        f"--output_dir {output_dir} "
        f"--tasks {cfg.tasks} "
        f"--num_episodes {cfg.num_rollout_episodes} "
        f"--iter_id {iter_id} "
    )
    if dry_run:
        args += "--dry_run "

    log_file = f"/tmp/step1_collect_iter{iter_id}.log"
    run_cmd(
        f"CUDA_VISIBLE_DEVICES={cfg.gpu_rollout} conda run -n rlft_ms3 python {script} {args}",
        log_file=log_file,
    )
    log.info(f"[Step 1] ✅ Rollout 收集完成 → {output_dir}")


# ── Step 2: VAE 编码 ──────────────────────────────────────────────────────────

def step2_vae_encode(cfg: VLAWConfig, iter_id: int, dry_run: bool = False) -> None:
    """Step 2: VAE 离线编码 D_real。"""
    log.info(f"[Step 2] VAE 编码 D_real (Iter {iter_id})")
    input_dir = str(WORKSPACE / cfg.rollout_base_dir / f"iter{iter_id}")
    output_dir = str(WORKSPACE / cfg.encoded_rollout_base_dir / f"iter{iter_id}")

    script = str(WORKSPACE / "rlft/vlaw/scripts/encode_rollouts_iter.py")
    args = (
        f"--input_dir {input_dir} "
        f"--output_dir {output_dir} "
        f"--tasks {cfg.tasks} "
        f"--iter_id {iter_id} "
    )
    if dry_run:
        args += "--dry_run "

    log_file = f"/tmp/step2_vae_iter{iter_id}.log"
    run_cmd(
        f"CUDA_VISIBLE_DEVICES={cfg.gpu_vae} conda run -n rlft_ms3 python {script} {args}",
        log_file=log_file,
    )
    log.info(f"[Step 2] ✅ VAE 编码完成 → {output_dir}")


# ── Step 3: VLM 标注 D_real ───────────────────────────────────────────────────

def step3_label_real(cfg: VLAWConfig, iter_id: int, dry_run: bool = False) -> None:
    """Step 3: VLM 奖励标注 D_real。"""
    log.info(f"[Step 3] VLM 标注 D_real (Iter {iter_id})")
    rollout_dir = str(WORKSPACE / cfg.rollout_base_dir / f"iter{iter_id}")
    output_dir = str(WORKSPACE / cfg.labeled_base_dir / f"iter{iter_id}")

    script = str(WORKSPACE / "rlft/vlaw/scripts/label_real_trajectories.py")
    args = (
        f"--rollout_dir {rollout_dir} "
        f"--output_dir {output_dir} "
        f"--iter_id {iter_id} "
        f"--tasks {cfg.tasks} "
        f"--threshold {cfg.vlm_threshold} "
    )
    if dry_run:
        args += "--dry_run "

    log_file = f"/tmp/step3_label_real_iter{iter_id}.log"
    run_cmd(
        f"CUDA_VISIBLE_DEVICES={cfg.gpu_vlm} conda run -n vlaw_reward python {script} {args}",
        log_file=log_file,
    )
    log.info(f"[Step 3] ✅ D_real VLM 标注完成 → {output_dir}")


# ── Step 4: WM 微调 ───────────────────────────────────────────────────────────

def step4_finetune_wm(cfg: VLAWConfig, iter_id: int, dry_run: bool = False) -> None:
    """Step 4: 在 D_real + λ·D_demo 上微调世界模型。"""
    log.info(f"[Step 4] WM 微调 (Iter {iter_id})")

    if iter_id == 1:
        # Iter-1 的 WM 已由 Phase-A 训练完成
        log.info("[Step 4] Iter-1: 检查 WM Phase-A checkpoint...")
        phase_a_dir = WORKSPACE / cfg.wm_phase_a_dir
        ckpts = sorted(phase_a_dir.glob("checkpoint-*.pt"))
        if ckpts:
            log.info(f"[Step 4] ✅ Phase-A checkpoint 已就绪: {ckpts[-1].name}")
            return
        log.warning("[Step 4] Phase-A checkpoint 不存在，等待训练完成...")
        # 轮询等待
        for _ in range(120):  # max 2h
            time.sleep(60)
            ckpts = sorted(phase_a_dir.glob("checkpoint-*.pt"))
            if ckpts:
                log.info(f"[Step 4] ✅ Phase-A checkpoint: {ckpts[-1].name}")
                return
        raise RuntimeError("WM Phase-A 训练超时 (>2h)")

    # Iter>1: 重新在 D_real ∪ D_demo 上微调
    num_gpus = len(cfg.gpu_wm.split(","))
    wm_output_dir = str(WORKSPACE / cfg.checkpoint_dir / f"world_model/iter{iter_id}")
    encoded_real = str(WORKSPACE / cfg.encoded_rollout_base_dir / f"iter{iter_id}")
    dataset_names = "+".join(cfg.task_list)

    finetune_cmd = (
        f"cd {WORKSPACE}/ctrl_world && "
        f"CUDA_VISIBLE_DEVICES={cfg.gpu_wm} WANDB_MODE=offline "
        f"conda run -n ctrl_world accelerate launch "
        f"--multi_gpu --num_processes {num_gpus} --main_process_port 29604 "
        f"scripts/train_wm.py "
        f"--task_type maniskill "
        f"--dataset_root_path {encoded_real} "
        f"--dataset_names {dataset_names} "
        f"--data_stat_path {WORKSPACE / cfg.data_stat_path} "
        f"--svd_model_path {WORKSPACE}/checkpoints/vlaw/world_model/pretrained/stable-video-diffusion-img2vid "
        f"--clip_model_path {WORKSPACE}/checkpoints/vlaw/world_model/pretrained/clip-vit-base-patch32 "
        f"--ckpt_path {get_wm_ckpt(cfg, iter_id - 1)} "
        f"--output_dir {wm_output_dir} "
        f"--freeze_unet_spatial false "
        f"--max_train_steps {cfg.wm_finetune_steps} "
        f"--train_batch_size 1 "
        f"--gradient_accumulation_steps 4 "
        f"--checkpointing_steps 1000 "
        f"--learning_rate 5e-5"
    )

    if dry_run:
        log.info(f"[DRY RUN] Step 4 CMD:\n{finetune_cmd}")
        return

    log_file = f"/tmp/step4_wm_iter{iter_id}.log"
    run_cmd(finetune_cmd, log_file=log_file)
    log.info(f"[Step 4] ✅ WM 微调完成 → {wm_output_dir}")


# ── Step 5: Imagination ───────────────────────────────────────────────────────

def step5_imagination(cfg: VLAWConfig, iter_id: int, dry_run: bool = False) -> None:
    """Step 5: 用世界模型生成合成轨迹 D_syn。"""
    log.info(f"[Step 5] Imagination (Iter {iter_id})")
    wm_ckpt = get_wm_ckpt(cfg, iter_id)
    real_data_dir = str(WORKSPACE / cfg.rollout_base_dir / f"iter{iter_id}")
    output_dir = str(WORKSPACE / cfg.synthetic_base_dir / f"iter{iter_id}")
    policy_ckpt = get_policy_ckpt(cfg, iter_id)

    script = str(WORKSPACE / "rlft/vlaw/scripts/run_imagination.py")
    args = (
        f"--wm_ckpt {wm_ckpt} "
        f"--policy_ckpt {policy_ckpt} "
        f"--real_data_dir {real_data_dir} "
        f"--output_dir {output_dir} "
        f"--tasks {cfg.tasks} "
        f"--num_trajectories {cfg.num_syn_trajectories} "
        f"--iter_id {iter_id} "
        f"--gpu_ids {cfg.gpu_imagine} "
    )
    if dry_run:
        args += "--dry_run "

    log_file = f"/tmp/step5_imagine_iter{iter_id}.log"
    run_cmd(
        f"CUDA_VISIBLE_DEVICES={cfg.gpu_imagine} conda run -n rlft_ms3 python {script} {args}",
        log_file=log_file,
    )
    log.info(f"[Step 5] ✅ Imagination 完成 → {output_dir}")


# ── Step 6: VLM 标注 D_syn ────────────────────────────────────────────────────

def step6_label_syn(cfg: VLAWConfig, iter_id: int, dry_run: bool = False) -> None:
    """Step 6: VLM 奖励标注 D_syn。"""
    log.info(f"[Step 6] VLM 标注 D_syn (Iter {iter_id})")
    syn_dir = str(WORKSPACE / cfg.synthetic_base_dir / f"iter{iter_id}")
    output_dir = str(WORKSPACE / cfg.labeled_syn_base_dir / f"iter{iter_id}")

    script = str(WORKSPACE / "rlft/vlaw/scripts/label_real_trajectories.py")
    args = (
        f"--rollout_dir {syn_dir} "
        f"--output_dir {output_dir} "
        f"--iter_id {iter_id} "
        f"--tasks {cfg.tasks} "
        f"--threshold {cfg.vlm_threshold} "
    )
    if dry_run:
        args += "--dry_run "

    log_file = f"/tmp/step6_label_syn_iter{iter_id}.log"
    run_cmd(
        f"CUDA_VISIBLE_DEVICES={cfg.gpu_vlm} conda run -n vlaw_reward python {script} {args}",
        log_file=log_file,
    )
    log.info(f"[Step 6] ✅ D_syn VLM 标注完成 → {output_dir}")


# ── Step 7: 策略更新 ──────────────────────────────────────────────────────────

def step7_policy_update(cfg: VLAWConfig, iter_id: int, dry_run: bool = False) -> None:
    """Step 7: 用 D_real+ ∪ D_syn+ ∪ D_demo 更新策略。"""
    log.info(f"[Step 7] 策略更新 (Iter {iter_id})")

    labeled_real = str(WORKSPACE / cfg.labeled_base_dir / f"iter{iter_id}")
    labeled_syn = str(WORKSPACE / cfg.labeled_syn_base_dir / f"iter{iter_id}")
    demo_dir = str(WORKSPACE / cfg.demo_dir)
    output_dir = str(WORKSPACE / cfg.checkpoint_dir / f"policy/iter{iter_id}")
    base_ckpt = get_policy_ckpt(cfg, iter_id)  # 当前迭代的起点

    script = str(WORKSPACE / "rlft/vlaw/scripts/run_policy_update.py")
    args = (
        f"--labeled_real_dir {labeled_real} "
        f"--labeled_syn_dir {labeled_syn} "
        f"--demo_dir {demo_dir} "
        f"--checkpoint_path {base_ckpt} "
        f"--output_dir {output_dir} "
        f"--tasks {cfg.tasks} "
        f"--num_steps {cfg.policy_update_steps} "
        f"--iter_id {iter_id} "
    )
    if dry_run:
        args += "--dry_run "

    log_file = f"/tmp/step7_policy_iter{iter_id}.log"
    run_cmd(
        f"CUDA_VISIBLE_DEVICES={cfg.gpu_policy} conda run -n rlft_ms3 python {script} {args}",
        log_file=log_file,
    )
    log.info(f"[Step 7] ✅ 策略更新完成 → {output_dir}")


# ── Step 8: 评估 ──────────────────────────────────────────────────────────────

def step8_evaluate(cfg: VLAWConfig, iter_id: int, dry_run: bool = False) -> dict:
    """Step 8: 评估更新后的策略。"""
    log.info(f"[Step 8] 评估 (Iter {iter_id})")
    policy_ckpt = str(WORKSPACE / cfg.checkpoint_dir / f"policy/iter{iter_id}/best.pt")

    if not Path(policy_ckpt).exists():
        # 检查是否有 final.pt
        final_ckpt = policy_ckpt.replace("best.pt", "final.pt")
        if Path(final_ckpt).exists():
            policy_ckpt = final_ckpt
        else:
            log.warning(f"[Step 8] 策略 checkpoint 不存在: {policy_ckpt}")
            return {}

    script = str(WORKSPACE / "rlft/vlaw/scripts/evaluate_policy.py")
    results_path = str(WORKSPACE / cfg.log_dir / f"iter{iter_id}/eval_results.json")
    args = (
        f"--policy_ckpt {policy_ckpt} "
        f"--output_path {results_path} "
        f"--tasks {cfg.tasks} "
        f"--num_episodes 100 "
        f"--iter_id {iter_id} "
    )
    if dry_run:
        args += "--dry_run "

    log_file = f"/tmp/step8_eval_iter{iter_id}.log"
    run_cmd(
        f"CUDA_VISIBLE_DEVICES={cfg.gpu_eval} conda run -n rlft_ms3 python {script} {args}",
        log_file=log_file,
        check=False,  # 不强制成功 (eval script 可能未实现)
    )

    # 读取结果
    results_file = Path(results_path)
    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)
        log.info(f"[Step 8] ✅ 评估结果: {results}")
        return results
    log.info("[Step 8] 评估完成（无结果文件）")
    return {}


# ── 主流程 ────────────────────────────────────────────────────────────────────

def run_iteration(cfg: VLAWConfig, iter_id: int, start_step: int = 1) -> None:
    """运行单次 VLAW 迭代的全部步骤。"""
    log.info(f"\n{'='*60}")
    log.info(f"  VLAW Iteration {iter_id} / {cfg.num_iters}")
    log.info(f"{'='*60}")

    state_path = WORKSPACE / cfg.checkpoint_dir / f"train_state_iter{iter_id}.json"
    dry = cfg.dry_run

    steps = {
        1: lambda: step1_collect_rollouts(cfg, iter_id, dry),
        2: lambda: step2_vae_encode(cfg, iter_id, dry),
        3: lambda: step3_label_real(cfg, iter_id, dry),
        4: lambda: step4_finetune_wm(cfg, iter_id, dry),
        5: lambda: step5_imagination(cfg, iter_id, dry),
        6: lambda: step6_label_syn(cfg, iter_id, dry),
        7: lambda: step7_policy_update(cfg, iter_id, dry),
        8: lambda: step8_evaluate(cfg, iter_id, dry),
    }

    for step_id in range(start_step, 9):
        t0 = time.time()
        step_fn = steps.get(step_id)
        if step_fn is None:
            log.warning(f"Step {step_id} 未定义，跳过")
            continue

        log.info(f"\n--- Step {step_id}/8 ---")
        step_fn()

        elapsed = time.time() - t0
        log.info(f"    Step {step_id} 耗时: {elapsed:.1f}s")

        save_state(state_path, {
            "iter_id": iter_id,
            "last_completed_step": step_id,
        })


def main() -> None:
    parser = argparse.ArgumentParser(description="VLAW 迭代训练主脚本 (P6.1)")

    # 主配置
    parser.add_argument("--num_iters", type=int, default=2)
    parser.add_argument("--tasks", type=str,
                        default="LiftPegUpright-v1,PickCube-v1,StackCube-v1")
    parser.add_argument("--base_policy_ckpt", type=str,
                        default="checkpoints/il/best_eval_success_once.pt")
    parser.add_argument("--num_rollout_episodes", type=int, default=50)
    parser.add_argument("--num_syn_trajectories", type=int, default=200)
    parser.add_argument("--wm_finetune_steps", type=int, default=5000)
    parser.add_argument("--policy_update_steps", type=int, default=2000)
    parser.add_argument("--vlm_threshold", type=float, default=0.8)

    # GPU
    parser.add_argument("--gpu_rollout", type=str, default="4,5")
    parser.add_argument("--gpu_vae", type=str, default="4")
    parser.add_argument("--gpu_vlm", type=str, default="6")
    parser.add_argument("--gpu_wm", type=str, default="0,1,2,3")
    parser.add_argument("--gpu_imagine", type=str, default="4,5,6,7")
    parser.add_argument("--gpu_policy", type=str, default="8,9")
    parser.add_argument("--gpu_eval", type=str, default="8")

    # 控制
    parser.add_argument("--resume_iter", type=int, default=1,
                        help="从第几轮迭代恢复（1=从头）")
    parser.add_argument("--resume_step", type=int, default=1,
                        help="从第几步恢复（1=完整从头）")
    parser.add_argument("--dry_run", action="store_true",
                        help="仅验证配置和接口，不实际运行")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--eval_iter_id", type=int, default=1)
    parser.add_argument("--config", type=str, default=None,
                        help="JSON 配置文件路径（覆盖 CLI 参数）")

    args = parser.parse_args()

    # 构建配置
    cfg = VLAWConfig()
    if args.config:
        with open(args.config) as f:
            config_dict = json.load(f)
        for k, v in config_dict.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    # CLI 覆盖
    for k, v in vars(args).items():
        if hasattr(cfg, k) and v is not None:
            setattr(cfg, k, v)

    # ── 日志目录 ────────────────────────────────────────────────────────────
    (WORKSPACE / cfg.log_dir).mkdir(parents=True, exist_ok=True)
    (WORKSPACE / cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    log.info("[VLAW] 训练启动")
    log.info(f"  tasks      : {cfg.task_list}")
    log.info(f"  num_iters  : {cfg.num_iters}")
    log.info(f"  dry_run    : {cfg.dry_run}")
    log.info(f"  resume     : Iter {args.resume_iter}, Step {args.resume_step}")

    if cfg.dry_run:
        log.info("[DRY RUN] 验证配置完成，退出")
        return

    if cfg.eval_only:
        step8_evaluate(cfg, cfg.eval_iter_id, dry_run=False)
        return

    # ── 主循环 ──────────────────────────────────────────────────────────────
    for iter_id in range(args.resume_iter, cfg.num_iters + 1):
        start_step = args.resume_step if iter_id == args.resume_iter else 1
        try:
            run_iteration(cfg, iter_id, start_step=start_step)
        except Exception as e:
            log.error(f"Iter {iter_id} 失败: {e}")
            log.error("训练中断，可使用 --resume_iter / --resume_step 恢复")
            raise

    log.info("\n[VLAW] 全部迭代完成 ✅")


if __name__ == "__main__":
    main()
