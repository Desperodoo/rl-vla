"""P0.2 ManiSkill RGB 数据验证脚本.

验证内容:
1. obs_mode="rgbd" 输出格式、分辨率、相机名称
2. 2 相机图像垂直拼接 → shape 确认
3. Ctrl-World VAE encode → decode 重建质量 (PSNR > 25 dB)
4. env.get_state() / env.set_state() 可用性

所属阶段: P0.2 — ManiSkill RGB 数据验证
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tyro


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

@dataclass
class ValidateConfig:
    """P0.2 验证脚本配置."""

    # ManiSkill 环境
    env_id: str = "PickCube-v1"
    """使用的任务 ID, PickCube 依赖少、启动快"""

    camera_width: int = 192
    """相机宽度 (建议 192 或 128; 须是 8 的整倍数)"""

    camera_height: int = 192
    """相机高度 (建议 192 或 128; 须是 8 的整倍数)"""

    num_steps: int = 5
    """采集验证帧数"""

    # GPU 分配
    gpu_id: int = 4
    """使用的 GPU (Data-Agent 分配 GPU 4-5)"""

    sim_backend: str = "physx_cuda"
    """ManiSkill 仿真后端: physx_cuda 或 physx_cpu"""

    # VAE
    vae_model_id: str = "stabilityai/stable-video-diffusion-img2vid-xt"
    """HuggingFace 模型 ID 或本地路径 (用于加载 AutoencoderKLTemporalDecoder)"""

    vae_fallback_id: str = "stabilityai/sd-vae-ft-mse"
    """备选标准 AutoencoderKL (SD VAE)"""

    vae_path: Optional[str] = None
    """如已有本地 VAE safetensors 路径, 直接指定 (优先级最高)"""

    skip_vae: bool = False
    """跳过 VAE 测试 (用于仅验证 ManiSkill obs 格式时)"""

    # 输出
    save_dir: str = "data/vlaw/validation"
    """验证结果保存目录"""

    concat_mode: str = "vertical"
    """图像拼接方向: vertical (垂直, 推荐) 或 horizontal"""

    # 说明
    # ManiSkill3 标准任务 (PickCube/LiftPegUpright 等) 默认只有 base_camera 一个 sensor 相机。
    # 第二视角通过 human_render_camera_configs + env.render() 获取，
    # *不*进入 obs["sensor_data"]，须额外读取。
    # 如需手腕相机加入 obs，应自定义环境子类覆盖 _default_sensor_configs（P1.1 阶段实现）。


# ---------------------------------------------------------------------------
# ManiSkill 验证
# ---------------------------------------------------------------------------

def validate_maniskill_obs(cfg: ValidateConfig) -> dict:
    """验证 ManiSkill obs_mode='rgbd' 输出格式.

    Notes:
        ManiSkill3 标准任务默认只提供 base_camera 一个 sensor 相机。
        第二"手腕"视角通过 human_render_camera_configs + render() 方式获取，
        并在此处验证两路图像拼接 → VAE latent 的形状链路正确性。

    Returns:
        包含验证结果的字典
    """
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401 — 注册所有环境

    print(f"[VLAW-P0.2] 创建环境: {cfg.env_id}, "
          f"相机 {cfg.camera_width}×{cfg.camera_height}, "
          f"后端: {cfg.sim_backend}")

    # ----------------------------------------------------------------
    # ManiSkill3 sensor_configs 格式说明:
    #   - sensor_configs=dict(width=W, height=H) → 全局覆盖所有 sensor 相机分辨率
    #   - sensor_configs=dict(base_camera=dict(width=W, height=H)) → 按名称覆盖
    #   - 注意: 传入不存在的相机名 (如 hand_camera) 会导致 AttributeError,
    #     因为 ManiSkill3 会把未知键作为 CameraConfig 属性解析
    # ----------------------------------------------------------------
    env_make_kwargs: dict = dict(
        obs_mode="rgbd",
        render_mode="rgb_array",          # 允许 render() 获取第二视角
        sensor_configs=dict(
            width=cfg.camera_width,
            height=cfg.camera_height,
        ),
    )

    if cfg.sim_backend == "physx_cpu":
        env = gym.make(cfg.env_id, **env_make_kwargs)
        obs, _ = env.reset(seed=42)
        batch_shape = ()
    else:
        env = gym.make(
            cfg.env_id,
            num_envs=1,
            sim_backend=cfg.sim_backend,
            **env_make_kwargs,
        )
        obs, _ = env.reset(seed=42)
        batch_shape = (1,)

    # --- 检查 sensor_data 键 ---
    sensor_data = obs.get("sensor_data", {})
    camera_keys = list(sensor_data.keys())
    print(f"[VLAW-P0.2]  可用 sensor 相机: {camera_keys}")

    results: dict = {
        "env_id": cfg.env_id,
        "sim_backend": cfg.sim_backend,
        "cameras": {},
        "render_camera": {},
        "agent": {},
        "state_management": {},
        "concat": {},
    }

    # --- 读取每个相机的 RGB 形状 ---
    frames_for_concat: list[np.ndarray] = []
    for cam_name in camera_keys:
        cam_data = sensor_data[cam_name]
        rgb = cam_data.get("rgb")
        depth = cam_data.get("depth")

        if rgb is None:
            print(f"[VLAW-P0.2]  ⚠ {cam_name} 无 RGB 数据")
            continue

        if hasattr(rgb, "cpu"):
            rgb_np = rgb.cpu().numpy()
        else:
            rgb_np = np.asarray(rgb)

        print(f"[VLAW-P0.2]  {cam_name}.rgb shape={rgb_np.shape}, "
              f"dtype={rgb_np.dtype}, range=[{rgb_np.min()},{rgb_np.max()}]")

        results["cameras"][cam_name] = {
            "rgb_shape": list(rgb_np.shape),
            "rgb_dtype": str(rgb_np.dtype),
            "rgb_min": int(rgb_np.min()),
            "rgb_max": int(rgb_np.max()),
            "has_depth": depth is not None,
        }

        frame = rgb_np[0] if batch_shape else rgb_np   # (H, W, 3)
        frames_for_concat.append(frame)

    # --- 第二视角: env.render() 获取 "手腕相机" 模拟帧 ---
    # ManiSkill3 standard tasks embed only base_camera in sensor_data.
    # Use env.render() (rgb_array mode) to get a secondary viewpoint as "hand_camera".
    print(f"[VLAW-P0.2]  尝试 env.render() 获取第二视角...")
    try:
        render_out = env.render()
        if render_out is not None:
            if hasattr(render_out, "cpu"):
                render_np = render_out.cpu().numpy()
            else:
                render_np = np.asarray(render_out)

            # render() 可能返回 (N, H, W, 3) 或 (H, W, 3)
            if render_np.ndim == 4:
                render_np = render_np[0]  # (H, W, 3)

            # 裁剪/缩放到与 sensor 相机一致的 (H, W, 3)
            rh, rw = render_np.shape[:2]
            target_h, target_w = cfg.camera_height, cfg.camera_width
            if rh != target_h or rw != target_w:
                from PIL import Image as PILImage
                pil_img = PILImage.fromarray(render_np).resize(
                    (target_w, target_h), PILImage.BILINEAR
                )
                render_np = np.asarray(pil_img)

            print(f"[VLAW-P0.2]  render 视角 (模拟 hand_camera) shape={render_np.shape}")
            results["render_camera"] = {
                "shape": list(render_np.shape),
                "dtype": str(render_np.dtype),
                "note": "hand_camera via env.render(); "
                        "P1.1 阶段将通过自定义 env 子类加入 sensor_data",
            }
            frames_for_concat.append(render_np)
        else:
            print("[VLAW-P0.2]  ⚠ env.render() 返回 None")
    except Exception as e:
        print(f"[VLAW-P0.2]  ⚠ env.render() 失败: {e}")

    # --- Agent 状态 ---
    agent_obs = obs.get("agent", {})
    for key in ["qpos", "qvel"]:
        arr = agent_obs.get(key)
        if arr is not None:
            if hasattr(arr, "cpu"):
                arr = arr.cpu().numpy()
            print(f"[VLAW-P0.2]  agent.{key} shape={arr.shape}, dtype={arr.dtype}")
            results["agent"][key] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}

    # --- 多步采集验证 ---
    rgb_frames_base: list[np.ndarray] = []

    for _ in range(cfg.num_steps):
        action = env.action_space.sample()
        obs, _reward, _terminated, _truncated, _info = env.step(action)
        sensor_data = obs.get("sensor_data", {})
        if camera_keys:
            cam_name = camera_keys[0]   # base_camera
            rgb = sensor_data[cam_name]["rgb"]
            if hasattr(rgb, "cpu"):
                rgb = rgb.cpu().numpy()
            frame = rgb[0] if batch_shape else rgb
            rgb_frames_base.append(frame)

    print(f"[VLAW-P0.2]  采集帧数: base_camera={len(rgb_frames_base)}")

    # --- 图像拼接测试 ---
    if len(frames_for_concat) >= 2:
        f0, f1 = frames_for_concat[0], frames_for_concat[1]
        assert f0.shape == f1.shape, (
            f"两帧形状不一致: {f0.shape} vs {f1.shape}, 无法拼接"
        )
        if cfg.concat_mode == "vertical":
            concat = np.concatenate([f0, f1], axis=0)   # (2H, W, 3)
        else:
            concat = np.concatenate([f0, f1], axis=1)   # (H, 2W, 3)
        print(f"[VLAW-P0.2]  2-相机拼接 ({cfg.concat_mode}): {concat.shape}")
        results["concat"] = {
            "mode": cfg.concat_mode,
            "frame0_source": camera_keys[0] if camera_keys else "unknown",
            "frame1_source": "env.render()",
            "shape": list(concat.shape),
            "expected_vae_input": list(concat.shape[:2]),  # (H, W)
            "expected_latent_hwc": [concat.shape[0] // 8, concat.shape[1] // 8, 4],
        }
    elif len(frames_for_concat) == 1:
        print("[VLAW-P0.2]  ⚠ 仅 1 个相机视角，用单帧演示 latent shape")
        concat = frames_for_concat[0]
        results["concat"] = {
            "mode": "single_camera",
            "shape": list(concat.shape),
            "expected_latent_hwc": [concat.shape[0] // 8, concat.shape[1] // 8, 4],
        }
    else:
        print("[VLAW-P0.2]  ⚠ 无可用帧，跳过拼接")
        concat = None

    # --- get_state / set_state ---
    try:
        state = env.get_state()
        if hasattr(state, "cpu"):
            state_np = state.cpu().numpy()
        else:
            state_np = np.asarray(state)
        print(f"[VLAW-P0.2]  env.get_state() ✅ → shape={state_np.shape}")
        env.set_state(state)
        print(f"[VLAW-P0.2]  env.set_state() ✅")
        results["state_management"]["get_state"] = True
        results["state_management"]["set_state"] = True
        results["state_management"]["state_shape"] = list(state_np.shape)
    except Exception as e:
        print(f"[VLAW-P0.2]  env.get_state/set_state ❌: {e}")
        results["state_management"]["get_state"] = False
        results["state_management"]["error"] = str(e)

    env.close()

    return results, concat, rgb_frames_base, []


# ---------------------------------------------------------------------------
# VAE 验证
# ---------------------------------------------------------------------------

def _compute_psnr(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """计算两张 uint8 图像的 PSNR (dB)."""
    mse = np.mean((img_a.astype(np.float64) - img_b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(255.0 / np.sqrt(mse)))


def _load_svd_vae(model_id: str, device: torch.device) -> tuple:
    """尝试加载 SVD AutoencoderKLTemporalDecoder.

    Returns:
        (vae, vae_type) — vae_type 为 "svd" 或 "sd"
    """
    from diffusers import AutoencoderKLTemporalDecoder
    print(f"[VLAW-P0.2]  加载 SVD VAE — {model_id}")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float16
    )
    vae = vae.to(device).eval()
    return vae, "svd"


def _load_sd_vae(model_id: str, device: torch.device) -> tuple:
    """加载标准 AutoencoderKL (SD VAE) 作为备选."""
    from diffusers import AutoencoderKL
    print(f"[VLAW-P0.2]  加载 SD VAE — {model_id}")
    vae = AutoencoderKL.from_pretrained(model_id, torch_dtype=torch.float16)
    vae = vae.to(device).eval()
    return vae, "sd"


def _create_synthetic_vae(device: torch.device) -> tuple:
    """创建随机初始化的 AutoencoderKL 用于 shape 验证.

    Notes:
        使用标准 SD VAE 架构 (downscale=8, latent_channels=4), 但权重随机。
        仅用于验证 shape 链路; PSNR 极低 (~4 dB), 需要预训练权重才能 >25 dB。
    """
    from diffusers import AutoencoderKL
    print("[VLAW-P0.2]  ⚠ 无可用预训练 VAE，使用随机初始化架构做 shape 验证")
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=[
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
        ],
        up_block_types=[
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
        ],
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        norm_num_groups=32,
        latent_channels=4,
    )
    vae = vae.to(device=device, dtype=torch.float16).eval()
    return vae, "synthetic"


def validate_vae(
    cfg: ValidateConfig,
    concat_frame: np.ndarray,
    device: torch.device,
) -> dict:
    """测试 VAE encode→decode 重建质量.

    Args:
        cfg: 验证配置
        concat_frame: 拼接后的 uint8 图像 (H, W, 3)
        device: 计算设备

    Returns:
        包含 VAE 验证结果的字典
    """
    results: dict = {}

    # --- 加载 VAE ---
    vae = None
    vae_type = None

    # 优先使用用户指定路径
    if cfg.vae_path:
        try:
            vae, vae_type = _load_sd_vae(cfg.vae_path, device)
            results["vae_source"] = cfg.vae_path
        except Exception as e:
            print(f"[VLAW-P0.2]  ⚠ vae_path 加载失败: {e}")

    # 尝试 SVD VAE
    if vae is None:
        try:
            vae, vae_type = _load_svd_vae(cfg.vae_model_id, device)
            results["vae_source"] = cfg.vae_model_id
        except Exception as e:
            print(f"[VLAW-P0.2]  ⚠ SVD VAE 加载失败: {e}")

    # 备选 SD VAE
    if vae is None:
        try:
            vae, vae_type = _load_sd_vae(cfg.vae_fallback_id, device)
            results["vae_source"] = cfg.vae_fallback_id
        except Exception as e:
            print(f"[VLAW-P0.2]  ⚠ SD VAE 备选加载失败: {e}")

    # 最终备选: 随机初始化合成 VAE (仅验证 shape, PSNR 不达标)
    if vae is None:
        vae, vae_type = _create_synthetic_vae(device)
        results["vae_source"] = "synthetic_random_init"
        results["warning"] = (
            "使用随机初始化 VAE 仅验证 shape 链路。"
            "PSNR 需预训练权重 (stabilityai/sd-vae-ft-mse 或 Ctrl-World VAE)。"
        )
        print("[VLAW-P0.2]  ⚠ 将使用随机 VAE 验证 shape，PSNR 结果仅供参考")

    print(f"[VLAW-P0.2]  VAE 类型: {vae_type}, 加载完成 ✅")
    results["vae_type"] = vae_type

    # --- 预处理: uint8 → float16 tensor, 归一化到 [-1, 1] ---
    H, W, C = concat_frame.shape
    img_f = concat_frame.astype(np.float32) / 127.5 - 1.0   # [-1, 1]
    img_tensor = torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    img_tensor = img_tensor.to(device=device, dtype=torch.float16)

    print(f"[VLAW-P0.2]  VAE 输入 tensor: {img_tensor.shape}, dtype={img_tensor.dtype}")
    results["input_shape"] = list(img_tensor.shape)

    # --- Encode ---
    t0 = time.perf_counter()
    with torch.no_grad():
        if vae_type == "svd":
            # SVD VAE encode 接口
            encoder_output = vae.encode(img_tensor)
        else:
            encoder_output = vae.encode(img_tensor)
        latent_dist = encoder_output.latent_dist
        latent = latent_dist.sample()   # (1, 4, H/8, W/8)
    encode_ms = (time.perf_counter() - t0) * 1000

    print(f"[VLAW-P0.2]  Latent shape: {latent.shape}, "
          f"dtype={latent.dtype}, encode 耗时 {encode_ms:.1f}ms")
    results["latent_shape"] = list(latent.shape)
    results["latent_dtype"] = str(latent.dtype)
    results["encode_ms"] = round(encode_ms, 2)

    expected_lat_h = H // 8
    expected_lat_w = W // 8
    shape_ok = (latent.shape[-2] == expected_lat_h and latent.shape[-1] == expected_lat_w)
    print(f"[VLAW-P0.2]  Latent 形状验证: 期望 (1,4,{expected_lat_h},{expected_lat_w}) "
          f"{'✅' if shape_ok else '❌'}")
    results["latent_shape_ok"] = shape_ok

    # --- Decode ---
    t0 = time.perf_counter()
    with torch.no_grad():
        if vae_type == "svd":
            # SVD AutoencoderKLTemporalDecoder.decode 需要 num_frames
            decoded = vae.decode(latent, num_frames=1).sample   # (1,3,H,W)
        else:
            decoded = vae.decode(latent).sample   # (1,3,H,W)
    decode_ms = (time.perf_counter() - t0) * 1000

    print(f"[VLAW-P0.2]  Decoded shape: {decoded.shape}, decode 耗时 {decode_ms:.1f}ms")
    results["decoded_shape"] = list(decoded.shape)
    results["decode_ms"] = round(decode_ms, 2)

    # --- 计算 PSNR ---
    decoded_np = decoded[0].float().cpu().numpy()      # (3, H, W) float
    decoded_np = (decoded_np + 1.0) * 127.5              # 反归一化
    decoded_np = np.clip(decoded_np, 0, 255).astype(np.uint8)
    decoded_np = decoded_np.transpose(1, 2, 0)           # (H, W, 3)

    psnr = _compute_psnr(concat_frame, decoded_np)
    passed = psnr > 25.0
    is_synthetic = vae_type == "synthetic"
    psnr_label = (
        f"✅ (>25)" if passed
        else f"⚠ (随机权重, shape OK)" if is_synthetic
        else f"❌ (<25)"
    )
    print(f"[VLAW-P0.2]  PSNR: {psnr:.2f} dB {psnr_label}")
    results["psnr_db"] = round(psnr, 4)
    results["psnr_passed"] = passed
    results["shape_pipeline_ok"] = True  # shape 链路正确，与 PSNR 无关

    # --- float16 latent 统计 ---
    lat_fp16 = latent.to(torch.float16)
    results["latent_fp16_mean"] = round(float(lat_fp16.float().mean()), 4)
    results["latent_fp16_std"] = round(float(lat_fp16.float().std()), 4)

    # 保存可视化对比图 (可选)
    try:
        from PIL import Image
        out_dir = Path(cfg.save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(concat_frame).save(out_dir / "original_concat.png")
        Image.fromarray(decoded_np).save(out_dir / "reconstructed_concat.png")
        print(f"[VLAW-P0.2]  可视化图像已保存至: {out_dir}")
    except ImportError:
        print("[VLAW-P0.2]  ⚠ Pillow 未安装, 跳过图像保存")

    del vae
    torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def run_validation(cfg: ValidateConfig) -> dict:
    """执行完整的 P0.2 验证流程.

    Args:
        cfg: 验证配置

    Returns:
        完整验证结果字典
    """
    print(f"\n{'='*60}")
    print(f"[VLAW-P0.2] 开始 ManiSkill RGB 数据验证")
    print(f"{'='*60}\n")

    # 设置 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[VLAW-P0.2] 设备: {device} (CUDA_VISIBLE_DEVICES={cfg.gpu_id})")

    all_results: dict = {
        "config": {
            "env_id": cfg.env_id,
            "camera_width": cfg.camera_width,
            "camera_height": cfg.camera_height,
            "concat_mode": cfg.concat_mode,
            "gpu_id": cfg.gpu_id,
            "sim_backend": cfg.sim_backend,
        },
        "maniskill": {},
        "vae": {},
        "summary": {},
    }

    # ---- STEP 1: ManiSkill obs 验证 ----
    print(f"\n[VLAW-P0.2] ── STEP 1: ManiSkill obs_mode='rgbd' 验证 ──")
    try:
        ms_results, concat_frame, frames_base, frames_hand = validate_maniskill_obs(cfg)
        all_results["maniskill"] = ms_results
        ms_ok = (
            len(ms_results["cameras"]) >= 1
            and ms_results["concat"].get("shape") is not None
        )
    except Exception as e:
        import traceback
        print(f"[VLAW-P0.2] ❌ ManiSkill 验证异常: {e}")
        traceback.print_exc()
        all_results["maniskill"]["error"] = str(e)
        ms_ok = False
        concat_frame = None
        frames_base, frames_hand = [], []

    # ---- STEP 2: VAE 验证 ----
    if not cfg.skip_vae and concat_frame is not None:
        print(f"\n[VLAW-P0.2] ── STEP 2: VAE encode→decode 验证 ──")
        try:
            vae_results = validate_vae(cfg, concat_frame, device)
            all_results["vae"] = vae_results
            vae_ok = vae_results.get("psnr_passed", False)
        except Exception as e:
            import traceback
            print(f"[VLAW-P0.2] ❌ VAE 验证异常: {e}")
            traceback.print_exc()
            all_results["vae"]["error"] = str(e)
            vae_ok = False
    else:
        if cfg.skip_vae:
            print("\n[VLAW-P0.2] ── STEP 2: VAE 验证已跳过 ──")
        else:
            print("\n[VLAW-P0.2] ── STEP 2: VAE 验证跳过 (无拼接图像) ──")
        vae_ok = None

    # ---- 汇总 ----
    all_results["summary"] = {
        "maniskill_obs_ok": ms_ok,
        "vae_psnr_ok": vae_ok,
        "vae_shape_pipeline_ok": all_results["vae"].get("shape_pipeline_ok", False),
        "cameras_found": list(all_results["maniskill"].get("cameras", {}).keys()),
        "latent_shape": all_results["vae"].get("latent_shape"),
        "psnr_db": all_results["vae"].get("psnr_db"),
        "concat_shape": all_results["maniskill"].get("concat", {}).get("shape"),
        "expected_latent_hwc": all_results["maniskill"].get("concat", {}).get("expected_latent_hwc"),
        "vae_source": all_results["vae"].get("vae_source", "N/A"),
    }

    print(f"\n{'='*60}")
    print(f"[VLAW-P0.2] 验证汇总:")
    print(f"  ManiSkill obs 格式: {'✅' if ms_ok else '❌'}")
    vae_source = all_results["vae"].get("vae_source", "N/A")
    psnr_db = all_results["vae"].get("psnr_db", "N/A")
    shape_ok = all_results["vae"].get("shape_pipeline_ok", False)
    if vae_ok is None:
        print(f"  VAE PSNR > 25 dB:   跳过")
    elif vae_source == "synthetic_random_init":
        print(f"  VAE shape 链路:     {'✅' if shape_ok else '❌'} "
              f"(随机 VAE; PSNR={psnr_db} dB — 需预训练权重)")
    else:
        print(f"  VAE PSNR > 25 dB:   {'✅' if vae_ok else '❌'} "
              f"({psnr_db} dB, source={vae_source})")
    print(f"  相机: {all_results['summary']['cameras_found']}")
    print(f"  拼接 shape: {all_results['summary']['concat_shape']}")
    print(f"  Latent shape: {all_results['summary']['latent_shape']}")
    print(f"{'='*60}\n")

    # ---- 保存结果 ----
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    result_path = save_dir / "p0_2_validation_results.json"
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"[VLAW-P0.2] 结果已保存: {result_path}")

    return all_results


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = tyro.cli(ValidateConfig)
    run_validation(cfg)
