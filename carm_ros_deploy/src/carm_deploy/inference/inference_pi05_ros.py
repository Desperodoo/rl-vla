#!/usr/bin/env python3
"""Standalone ROS inference loop for LeRobot/OpenPI pi0.5/pi05 policies."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np
import rospy
import torch
from einops import rearrange

carm_deploy_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, carm_deploy_root)
rl_vla_root = os.path.dirname(os.path.dirname(os.path.dirname(carm_deploy_root)))
sys.path.insert(0, rl_vla_root)

from core.env_ros import RealEnvironment
from core.safety_controller import SafetyController
from inference.inference_logger import InferenceLogger
from inference.inference_recorder import InferenceRecorder
from inference.policy_loader_pi05 import LeRobotPi05Policy
from utils.keyboard_intervention import KeyboardInterventionHandler
from utils.trajectory_interpolator import ActionChunkManager, TrajectoryInterpolator


class InferencePi05Node:
    """Dedicated inference node for joint-space pi0.5/pi05 policies."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.desire_inference_freq = float(config.get("desire_inference_freq", 15.0))
        self.control_freq = int(config.get("control_freq", 50))
        self.execution_mode = config.get("execution_mode", "receding_horizon")
        self.max_active_chunks = config.get("max_active_chunks")
        self.crossfade_steps = int(config.get("crossfade_steps", 0))
        self.temporal_factor_k = float(config.get("temporal_factor_k", 0.05))
        self.chunk_time_base = config.get("chunk_time_base", "sys_time")
        self.truncate_at_act_horizon = bool(config.get("truncate_at_act_horizon", True))
        self.check_workspace = bool(config.get("check_workspace", False))
        self.record_inference_enabled = bool(config.get("record_inference", False))
        self.intervention_enabled = bool(config.get("intervention", False))
        self.max_steps = int(config.get("max_steps", 99999))
        self.running = True
        self.episode_started = False
        self.waiting_start = self.record_inference_enabled
        self.episode_paused = self.waiting_start
        self.pending_save = False
        self.step_count = 0
        self.control_step_count = 0
        self.latest_obs: Optional[Dict[str, Any]] = None
        self.lock_tfs = threading.Lock()
        self._shutdown_called = False
        self._last_control_time = None
        self._control_hz_ema = None
        self._last_gripper_log_time = 0.0
        self._last_gripper_value = None

        rospy.loginfo("Initializing RealEnvironment for pi05 inference...")
        self.env = RealEnvironment(config)
        rospy.loginfo("Initializing LeRobot pi05 policy...")
        self.policy = self._create_policy(config)
        self._pred_horizon = getattr(self.policy, "pred_horizon", 16)
        self._act_horizon = int(config.get("act_horizon", self._pred_horizon))
        self._action_dim_full = getattr(self.policy, "action_dim_full", getattr(self.policy, "action_dim", 7))

        self.safety_controller = self._create_safety_controller(config)
        self.logger = self._create_logger(config)
        self._setup_logger_metadata(config)

        self.action_manager = ActionChunkManager(
            temporal_factor_k=self.temporal_factor_k,
            execution_mode=self.execution_mode,
            max_active_chunks=self.max_active_chunks,
            crossfade_steps=self.crossfade_steps,
        )

        self.intervention_handler = None
        self.inference_recorder = None
        if self.intervention_enabled or self.record_inference_enabled:
            self._init_intervention_and_recording(config)

        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.inference_thread.start()

    def _create_policy(self, config: Dict[str, Any]) -> LeRobotPi05Policy:
        pretrain_path = config.get("pretrain", "")
        if not pretrain_path:
            rospy.logerr("No pretrained pi05 policy specified. Use --pretrain.")
            raise SystemExit(1)
        policy = LeRobotPi05Policy(config)
        policy.load_model(pretrain_path)
        return policy

    def _create_safety_controller(self, config: Dict[str, Any]) -> SafetyController:
        safety_config_path = config.get("safety_config", "")
        if safety_config_path and os.path.exists(safety_config_path):
            rospy.loginfo(f"Loading safety config from: {safety_config_path}")
            return SafetyController.from_config(safety_config_path)
        rospy.logwarn("No safety config found, using default safety limits")
        return SafetyController()

    def _create_logger(self, config: Dict[str, Any]) -> InferenceLogger:
        log_dir = config.get("log_dir", "")
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            return InferenceLogger(log_dir=log_dir, save_images=bool(config.get("save_images", False)))
        from utils.paths import ensure_dir, get_inference_logs_dir

        default_log_dir = ensure_dir(get_inference_logs_dir())
        return InferenceLogger(log_dir=default_log_dir, save_images=bool(config.get("save_images", False)))

    def _setup_logger_metadata(self, config: Dict[str, Any]):
        model_config = {
            "path": config.get("pretrain", ""),
            "algorithm": getattr(self.policy, "algorithm", "lerobot_pi05"),
            "action_mode": getattr(self.policy, "action_representation", "joint_absolute_gripper"),
            "state_mode": getattr(self.policy, "state_mode", "joint_only"),
            "obs_horizon": getattr(self.policy, "obs_horizon", 1),
            "pred_horizon": getattr(self.policy, "pred_horizon", 16),
            "action_dim": getattr(self.policy, "action_dim", 7),
            "action_dim_full": getattr(self.policy, "action_dim_full", 7),
            "control_mode": getattr(self.policy, "control_mode", "joint"),
        }
        control_config = {
            "control_freq": self.control_freq,
            "command_mode": "joint",
        }
        execution_config = {
            "mode": self.execution_mode,
            "act_horizon": self._act_horizon,
            "max_active_chunks": self.max_active_chunks,
            "crossfade_steps": self.crossfade_steps,
            "truncate_at_act_horizon": self.truncate_at_act_horizon,
            "temporal_factor_k": self.temporal_factor_k,
            "desire_inference_freq": self.desire_inference_freq,
        }
        safety_config = {
            "config_path": config.get("safety_config", ""),
            "check_workspace": self.check_workspace,
        }
        self.logger.set_metadata(
            model_path=config.get("pretrain", ""),
            model_config=model_config,
            control_config=control_config,
            execution_config=execution_config,
            safety_config=safety_config,
        )

    def _init_intervention_and_recording(self, config: Dict[str, Any]):
        if self.intervention_enabled:
            self.intervention_handler = KeyboardInterventionHandler(
                xyz_scale=config.get("intervention_xyz_scale", 0.005),
                gripper_open=config.get("intervention_gripper_open", 1.0),
                gripper_close=config.get("intervention_gripper_close", 0.0),
                mode=config.get("intervention_mode", "replace"),
            )
            self.intervention_handler.set_record_callback(self._handle_record_action)
            self.intervention_handler.set_quit_callback(self._handle_quit)
            self.intervention_handler.start()

        if self.record_inference_enabled:
            record_dir = config.get("record_dir") or config.get("log_dir", "")
            if not record_dir:
                from utils.paths import get_inference_logs_dir

                record_dir = get_inference_logs_dir()
            self.inference_recorder = InferenceRecorder(
                output_dir=record_dir,
                pred_horizon=self._pred_horizon,
                action_dim=self._action_dim_full,
                image_size=getattr(self.policy, "target_image_size", (224, 224)),
            )
            rospy.loginfo("Multi-episode recording mode enabled; press R to start/stop, Y/N to save/discard.")

    def _handle_quit(self):
        rospy.loginfo("Quit requested via keyboard")
        self.running = False

    def _handle_record_action(self, action: str):
        if action == "toggle":
            if self.pending_save:
                rospy.logwarn("Please confirm save first (Y/N)")
                return
            if self.waiting_start:
                self._start_new_episode()
            else:
                self._stop_current_episode()
        elif action == "confirm":
            if self.pending_save:
                self._confirm_save_episode(save=True)
        elif action == "discard":
            if self.pending_save:
                self._confirm_save_episode(save=False)

    def _start_new_episode(self):
        self.pending_save = False
        with self.lock_tfs:
            self.action_manager.clear()
        self.policy.reset()
        self.waiting_start = False
        self.episode_paused = False
        self.step_count = 0
        if self.inference_recorder is not None:
            self.inference_recorder.start_recording()
        rospy.loginfo("Episode started for pi05 inference.")

    def _stop_current_episode(self):
        self.episode_paused = True
        self.pending_save = True
        if self.inference_recorder is not None:
            self.inference_recorder.stop_recording()
        rospy.loginfo("Episode stopped; waiting for save/discard confirmation.")

    def _confirm_save_episode(self, save: bool):
        if self.inference_recorder is not None:
            if save:
                path = self.inference_recorder.confirm_save()
                if path:
                    rospy.loginfo(f"Saved recorded episode to: {path}")
            else:
                self.inference_recorder.discard()
                rospy.loginfo("Discarded recorded episode.")
        self.pending_save = False
        self.waiting_start = True
        self.episode_paused = True
        try:
            self.env.init_status()
        except Exception as exc:
            rospy.logerr(f"Failed to reinitialize arm: {exc}")

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        target_h, target_w = getattr(self.policy, "target_image_size", (224, 224))
        image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return rearrange(image, "h w c -> c h w")

    def _normalize_images(self, obs: Dict[str, Any]) -> np.ndarray:
        return self._preprocess_image(obs["images"][0])

    def _apply_ee_intervention(self, actions: np.ndarray) -> tuple[np.ndarray, Optional[np.ndarray]]:
        if not self.intervention_enabled or self.intervention_handler is None:
            return actions, None
        intervention = self.intervention_handler.get_intervention()
        if intervention is None:
            return actions, None
        delta, mode, _ = intervention
        updated = actions.copy()
        mask = np.zeros_like(updated, dtype=bool)
        xyz_delta = np.asarray(delta[:3], dtype=np.float32)
        if mode == "replace":
            updated[:, :3] = xyz_delta[None, :]
        else:
            updated[:, :3] += xyz_delta[None, :]
        mask[:, :3] = True
        if getattr(self.intervention_handler, "_current_gripper", None) is not None and updated.shape[1] > 7:
            updated[:, 7] = float(self.intervention_handler._current_gripper)
            mask[:, 7] = True
        return updated, mask

    def _apply_safety(self, actions: np.ndarray) -> tuple[np.ndarray, bool, list[str]]:
        safe_actions = actions.copy()
        warnings: list[str] = []
        clipped_any = False
        current_pose = np.asarray(self.latest_obs["qpos_end"], dtype=np.float32)
        for idx in range(len(safe_actions)):
            candidate = safe_actions[idx].copy()
            clipped_pose, ws_warnings = self.safety_controller.check_workspace(candidate[:7])
            if ws_warnings:
                clipped_any = True
                candidate[:7] = clipped_pose[:7]
                if idx == 0:
                    warnings.extend(ws_warnings)
            gripper_action = np.array([0, 0, 0, 0, 0, 0, candidate[7] if len(candidate) > 7 else current_pose[-1]], dtype=np.float32)
            clipped_gripper, grip_warnings = self.safety_controller.check_joint_limits(gripper_action)
            if grip_warnings:
                clipped_any = True
                if len(candidate) > 7:
                    candidate[7] = clipped_gripper[6]
                if idx == 0:
                    warnings.extend(grip_warnings)
            safe_actions[idx] = candidate
            current_pose = candidate
        return safe_actions, clipped_any, warnings

    def _append_chunk(self, actions: np.ndarray, chunk_base_time: float):
        tf = TrajectoryInterpolator()
        action_interval = 1.0 / self.control_freq
        num_actions_to_add = min(self._act_horizon, len(actions)) if self.truncate_at_act_horizon else len(actions)
        for i in range(num_actions_to_add):
            tf.append(chunk_base_time + i * action_interval, actions[i].tolist())
        with self.lock_tfs:
            self.action_manager.add_trajectory(tf)

    def _inference_loop(self):
        rospy.loginfo("pi05 inference thread started")
        self.policy.reset()
        desire_period = 1.0 / self.desire_inference_freq
        with torch.inference_mode():
            while self.running and not rospy.is_shutdown():
                if self.episode_paused or self.waiting_start:
                    time.sleep(0.1)
                    continue
                self.latest_obs = self.env.get_observation()
                if self.latest_obs is None:
                    time.sleep(0.1)
                    rospy.loginfo_throttle(5.0, "Waiting for observation...")
                    continue
                if not self.episode_started:
                    self.logger.start_episode()
                    self.episode_started = True
                step_start = time.time()
                try:
                    qpos_joint = np.asarray(self.latest_obs["qpos_joint"], dtype=np.float32)
                    qpos_end = np.asarray(self.latest_obs["qpos_end"], dtype=np.float32)
                    state = self.policy.build_state_from_obs(qpos_joint, qpos_end)
                    qpos = torch.from_numpy(state).float().to(self.policy.device)
                    ee_pose = torch.from_numpy(qpos_end[:7]).float().to(self.policy.device)
                    curr_image = torch.from_numpy(self._normalize_images(self.latest_obs)).float().to(self.policy.device)

                    infer_start = time.time()
                    ret = self.policy.predict_action_chunk({"qpos": qpos, "ee_pose": ee_pose, "image": curr_image, "task": self.config.get("task", "pick and place")})
                    inference_time = time.time() - infer_start
                    all_actions = ret["a_hat"].squeeze(0).detach().cpu().numpy()
                    action_model = all_actions.copy()

                    all_actions, intervention_mask = self._apply_ee_intervention(all_actions)
                    all_actions, safety_clipped, safety_events = self._apply_safety(all_actions)
                    action_intervened = all_actions.copy()

                    if self.record_inference_enabled and self.inference_recorder is not None and self.inference_recorder.is_recording:
                        self.inference_recorder.record_step(
                            obs=self.latest_obs,
                            action_model=action_model,
                            action_intervened=action_intervened,
                            intervention_mask=intervention_mask,
                            timestamp=time.time(),
                        )

                    chunk_base_time = self.latest_obs.get("stamp", time.time()) if self.chunk_time_base == "obs_stamp" else time.time()
                    self._append_chunk(all_actions, chunk_base_time)
                    self.logger.log_step(
                        timestamp=time.time(),
                        obs=self.latest_obs,
                        raw_action=action_model[0],
                        executed_action=all_actions[0],
                        inference_time=inference_time,
                        safety_clipped=safety_clipped,
                        safety_warnings=safety_events if safety_events else None,
                    )
                    self.step_count += 1
                    rospy.loginfo_throttle(
                        5.0,
                        f"pi05 step {self.step_count}, inference={inference_time:.4f}s, actions={all_actions.shape}",
                    )
                    if self.step_count >= self.max_steps:
                        rospy.logwarn(f"Reached max_steps ({self.max_steps}), auto-stopping episode...")
                        self._stop_current_episode()
                except Exception as exc:
                    import traceback

                    rospy.logerr(f"Error in pi05 inference: {exc}")
                    rospy.logerr(traceback.format_exc())
                wait_tm = desire_period - (time.time() - step_start)
                if wait_tm > 0:
                    time.sleep(wait_tm)

    def control_loop(self):
        rospy.loginfo("pi05 control loop started")
        control_period = 1.0 / self.control_freq
        while self.running and not rospy.is_shutdown():
            if self.episode_paused or self.waiting_start:
                time.sleep(0.05)
                continue
            query_time = time.time()
            with self.lock_tfs:
                action = self.action_manager.get_fused_action(query_time)
            if action is None:
                time.sleep(0.02)
                continue

            if self._last_control_time is not None:
                dt = query_time - self._last_control_time
                if dt > 0:
                    inst_hz = 1.0 / dt
                    self._control_hz_ema = inst_hz if self._control_hz_ema is None else 0.2 * inst_hz + 0.8 * self._control_hz_ema
            self._last_control_time = query_time

            grip_val = float(action[-1]) if len(action) > 0 else None
            now = time.time()
            if grip_val is not None and (now - self._last_gripper_log_time) >= 5.0:
                delta = None if self._last_gripper_value is None else (grip_val - self._last_gripper_value)
                hz_str = f"{self._control_hz_ema:.1f}Hz" if self._control_hz_ema is not None else "n/a"
                rospy.loginfo(f"pi05 gripper cmd: {grip_val:.4f}, delta: {delta if delta is not None else 'n/a'}, control_hz: {hz_str}")
                self._last_gripper_value = grip_val
                self._last_gripper_log_time = now

            self.env.end_control_nostep(action)
            self.control_step_count += 1
            time.sleep(control_period)

    def shutdown(self):
        if self._shutdown_called:
            return
        self._shutdown_called = True
        rospy.loginfo("Shutting down InferencePi05Node...")
        self.running = False
        if self.inference_thread.is_alive():
            self.inference_thread.join(timeout=2.0)
        if self.intervention_handler is not None:
            self.intervention_handler.stop()
        if self.inference_recorder is not None:
            if self.inference_recorder.is_recording:
                self.inference_recorder.stop_recording()
            if self.inference_recorder.is_pending_save:
                rospy.logwarn("Discarding unsaved recording data on shutdown")
                self.inference_recorder.discard()
        if self.episode_started:
            log_path = self.logger.end_episode()
            if log_path:
                rospy.loginfo(f"Inference log saved to: {log_path}")
        self.env.shutdown()
        rospy.loginfo("InferencePi05Node shutdown complete")


def parse_args():
    parser = argparse.ArgumentParser(description="CARM pi0.5/pi05 Policy Inference (ROS)")
    parser.add_argument("--robot_ip", type=str, default="10.42.0.101", help="Robot IP address")
    parser.add_argument("--robot_mode", type=int, default=4, help="Control mode (0=IDLE, 1=POSITION, 2=MIT, 3=DRAG, 4=PF)")
    parser.add_argument("--robot_tau", type=float, default=10, help="Gripper torque")
    parser.add_argument("--arm_init_pose", type=float, nargs=7, default=[0.2475, 0.0014, 0.3251, 0.9996, -0.0034, 0.0255, -0.0074], help="Initial end effector pose [x,y,z,qx,qy,qz,qw]")
    parser.add_argument("--arm_init_gripper", type=float, default=0.078, help="Initial gripper position")
    parser.add_argument("--camera_topics", type=str, default="/camera/color/image_raw", help="Camera topic(s), comma separated")
    parser.add_argument("--sync_slop", type=float, default=0.02, help="Image sync tolerance in seconds")
    parser.add_argument("--pretrain", type=str, default="", help="Path to LeRobot/OpenPI pretrained policy dir")
    parser.add_argument("--dataset_root", type=str, default="/mnt/disk_2/wjz/runs/pi05_full_export/train", help="LeRobot dataset root used to load metadata for policy construction")
    parser.add_argument("--repo_id", type=str, default="carm/pi05_local", help="LeRobot dataset repo_id used with dataset_root")
    parser.add_argument("--peft_adapter_path", type=str, default="", help="Optional PEFT adapter path")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device")
    parser.add_argument("--state_mode", type=str, default="joint_only", choices=["joint_only", "ee_only", "both"], help="State construction mode")
    parser.add_argument("--control_mode", type=str, default="joint", choices=["joint"], help="Control mode for pi05 runtime")
    parser.add_argument("--action_representation", type=str, default="joint_absolute_gripper", help="Human-readable action contract label")
    parser.add_argument("--tokenizer_path_override", type=str, default="/home/wjz/.cache/huggingface/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c", help="Local tokenizer path override")
    parser.add_argument("--task", type=str, default="pick and place", help="Task text passed to the PI05 tokenizer/processor")
    parser.add_argument("--desire_inference_freq", type=float, default=15, help="Desired inference frequency")
    parser.add_argument("--temporal_factor_k", type=float, default=0.05, help="Temporal factor for action fusion")
    parser.add_argument("--execution_mode", type=str, default="receding_horizon", choices=["temporal_ensemble", "receding_horizon"], help="Action chunk execution mode")
    parser.add_argument("--max_active_chunks", type=int, default=None, help="Max active chunks in manager")
    parser.add_argument("--crossfade_steps", type=int, default=0, help="Crossfade smoothing steps when switching chunks")
    parser.add_argument("--truncate_at_act_horizon", action="store_true", default=True, help="Truncate action chunk at act_horizon")
    parser.add_argument("--act_horizon", type=int, default=8, help="Action horizon for chunk truncation")
    parser.add_argument("--chunk_time_base", type=str, default="sys_time", choices=["sys_time", "obs_stamp"], help="Chunk base time")
    parser.add_argument("--control_freq", type=int, default=50, help="Control loop frequency in Hz")
    parser.add_argument("--safety_config", type=str, default="", help="Path to safety config JSON file")
    parser.add_argument("--log_dir", type=str, default="", help="Directory to save inference logs")
    parser.add_argument("--save_images", action="store_true", help="Save images in inference log")
    parser.add_argument("--vis", action="store_true", default=True, help="Visualize images in OpenCV window")
    parser.add_argument("--record_inference", action="store_true", help="Enable inference data recording")
    parser.add_argument("--intervention", action="store_true", help="Enable keyboard intervention during inference")
    parser.add_argument("--intervention_mode", type=str, default="replace", choices=["replace", "additive"], help="Intervention mode")
    parser.add_argument("--intervention_xyz_scale", type=float, default=0.01, help="XYZ movement scale per keypress in meters")
    parser.add_argument("--intervention_gripper_open", type=float, default=1.0, help="Gripper open value for intervention")
    parser.add_argument("--intervention_gripper_close", type=float, default=0.0, help="Gripper close value for intervention")
    parser.add_argument("--record_dir", type=str, default="", help="Directory to save recorded inference data")
    parser.add_argument("--max_steps", type=int, default=99999, help="Maximum steps per episode")
    return parser.parse_args(args=rospy.myargv()[1:])


def main():
    rospy.init_node("carm_inference_pi05", anonymous=True)
    args = parse_args()
    config = vars(args)

    for key in [
        "robot_ip", "robot_mode", "robot_tau", "arm_init_pose", "arm_init_gripper",
        "camera_topics", "sync_slop", "pretrain", "dataset_root", "repo_id", "peft_adapter_path", "device",
        "state_mode", "control_mode", "action_representation", "tokenizer_path_override", "task",
        "desire_inference_freq", "temporal_factor_k", "execution_mode", "max_active_chunks",
        "crossfade_steps", "truncate_at_act_horizon", "act_horizon", "chunk_time_base",
        "control_freq", "safety_config", "log_dir", "save_images", "vis",
        "record_inference", "intervention", "intervention_mode", "intervention_xyz_scale",
        "intervention_gripper_open", "intervention_gripper_close", "record_dir", "max_steps",
    ]:
        if rospy.has_param(f"~{key}"):
            config[key] = rospy.get_param(f"~{key}")

    if isinstance(config["camera_topics"], str):
        config["camera_topics"] = config["camera_topics"].split(",")
    if isinstance(config.get("arm_init_pose"), str):
        config["arm_init_pose"] = [float(x) for x in config["arm_init_pose"].split()]
    if isinstance(config.get("arm_init_gripper"), str):
        config["arm_init_gripper"] = float(config["arm_init_gripper"])

    if not config.get("safety_config"):
        default_safety = os.path.join(carm_deploy_root, "safety_config.json")
        config["safety_config"] = default_safety
    config["safety_config"] = os.path.expandvars(os.path.expanduser(config["safety_config"]))
    if not os.path.exists(config["safety_config"]):
        rospy.logfatal("Safety config missing: %s", config["safety_config"])
        raise SystemExit(1)

    rospy.loginfo("=" * 60)
    rospy.loginfo("CARM pi0.5/pi05 Policy Inference Node")
    rospy.loginfo("=" * 60)
    rospy.loginfo(f"Robot IP: {config['robot_ip']}")
    rospy.loginfo(f"Camera topics: {config['camera_topics']}")
    rospy.loginfo(f"Pretrain: {config['pretrain']}")
    rospy.loginfo(f"Control mode: {config['control_mode']}")
    rospy.loginfo(f"Action representation: {config['action_representation']}")
    rospy.loginfo("=" * 60)

    node = InferencePi05Node(config)
    shutdown_in_progress = False

    def signal_handler(signum, frame):
        nonlocal shutdown_in_progress
        if shutdown_in_progress:
            rospy.logwarn("Force exit requested, exiting immediately...")
            sys.exit(1)
        shutdown_in_progress = True
        rospy.loginfo("\nReceived shutdown signal, cleaning up...")
        node.shutdown()
        rospy.signal_shutdown("User interrupted")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    rospy.on_shutdown(node.shutdown)

    try:
        node.control_loop()
    except KeyboardInterrupt:
        rospy.loginfo("Interrupted by user")
    except Exception as exc:
        rospy.logerr(f"Unexpected error: {exc}")
    finally:
        if not shutdown_in_progress:
            node.shutdown()


if __name__ == "__main__":
    main()
