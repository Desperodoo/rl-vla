#!/usr/bin/env python3
"""
InferenceNode — the core real-robot inference controller.

Extracted from the monolithic inference_ros.py.  Responsibilities:
  - Wire together env, policy, safety, action processor, logger, recorder
  - Inference thread: observe → predict → process → chunk
  - Control loop: query fused action → send to robot
  - Episode lifecycle (multi-episode recording)
"""

import os
import threading
import time
import numpy as np
import torch
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple

import rospy

from rlft.utils.pose_utils import (
    pose_to_transform_matrix,
    apply_relative_transform,
)

from inference.policy_loader import RealPolicy
from inference.inference_logger import InferenceLogger
from inference.inference_recorder import InferenceRecorder
from inference.action_processor import ActionProcessor, SafetyResult
from inference.config import InferenceConfig

from core.env_ros import RealEnvironment
from core.safety_controller import SafetyController

from utils.trajectory_interpolator import VecTF, ActionChunkManager
from utils.keyboard_intervention import KeyboardInterventionHandler, InterventionApplier
from utils.timeline_logger import TimelineLogger


class InferenceNode:
    """ROS inference node with human intervention and data recording support."""

    def __init__(self, config):
        """
        Args:
            config: dict or InferenceConfig.
        """
        # Normalise config to InferenceConfig
        if isinstance(config, InferenceConfig):
            self.cfg = config
        else:
            self.cfg = InferenceConfig.from_dict(config)
        # Keep a dict view for sub-modules that still expect a dict
        self.config = self.cfg.to_dict()

        # Core parameters
        self.temporal_factor_k = self.cfg.temporal_factor_k
        self.desire_inference_freq = self.cfg.desire_inference_freq
        self.pos_lookahead_step = self.cfg.pos_lookahead_step
        self.pos_lookahead_duration = self.cfg.pos_lookahead_duration
        self.check_workspace = True
        self.teleop_scale = self.cfg.teleop_scale  # 1.0 (fixed)
        self.inference_speed_scale = self.cfg.inference_speed_scale
        self.control_freq = self.cfg.control_freq
        self.execution_mode = self.cfg.execution_mode
        self.max_active_chunks = self.cfg.max_active_chunks
        self.crossfade_steps = self.cfg.crossfade_steps
        self.truncate_at_act_horizon = self.cfg.truncate_at_act_horizon

        # ---- Environment ----
        rospy.loginfo("Initializing environment...")
        self.env = RealEnvironment(self.config)

        # ---- Policy ----
        rospy.loginfo("Initializing policy...")
        self.policy = self._create_policy(self.config)

        # ---- Safety controller ----
        self.safety_controller = self._create_safety_controller(self.config)

        # ---- Logger ----
        self.logger = self._create_logger(self.config)
        self.episode_started = False

        # ---- Timeline ----
        self.timeline_enabled = self.cfg.timeline_enabled
        self.timeline_control_stride = self.cfg.timeline_control_stride
        self.chunk_time_base = self.cfg.chunk_time_base
        self.timeline_logger: Optional[TimelineLogger] = None

        # Horizons from policy
        self._act_horizon = self.cfg.act_horizon or getattr(self.policy, 'pred_horizon', 8)
        self._pred_horizon = getattr(self.policy, 'pred_horizon', 16)
        self._obs_horizon = getattr(self.policy, 'obs_horizon', 2)
        self._action_dim_full = getattr(self.policy, 'action_dim_full', 15)

        # Create timeline logger
        if self.timeline_enabled:
            timeline_path = self.cfg.timeline_log
            if not timeline_path:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                timeline_path = os.path.join(
                    self.logger.log_dir, f'timeline_{timestamp}.jsonl',
                )
            self.timeline_logger = TimelineLogger(
                timeline_path,
                control_log_interval=self.timeline_control_stride * 10,
            )
            self._timeline_path = timeline_path
            self.timeline_logger.log(
                'init',
                desire_inference_freq=self.desire_inference_freq,
                temporal_factor_k=self.temporal_factor_k,
                pos_lookahead_step=self.pos_lookahead_step,
                pos_lookahead_duration=self.pos_lookahead_duration,
                chunk_time_base=self.chunk_time_base,
                act_horizon=self._act_horizon,
                pred_horizon=self._pred_horizon,
                obs_horizon=self._obs_horizon,
                execution_mode=self.execution_mode,
                max_active_chunks=self.max_active_chunks,
                crossfade_steps=self.crossfade_steps,
                truncate_at_act_horizon=self.truncate_at_act_horizon,
                teleop_scale=self.teleop_scale,
                control_freq=self.control_freq,
            )
        else:
            self._timeline_path = None

        # ---- Logger metadata ----
        self._setup_logger_metadata(self.config)

        # ---- Action chunk manager ----
        self.action_manager = ActionChunkManager(
            temporal_factor_k=self.temporal_factor_k,
            execution_mode=self.execution_mode,
            max_active_chunks=self.max_active_chunks,
            crossfade_steps=self.crossfade_steps,
        )
        self.lock_tfs = threading.Lock()

        rospy.loginfo(
            f"ActionChunkManager: mode={self.execution_mode}, "
            f"max_active_chunks={self.max_active_chunks}, "
            f"crossfade_steps={self.crossfade_steps}, "
            f"truncate_at_act_horizon={self.truncate_at_act_horizon}",
        )

        # ---- Action processor (Phase 2 extraction) ----
        self.action_processor = ActionProcessor(
            action_dim_full=self._action_dim_full,
            safety_controller=self.safety_controller,
            inference_speed_scale=self.inference_speed_scale,
            check_workspace=self.check_workspace,
        )

        # ---- Runtime state ----
        self.running = True
        self.latest_obs = None
        self.pos_lookahead_step_start_idx = 0
        self.step_count = 0
        self.last_action = None
        self.control_step_count = 0
        self._last_control_time: Optional[float] = None
        self._control_hz_ema: Optional[float] = None
        self._last_gripper_value: Optional[float] = None
        self._last_gripper_log_time = 0.0

        # Episode lifecycle
        self.record_inference_enabled = self.cfg.record_inference
        self.waiting_start = self.record_inference_enabled
        self.episode_paused = self.waiting_start
        self.pending_save = False
        self.max_steps = self.cfg.max_steps

        # ---- Intervention & recording ----
        self.intervention_enabled = self.cfg.intervention
        self.intervention_handler: Optional[KeyboardInterventionHandler] = None
        self.inference_recorder: Optional[InferenceRecorder] = None

        if self.intervention_enabled or self.record_inference_enabled:
            self._init_intervention_and_recording(self.config)

        # ---- Start inference thread ----
        self.inference_thread = threading.Thread(
            target=self._inference_loop, daemon=True,
        )
        self.inference_thread.start()
        rospy.loginfo("InferenceNode initialized")

    # =====================================================================
    # Factory helpers
    # =====================================================================

    def _create_policy(self, config: dict) -> RealPolicy:
        pretrain_path = config.get('pretrain', '')
        if not pretrain_path:
            rospy.logerr("No pretrain model specified! Use --pretrain.")
            raise SystemExit(1)
        if not os.path.exists(pretrain_path):
            rospy.logerr(f"Pretrain model not found: {pretrain_path}")
            raise SystemExit(1)
        rospy.loginfo(f"Loading policy from: {pretrain_path}")
        policy = RealPolicy(config)
        policy.load_model(pretrain_path)
        return policy

    def _create_safety_controller(self, config: dict) -> SafetyController:
        path = config.get('safety_config', '')
        if path and os.path.exists(path):
            rospy.loginfo(f"Loading safety config from: {path}")
            return SafetyController.from_config(path)
        rospy.logwarn("No safety config found, using default safety limits")
        return SafetyController()

    def _create_logger(self, config: dict) -> InferenceLogger:
        log_dir = config.get('log_dir', '')
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            return InferenceLogger(log_dir=log_dir)
        from utils.paths import get_inference_logs_dir, ensure_dir
        return InferenceLogger(log_dir=ensure_dir(get_inference_logs_dir()))

    def _setup_logger_metadata(self, config: dict) -> None:
        pretrain_path = config.get('pretrain', '')
        model_config = {
            'path': pretrain_path,
            'algorithm': getattr(self.policy, 'algorithm', 'unknown'),
            'action_mode': 'full' if self._action_dim_full == 15 else 'ee_only',
            'state_mode': getattr(self.policy, 'state_mode', 'joint_only'),
            'obs_horizon': getattr(self.policy, 'obs_horizon', 2),
            'pred_horizon': getattr(self.policy, 'pred_horizon', 16),
            'action_dim': getattr(self.policy, 'action_dim', 13),
            'action_dim_full': self._action_dim_full,
            'visual_encoder_type': config.get('visual_encoder_type', 'unknown'),
            'use_ema': getattr(self.policy, 'use_ema', False),
            'num_inference_steps': getattr(self.policy, 'num_inference_steps', 10),
        }
        normalizer_config = {
            'enabled': getattr(self.policy, 'normalize_actions', False),
            'mode': getattr(self.policy, 'action_norm_mode', 'standard'),
        }
        if hasattr(self.policy, 'action_normalizer') and self.policy.action_normalizer is not None:
            normalizer = self.policy.action_normalizer
            if hasattr(normalizer, 'stats') and normalizer.stats:
                normalizer_config['action_stats'] = {
                    'mean': normalizer.stats.get('mean', []),
                    'std': normalizer.stats.get('std', []),
                }
        control_config = {
            'control_freq': self.control_freq,
            'teleop_scale': self.teleop_scale,
            'gripper_hysteresis_window': getattr(
                self.policy, 'gripper_hysteresis_window', 1,
            ),
        }
        execution_config = {
            'mode': self.execution_mode,
            'act_horizon': self._act_horizon,
            'max_active_chunks': self.max_active_chunks,
            'crossfade_steps': self.crossfade_steps,
            'truncate_at_act_horizon': self.truncate_at_act_horizon,
            'temporal_factor_k': self.temporal_factor_k,
            'pos_lookahead_step': self.pos_lookahead_step,
            'chunk_time_base': self.chunk_time_base,
            'desire_inference_freq': self.desire_inference_freq,
        }
        safety_config = {
            'config_path': config.get('safety_config', ''),
            'check_workspace': self.check_workspace,
            'max_relative_translation': 0.1,
        }
        self.logger.set_metadata(
            model_path=pretrain_path,
            model_config=model_config,
            normalizer_config=normalizer_config,
            control_config=control_config,
            execution_config=execution_config,
            safety_config=safety_config,
        )
        rospy.loginfo("Logger metadata configured for run_info.json")

    # =====================================================================
    # Intervention & recording setup
    # =====================================================================

    def _init_intervention_and_recording(self, config: dict) -> None:
        if self.intervention_enabled:
            self.intervention_handler = KeyboardInterventionHandler(
                xyz_scale=config.get('intervention_xyz_scale', 0.005),
                gripper_open=config.get('intervention_gripper_open', 1.0),
                gripper_close=config.get('intervention_gripper_close', 0.0),
                mode=config.get('intervention_mode', 'replace'),
            )
            self.intervention_handler.set_record_callback(self._handle_record_action)
            self.intervention_handler.set_quit_callback(
                lambda: setattr(self, 'running', False),
            )
            self.intervention_handler.start()
            rospy.loginfo("Keyboard intervention enabled")

        if self.record_inference_enabled:
            record_dir = config.get('record_dir', '') or config.get('log_dir', '')
            if not record_dir:
                from utils.paths import get_inference_logs_dir
                record_dir = get_inference_logs_dir()
            action_dim = getattr(self.policy, 'action_dim_full', 15)
            self.inference_recorder = InferenceRecorder(
                output_dir=record_dir,
                pred_horizon=self._pred_horizon,
                action_dim=action_dim,
            )
            rospy.loginfo(f"Inference recording enabled, output_dir: {record_dir}")
            rospy.loginfo("=" * 60)
            rospy.loginfo("Multi-episode recording mode enabled")
            rospy.loginfo("Press 'R' to start recording an episode")
            rospy.loginfo(
                "Press 'R' again to stop and choose to save (Y) or discard (N)",
            )
            rospy.loginfo("After save/discard, arm will return to init position")
            rospy.loginfo("Press 'R' to start next episode")
            rospy.loginfo("Press Ctrl+C to quit")
            rospy.loginfo("=" * 60)

    # =====================================================================
    # Episode state machine
    # =====================================================================

    def _handle_record_action(self, action: str) -> None:
        if action == 'toggle':
            if self.pending_save:
                rospy.logwarn("Please confirm save first (Y/N)")
                return
            if self.waiting_start:
                self._start_new_episode()
            else:
                self._stop_current_episode()
        elif action == 'confirm':
            if self.pending_save:
                self._confirm_save_episode(save=True)
            else:
                rospy.logwarn("No episode waiting for save")
        elif action == 'discard':
            if self.pending_save:
                self._confirm_save_episode(save=False)
            else:
                rospy.logwarn("No episode to discard")

    def _start_new_episode(self) -> None:
        self.pending_save = False
        with self.lock_tfs:
            self.action_manager.clear()
        if hasattr(self, 'policy') and self.policy is not None:
            self.policy.reset()
            rospy.loginfo("Policy state reset")
        self.waiting_start = False
        self.episode_paused = False
        if self.inference_recorder:
            self.inference_recorder.start_recording()
        self.step_count = 0
        rospy.loginfo("=" * 60)
        rospy.loginfo("Episode started! Robot is now under policy control.")
        rospy.loginfo("Press 'R' to stop recording")
        rospy.loginfo("=" * 60)

    def _stop_current_episode(self) -> None:
        self.episode_paused = True
        self.pending_save = True
        if self.inference_recorder:
            self.inference_recorder.stop_recording()
        rospy.loginfo("=" * 60)
        rospy.loginfo(f"Episode stopped - {self.step_count} steps recorded")
        rospy.loginfo("Save this episode? Press 'Y' to save, 'N' to discard")
        rospy.loginfo("=" * 60)

    def _confirm_save_episode(self, save: bool) -> None:
        if save and self.inference_recorder:
            filepath = self.inference_recorder.confirm_save()
            if filepath:
                rospy.loginfo(f"Episode saved to: {filepath}")
        elif not save and self.inference_recorder:
            self.inference_recorder.discard()
            rospy.loginfo("Episode discarded")
        self.pending_save = False
        rospy.loginfo("Returning to initial position...")
        self._reinitialize_arm()
        self.waiting_start = True
        self.episode_paused = True
        rospy.loginfo("=" * 60)
        rospy.loginfo("Ready for next episode. Press 'R' to start recording")
        rospy.loginfo("=" * 60)

    def _reinitialize_arm(self) -> None:
        try:
            self.env.init_status()
            rospy.loginfo("Arm returned to initial position")
        except Exception as e:
            rospy.logerr(f"Failed to reinitialize arm: {e}")

    # =====================================================================
    # Inference loop — decomposed private helpers
    # =====================================================================

    def _log_obs_timing(self, obs: dict, t_obs_ready: float) -> None:
        if self.timeline_logger is None:
            return
        obs_stamp = obs.get('stamp')
        delta = (t_obs_ready - obs_stamp) if obs_stamp is not None else None
        self.timeline_logger.log(
            'obs', obs_stamp_ros=obs_stamp,
            t_obs_ready_sys=t_obs_ready, delta_obs=delta,
        )

    def _log_inference_timing(self, t_start: float, duration: float) -> None:
        if self.timeline_logger is None:
            return
        self.timeline_logger.log(
            'inference',
            t_infer_start=t_start,
            t_infer_end=t_start + duration,
            inference_time=duration,
        )

    def _apply_intervention(
        self, all_actions: np.ndarray,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply keyboard intervention if enabled. Returns (actions, mask)."""
        mask = None
        if not (self.intervention_enabled and self.intervention_handler is not None):
            return all_actions, mask
        intervention = self.intervention_handler.get_intervention()
        if intervention is None:
            return all_actions, mask
        all_actions, mask = InterventionApplier.apply_to_action_chunk(
            all_actions, intervention, action_format='ee_delta',
        )
        rospy.loginfo_throttle(
            2.0, f"Intervention applied: mask={mask[0].sum()} dims",
        )
        return all_actions, mask

    def _record_step(
        self,
        obs: dict,
        action_model: np.ndarray,
        action_intervened: np.ndarray,
        intervention_mask: Optional[np.ndarray],
    ) -> None:
        if not (self.record_inference_enabled and self.inference_recorder is not None):
            return
        if not self.inference_recorder.is_recording:
            return
        self.inference_recorder.record_step(
            obs=obs,
            action_model=action_model,
            action_intervened=action_intervened,
            intervention_mask=intervention_mask,
            timestamp=time.time(),
        )

    def _post_action_chunk(
        self,
        abs_actions: np.ndarray,
        obs: dict,
        t_obs_ready: float,
    ) -> None:
        """Create a VecTF trajectory and add it to the ActionChunkManager."""
        obs_stamp = obs.get('stamp')
        if self.chunk_time_base == 'obs_stamp' and obs_stamp is not None:
            chunk_base_time = obs_stamp
        else:
            chunk_base_time = time.time()

        tf = VecTF({})
        action_interval = 1.0 / self.control_freq

        if self.truncate_at_act_horizon:
            n = min(self._act_horizon, len(abs_actions))
        else:
            n = len(abs_actions)

        self.pos_lookahead_step_start_idx += 1
        chunk_targets: List[float] = []

        for i in range(n):
            if self.pos_lookahead_step == 1:
                target_time = chunk_base_time + i * action_interval
            else:
                if self.pos_lookahead_step_start_idx % self.pos_lookahead_step == 0:
                    target_time = chunk_base_time + i * action_interval
                else:
                    target_time = chunk_base_time + i * self.pos_lookahead_duration
            tf.append(target_time, abs_actions[i].tolist())
            chunk_targets.append(target_time)

        with self.lock_tfs:
            chunk_id = self.action_manager.add_trajectory(tf)

        if self.timeline_logger is not None:
            delta_chunk_obs = None
            if obs_stamp is not None:
                delta_chunk_obs = chunk_base_time - obs_stamp
            self.timeline_logger.log(
                'chunk',
                chunk_id=chunk_id,
                chunk_base_time=chunk_base_time,
                obs_stamp_ros=obs_stamp,
                t_obs_ready_sys=t_obs_ready,
                action_interval=action_interval,
                pred_horizon=len(abs_actions),
                act_horizon=self._act_horizon,
                num_actions_added=n,
                truncated=self.truncate_at_act_horizon,
                delta_chunk_obs=delta_chunk_obs,
                chunk_targets=chunk_targets,
            )

    def _log_step(
        self,
        obs: dict,
        raw_action: np.ndarray,
        executed_action: np.ndarray,
        inference_time: float,
        safety: SafetyResult,
    ) -> None:
        self.logger.log_step(
            timestamp=time.time(),
            obs=obs,
            raw_action=raw_action,
            executed_action=executed_action,
            inference_time=inference_time,
            safety_clipped=safety.clipped,
            safety_warnings=safety.events if safety.events else None,
        )

    def _check_max_steps(self) -> None:
        if self.step_count >= self.max_steps:
            rospy.logwarn(
                f"Reached max_steps ({self.max_steps}), auto-stopping episode...",
            )
            self._stop_current_episode()

    # =====================================================================
    # Inference loop (main)
    # =====================================================================

    def _inference_loop(self) -> None:
        """Inference thread main loop (runs as a daemon thread)."""
        rospy.loginfo("Inference thread started")
        if hasattr(self, 'policy') and self.policy is not None:
            self.policy.reset()
            rospy.loginfo("Policy state initialized (reset)")

        desire_period = 1.0 / self.desire_inference_freq
        consecutive_errors = 0
        MAX_CONSECUTIVE_ERRORS = 10  # SEC-1 fix

        with torch.inference_mode():
            while self.running and not rospy.is_shutdown():
                if self.episode_paused or self.waiting_start:
                    time.sleep(0.1)
                    continue

                # 1. Observe
                self.latest_obs = self.env.get_observation()
                if self.latest_obs is None:
                    time.sleep(0.5)
                    rospy.loginfo_throttle(5.0, "Waiting for observation...")
                    continue

                t_obs_ready = time.time()
                self._log_obs_timing(self.latest_obs, t_obs_ready)

                if not self.episode_started:
                    self.logger.start_episode(timeline_path=self._timeline_path)
                    self.episode_started = True
                    rospy.loginfo("Episode started, logging enabled")

                last_start = time.time()

                try:
                    # 2. Prepare input — BUG-1 fix: pass raw HWC image
                    qpos_joint = np.array(self.latest_obs['qpos_joint'])
                    qpos_end = np.array(self.latest_obs['qpos_end'])

                    if hasattr(self.policy, 'build_state_from_obs'):
                        state = self.policy.build_state_from_obs(qpos_joint, qpos_end)
                    else:
                        state = qpos_joint.astype(np.float32)
                    qpos_t = torch.from_numpy(state).float().cuda().unsqueeze(0)

                    # Raw HWC image — RealPolicy handles resize + CHW
                    image_raw = self.latest_obs['images'][0]
                    image_t = torch.from_numpy(image_raw).float().cuda()

                    qpos_end_list = qpos_end.tolist()

                    # 3. Inference
                    t_infer = time.time()
                    ret = self.policy({"qpos": qpos_t, "image": image_t})
                    inference_time = time.time() - t_infer
                    self._log_inference_timing(t_infer, inference_time)

                    all_actions = ret['a_hat'].squeeze(0).cpu().numpy()

                    # 4. Action processing pipeline
                    all_actions = self.action_processor.apply_speed_scale(all_actions)
                    safety_result = self.action_processor.apply_safety_checks(
                        all_actions, qpos_end_list,
                    )
                    all_actions = safety_result.actions

                    # 5. Intervention
                    action_model = all_actions.copy()
                    all_actions, intervention_mask = self._apply_intervention(all_actions)
                    action_intervened = all_actions.copy()

                    # 6. Record
                    self._record_step(
                        self.latest_obs, action_model,
                        action_intervened, intervention_mask,
                    )

                    self.last_action = all_actions[0].copy()
                    raw_action_for_log = all_actions[0].copy()

                    # 7. Relative → absolute
                    abs_actions = self.action_processor.convert_to_absolute(
                        all_actions, qpos_end_list,
                    )

                    # 8. Post chunk
                    self._post_action_chunk(abs_actions, self.latest_obs, t_obs_ready)

                    # 9. Log
                    self._log_step(
                        self.latest_obs, raw_action_for_log,
                        abs_actions[0], inference_time, safety_result,
                    )

                    self.step_count += 1
                    consecutive_errors = 0  # SEC-1: reset on success
                    rospy.loginfo_throttle(
                        5.0,
                        f"Step {self.step_count}, Inference: {inference_time:.4f}s, "
                        f"Actions: {abs_actions.shape}",
                    )
                    self._check_max_steps()

                except Exception as e:
                    consecutive_errors += 1
                    import traceback
                    rospy.logerr(
                        f"Inference error ({consecutive_errors}/"
                        f"{MAX_CONSECUTIVE_ERRORS}): {e}",
                    )
                    rospy.logerr(traceback.format_exc())
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        rospy.logerr(
                            "Too many consecutive errors, stopping inference",
                        )
                        self.running = False
                        break

                wait_tm = desire_period - (time.time() - last_start)
                if wait_tm > 0:
                    time.sleep(wait_tm)

    # =====================================================================
    # Control loop
    # =====================================================================

    def control_loop(self) -> None:
        """Control main loop — runs on the main thread."""
        rospy.loginfo("Control loop started")
        control_period = 1.0 / self.control_freq
        rospy.loginfo(
            f"Control frequency: {self.control_freq}Hz "
            f"(period={control_period:.4f}s)",
        )

        while self.running and not rospy.is_shutdown():
            if self.episode_paused or self.waiting_start:
                time.sleep(0.05)
                continue

            tm = time.time()
            meta = None
            with self.lock_tfs:
                if self.timeline_logger is not None:
                    action, meta = self.action_manager.get_fused_action_with_meta(tm)
                else:
                    action = self.action_manager.get_fused_action(tm)

            if action is None:
                time.sleep(0.02)
                continue

            # EMA control frequency estimation
            if self._last_control_time is not None:
                dt = tm - self._last_control_time
                if dt > 0:
                    inst_hz = 1.0 / dt
                    if self._control_hz_ema is None:
                        self._control_hz_ema = inst_hz
                    else:
                        self._control_hz_ema = 0.2 * inst_hz + 0.8 * self._control_hz_ema
            self._last_control_time = tm

            # Throttled gripper logging
            grip_val = float(action[-1]) if len(action) > 0 else None
            now = time.time()
            if grip_val is not None and (now - self._last_gripper_log_time) >= 5.0:
                delta = (
                    None if self._last_gripper_value is None
                    else (grip_val - self._last_gripper_value)
                )
                hz_str = (
                    f"{self._control_hz_ema:.1f}Hz"
                    if self._control_hz_ema is not None else "n/a"
                )
                rospy.loginfo(
                    f"Gripper cmd: {grip_val:.4f}, delta: "
                    f"{delta if delta is not None else 'n/a'}, control_hz: {hz_str}",
                )
                self._last_gripper_value = grip_val
                self._last_gripper_log_time = now

            # Execute
            self.env.end_control_nostep(action)

            if (
                self.timeline_logger is not None
                and (self.control_step_count % self.timeline_control_stride == 0)
            ):
                self.timeline_logger.log(
                    'control',
                    query_time=tm,
                    t_send_sys=time.time(),
                    candidate_timestamps=meta.get('candidate_timestamps', []) if meta else [],
                    weights=meta.get('weights', []) if meta else [],
                    num_candidates=meta.get('num_candidates', 0) if meta else 0,
                    used_chunk_ids=meta.get('used_chunk_ids', []) if meta else [],
                )
            self.control_step_count += 1
            time.sleep(control_period)

    # =====================================================================
    # Shutdown
    # =====================================================================

    def shutdown(self) -> None:
        if hasattr(self, '_shutdown_called') and self._shutdown_called:
            return
        self._shutdown_called = True
        rospy.loginfo("Shutting down InferenceNode...")
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

        if self.timeline_logger is not None:
            self.timeline_logger.close()

        self.env.shutdown()
        rospy.loginfo("InferenceNode shutdown complete")
