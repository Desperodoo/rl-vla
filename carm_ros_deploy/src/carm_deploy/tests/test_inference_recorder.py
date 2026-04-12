import json
import importlib.util
from pathlib import Path

import h5py
import numpy as np

_MODULE_PATH = Path(__file__).resolve().parents[1] / 'inference' / 'inference_recorder.py'
_SPEC = importlib.util.spec_from_file_location('real_inference_recorder', _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
InferenceDatasetConverter = _MODULE.InferenceDatasetConverter
InferenceRecorder = _MODULE.InferenceRecorder


def _make_obs() -> dict:
    return {
        'images': [np.zeros((8, 8, 3), dtype=np.uint8)],
        'qpos_joint': np.zeros(7, dtype=np.float32),
        'qpos_end': np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0, 0.5], dtype=np.float32),
        'qpos': np.zeros(15, dtype=np.float32),
        'gripper': 0.5,
    }


def _make_action_chunk(offset: float) -> np.ndarray:
    chunk = np.zeros((4, 8), dtype=np.float32)
    chunk[:, 0] = 0.1 + offset
    chunk[:, 1] = 0.2
    chunk[:, 2] = 0.3
    chunk[:, 6] = 1.0
    chunk[:, 7] = 0.5
    return chunk


def _write_run_info(log_dir: Path, episode_name: str) -> Path:
    run_info_path = log_dir / 'run_info_20260411_205016.json'
    run_info = {
        'files': {
            'episode_hdf5': [episode_name],
            'timeline': 'timeline_20260411_205014.jsonl',
            'run_info': run_info_path.name,
        },
        'summary': {
            'avg_inference_time': 0.01,
            'max_inference_time': 0.02,
            'safety_clips': 0,
            'safety_clip_rate': 0.0,
        },
        'execution': {
            'desire_inference_freq': 10,
            'act_horizon': 4,
            'truncate_at_act_horizon': True,
        },
        'control': {
            'control_freq': 50,
        },
        'model': {
            'path': 'dummy.ckpt',
            'algorithm': 'consistency_flow',
            'pred_horizon': 4,
            'action_dim_full': 8,
        },
    }
    run_info_path.write_text(json.dumps(run_info), encoding='utf-8')
    return run_info_path


def test_inference_recorder_writes_execution_only_schema(tmp_path):
    recorder = InferenceRecorder(
        output_dir=str(tmp_path),
        pred_horizon=4,
        action_dim=8,
        image_size=(8, 8),
        max_steps=16,
        camera_topics=['/cam0'],
        camera_names=['cam0'],
        primary_camera='cam0',
    )

    assert recorder.start_recording() is True
    for step in range(3):
        model = _make_action_chunk(offset=0.01 * step)
        executed = model.copy()
        if step == 1:
            executed[0, 0] += 0.05
        assert recorder.record_step(_make_obs(), model, executed, timestamp=float(step)) is True

    assert recorder.stop_recording() is True
    path = recorder.confirm_save(success=True, outcome_label='success')
    assert path is not None

    with h5py.File(path, 'r') as f:
        assert 'action_model' in f
        assert 'action_executed' in f
        assert 'action' in f
        assert 'action_intervened' not in f
        assert 'intervention_mask' not in f
        assert 'has_intervention' not in f.attrs
        assert 'intervention_ratio' not in f.attrs
        np.testing.assert_allclose(f['action'][:], f['action_executed'][:, 0, :])


def test_inference_dataset_converter_uses_action_executed_schema(tmp_path):
    log_dir = tmp_path / 'inference_logs'
    log_dir.mkdir()

    recorder = InferenceRecorder(
        output_dir=str(log_dir),
        pred_horizon=4,
        action_dim=8,
        image_size=(8, 8),
        max_steps=16,
        camera_topics=['/cam0'],
        camera_names=['cam0'],
        primary_camera='cam0',
    )
    recorder.start_recording()
    for step in range(5):
        model = _make_action_chunk(offset=0.01 * step)
        executed = model.copy()
        if step == 2:
            executed[0, 1] += 0.02
        recorder.record_step(_make_obs(), model, executed, timestamp=float(step))
    recorder.stop_recording()
    episode_path = Path(recorder.confirm_save(success=True, outcome_label='success'))

    _write_run_info(log_dir, episode_path.name)

    output_path = tmp_path / 'staging' / 'episode_0001.hdf5'
    record = InferenceDatasetConverter.convert_to_training_format(
        input_path=str(episode_path),
        output_path=str(output_path),
        admission_policy='episode',
        min_steps=4,
    )

    assert record['converted'] is True
    with h5py.File(output_path, 'r') as f:
        assert f.attrs['dataset_type'] == 'inference_staging'
        assert f.attrs['staging_schema_version'] == InferenceDatasetConverter.STAGING_SCHEMA_VERSION
        assert f.attrs['action_source_used'] == 'action_executed[:,0,:]'
        assert int(f.attrs['source_num_steps']) == 5
        assert 'filtered_intervention' not in f.attrs
        assert 'intervention_ratio_raw' not in f.attrs
        assert 'gold_max_safety_clip_rate' in f.attrs
        assert 'silver_max_safety_clip_rate' in f.attrs
        assert f['action'].shape == (5, 8)

    sidecar_path = output_path.with_suffix('.meta.json')
    sidecar = json.loads(sidecar_path.read_text(encoding='utf-8'))
    assert sidecar['conversion']['gold_max_safety_clip_rate'] == InferenceDatasetConverter.DEFAULT_GOLD_MAX_SAFETY_CLIP_RATE
    assert sidecar['stats']['action_source_used'] == 'action_executed[:,0,:]'


def test_inference_recorder_writes_hitl_provenance(tmp_path):
    recorder = InferenceRecorder(
        output_dir=str(tmp_path),
        pred_horizon=4,
        action_dim=8,
        image_size=(8, 8),
        max_steps=16,
        camera_topics=['/cam0'],
        camera_names=['cam0'],
        primary_camera='cam0',
        hitl_enabled=True,
        hitl_mode='candidate',
    )

    assert recorder.start_recording() is True
    model = _make_action_chunk(offset=0.0)
    executed = model.copy()
    human = np.full((4, 8), 0.2, dtype=np.float32)
    shared = np.full((4, 8), 0.3, dtype=np.float32)
    hitl_data = {
        'action_policy_chunk': model,
        'action_human_chunk': human,
        'action_shared_chunk': shared,
        'action_sched_candidate': np.full((8,), 0.4, dtype=np.float32),
        'action_exec_candidate': np.full((8,), 0.5, dtype=np.float32),
        'action_human_direct_target': np.full((8,), 0.6, dtype=np.float32),
        'action_human_sched_target': np.full((8,), 0.65, dtype=np.float32),
        'action_human_exec_target': np.full((8,), 0.66, dtype=np.float32),
        'action_live_execute_target': np.full((8,), 0.7, dtype=np.float32),
        'hitl_human_active': True,
        'hitl_human_valid': True,
        'hitl_signal_age_ms': 12.0,
        'hitl_human_history_count': 3,
        'hitl_human_history_span_ms': 80.0,
        'hitl_human_history_usable': True,
        'hitl_human_rollout_step_count': 4,
        'hitl_human_rollout_dt_ms': 20.0,
        'hitl_human_linear_velocity': np.array([0.1, 0.0, 0.0], dtype=np.float32),
        'hitl_human_angular_velocity': np.array([0.0, 0.0, 0.2], dtype=np.float32),
        'hitl_human_gripper_velocity': 0.03,
        'hitl_policy_sequence': 3,
        'hitl_human_sequence': 7,
        'hitl_shared_source': 1,
        'hitl_shared_valid_mask': True,
        'hitl_live_execute_source': 1,
    }
    assert recorder.record_step(_make_obs(), model, executed, timestamp=0.0, hitl_data=hitl_data) is True
    assert recorder.record_control_step(
        query_time=1.25,
        t_send_sys=1.30,
        execute_source='human_scheduled',
        human_execute_mode='scheduled',
        live_execute_target=np.full((8,), 0.7, dtype=np.float32),
        human_direct_target=np.full((8,), 0.6, dtype=np.float32),
        human_sched_target=np.full((8,), 0.65, dtype=np.float32),
        human_exec_target=np.full((8,), 0.66, dtype=np.float32),
        shared_source='human',
    ) is True
    assert recorder.stop_recording() is True
    path = recorder.confirm_save(success=True, outcome_label='success')

    with h5py.File(path, 'r') as f:
        assert bool(f.attrs['hitl_enabled']) is True
        assert f.attrs['hitl_mode'] == 'candidate'
        assert f.attrs['hitl_arbitration_mode'] == 'source_select'
        np.testing.assert_allclose(f['action_policy_chunk'][0], model)
        np.testing.assert_allclose(f['action_human_chunk'][0], human)
        np.testing.assert_allclose(f['action_shared_chunk'][0], shared)
        np.testing.assert_allclose(f['action_sched_candidate'][0], np.full((8,), 0.4))
        np.testing.assert_allclose(f['action_exec_candidate'][0], np.full((8,), 0.5))
        np.testing.assert_allclose(f['action_human_direct_target'][0], np.full((8,), 0.6))
        np.testing.assert_allclose(f['action_human_sched_target'][0], np.full((8,), 0.65))
        np.testing.assert_allclose(f['action_human_exec_target'][0], np.full((8,), 0.66))
        np.testing.assert_allclose(f['action_live_execute_target'][0], np.full((8,), 0.7))
        assert bool(f['hitl_human_active'][0]) is True
        assert bool(f['hitl_human_valid'][0]) is True
        assert int(f['hitl_human_history_count'][0]) == 3
        assert bool(f['hitl_human_history_usable'][0]) is True
        np.testing.assert_allclose(f['hitl_human_linear_velocity'][0], np.array([0.1, 0.0, 0.0]))
        np.testing.assert_allclose(f['hitl_human_angular_velocity'][0], np.array([0.0, 0.0, 0.2]))
        assert float(f['hitl_human_gripper_velocity'][0]) == 0.03
        assert int(f['hitl_shared_source'][0]) == 1
        assert int(f['hitl_live_execute_source'][0]) == 1
        assert bool(f.attrs['control_provenance_aligned']) is True
        assert int(f.attrs['num_control_steps']) == 1
        control_grp = f['control_provenance']
        assert control_grp['execute_source'][0].decode('utf-8') == 'human_scheduled'
        assert control_grp['human_execute_mode'][0].decode('utf-8') == 'scheduled'
        assert control_grp['shared_source'][0].decode('utf-8') == 'human'
        np.testing.assert_allclose(control_grp['human_direct_target'][0], np.full((8,), 0.6))
        np.testing.assert_allclose(control_grp['human_sched_target'][0], np.full((8,), 0.65))
        np.testing.assert_allclose(control_grp['human_exec_target'][0], np.full((8,), 0.66))
        np.testing.assert_allclose(control_grp['live_execute_target'][0], np.full((8,), 0.7))


def test_convert_directory_skips_old_incompatible_rollout(tmp_path):
    input_dir = tmp_path / 'input'
    input_dir.mkdir()

    old_path = input_dir / 'inference_episode_0001_legacy.hdf5'
    with h5py.File(old_path, 'w') as f:
        obs = f.create_group('observations')
        obs.create_dataset('images', data=np.zeros((1, 8, 8, 3), dtype=np.uint8))
        obs.create_dataset('qpos_joint', data=np.zeros((1, 7), dtype=np.float32))
        obs.create_dataset('qpos_end', data=np.zeros((1, 8), dtype=np.float32))
        obs.create_dataset('qpos', data=np.zeros((1, 15), dtype=np.float32))
        obs.create_dataset('gripper', data=np.zeros((1,), dtype=np.float32))
        obs.create_dataset('timestamps', data=np.zeros((1,), dtype=np.float64))
        f.create_dataset('action', data=np.zeros((1, 8), dtype=np.float32))
        f.create_dataset('action_model', data=np.zeros((1, 4, 8), dtype=np.float32))
        f.attrs['num_steps'] = 1
        f.attrs['pred_horizon'] = 4
        f.attrs['action_dim'] = 8

    recorder = InferenceRecorder(
        output_dir=str(input_dir),
        pred_horizon=4,
        action_dim=8,
        image_size=(8, 8),
        max_steps=16,
        camera_topics=['/cam0'],
        camera_names=['cam0'],
        primary_camera='cam0',
    )
    recorder.start_recording()
    for step in range(4):
        model = _make_action_chunk(offset=0.01 * step)
        recorder.record_step(_make_obs(), model, model, timestamp=float(step))
    recorder.stop_recording()
    new_episode_path = Path(recorder.confirm_save(success=True, outcome_label='success'))
    _write_run_info(input_dir, new_episode_path.name)

    records = InferenceDatasetConverter.convert_directory_to_training_format(
        input_dir=str(input_dir),
        output_dir=str(tmp_path / 'out'),
        admission_policy='episode',
        min_steps=2,
    )

    assert len(records) == 2
    error_records = [record for record in records if not record['converted']]
    ok_records = [record for record in records if record['converted']]
    assert len(error_records) == 1
    assert error_records[0]['admission_reason'] == 'conversion_error:KeyError'
    assert 'action_executed' in error_records[0]['error']
    assert len(ok_records) == 1
