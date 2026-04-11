#!/usr/bin/env python3
"""Convert recorded CARM inference episodes into a training staging directory."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CARM_DEPLOY_ROOT = ROOT / 'carm_ros_deploy' / 'src' / 'carm_deploy'
for p in (str(ROOT), str(CARM_DEPLOY_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from inference.inference_recorder import InferenceDatasetConverter


def _validate_thresholds(
    gold_max_safety_clip_rate: float,
    silver_max_safety_clip_rate: float,
) -> None:
    if not 0.0 <= gold_max_safety_clip_rate <= 1.0:
        raise ValueError('gold_max_safety_clip_rate must be within [0, 1]')
    if not 0.0 <= silver_max_safety_clip_rate <= 1.0:
        raise ValueError('silver_max_safety_clip_rate must be within [0, 1]')
    if gold_max_safety_clip_rate > silver_max_safety_clip_rate:
        raise ValueError('gold_max_safety_clip_rate must be <= silver_max_safety_clip_rate')


def main() -> None:
    parser = argparse.ArgumentParser(description='Convert inference_episode files into training staging episodes')
    parser.add_argument('--input_dir', type=str, default='inference_logs', help='Directory containing inference_episode_*.hdf5')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to write converted episode_*.hdf5 files')
    parser.add_argument('--admission_policy', choices=['none', 'episode'], default='none', help='Apply no admission filter or episode-level admission')
    parser.add_argument('--policy_version', type=str, default=InferenceDatasetConverter.ADMISSION_POLICY_VERSION, help='Admission policy version string written into metadata')
    parser.add_argument('--min_steps', type=int, default=InferenceDatasetConverter.DEFAULT_MIN_STEPS, help='Minimum kept steps for episode admission')
    parser.add_argument('--gold_max_safety_clip_rate', type=float, default=InferenceDatasetConverter.DEFAULT_GOLD_MAX_SAFETY_CLIP_RATE, help='Maximum safety clip rate for gold bucket')
    parser.add_argument('--silver_max_safety_clip_rate', type=float, default=InferenceDatasetConverter.DEFAULT_SILVER_MAX_SAFETY_CLIP_RATE, help='Maximum safety clip rate for silver bucket')
    parser.add_argument('--drop_failed_episode', action='store_true', help='Do not write staging HDF5 for episodes that fail admission')
    args = parser.parse_args()

    _validate_thresholds(
        gold_max_safety_clip_rate=args.gold_max_safety_clip_rate,
        silver_max_safety_clip_rate=args.silver_max_safety_clip_rate,
    )

    converted_records = InferenceDatasetConverter.convert_directory_to_training_format(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        admission_policy=args.admission_policy,
        policy_version=args.policy_version,
        min_steps=args.min_steps,
        gold_max_safety_clip_rate=args.gold_max_safety_clip_rate,
        silver_max_safety_clip_rate=args.silver_max_safety_clip_rate,
        drop_failed_episode=args.drop_failed_episode,
    )

    reason_counts: dict[str, int] = {}
    bucket_counts: dict[str, int] = {}
    for record in converted_records:
        reason = str(record.get('admission_reason', 'unknown'))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        bucket = str(record.get('admission_bucket', 'unknown'))
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    metadata = {
        'input_dir': os.path.expandvars(os.path.expanduser(args.input_dir)),
        'output_dir': os.path.expandvars(os.path.expanduser(args.output_dir)),
        'num_input_files': len(converted_records),
        'num_written_files': sum(1 for record in converted_records if record.get('converted')),
        'num_failed_admission': sum(1 for record in converted_records if not record.get('admission_pass')),
        'num_gold': bucket_counts.get('gold', 0),
        'num_silver': bucket_counts.get('silver', 0),
        'num_reject': bucket_counts.get('reject', 0),
        'num_success_labeled': sum(1 for record in converted_records if record.get('success') is True),
        'num_failure_labeled': sum(1 for record in converted_records if record.get('success') is False),
        'converted_files': [os.path.basename(record['output_path']) for record in converted_records if record.get('converted')],
        'episode_sidecars': [os.path.basename(record['sidecar_path']) for record in converted_records],
        'action_source_used': 'action_executed[:,0,:]',
        'admission_policy': args.admission_policy,
        'policy_version': args.policy_version,
        'min_steps': args.min_steps,
        'gold_max_safety_clip_rate': args.gold_max_safety_clip_rate,
        'silver_max_safety_clip_rate': args.silver_max_safety_clip_rate,
        'drop_failed_episode': args.drop_failed_episode,
        'admission_bucket_counts': bucket_counts,
        'admission_reason_counts': reason_counts,
        'episodes': converted_records,
    }
    metadata_path = Path(args.output_dir).expanduser() / 'conversion_metadata.json'
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"Processed {len(converted_records)} files into {args.output_dir}")
    print(f"Wrote metadata to {metadata_path}")


if __name__ == '__main__':
    main()
