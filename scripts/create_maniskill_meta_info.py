#!/usr/bin/env python3
"""VLAW P2.1 — 生成 ManiSkill 动作归一化统计量 (stat.json).

用法:
    conda run -n ctrl_world python scripts/create_maniskill_meta_info.py \
        --data_dir data/vlaw/demos \
        --output_dir data/vlaw/meta_info/maniskill

输出:
    data/vlaw/meta_info/maniskill/stat.json
        {"state_01": [7 floats], "state_99": [7 floats]}
"""

import argparse
import sys
from pathlib import Path

# 将项目根目录加入 Python 路径
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "ctrl_world"))


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 ManiSkill 动作统计量")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/vlaw/demos",
        help="包含 HDF5 轨迹文件的目录",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/vlaw/meta_info/maniskill",
        help="stat.json 输出目录",
    )
    args = parser.parse_args()

    from dataset.dataset_maniskill import create_meta_info

    stat_path = create_meta_info(
        data_dir=args.data_dir,
        output_dir=str(Path(args.output_dir).parent),
        dataset_name=Path(args.output_dir).name,
    )
    print(f"✅ 元信息已生成: {stat_path}")


if __name__ == "__main__":
    main()
