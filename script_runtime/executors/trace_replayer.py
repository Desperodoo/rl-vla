"""Replay helper for existing timeline and trace logs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


class TraceReplayer:
    """Loads JSONL traces from timeline or script-runtime outputs."""

    def load_jsonl(self, path: str | Path) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    def summarize(self, rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        rows = list(rows)
        failures = [row for row in rows if row.get("failure_code") not in (None, "", "NONE")]
        skills = sorted({row.get("skill_name", "unknown") for row in rows})
        return {
            "num_events": len(rows),
            "num_failures": len(failures),
            "skills": skills,
        }
