from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_PI05_TASK_SEMANTICS = {
    "task": "pick_and_place_tape_into_cup",
    "description": "Pick up the black tape roll and place it into the blue cup.",
    "subtasks": [
        {
            "name": "pick_tape",
            "instruction": "Pick up the black tape roll.",
        },
        {
            "name": "place_tape_in_cup",
            "instruction": "Place the tape roll into the blue cup.",
        },
    ],
}


@dataclass(frozen=True)
class Pi05SubtaskSemantics:
    name: str
    instruction: str


@dataclass(frozen=True)
class Pi05TaskSemantics:
    task: str
    description: str
    subtasks: tuple[Pi05SubtaskSemantics, ...]

    @property
    def subtask_names(self) -> tuple[str, ...]:
        return tuple(subtask.name for subtask in self.subtasks)

    def instruction_for(self, subtask_name: str) -> str:
        for subtask in self.subtasks:
            if subtask.name == subtask_name:
                return subtask.instruction
        raise KeyError(f"Unknown subtask {subtask_name!r}; expected one of {self.subtask_names}")

    def prompt_for(self, subtask_name: str) -> str:
        return f"{self.description} Current subtask: {self.instruction_for(subtask_name)}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "description": self.description,
            "subtasks": [
                {"name": subtask.name, "instruction": subtask.instruction}
                for subtask in self.subtasks
            ],
        }


def _parse_task_semantics(data: dict[str, Any]) -> Pi05TaskSemantics:
    task = str(data.get("task", "")).strip()
    description = str(data.get("description", "")).strip()
    raw_subtasks = data.get("subtasks", [])
    if not task:
        raise ValueError("Task semantics must include a non-empty 'task'")
    if not description:
        raise ValueError("Task semantics must include a non-empty 'description'")
    if not isinstance(raw_subtasks, list) or not raw_subtasks:
        raise ValueError("Task semantics must include a non-empty 'subtasks' list")

    subtasks: list[Pi05SubtaskSemantics] = []
    seen: set[str] = set()
    for raw in raw_subtasks:
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid subtask entry: {raw!r}")
        name = str(raw.get("name", "")).strip()
        instruction = str(raw.get("instruction", "")).strip()
        if not name or not instruction:
            raise ValueError(f"Subtask entries need non-empty name/instruction: {raw!r}")
        if name in seen:
            raise ValueError(f"Duplicate subtask name: {name}")
        seen.add(name)
        subtasks.append(Pi05SubtaskSemantics(name=name, instruction=instruction))

    return Pi05TaskSemantics(task=task, description=description, subtasks=tuple(subtasks))


def load_pi05_task_semantics(path: str | Path | None = None) -> Pi05TaskSemantics:
    if path is None:
        return _parse_task_semantics(DEFAULT_PI05_TASK_SEMANTICS)

    semantics_path = Path(path).expanduser().resolve()
    with open(semantics_path) as handle:
        data = json.load(handle)
    return _parse_task_semantics(data)
