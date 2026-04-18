"""Registry for executable skills."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

from .skill_base import Skill


class SkillRegistry:
    """Simple registry used by tasks and executor nodes."""

    def __init__(self):
        self._skills: Dict[str, Skill] = {}

    def register(self, skill: Skill) -> Skill:
        self._skills[skill.name] = skill
        return skill

    def register_many(self, skills: Iterable[Skill]) -> None:
        for skill in skills:
            self.register(skill)

    def get(self, name: str) -> Skill:
        if name not in self._skills:
            raise KeyError(f"Unknown skill: {name}")
        return self._skills[name]

    def maybe_get(self, name: str) -> Optional[Skill]:
        return self._skills.get(name)

    def names(self) -> Iterable[str]:
        return sorted(self._skills)
