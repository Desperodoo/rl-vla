"""Task abstraction for composed script policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ScriptTask:
    """High-level task wrapping a behavior tree or equivalent executor."""

    name: str
    goal: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    root: Optional[Any] = None

    def build(self, registry: Any) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError
