"""Uniform adapter interface for learned modules exported from rlft."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence


@dataclass
class LearnedModuleAdapter:
    """Base adapter used for proposal, scoring, residuals, and risk checks."""

    name: str = "learned_module"

    def predict(self, obs: Any, goal: Any, context: Any) -> Any:
        raise NotImplementedError

    def score(self, candidates: Sequence[Any], obs: Any, context: Any) -> List[float]:
        raise NotImplementedError

    def success_prob(self, obs: Any, context: Any) -> float:
        raise NotImplementedError

    def risk_prob(self, obs: Any, context: Any) -> float:
        raise NotImplementedError


@dataclass
class NullLearnedModuleAdapter(LearnedModuleAdapter):
    """Fallback adapter that leaves control flow unchanged."""

    name: str = "null_learned_module"

    def predict(self, obs: Any, goal: Any, context: Any) -> Any:
        return None

    def score(self, candidates: Sequence[Any], obs: Any, context: Any) -> List[float]:
        return [0.0 for _ in candidates]

    def success_prob(self, obs: Any, context: Any) -> float:
        return 0.0

    def risk_prob(self, obs: Any, context: Any) -> float:
        return 0.0
