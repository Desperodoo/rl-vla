"""Learned-module backed helper skills."""

from __future__ import annotations

from script_runtime.core.failure_codes import FailureCode
from script_runtime.core.result_types import SkillResult
from script_runtime.core.skill_base import Skill, SkillContext


class ScorePlaceCandidates(Skill):
    def __init__(self):
        super().__init__(name="ScorePlaceCandidates", timeout_s=0.5, failure_code=FailureCode.NO_OBJECT_DETECTED)

    def run(self, context: SkillContext) -> SkillResult:
        candidates = context.world_state.learned.place_candidates
        learned = context.adapters.get("learned")
        if learned is None or not candidates:
            return SkillResult.success(place_candidates=candidates)
        scores = learned.score(candidates, context.world_state, context)
        ranked = []
        for candidate, score in zip(candidates, scores):
            item = dict(candidate)
            item["score"] = score
            ranked.append(item)
        ranked.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        context.blackboard.update_world(learned={"place_candidates": ranked})
        return SkillResult.success(place_candidates=ranked)


class SuccessRiskCheck(Skill):
    def __init__(self):
        super().__init__(name="SuccessRiskCheck", timeout_s=0.2, failure_code=FailureCode.UNKNOWN)

    def run(self, context: SkillContext) -> SkillResult:
        learned = context.adapters.get("learned")
        if learned is None:
            return SkillResult.success(success_probability=0.0, risk_score=0.0)
        success_probability = learned.success_prob(context.world_state, context)
        risk_score = learned.risk_prob(context.world_state, context)
        context.blackboard.update_world(
            learned={
                "success_probability": success_probability,
                "risk_score": risk_score,
            }
        )
        return SkillResult.success(success_probability=success_probability, risk_score=risk_score)
