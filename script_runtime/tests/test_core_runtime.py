from script_runtime.core import FailureCode, Skill, SkillContext, SkillRegistry, SkillStatus
from script_runtime.core.result_types import SkillResult


class DummySkill(Skill):
    def __init__(self):
        super().__init__(name="DummySkill", timeout_s=1.0, failure_code=FailureCode.UNKNOWN)

    def run(self, context):
        return SkillResult.success(answer=42)


def test_blackboard_snapshot_contains_world_and_scratch(blackboard):
    blackboard.set("foo", "bar")
    snap = blackboard.snapshot()
    assert "world_state" in snap
    assert snap["scratch"]["foo"] == "bar"


def test_skill_registry_roundtrip():
    registry = SkillRegistry()
    skill = DummySkill()
    registry.register(skill)
    assert registry.get("DummySkill") is skill


def test_skill_execute_emits_success_trace(blackboard, adapters, trace_recorder):
    skill = DummySkill()
    context = SkillContext(
        blackboard=blackboard,
        adapters=adapters,
        task_id="task-1",
        trace_sink=trace_recorder,
    )
    result = skill.execute(context)
    assert result.status == SkillStatus.SUCCESS
    assert trace_recorder.events[-1].skill_name == "DummySkill"
    assert trace_recorder.events[-1].result == "SUCCESS"
