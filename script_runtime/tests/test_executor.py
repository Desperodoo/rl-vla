from script_runtime.core import FailureCode, Skill, SkillContext, SkillRegistry, SkillStatus
from script_runtime.core.result_types import SkillResult
from script_runtime.executors import RetryNode, SequenceNode, SkillNode, TraceRecorder, TreeExecutor


class FlakySkill(Skill):
    def __init__(self):
        super().__init__(name="FlakySkill", timeout_s=1.0, failure_code=FailureCode.TIMEOUT)
        self.calls = 0

    def run(self, context):
        self.calls += 1
        if self.calls < 2:
            return SkillResult.failure(FailureCode.TIMEOUT, message="first attempt fails")
        return SkillResult.success(call_count=self.calls)


class AlwaysSuccessSkill(Skill):
    def __init__(self):
        super().__init__(name="AlwaysSuccessSkill")

    def run(self, context):
        return SkillResult.success()


def test_retry_node_recovers_on_second_attempt(blackboard, adapters):
    registry = SkillRegistry()
    registry.register(FlakySkill())
    root = RetryNode("retry_flaky", SkillNode("flaky", "FlakySkill"), max_attempts=2)
    executor = TreeExecutor(root=root, registry=registry)
    context = SkillContext(blackboard=blackboard, adapters=adapters, task_id="retry-task")
    result = executor.run(context)
    assert result.status == SkillStatus.SUCCESS


def test_sequence_stops_on_first_failure(blackboard, adapters):
    registry = SkillRegistry()
    registry.register(FlakySkill())
    registry.register(AlwaysSuccessSkill())
    root = SequenceNode(
        "seq",
        [
            SkillNode("flaky", "FlakySkill"),
            SkillNode("success", "AlwaysSuccessSkill"),
        ],
    )
    executor = TreeExecutor(root=root, registry=registry)
    context = SkillContext(blackboard=blackboard, adapters=adapters, task_id="seq-task")
    result = executor.run(context)
    assert result.status == SkillStatus.FAILURE
    assert result.failure_code == FailureCode.TIMEOUT
