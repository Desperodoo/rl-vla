from script_runtime.core import FailureCode, Skill, SkillContext, SkillRegistry, SkillStatus
from script_runtime.core.result_types import SkillResult
from script_runtime.executors import TreeExecutor
from script_runtime.skills.checks import CheckContact, CheckGrasp, CheckSceneReady, CheckTaskSuccess, WaitForObjectStable
from script_runtime.skills.gripper import ExecuteGraspPhase, OpenGripper
from script_runtime.skills.motion import GoHome, GoPregrasp, Lift, PlaceApproach, PlaceRelease, Retreat
from script_runtime.skills.perception import GetGraspCandidates, GetObjectPose, ReacquirePerception
from script_runtime.skills.recovery import HumanTakeover, RetryWithNextCandidate, SafeRetreat
from script_runtime.tasks.pick_place import PickPlaceTask


class StubMoveJ(GoHome):
    def __init__(self):
        super().__init__(home_joints=[0.0] * 6)


class StubMoveL(GoPregrasp):
    def __init__(self, name, key):
        super().__init__(target_pose=None)
        self.name = name
        self.key = key

    def run(self, context: SkillContext):
        pose = context.blackboard.get(self.key)
        if pose is None:
            return SkillResult.failure(FailureCode.NO_IK, message=f"missing {self.key}")
        return super().run(context)


def build_registry():
    registry = SkillRegistry()
    registry.register(CheckSceneReady())
    registry.register(GetObjectPose())
    registry.register(GetGraspCandidates())
    registry.register(ReacquirePerception())
    registry.register(CheckGrasp())
    registry.register(CheckTaskSuccess())
    registry.register(ExecuteGraspPhase())
    registry.register(OpenGripper())
    registry.register(SafeRetreat())
    registry.register(RetryWithNextCandidate())
    registry.register(HumanTakeover())

    registry.register(StubMoveL("GoPregrasp", "pregrasp_pose"))
    lift = StubMoveL("Lift", "lift_pose")
    registry.register(lift)
    place = StubMoveL("PlaceApproach", "place_pose")
    registry.register(place)
    place_release = StubMoveL("PlaceRelease", "place_release_pose")
    registry.register(place_release)
    retreat = StubMoveL("Retreat", "retreat_pose")
    registry.register(retreat)
    registry.register(StubMoveJ())
    return registry


def seed_nominal_blackboard(blackboard):
    blackboard.set("object_pose", [0.1, 0.0, 0.2, 0, 0, 0, 1, 0.04])
    blackboard.set("grasp_candidates", [{"pose": [0.1, 0.0, 0.2, 0, 0, 0, 1, 0.04]}])
    blackboard.set("pregrasp_pose", [0.2, 0.0, 0.3, 0, 0, 0, 1, 0.04])
    blackboard.set("lift_pose", [0.2, 0.0, 0.35, 0, 0, 0, 1, 0.02])
    blackboard.set("place_pose", [0.3, 0.1, 0.2, 0, 0, 0, 1, 0.02])
    blackboard.set("place_release_pose", [0.3, 0.1, 0.18, 0, 0, 0, 1, 0.02])
    blackboard.set("retreat_pose", [0.25, 0.1, 0.3, 0, 0, 0, 1, 0.07])
    blackboard.update_world(scene={"grasped": True})


def test_pick_place_nominal_flow(blackboard, adapters):
    registry = build_registry()
    seed_nominal_blackboard(blackboard)
    task = PickPlaceTask()
    root = task.build(registry)
    executor = TreeExecutor(root=root, registry=registry)
    context = SkillContext(blackboard=blackboard, adapters=adapters, task_id="pick-place-1")
    result = executor.run(context)
    assert result.status == SkillStatus.SUCCESS


def test_pick_place_fails_without_candidates(blackboard, adapters):
    registry = build_registry()
    blackboard.set("object_pose", [0.1, 0.0, 0.2, 0, 0, 0, 1, 0.04])
    task = PickPlaceTask()
    root = task.build(registry)
    executor = TreeExecutor(root=root, registry=registry)
    context = SkillContext(blackboard=blackboard, adapters=adapters, task_id="pick-place-2")
    result = executor.run(context)
    assert result.status == SkillStatus.FAILURE
    assert result.failure_code == FailureCode.NO_GRASP_CANDIDATE
