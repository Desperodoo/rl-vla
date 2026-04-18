from script_runtime.core import SkillStatus
from script_runtime.session import build_pick_place_session


class StubManiSkillBridge:
    def __init__(self, **kwargs):
        self.connected = False
        self.current_pose = [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]
        self.grasped = False
        self.kwargs = kwargs

    def connect(self):
        self.connected = True
        return {"ok": True}

    def initialize(self):
        self.connected = True
        return {"ok": True}

    def shutdown(self):
        self.connected = False
        return {"ok": True}

    def move_j(self, joints, speed=1.0):
        return {"ok": True, "command": {"type": "move_j", "joints": list(joints), "speed": speed}}

    def move_l(self, pose, speed=1.0):
        self.current_pose = list(pose)
        return {"ok": True, "command": {"type": "move_l", "pose": list(pose), "speed": speed}}

    def servo_delta(self, delta_pose):
        return {"ok": True, "command": {"type": "servo_delta", "delta_pose": list(delta_pose)}}

    def stop(self):
        return {"ok": True, "command": {"type": "stop"}}

    def open_gripper(self, width=None):
        self.grasped = False
        return {"ok": True, "command": {"type": "open_gripper", "width": width}}

    def close_gripper(self, width=None, guarded=False):
        self.grasped = True
        return {"ok": True, "command": {"type": "close_gripper", "width": width, "guarded": guarded}}

    def execute_grasp_phase(self, target_pose, context=None):
        self.current_pose = list(target_pose)
        self.grasped = True
        return {
            "ok": True,
            "action": "execute_grasp_phase",
            "command": {"type": "execute_grasp_phase", "target_pose": list(target_pose)},
        }

    def get_status(self):
        return {
            "ok": True,
            "connected": self.connected,
            "fault": False,
            "joint_positions": [0.0] * 8,
            "eef_pose": list(self.current_pose),
            "gripper_width": 0.0 if self.grasped else 0.04,
            "mode": "pd_ee_pose",
            "success": True,
            "is_grasped": self.grasped,
        }

    def refresh_world(self, blackboard):
        blackboard.update_world(
            robot={
                "joint_positions": [0.0] * 8,
                "eef_pose": list(self.current_pose),
                "gripper_width": 0.0 if self.grasped else 0.04,
                "fault_flag": False,
                "mode": "pd_ee_pose",
            },
            scene={"grasped": self.grasped},
        )
        return self.get_status()

    def get_object_pose(self):
        return [0.1, 0.0, 0.05, 0.0, 0.0, 0.0, 1.0]

    def get_grasp_candidates(self):
        return [
            {
                "pose": [0.1, 0.0, 0.05, 0.0, 0.0, 0.0, 1.0],
                "pregrasp_pose": [0.1, 0.0, 0.13, 0.0, 0.0, 0.0, 1.0],
                "score": 1.0,
            }
        ]

    def is_grasped(self):
        return self.grasped


def test_pick_place_session_can_use_maniskill_oracle_bridge():
    config = {
        "runtime": {"task_id": "maniskill-oracle-test"},
        "execution": {"active_source": "policy", "control_owner": "script_runtime"},
        "task_goal": {"task_name": "pick_cube_validation"},
        "scene": {
            "workspace_ready": True,
            "tracking_lost": False,
            "depth_anomaly": False,
            "detection_confidence": 1.0,
            "calibration_version": "oracle_state",
        },
        "poses": {
            "home_joints": [0.0] * 6,
            "pregrasp_pose": [0.1, 0.0, 0.13, 0.0, 0.0, 0.0, 1.0],
            "lift_pose": [0.1, 0.0, 0.20, 0.0, 0.0, 0.0, 1.0],
            "place_pose": [0.2, 0.1, 0.10, 0.0, 0.0, 0.0, 1.0],
            "retreat_pose": [0.2, 0.1, 0.18, 0.0, 0.0, 0.0, 1.0],
        },
        "gripper": {"open_width": 0.04, "close_width": 0.0},
    }
    bridge = StubManiSkillBridge()
    session = build_pick_place_session(config=config, sdk_bridge=bridge)
    session.adapters["maniskill"] = bridge
    result = session.run()

    assert result.status == SkillStatus.SUCCESS
    assert session.blackboard.world_state.scene.object_pose is not None
    assert session.blackboard.world_state.scene.grasped is False
