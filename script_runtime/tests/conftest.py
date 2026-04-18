"""Fixtures for script_runtime tests."""

import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from script_runtime.adapters import CameraBridge, CameraSnapshot, MockSDKBridge, NullLearnedModuleAdapter, RosBridge
from script_runtime.core import TaskBlackboard, WorldState
from script_runtime.executors import TraceRecorder


@pytest.fixture()
def blackboard():
    world = WorldState()
    world.perception.detection_confidence = 0.9
    world.scene.workspace_ready = True
    return TaskBlackboard(world)


@pytest.fixture()
def adapters():
    return {
        "sdk": MockSDKBridge(),
        "ros": RosBridge(),
        "camera": CameraBridge(CameraSnapshot(calibration_version="test-calib")),
        "learned": NullLearnedModuleAdapter(),
    }


@pytest.fixture()
def trace_recorder():
    return TraceRecorder()
