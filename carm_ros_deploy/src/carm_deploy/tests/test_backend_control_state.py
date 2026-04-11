"""Tests for lower-machine backend control_state behavior."""

import importlib
import os
import sys


BACKEND_SERVER_DIR = "/home/amax/rl-vla/backend/carm_backend/server"
if BACKEND_SERVER_DIR not in sys.path:
    sys.path.insert(0, BACKEND_SERVER_DIR)

import server  # noqa: E402
joystick_api = importlib.import_module("api.joystick")  # noqa: E402
joystick_basic = importlib.import_module("basic.joystick")  # noqa: E402


class FakeController:
    available = True

    def __init__(self):
        self._state = {
            "local_control_enabled": True,
            "control_owner": "lower_machine",
            "updated_at": 1.0,
        }

    def get_control_state(self):
        return dict(self._state)

    def set_control_state(self, local_control_enabled, control_owner):
        self._state = {
            "local_control_enabled": bool(local_control_enabled),
            "control_owner": control_owner,
            "updated_at": 2.0,
        }
        return dict(self._state)


class TestBackendControlStateHelpers:
    def test_validate_control_state(self):
        assert joystick_basic.validate_control_state(True, "lower_machine") is True
        assert joystick_basic.validate_control_state(False, "upper_machine") is True
        assert joystick_basic.validate_control_state(True, "upper_machine") is False
        assert joystick_basic.validate_control_state(False, "lower_machine") is False


class TestBackendControlStateApi:
    def setup_method(self):
        self.client = server.app.test_client()
        self.fake = FakeController()
        joystick_api.DEPENDENCIES_AVAILABLE = True
        joystick_api.get_joystick_controller = lambda: self.fake

    def test_get_control_state(self):
        resp = self.client.get("/api/joystick/control_state")
        body = resp.get_json()
        assert body["code"] == 1
        assert body["data"]["local_control_enabled"] is True
        assert body["data"]["control_owner"] == "lower_machine"

    def test_post_control_state(self):
        resp = self.client.post(
            "/api/joystick/control_state",
            json={"local_control_enabled": False, "control_owner": "upper_machine"},
        )
        body = resp.get_json()
        assert body["code"] == 1
        assert body["data"]["local_control_enabled"] is False
        assert body["data"]["control_owner"] == "upper_machine"

    def test_post_control_state_invalid_combination(self):
        def _raise(_, __):
            raise ValueError("invalid control_state combination")

        self.fake.set_control_state = _raise
        resp = self.client.post(
            "/api/joystick/control_state",
            json={"local_control_enabled": True, "control_owner": "upper_machine"},
        )
        body = resp.get_json()
        assert body["code"] == 0
