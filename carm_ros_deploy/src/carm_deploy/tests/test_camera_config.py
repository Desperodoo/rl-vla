import os
import sys
import types

import pytest


_CARM_DEPLOY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CARM_DEPLOY_ROOT not in sys.path:
    sys.path.insert(0, _CARM_DEPLOY_ROOT)

if 'rospy' not in sys.modules:
    _rospy_mock = types.ModuleType('rospy')
    _rospy_mock.logwarn = lambda *a, **kw: None
    sys.modules['rospy'] = _rospy_mock

from core.camera_config import (
    DEFAULT_CAMERA_TOPICS,
    DEFAULT_SYNC_SLOP,
    resolve_camera_config,
    topic_to_camera_name,
)


class TestTopicToCameraName:
    def test_derives_stable_name(self):
        assert topic_to_camera_name('/wrist/color/image_raw', 0) == 'wrist_color_image_raw'

    def test_falls_back_when_empty(self):
        assert topic_to_camera_name('', 2) == 'camera_2'


class TestResolveCameraConfig:
    def test_defaults(self):
        cfg = resolve_camera_config({})
        assert cfg.topics == DEFAULT_CAMERA_TOPICS
        assert cfg.names == ['camera_color_image_raw']
        assert cfg.primary_name == 'camera_color_image_raw'
        assert cfg.sync_slop_sec == DEFAULT_SYNC_SLOP

    def test_normalizes_csv_inputs(self):
        cfg = resolve_camera_config(
            {
                'camera_topics': '/cam1/color/image_raw, /cam2/color/image_raw',
                'camera_names': ' wrist , third ',
                'primary_camera': 'third',
                'sync_slop': '0.03',
            }
        )
        assert cfg.topics == ['/cam1/color/image_raw', '/cam2/color/image_raw']
        assert cfg.names == ['wrist', 'third']
        assert cfg.primary_name == 'third'
        assert cfg.sync_slop_sec == 0.03

    def test_mismatched_names_fall_back_to_topic_names(self):
        cfg = resolve_camera_config(
            {
                'camera_topics': ['/cam1/color/image_raw', '/cam2/color/image_raw'],
                'camera_names': ['only_one'],
            }
        )
        assert cfg.names == ['cam1_color_image_raw', 'cam2_color_image_raw']
        assert cfg.primary_name == 'cam1_color_image_raw'

    def test_invalid_primary_falls_back_to_first(self):
        cfg = resolve_camera_config(
            {
                'camera_topics': ['/cam1/color/image_raw'],
                'camera_names': ['wrist'],
                'primary_camera': 'third',
            }
        )
        assert cfg.primary_name == 'wrist'

    def test_sync_slop_must_be_positive(self):
        with pytest.raises(ValueError, match='sync_slop must be > 0'):
            resolve_camera_config({'sync_slop': 0})

    def test_ros_params_override_raw_config(self):
        params = {
            '~camera_topics': '/ros/camera/color/image_raw',
            '~primary_camera': 'ros_camera_color_image_raw',
        }
        cfg = resolve_camera_config(
            {'camera_topics': '/cli/camera/color/image_raw'},
            ros_has_param=params.__contains__,
            ros_get_param=params.__getitem__,
        )
        assert cfg.topics == ['/ros/camera/color/image_raw']
        assert cfg.primary_name == 'ros_camera_color_image_raw'
