"""ROS/logging compatibility layer.

Provides log_info, log_warn, log_err that use rospy when available,
falling back to stdlib logging otherwise. Eliminates the duplicated
try/except rospy pattern across inference_recorder.py and policy_loader.py.
"""

import logging

try:
    import rospy
    _HAS_ROSPY = True
except ImportError:
    _HAS_ROSPY = False

_logger = logging.getLogger("carm_deploy")


def log_info(msg: str) -> None:
    if _HAS_ROSPY:
        rospy.loginfo(msg)
    else:
        _logger.info(msg)


def log_warn(msg: str) -> None:
    if _HAS_ROSPY:
        rospy.logwarn(msg)
    else:
        _logger.warning(msg)


def log_err(msg: str) -> None:
    if _HAS_ROSPY:
        rospy.logerr(msg)
    else:
        _logger.error(msg)
