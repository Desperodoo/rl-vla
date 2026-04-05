from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence


DEFAULT_CAMERA_TOPICS = ['/camera/color/image_raw']
DEFAULT_SYNC_SLOP = 0.02


@dataclass(frozen=True)
class CameraConfig:
    topics: list[str]
    names: list[str]
    primary_name: str
    sync_slop_sec: float

    @property
    def primary_index(self) -> int:
        return self.names.index(self.primary_name)

    def to_runtime_dict(self) -> dict[str, Any]:
        return {
            'camera_topics': list(self.topics),
            'camera_names': list(self.names),
            'primary_camera': self.primary_name,
            'sync_slop': self.sync_slop_sec,
        }


RosParamGetter = Callable[[str], Any]
RosHasParam = Callable[[str], bool]
RosWarnLogger = Callable[[str], None]


def topic_to_camera_name(topic: str, index: int) -> str:
    name = str(topic).strip('/').replace('/', '_').replace('-', '_')
    if not name:
        name = f'camera_{index}'
    return name


def _normalize_csv_or_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = value.split(',')
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        items = list(value)
    else:
        items = [value]
    return [str(item).strip() for item in items if str(item).strip()]


def normalize_camera_topics(value: Any) -> list[str]:
    topics = _normalize_csv_or_list(value)
    return topics or list(DEFAULT_CAMERA_TOPICS)


def normalize_camera_names(value: Any) -> list[str]:
    return _normalize_csv_or_list(value)


def _resolve_raw_value(
    raw_config: Optional[Mapping[str, Any]],
    key: str,
    ros_has_param: Optional[RosHasParam],
    ros_get_param: Optional[RosParamGetter],
) -> Any:
    if raw_config is not None and key in raw_config and raw_config[key] is not None:
        explicit_value = raw_config[key]
    else:
        explicit_value = None

    if ros_has_param is not None and ros_get_param is not None and ros_has_param(f'~{key}'):
        return ros_get_param(f'~{key}')
    if explicit_value is not None:
        return explicit_value
    return None


def resolve_camera_config(
    raw_config: Optional[Mapping[str, Any]] = None,
    *,
    ros_has_param: Optional[RosHasParam] = None,
    ros_get_param: Optional[RosParamGetter] = None,
    logwarn: Optional[RosWarnLogger] = None,
) -> CameraConfig:
    topics_raw = _resolve_raw_value(raw_config, 'camera_topics', ros_has_param, ros_get_param)
    names_raw = _resolve_raw_value(raw_config, 'camera_names', ros_has_param, ros_get_param)
    primary_raw = _resolve_raw_value(raw_config, 'primary_camera', ros_has_param, ros_get_param)
    sync_slop_raw = _resolve_raw_value(raw_config, 'sync_slop', ros_has_param, ros_get_param)

    topics = normalize_camera_topics(topics_raw)
    names = normalize_camera_names(names_raw)
    if not names:
        names = [topic_to_camera_name(topic, idx) for idx, topic in enumerate(topics)]
    elif len(names) != len(topics):
        if logwarn is not None:
            logwarn(
                f"camera_names count ({len(names)}) does not match camera_topics count ({len(topics)}); "
                'falling back to topic-derived names'
            )
        names = [topic_to_camera_name(topic, idx) for idx, topic in enumerate(topics)]

    primary_name = str(primary_raw).strip() if primary_raw is not None else ''
    if not primary_name:
        primary_name = names[0]
    elif primary_name not in names:
        if logwarn is not None:
            logwarn(
                f"primary_camera '{primary_name}' not found in camera_names {names}, "
                f"fallback to '{names[0]}'"
            )
        primary_name = names[0]

    sync_slop = DEFAULT_SYNC_SLOP if sync_slop_raw is None else float(sync_slop_raw)
    if sync_slop <= 0:
        raise ValueError(f'sync_slop must be > 0, got {sync_slop}')

    return CameraConfig(
        topics=topics,
        names=names,
        primary_name=primary_name,
        sync_slop_sec=sync_slop,
    )
