"""session action stream and error logging for debugging / replay"""

from __future__ import annotations

import json
import logging
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, TextIO

from pyray import Transform

from builder.heightfield.heightmap_catalog import HeightmapId
from builder.scene.scene_types import MeshId


actions_filename: str = "actions.jsonl"
error_log_filename: str = "builder.log"
logger_name: str = "builder"


def default_logs_dir() -> Path:
    """logs directory under the process working directory"""
    return Path.cwd() / "logs"


def utc_now_iso() -> str:
    """current utc timestamp as iso-8601"""
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    """convert common domain values to json-serializable form"""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, MeshId):
        return value.name
    if isinstance(value, HeightmapId):
        return value.name
    if isinstance(value, IntEnum):
        return value.name
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, Transform):
        return transform_payload(value)
    return repr(value)


def transform_payload(transform: Transform) -> dict[str, list[float]]:
    """json-friendly local transform components"""
    return {
        "translation": [
            float(transform.translation.x),
            float(transform.translation.y),
            float(transform.translation.z),
        ],
        "rotation": [
            float(transform.rotation.x),
            float(transform.rotation.y),
            float(transform.rotation.z),
            float(transform.rotation.w),
        ],
        "scale": [
            float(transform.scale.x),
            float(transform.scale.y),
            float(transform.scale.z),
        ],
    }


@dataclass
class SessionJournal:
    """cleared-on-start action jsonl plus error log to stderr and file"""

    logs_dir: Path = field(default_factory=default_logs_dir)
    actions_path: Path = field(init=False)
    error_log_path: Path = field(init=False)
    _actions_file: TextIO | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.actions_path = self.logs_dir / actions_filename
        self.error_log_path = self.logs_dir / error_log_filename

    def begin_session(self) -> None:
        """create logs dir, truncate action + error files, attach stderr logging"""
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        if self._actions_file is not None:
            self._actions_file.close()
            self._actions_file = None
        self.actions_path.write_text("", encoding="utf-8")
        self.error_log_path.write_text("", encoding="utf-8")
        self._actions_file = self.actions_path.open("a", encoding="utf-8")
        configure_error_logging(self.error_log_path)
        record = self.record
        record("session_start", logs_dir=str(self.logs_dir.resolve()))

    def close(self) -> None:
        """flush and close the action stream"""
        if self._actions_file is not None:
            self._actions_file.close()
            self._actions_file = None

    def record(self, action_type: str, **fields: Any) -> None:
        """append one user-action event (not camera motion)"""
        if self._actions_file is None:
            return
        payload: dict[str, Any] = {
            "ts": utc_now_iso(),
            "type": action_type,
        }
        key: str
        value: Any
        for key, value in fields.items():
            payload[key] = json_safe(value)
        self._actions_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._actions_file.flush()


def configure_error_logging(error_log_path: Path) -> None:
    """log errors to stderr and the session error file"""
    log: logging.Logger = logging.getLogger(logger_name)
    log.handlers.clear()
    log.setLevel(logging.DEBUG)
    log.propagate = False
    formatter: logging.Formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stderr_handler: logging.StreamHandler[Any] = logging.StreamHandler()
    stderr_handler.setLevel(logging.INFO)
    stderr_handler.setFormatter(formatter)
    file_handler: logging.FileHandler = logging.FileHandler(
        error_log_path, mode="a", encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    log.addHandler(stderr_handler)
    log.addHandler(file_handler)


def builder_logger() -> logging.Logger:
    """return the shared builder logger"""
    return logging.getLogger(logger_name)


def log_exception(message: str, exc: BaseException | None = None) -> None:
    """write an error with traceback to stderr and the session log file"""
    log: logging.Logger = builder_logger()
    if exc is None:
        log.error("%s\n%s", message, traceback.format_exc())
        return
    log.error("%s: %s\n%s", message, exc, "".join(traceback.format_exception(exc)))
