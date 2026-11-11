"""Frozen walking-robot configurations (shared base, Go2) and the spaces derived from them.

The action is a per-control-step body displacement ``(forward m, left m,
turn rad)``. It is declared as ``joint_delta`` because Inspect Robots has no
base-motion control mode yet; the three "joints" are the body's planar
degrees of freedom, and the operating notes tell policies so. The embodiment
converts each displacement to a body velocity held for one control period.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, ClassVar, Literal, Self

import numpy as np
import numpy.typing as npt
from inspect_robots.spaces import (
    ActionSemantics,
    Box,
    CameraSpec,
    ObservationSpace,
    StateField,
    StateSpec,
)

Backend = Literal["sdk", "kinematic", "isaaclab"]
BACKENDS: tuple[str, ...] = ("sdk", "kinematic", "isaaclab")

#: Action dimension names, in order. Positive forward is the front of the body,
#: positive left is the robot's left, positive turn is counter-clockwise seen
#: from above.
DIM_LABELS: tuple[str, str, str] = ("base_forward", "base_left", "base_turn")
ACTION_DIM = len(DIM_LABELS)

#: Proprioception keys. ``base_pose`` is odometry ``(x m, y m, yaw rad)``
#: relative to where the robot stood at the last reset, ``base_vel`` is the
#: measured body velocity ``(forward m/s, left m/s, turn rad/s)`` and
#: ``body_rpy`` is the IMU roll, pitch and yaw in radians.
POSE_KEY = "base_pose"
VEL_KEY = "base_vel"
RPY_KEY = "body_rpy"

ACTION_SEMANTICS = ActionSemantics(
    control_mode="joint_delta",
    frame="base",
    dim_labels=DIM_LABELS,
)

STATE_SPEC = StateSpec(
    fields=(
        StateField(key=POSE_KEY, shape=(3,), unit="m,m,rad"),
        StateField(key=VEL_KEY, shape=(3,), unit="m/s,m/s,rad/s"),
        StateField(key=RPY_KEY, shape=(3,), unit="rad"),
    )
)


@dataclass(frozen=True)
class LeggedConfig:
    """Settings shared by every walking robot. Every field is a scalar so ``-E key=value`` works.

    Speed caps bound the action box: the per-step displacement limit is the
    speed cap divided by ``control_hz``. ``command_timeout_s`` is the watchdog
    grace after a step's control period ends; when no new step arrives in that
    window (for example while an LLM is thinking) the commanded velocity drops
    to zero. ``geofence_m`` ends the trial and stops the robot once odometry
    puts it farther than this from the reset point; ``None`` disables it.
    Subclasses set robot-specific defaults and add their transport fields.
    """

    #: Isaac Lab camera mount: the body link and its offset in metres.
    ISAAC_CAMERA_LINK: ClassVar[str] = "base"
    ISAAC_CAMERA_POS: ClassVar[tuple[float, float, float]] = (0.32, 0.0, 0.05)

    backend: Backend = "sdk"
    control_hz: float = 10.0
    stream_hz: float = 50.0
    max_forward_speed: float = 0.5
    max_backward_speed: float = 0.3
    max_lateral_speed: float = 0.3
    max_turn_rate: float = 1.0
    command_timeout_s: float = 0.1
    geofence_m: float | None = 3.0
    camera_name: str = "front_cam"
    camera_height: int = 480
    camera_width: int = 640
    unattended: bool = False
    docs_extra: str = ""
    # Isaac Lab backend.
    isaac_task: str = "Isaac-Velocity-Flat-Unitree-Go2-Play-v0"
    isaac_checkpoint: str = ""
    headless: bool = True
    #: Place coloured boxes in the Isaac scene so camera-driven policies have targets.
    isaac_props: bool = True

    def make_action_box(self) -> Box:
        """The action space this config describes."""
        return action_box(self)

    def make_observation_space(self) -> ObservationSpace:
        """The observation space this config describes."""
        return observation_space(self)

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}; got {self.backend!r}")
        for name in (
            "control_hz",
            "stream_hz",
            "max_forward_speed",
            "max_backward_speed",
            "max_lateral_speed",
            "max_turn_rate",
        ):
            _require_positive(name, getattr(self, name))
        if self.stream_hz < self.control_hz:
            raise ValueError("stream_hz must be >= control_hz")
        if not math.isfinite(self.command_timeout_s) or self.command_timeout_s < 0:
            raise ValueError("command_timeout_s must be finite and >= 0")
        if self.geofence_m is not None:
            _require_positive("geofence_m", self.geofence_m)
        if self.camera_height <= 0 or self.camera_width <= 0:
            raise ValueError("camera_height and camera_width must be > 0")
        if not self.camera_name.strip():
            raise ValueError("camera_name must be non-empty")

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> Self:
        """Build from CLI-style kwargs, coercing strings and rejecting unknown keys."""
        known = {item.name: item for item in fields(cls)}
        unknown = sorted(set(kwargs) - set(known))
        if unknown:
            raise TypeError(f"unknown {cls.__name__} option(s): {', '.join(unknown)}")
        coerced: dict[str, Any] = {}
        for key, value in kwargs.items():
            coerced[key] = _coerce(key, value, cls.__dataclass_fields__[key].default)
        return cls(**coerced)

    @property
    def low(self) -> npt.NDArray[np.float64]:
        """Per-step displacement lower bounds ``(back m, right m, clockwise rad)``."""
        return -np.asarray(
            (self.max_backward_speed, self.max_lateral_speed, self.max_turn_rate),
            dtype=np.float64,
        ) / float(self.control_hz)

    @property
    def high(self) -> npt.NDArray[np.float64]:
        """Per-step displacement upper bounds ``(forward m, left m, counter-clockwise rad)``."""
        return np.asarray(
            (self.max_forward_speed, self.max_lateral_speed, self.max_turn_rate),
            dtype=np.float64,
        ) / float(self.control_hz)


@dataclass(frozen=True)
class Go2Config(LeggedConfig):
    """Unitree Go2 Air walked through its own sport controller over WebRTC.

    The Air has no official SDK; the real backend uses the WebRTC connection the
    Unitree phone app uses. ``connection`` is ``local_sta`` (robot on the same
    network, at ``robot_ip``, or found by ``serial_number``) or ``local_ap``
    (laptop on the robot's own hotspot, 192.168.12.1). Firmware 1.1.15 and
    later also need the robot's AES key, read from the environment variable
    named by ``aes_key_env`` so it never appears on a command line.

    The robot must already be standing (``body_height`` at least
    ``min_body_height_m``); nothing here stands it up. Speed caps start well
    below the robot's limits.
    """

    ISAAC_CAMERA_LINK: ClassVar[str] = "base"
    ISAAC_CAMERA_POS: ClassVar[tuple[float, float, float]] = (0.32, 0.0, 0.05)

    stream_hz: float = 20.0
    max_forward_speed: float = 0.4
    max_backward_speed: float = 0.25
    max_lateral_speed: float = 0.25
    max_turn_rate: float = 0.8
    #: The rough-terrain policy (run on flat ground) follows turn commands; the
    #: pretrained flat-terrain Go2 policy stalls on in-place turns.
    isaac_task: str = "Isaac-Velocity-Rough-Unitree-Go2-Play-v0"
    connection: str = "local_sta"
    robot_ip: str = ""
    serial_number: str = ""
    aes_key_env: str = "GO2_AES_128_KEY"
    connect_timeout_s: float = 20.0
    request_timeout_s: float = 3.0
    state_stale_s: float = 0.5
    frame_stale_s: float = 1.0
    min_body_height_m: float = 0.2

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.connection not in GO2_CONNECTIONS:
            raise ValueError(
                f"connection must be one of {GO2_CONNECTIONS}; got {self.connection!r}"
            )
        for name in (
            "connect_timeout_s",
            "request_timeout_s",
            "state_stale_s",
            "frame_stale_s",
            "min_body_height_m",
        ):
            _require_positive(name, getattr(self, name))


GO2_CONNECTIONS: tuple[str, ...] = ("local_sta", "local_ap")


def _require_positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and > 0; got {value!r}")


_NONE_WORDS = frozenset({"none", "null", ""})
_TRUE_WORDS = frozenset({"1", "true", "yes", "on"})
_FALSE_WORDS = frozenset({"0", "false", "no", "off"})


def _coerce(key: str, value: Any, default: Any) -> Any:
    """Coerce a CLI string to the type implied by the field's default."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if key == "geofence_m":
        return None if text.lower() in _NONE_WORDS else float(text)
    if isinstance(default, bool):
        lowered = text.lower()
        if lowered in _TRUE_WORDS:
            return True
        if lowered in _FALSE_WORDS:
            return False
        raise ValueError(f"{key} must be a boolean; got {value!r}")
    if isinstance(default, int):
        return int(text)
    if isinstance(default, float):
        return float(text)
    return value


def action_box(cfg: LeggedConfig | None = None) -> Box:
    """Build the 3-D per-step body displacement action space."""
    resolved = cfg if cfg is not None else LeggedConfig()
    return Box(
        shape=(ACTION_DIM,), low=resolved.low, high=resolved.high, semantics=ACTION_SEMANTICS
    )


def observation_space(cfg: LeggedConfig | None = None) -> ObservationSpace:
    """Build the one-camera, odometry-and-IMU observation contract."""
    resolved = cfg if cfg is not None else LeggedConfig()
    return ObservationSpace(
        cameras=(
            CameraSpec(
                name=resolved.camera_name,
                height=resolved.camera_height,
                width=resolved.camera_width,
                channels=3,
            ),
        ),
        state=STATE_SPEC,
    )
