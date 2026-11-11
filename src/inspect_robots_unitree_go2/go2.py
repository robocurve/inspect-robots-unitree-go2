"""Go2 (Air) backend over the WebRTC link the Unitree phone app uses.

The Air has no official SDK. ``unitree_webrtc_connect`` (an asyncio library)
reaches the robot's sport controller over a WebRTC data channel. This module
keeps all of that behind two seams:

- ``WebRTCLink`` owns the asyncio loop (on its own thread) and the library
  connection. It sends sport requests, streams ``Move`` without waiting for a
  reply, subscribes to ``rt/lf/sportmodestate`` and keeps the latest camera
  frame. Every send first checks that the data channel is still open, because
  the library silently drops messages on a closed channel.
- ``Go2WebRTCDriver`` is the ``LeggedDriver`` the embodiment talks to. It never
  stands the robot up or lies it down: it refuses a robot that is not already
  standing, sends ``BalanceStand`` once (so ``Move`` is accepted), streams
  ``Move``, and refuses stale state or stale video.

Nothing documents whether the Go2 stops on its own when ``Move`` stops
arriving, so the embodiment's streamer is the watchdog: it sends
``Move(0, 0, 0)`` whenever a step's deadline passes, and ``disconnect`` sends
``StopMove``. If this host dies mid-move, the robot may keep its last
velocity: keep the area clear and a way to stop it at hand.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import math
import os
import threading
import time
from collections.abc import Callable, Coroutine, Mapping
from typing import Any, Protocol, TypeVar

import numpy as np
import numpy.typing as npt

from inspect_robots_unitree_go2.config import Go2Config
from inspect_robots_unitree_go2.driver import LeggedState

GO2_WEBRTC_INSTALL = (
    'pip install "inspect-robots-unitree-go2[hardware]"  (installs unitree_webrtc_connect)'
)

SPORT_TOPIC = "rt/api/sport/request"
STATE_TOPIC = "rt/lf/sportmodestate"
MOTION_SWITCHER_TOPIC = "rt/api/motion_switcher/request"

BALANCE_STAND = 1002
STOP_MOVE = 1003
MOVE = 1008
#: Motion switcher: read the current mode without changing it.
MOTION_MODE_QUERY = 1001

Vec = npt.NDArray[np.float64]
Frame = npt.NDArray[np.uint8]
StateCallback = Callable[[Mapping[str, Any]], None]
T = TypeVar("T")


class Go2Link(Protocol):
    """What the driver needs from the transport."""

    def request(self, topic: str, api_id: int, parameter: Mapping[str, Any] | None = None) -> Any:
        """Send a request and wait for its reply; raise on a non-zero status."""
        ...

    def send_move(self, forward: float, left: float, turn: float) -> None:
        """Send one ``Move`` without waiting for a reply."""
        ...

    def latest_frame(self) -> tuple[float, Frame] | None:
        """The newest camera frame (RGB) and its arrival time, if any."""
        ...

    def close(self) -> None:
        """Disconnect and stop the transport."""
        ...


# ---------------------------------------------------------------------- transport


class WebRTCLink:
    """Run a ``UnitreeWebRTCConnection`` on a private asyncio loop thread.

    ``conn`` is the library connection (or a test double with the same
    surface: ``connect``, ``disconnect``, ``datachannel.data_channel_opened``,
    ``datachannel.pub_sub.{publish_request_new, publish_without_callback,
    subscribe}`` and ``video.{switchVideoChannel, add_track_callback}``).
    """

    def __init__(self, conn: Any, *, clock: Callable[[], float], request_timeout_s: float) -> None:
        self._conn = conn
        self._clock = clock
        self._timeout = request_timeout_s
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="go2-webrtc", daemon=True
        )
        self._thread.start()
        self._frame_lock = threading.Lock()
        self._frame: tuple[float, Frame] | None = None
        self._ids = itertools.count(int(time.time() * 1000) % 1_000_000_000)
        self._closed = False

    # ------------------------------------------------------------------ lifecycle

    def start(self, on_state: StateCallback, *, connect_timeout_s: float, video: bool) -> None:
        """Connect, subscribe to sport state, and (optionally) turn on video."""

        def deliver(message: Mapping[str, Any]) -> None:
            data = message.get("data") if isinstance(message, Mapping) else None
            if isinstance(data, Mapping):
                on_state(data)

        async def go() -> None:
            await self._conn.connect()
            self._require_open()
            self._conn.datachannel.pub_sub.subscribe(STATE_TOPIC, deliver)
            if video:
                self._conn.video.add_track_callback(self._read_track)
                self._conn.video.switchVideoChannel(True)

        self._run(go(), connect_timeout_s)

    def close(self) -> None:
        """Disconnect (bounded wait), then stop the loop thread. Idempotent."""
        if self._closed:
            return
        self._closed = True
        try:
            self._run(self._conn.disconnect(), self._timeout)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=self._timeout)

    # ------------------------------------------------------------------ Go2Link

    def request(self, topic: str, api_id: int, parameter: Mapping[str, Any] | None = None) -> Any:
        """Send a request and wait up to ``request_timeout_s`` for its reply."""
        options: dict[str, Any] = {"api_id": api_id}
        if parameter is not None:
            options["parameter"] = dict(parameter)

        async def go() -> Any:
            self._require_open()
            return await self._conn.datachannel.pub_sub.publish_request_new(topic, options)

        reply = self._run(go(), self._timeout)
        code = _status_code(reply)
        if code not in (None, 0):
            raise RuntimeError(f"Go2 request {api_id} on {topic} failed with status {code}")
        return reply

    def send_move(self, forward: float, left: float, turn: float) -> None:
        """Send ``Move`` as a no-reply request (the library's streaming pattern)."""
        payload = {
            "header": {
                "identity": {"id": next(self._ids), "api_id": MOVE},
                "policy": {"priority": 0, "noreply": True},
            },
            "parameter": json.dumps({"x": forward, "y": left, "z": turn}),
            "binary": [],
        }

        async def go() -> None:
            self._require_open()
            self._conn.datachannel.pub_sub.publish_without_callback(SPORT_TOPIC, payload, "req")

        self._run(go(), self._timeout)

    def latest_frame(self) -> tuple[float, Frame] | None:
        """The newest camera frame and its arrival time on ``clock``."""
        with self._frame_lock:
            return self._frame

    # ------------------------------------------------------------------ helpers

    def _require_open(self) -> None:
        if not getattr(self._conn.datachannel, "data_channel_opened", False):
            raise RuntimeError("Go2 WebRTC data channel is closed")

    async def _read_track(self, track: Any) -> None:
        """Keep the newest frame from the video track until it ends."""
        while True:
            frame = await track.recv()
            image = np.asarray(frame.to_ndarray(format="rgb24"), dtype=np.uint8)
            with self._frame_lock:
                self._frame = (self._clock(), image)

    def _run(self, coro: Coroutine[Any, Any, T], timeout_s: float) -> T:
        if self._closed and not self._thread.is_alive():
            coro.close()
            raise RuntimeError("Go2 WebRTC link is closed")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=timeout_s)
        except TimeoutError:
            future.cancel()
            raise TimeoutError(f"Go2 WebRTC call timed out after {timeout_s} s") from None


def _status_code(reply: Any) -> Any:
    """``reply['data']['header']['status']['code']`` when present."""
    node = reply
    for key in ("data", "header", "status", "code"):
        if not isinstance(node, Mapping) or key not in node:
            return None
        node = node[key]
    return node


# ---------------------------------------------------------------------- driver


class Go2WebRTCDriver:
    """``LeggedDriver`` for a Go2 over a ``Go2Link``.

    ``on_sport_state`` runs on the link's loop thread; one lock guards the
    shared state.
    """

    def __init__(self, cfg: Go2Config, *, link: Go2Link, clock: Callable[[], float]) -> None:
        self._cfg = cfg
        self._link = link
        self._clock = clock
        self._lock = threading.Lock()
        self._state: tuple[float, LeggedState, float, Any] | None = None
        self._bad_sample: str | None = None
        self._disconnected = False

    # ------------------------------------------------------------------ state

    def on_sport_state(self, data: Mapping[str, Any]) -> None:
        """Store one ``rt/lf/sportmodestate`` sample with its arrival time.

        Position and velocity are in the odometry frame; the velocity is
        rotated into the body frame with the sample's IMU yaw.
        """
        try:
            imu = data["imu_state"]
            rpy = _finite_vec(imu["rpy"], 3, "imu_state.rpy")
            position = _finite_vec(data["position"], 3, "position")
            world_v = _finite_vec(data["velocity"], 3, "velocity")
            yaw_speed = _finite_float(data["yaw_speed"], "yaw_speed")
            height = _finite_float(data["body_height"], "body_height")
        except (KeyError, TypeError, ValueError) as exc:
            with self._lock:
                self._bad_sample = f"malformed sportmodestate sample: {exc!r}"
            return
        yaw = float(rpy[2])
        cos, sin = math.cos(yaw), math.sin(yaw)
        vx, vy = float(world_v[0]), float(world_v[1])
        state = LeggedState(
            pose=np.asarray((position[0], position[1], yaw), dtype=np.float64),
            velocity=np.asarray(
                (cos * vx + sin * vy, -sin * vx + cos * vy, yaw_speed),
                dtype=np.float64,
            ),
            rpy=rpy,
        )
        with self._lock:
            self._state = (self._clock(), state, height, data.get("mode"))
            self._bad_sample = None

    def _fresh_locked(self) -> tuple[LeggedState, float, Any]:
        if self._state is None:
            raise RuntimeError(self._bad_sample or "no Go2 sport state received yet")
        age = self._clock() - self._state[0]
        if age > self._cfg.state_stale_s:
            detail = f" (last error: {self._bad_sample})" if self._bad_sample else ""
            raise RuntimeError(f"Go2 sport state is {age:.2f} s old; stopping{detail}")
        return self._state[1], self._state[2], self._state[3]

    # ------------------------------------------------------------------ connect

    def wait_for_state(self, timeout_s: float, sleep: Callable[[float], None]) -> None:
        """Block until a fresh state sample arrives, or raise after ``timeout_s``."""
        deadline = self._clock() + timeout_s
        while True:
            with self._lock:
                try:
                    self._fresh_locked()
                    return
                except RuntimeError as exc:
                    error = exc
            if self._clock() >= deadline:
                raise RuntimeError(f"no fresh Go2 sport state within {timeout_s} s: {error}")
            sleep(0.05)

    def check_ready(self, sleep: Callable[[float], None]) -> None:
        """Refuse a robot that is not standing, then unlock ``Move`` with BalanceStand."""
        self.wait_for_state(self._cfg.connect_timeout_s, sleep)
        with self._lock:
            _, height, mode = self._fresh_locked()
        if height < self._cfg.min_body_height_m:
            raise RuntimeError(
                f"Go2 body height is {height:.2f} m (mode {mode}), below "
                f"{self._cfg.min_body_height_m} m: it is not standing. Stand it up with the "
                "remote or the app (then close the app) and retry."
            )
        self._link.request(SPORT_TOPIC, BALANCE_STAND)

    # ------------------------------------------------------------------ LeggedDriver

    def send_velocity(self, forward: float, left: float, turn: float) -> None:
        """Stream one ``Move``. The embodiment re-sends it at ``stream_hz``."""
        self._link.send_move(float(forward), float(left), float(turn))

    def stand(self) -> None:
        """Command zero velocity. The robot keeps balancing in place."""
        self._link.send_move(0.0, 0.0, 0.0)

    def read_state(self) -> LeggedState:
        """The latest fresh state; raises when it is missing or stale."""
        with self._lock:
            state, _, _ = self._fresh_locked()
        return LeggedState(
            pose=state.pose.copy(), velocity=state.velocity.copy(), rpy=state.rpy.copy()
        )

    def read_frame(self) -> Frame:
        """The latest front-camera frame, resized to the declared shape."""
        latest = self._link.latest_frame()
        if latest is None:
            raise RuntimeError("no Go2 camera frame received yet")
        stamp, frame = latest
        age = self._clock() - stamp
        if age > self._cfg.frame_stale_s:
            raise RuntimeError(f"Go2 camera frame is {age:.2f} s old")
        return resize_nearest(frame, self._cfg.camera_height, self._cfg.camera_width)

    def disconnect(self) -> None:
        """Send ``StopMove`` (best effort), then close the link. Idempotent."""
        if self._disconnected:
            return
        self._disconnected = True
        try:
            with contextlib.suppress(Exception):  # the link may already be gone
                self._link.request(SPORT_TOPIC, STOP_MOVE)
        finally:
            self._link.close()


# ---------------------------------------------------------------------- helpers


def _finite_vec(values: Any, size: int, name: str) -> Vec:
    vec = np.asarray(values, dtype=np.float64)
    if vec.shape != (size,) or not np.all(np.isfinite(vec)):
        raise ValueError(f"{name} is not {size} finite numbers: {values!r}")
    return vec


def _finite_float(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} is not finite: {value!r}")
    return number


def resize_nearest(frame: Frame, height: int, width: int) -> Frame:
    """Nearest-neighbour resize of an ``(H, W, 3)`` uint8 image (no OpenCV needed)."""
    image = np.asarray(frame, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3 or image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(f"Go2 camera frame has shape {image.shape}, expected (H, W, 3)")
    if image.shape[:2] == (height, width):
        return image
    rows = (np.arange(height) * image.shape[0]) // height
    cols = (np.arange(width) * image.shape[1]) // width
    return np.ascontiguousarray(image[rows][:, cols])


def read_aes_key(cfg: Go2Config, environ: Mapping[str, str] = os.environ) -> str | None:
    """The AES key from ``cfg.aes_key_env``, or None (firmware before 1.1.15 needs none)."""
    key = environ.get(cfg.aes_key_env, "").strip()
    if not key:
        return None
    if len(key) != 32 or any(c not in "0123456789abcdefABCDEF" for c in key):
        raise ValueError(f"{cfg.aes_key_env} must be 32 hex characters")
    return key


def connect_go2(
    cfg: Go2Config,
    conn: Any,
    *,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    video: bool = True,
    ready_check: bool = True,
) -> Go2WebRTCDriver:
    """Start a link on ``conn``, wire state to a driver, and check readiness.

    On any failure the link is closed before the error propagates.
    """
    link = WebRTCLink(conn, clock=clock, request_timeout_s=cfg.request_timeout_s)
    driver = Go2WebRTCDriver(cfg, link=link, clock=clock)
    try:
        link.start(driver.on_sport_state, connect_timeout_s=cfg.connect_timeout_s, video=video)
        if ready_check:
            driver.check_ready(sleep)
    except BaseException:
        with contextlib.suppress(Exception):
            link.close()
        raise
    return driver


def make_connection(cfg: Go2Config) -> Any:  # pragma: no cover - needs the library
    """Build the library connection the config describes (does not connect)."""
    try:
        from unitree_webrtc_connect.webrtc_driver import (
            UnitreeWebRTCConnection,
            WebRTCConnectionMethod,
        )
    except ImportError as exc:
        raise RuntimeError(
            f"Go2 backend needs unitree_webrtc_connect: {GO2_WEBRTC_INSTALL}"
        ) from exc
    key = read_aes_key(cfg)
    if cfg.connection == "local_ap":
        return UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalAP, aes_128_key=key)
    if not cfg.robot_ip and not cfg.serial_number:
        raise ValueError("connection=local_sta needs robot_ip or serial_number")
    return UnitreeWebRTCConnection(
        WebRTCConnectionMethod.LocalSTA,
        ip=cfg.robot_ip or None,
        serialNumber=cfg.serial_number or None,
        aes_128_key=key,
    )


def build_go2_webrtc_driver(cfg: Go2Config) -> Go2WebRTCDriver:  # pragma: no cover - hardware
    """Connect to a real Go2 and return a ready driver."""
    return connect_go2(cfg, make_connection(cfg), clock=time.monotonic, sleep=time.sleep)
