"""Go2 WebRTC backend against a fake library connection (no robot, no network)."""

from __future__ import annotations

import asyncio
import json
import math
from typing import Any

import numpy as np
import pytest

from inspect_robots_unitree_go2 import Go2Config, Go2Embodiment
from inspect_robots_unitree_go2.go2 import (
    BALANCE_STAND,
    MOVE,
    SPORT_TOPIC,
    STATE_TOPIC,
    STOP_MOVE,
    Go2WebRTCDriver,
    WebRTCLink,
    _status_code,
    connect_go2,
    read_aes_key,
    resize_nearest,
)


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


class FakePubSub:
    def __init__(self, conn: FakeConn) -> None:
        self.conn = conn
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.sent: list[tuple[str, Any, Any]] = []
        self.subscriptions: dict[str, Any] = {}

    async def publish_request_new(self, topic: str, options: dict[str, Any]) -> Any:
        self.requests.append((topic, options))
        if self.conn.hang_requests:
            await asyncio.sleep(10)
        return self.conn.reply

    def publish_without_callback(self, topic: str, data: Any = None, msg_type: Any = None) -> None:
        self.sent.append((topic, data, msg_type))

    def subscribe(self, topic: str, callback: Any = None) -> None:
        self.subscriptions[topic] = callback


class FakeDataChannel:
    def __init__(self, conn: FakeConn) -> None:
        self.data_channel_opened = False
        self.pub_sub = FakePubSub(conn)


class FakeVideo:
    def __init__(self) -> None:
        self.callbacks: list[Any] = []
        self.switched: list[bool] = []

    def add_track_callback(self, callback: Any) -> None:
        self.callbacks.append(callback)

    def switchVideoChannel(self, switch: bool) -> None:
        self.switched.append(switch)


class FakeConn:
    def __init__(self, *, opens: bool = True, connect_error: Exception | None = None) -> None:
        self.datachannel = FakeDataChannel(self)
        self.video = FakeVideo()
        self.opens = opens
        self.connect_error = connect_error
        self.disconnects = 0
        self.disconnect_error: Exception | None = None
        self.reply: Any = {"data": {"header": {"status": {"code": 0}}}}
        self.hang_requests = False
        self.on_connect: Any = None

    async def connect(self) -> None:
        if self.connect_error is not None:
            raise self.connect_error
        self.datachannel.data_channel_opened = self.opens

    async def disconnect(self) -> None:
        self.disconnects += 1
        self.datachannel.data_channel_opened = False
        if self.disconnect_error is not None:
            raise self.disconnect_error

    def push_state(self, data: Any) -> None:
        self.datachannel.pub_sub.subscriptions[STATE_TOPIC]({"topic": STATE_TOPIC, "data": data})


def sample(
    *, x: float = 1.0, y: float = 2.0, yaw: float = math.pi / 2, height: float = 0.32
) -> dict[str, Any]:
    return {
        "mode": 1,
        "position": [x, y, 0.3],
        "velocity": [0.0, 0.5, 0.0],
        "yaw_speed": 0.1,
        "body_height": height,
        "imu_state": {"rpy": [0.01, -0.02, yaw]},
    }


def cfg(**kw: Any) -> Go2Config:
    return Go2Config(backend="sdk", camera_height=4, camera_width=6, **kw)


class FakeFrame:
    def __init__(self, image: np.ndarray) -> None:
        self.image = image

    def to_ndarray(self, format: str) -> np.ndarray:
        assert format == "rgb24"
        return self.image


class FakeTrack:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames = list(frames)

    async def recv(self) -> FakeFrame:
        if not self.frames:
            raise EOFError("track ended")
        return FakeFrame(self.frames.pop(0))


def run_track(link: WebRTCLink, conn: FakeConn, frames: list[np.ndarray]) -> None:
    future = asyncio.run_coroutine_threadsafe(
        conn.video.callbacks[0](FakeTrack(frames)),
        link._loop,
    )
    with pytest.raises(EOFError):
        future.result(timeout=2)


# ---------------------------------------------------------------------- link


def test_link_connects_subscribes_streams_and_closes() -> None:
    clock = Clock()
    conn = FakeConn()
    link = WebRTCLink(conn, clock=clock, request_timeout_s=1.0)
    states: list[Any] = []
    link.start(states.append, connect_timeout_s=1.0, video=True)
    assert conn.video.switched == [True]
    conn.push_state({"mode": 1})
    conn.datachannel.pub_sub.subscriptions[STATE_TOPIC]({"topic": STATE_TOPIC, "data": "junk"})
    conn.datachannel.pub_sub.subscriptions[STATE_TOPIC]("not a mapping")
    assert states == [{"mode": 1}]

    reply = link.request(SPORT_TOPIC, BALANCE_STAND)
    assert reply == conn.reply
    link.request(SPORT_TOPIC, 9, {"name": "x"})
    assert conn.datachannel.pub_sub.requests == [
        (SPORT_TOPIC, {"api_id": BALANCE_STAND}),
        (SPORT_TOPIC, {"api_id": 9, "parameter": {"name": "x"}}),
    ]

    link.send_move(0.2, -0.1, 0.3)
    topic, payload, msg_type = conn.datachannel.pub_sub.sent[-1]
    assert (topic, msg_type) == (SPORT_TOPIC, "req")
    assert payload["header"]["identity"]["api_id"] == MOVE
    assert payload["header"]["policy"]["noreply"] is True
    assert json.loads(payload["parameter"]) == {"x": 0.2, "y": -0.1, "z": 0.3}

    assert link.latest_frame() is None
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    run_track(link, conn, [image])
    clock.t = 101.0
    latest = link.latest_frame()
    assert latest is not None and latest[0] == 100.0 and latest[1].shape == (2, 3, 3)

    link.close()
    link.close()  # idempotent
    assert conn.disconnects == 1
    with pytest.raises(RuntimeError, match="closed"):
        link.send_move(0.0, 0.0, 0.0)


def test_link_refuses_a_closed_channel_and_bad_status() -> None:
    conn = FakeConn(opens=False)
    link = WebRTCLink(conn, clock=Clock(), request_timeout_s=1.0)
    with pytest.raises(RuntimeError, match="data channel is closed"):
        link.start(lambda _: None, connect_timeout_s=1.0, video=False)
    conn.opens = True
    link.start(lambda _: None, connect_timeout_s=1.0, video=False)
    assert conn.video.switched == []
    conn.reply = {"data": {"header": {"status": {"code": 3104}}}}
    with pytest.raises(RuntimeError, match="status 3104"):
        link.request(SPORT_TOPIC, BALANCE_STAND)
    conn.datachannel.data_channel_opened = False
    with pytest.raises(RuntimeError, match="data channel is closed"):
        link.send_move(0.1, 0.0, 0.0)
    link.close()


def test_link_times_out_a_silent_request() -> None:
    conn = FakeConn()
    link = WebRTCLink(conn, clock=Clock(), request_timeout_s=0.05)
    link.start(lambda _: None, connect_timeout_s=1.0, video=False)
    conn.hang_requests = True
    with pytest.raises(TimeoutError, match="timed out"):
        link.request(SPORT_TOPIC, BALANCE_STAND)
    link.close()


def test_status_code_reads_nested_reply() -> None:
    assert _status_code(None) is None
    assert _status_code({"data": {"header": {}}}) is None
    assert _status_code({"data": {"header": {"status": {"code": 0}}}}) == 0


# ---------------------------------------------------------------------- driver


class FakeLink:
    def __init__(self) -> None:
        self.requests: list[int] = []
        self.moves: list[tuple[float, float, float]] = []
        self.frame: tuple[float, np.ndarray] | None = None
        self.closed = 0
        self.request_error: Exception | None = None

    def request(self, topic: str, api_id: int, parameter: Any = None) -> Any:
        if self.request_error is not None:
            raise self.request_error
        self.requests.append(api_id)
        return {}

    def send_move(self, forward: float, left: float, turn: float) -> None:
        self.moves.append((forward, left, turn))

    def latest_frame(self) -> tuple[float, np.ndarray] | None:
        return self.frame

    def close(self) -> None:
        self.closed += 1


def make_driver(**kw: Any) -> tuple[Go2WebRTCDriver, FakeLink, Clock]:
    clock = Clock()
    link = FakeLink()
    return Go2WebRTCDriver(cfg(**kw), link=link, clock=clock), link, clock


def test_driver_state_is_body_frame_and_fresh() -> None:
    driver, _, clock = make_driver()
    with pytest.raises(RuntimeError, match="no Go2 sport state"):
        driver.read_state()
    driver.on_sport_state(sample())
    state = driver.read_state()
    np.testing.assert_allclose(state.pose, (1.0, 2.0, math.pi / 2))
    # World +y at yaw 90 degrees is body forward.
    np.testing.assert_allclose(state.velocity, (0.5, 0.0, 0.1), atol=1e-12)
    np.testing.assert_allclose(state.rpy, (0.01, -0.02, math.pi / 2))
    clock.t += 0.6
    with pytest.raises(RuntimeError, match=r"0\.60 s old"):
        driver.read_state()
    driver.on_sport_state({"position": [0, 0, 0]})
    with pytest.raises(RuntimeError, match="last error: malformed"):
        driver.read_state()


@pytest.mark.parametrize(
    "bad",
    [
        {"imu_state": {"rpy": [0.0, 0.0]}},
        {"position": [0.0, float("nan"), 0.0]},
        {"yaw_speed": float("inf")},
        {"body_height": None},
    ],
)
def test_driver_rejects_malformed_samples(bad: dict[str, Any]) -> None:
    driver, _, _ = make_driver()
    driver.on_sport_state({**sample(), **bad})
    with pytest.raises(RuntimeError, match="malformed sportmodestate"):
        driver.read_state()


def test_driver_streams_move_and_stand() -> None:
    driver, link, _ = make_driver()
    driver.send_velocity(0.3, 0.1, -0.2)
    driver.stand()
    assert link.moves == [(0.3, 0.1, -0.2), (0.0, 0.0, 0.0)]


def test_driver_frames_are_fresh_and_resized() -> None:
    driver, link, clock = make_driver()
    with pytest.raises(RuntimeError, match="no Go2 camera frame"):
        driver.read_frame()
    link.frame = (clock.t, np.arange(8 * 12 * 3, dtype=np.uint8).reshape(8, 12, 3))
    assert driver.read_frame().shape == (4, 6, 3)
    clock.t += 1.5
    with pytest.raises(RuntimeError, match=r"1\.50 s old"):
        driver.read_frame()


def test_check_ready_refuses_a_lying_robot_and_unlocks_move() -> None:
    driver, link, clock = make_driver()
    sleeps: list[float] = []

    def sleep(dt: float) -> None:
        sleeps.append(dt)
        clock.t += dt
        if len(sleeps) == 3:
            driver.on_sport_state(sample(height=0.08))

    with pytest.raises(RuntimeError, match="not standing"):
        driver.check_ready(sleep)
    assert link.requests == []
    driver.on_sport_state(sample())
    driver.check_ready(sleep)
    assert link.requests == [BALANCE_STAND]


def test_wait_for_state_times_out() -> None:
    driver, _, clock = make_driver()

    def sleep(dt: float) -> None:
        clock.t += dt

    with pytest.raises(RuntimeError, match=r"within 0\.2 s"):
        driver.wait_for_state(0.2, sleep)


def test_disconnect_stops_then_closes_even_if_stop_fails() -> None:
    driver, link, _ = make_driver()
    driver.disconnect()
    driver.disconnect()
    assert link.requests == [STOP_MOVE] and link.closed == 1
    driver, link, _ = make_driver()
    link.request_error = RuntimeError("gone")
    driver.disconnect()
    assert link.closed == 1


# ---------------------------------------------------------------------- connect


def test_connect_go2_wires_state_and_checks_readiness() -> None:
    clock = Clock()
    conn = FakeConn()

    def sleep(dt: float) -> None:
        clock.t += dt
        conn.push_state(sample())

    driver = connect_go2(cfg(), conn, clock=clock, sleep=sleep, video=False)
    assert [opts["api_id"] for _, opts in conn.datachannel.pub_sub.requests] == [BALANCE_STAND]
    driver.disconnect()
    assert conn.disconnects == 1
    assert [opts["api_id"] for _, opts in conn.datachannel.pub_sub.requests][-1] == STOP_MOVE


def test_connect_go2_without_ready_check_and_closes_on_failure() -> None:
    clock = Clock()
    conn = FakeConn()
    driver = connect_go2(cfg(), conn, clock=clock, sleep=lambda _: None, ready_check=False)
    assert conn.datachannel.pub_sub.requests == []
    driver.disconnect()

    busy = FakeConn(connect_error=ConnectionError("RobotBusyError"))
    busy.disconnect_error = RuntimeError("also broken")
    with pytest.raises(ConnectionError, match="RobotBusy"):
        connect_go2(cfg(), busy, clock=clock, sleep=lambda _: None)
    assert busy.disconnects == 1


def test_connect_go2_waits_for_the_first_frame_and_gives_up_cleanly() -> None:
    clock = Clock()
    conn = FakeConn()

    def sleep(dt: float) -> None:
        clock.t += dt
        conn.push_state(sample())

    with pytest.raises(RuntimeError, match="no fresh Go2 camera frame within"):
        connect_go2(cfg(connect_timeout_s=0.2), conn, clock=clock, sleep=sleep)
    assert conn.datachannel.pub_sub.requests == []  # never reached BalanceStand
    assert conn.disconnects == 1


def test_wait_for_frame_returns_once_a_frame_arrives() -> None:
    driver, link, clock = make_driver()

    def sleep(dt: float) -> None:
        clock.t += dt
        link.frame = (clock.t, np.zeros((8, 12, 3), dtype=np.uint8))

    driver.wait_for_frame(1.0, sleep)
    assert link.frame is not None


def test_check_fresh_needs_both_state_and_frame() -> None:
    driver, link, clock = make_driver()
    with pytest.raises(RuntimeError, match="no Go2 sport state"):
        driver.check_fresh()
    driver.on_sport_state(sample())
    with pytest.raises(RuntimeError, match="no Go2 camera frame"):
        driver.check_fresh()
    link.frame = (clock.t, np.zeros((8, 12, 3), dtype=np.uint8))
    driver.check_fresh()
    clock.t += 0.8
    with pytest.raises(RuntimeError, match=r"sport state is 0\.80 s old"):
        driver.check_fresh()


# ---------------------------------------------------------------------- helpers


def test_resize_nearest() -> None:
    image = np.arange(2 * 2 * 3, dtype=np.uint8).reshape(2, 2, 3)
    assert resize_nearest(image, 2, 2) is not None
    np.testing.assert_array_equal(resize_nearest(image, 2, 2), image)
    big = resize_nearest(image, 4, 4)
    np.testing.assert_array_equal(big[0, 0], image[0, 0])
    np.testing.assert_array_equal(big[3, 3], image[1, 1])
    with pytest.raises(ValueError, match="expected"):
        resize_nearest(np.zeros((2, 2), dtype=np.uint8), 2, 2)


def test_read_aes_key_from_env_only() -> None:
    config = cfg()
    assert read_aes_key(config, {}) is None
    assert read_aes_key(config, {"GO2_AES_128_KEY": " " + "a" * 32 + " "}) == "a" * 32
    with pytest.raises(ValueError, match="32 hex"):
        read_aes_key(config, {"GO2_AES_128_KEY": "zz"})


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="connection"):
        Go2Config(connection="remote")
    with pytest.raises(ValueError, match="frame_stale_s"):
        Go2Config(frame_stale_s=0)
    assert Go2Embodiment().info.name == "go2_base"


def test_go2_embodiment_needs_a_go2_config() -> None:
    from inspect_robots_unitree_go2 import LeggedConfig

    with pytest.raises(TypeError, match="needs a Go2Config"):
        Go2Embodiment(LeggedConfig())
