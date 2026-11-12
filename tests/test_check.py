"""Read-only Go2 check against the fake connection: it must command nothing."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from inspect_robots_unitree_go2.check import (
    Listened,
    assess,
    format_report,
    listen,
    main,
    motion_mode_name,
)
from inspect_robots_unitree_go2.config import Go2Config
from inspect_robots_unitree_go2.go2 import MOTION_SWITCHER_TOPIC, SPORT_TOPIC
from test_go2 import Clock, FakeConn, run_track, sample


def cfg() -> Go2Config:
    return Go2Config(backend="sdk", camera_height=4, camera_width=6)


def mode_reply(name: str = "normal") -> dict[str, Any]:
    return {
        "data": {
            "header": {"status": {"code": 0}},
            "data": json.dumps({"name": name}),
        }
    }


def heard(
    *, height: float = 0.32, samples: int = 30, frame: bool = True, mode_ok: bool = True
) -> tuple[Listened, FakeConn]:
    clock = Clock()
    conn = FakeConn()
    conn.reply = mode_reply() if mode_ok else {"data": {"header": {"status": {"code": 7}}}}

    def sleep(seconds: float) -> None:
        for _ in range(samples):
            clock.t += seconds / max(samples, 1)
            conn.push_state(sample(height=height))
        if frame:
            run_track(link_of(seen), conn, [np.zeros((8, 12, 3), dtype=np.uint8)])

    def link_of(listened: Listened) -> Any:
        assert listened.driver is not None
        return listened.driver._link

    seen = listen(cfg(), conn, clock=clock, sleep=lambda s: None, seconds=1.0)
    sleep(1.0)
    return seen, conn


def test_ready_robot_and_nothing_sent_on_the_sport_topic() -> None:
    seen, conn = heard()
    checks = assess(cfg(), seen)
    assert [c.level for c in checks] == ["ok"] * 6
    assert "READY" in format_report(checks)
    pub = conn.datachannel.pub_sub
    assert all(topic != SPORT_TOPIC for topic, _ in pub.requests)
    assert all(topic != SPORT_TOPIC for topic, _, _ in pub.sent)
    assert [topic for topic, _ in pub.requests] == [MOTION_SWITCHER_TOPIC]
    assert seen.motion_mode == "normal"
    assert seen.close is not None
    seen.close()
    assert conn.disconnects == 1


def test_lying_robot_slow_state_no_frame_and_failed_mode_query() -> None:
    seen, _ = heard(height=0.08, samples=1, frame=False, mode_ok=False)
    by_name = {c.name: c for c in assess(cfg(), seen)}
    assert by_name["sport state"].level == "warn"
    assert by_name["standing"].level == "fail"
    assert by_name["motion mode"].level == "warn"
    assert by_name["camera"].level == "fail"
    assert "NOT READY: standing, camera" in format_report(list(by_name.values()))


def test_no_state_at_all() -> None:
    seen, _ = heard(samples=0, frame=False)
    by_name = {c.name: c for c in assess(cfg(), seen)}
    assert by_name["sport state"].level == "fail"
    assert by_name["standing"].detail == "no body_height in the sport state"


def test_connection_failure_is_reported() -> None:
    conn = FakeConn(connect_error=ConnectionError("RobotBusyError"))
    conn.disconnect_error = RuntimeError("cleanup broke")
    seen = listen(cfg(), conn, clock=Clock(), sleep=lambda s: None, seconds=1.0)
    assert seen.error is not None and "RobotBusy" in seen.error
    assert [c.level for c in assess(cfg(), seen)] == ["fail"]
    assert assess(cfg(), Listened()) == assess(cfg(), Listened(error="not connected"))


def test_motion_mode_name_accepts_a_dict_payload() -> None:
    assert motion_mode_name({"data": {"data": {"name": "ai"}}}) == "ai"


def test_main_prints_report_json_and_exit_code(capsys: pytest.CaptureFixture[str]) -> None:
    seen, conn = heard()
    assert main(["-E", "robot_ip=10.0.0.2"], listen_fn=lambda c, s: seen) == 0
    assert "READY" in capsys.readouterr().out
    assert conn.disconnects == 1
    failed = Listened(error="could not connect")
    assert main(["--json"], listen_fn=lambda c, s: failed) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["name"] == "aes key" and payload[1]["level"] == "fail"


def test_main_rejects_bad_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit):
        main(["--seconds", "0"], listen_fn=lambda c, s: Listened())
    with pytest.raises(SystemExit, match="key=value"):
        main(["-E", "oops"], listen_fn=lambda c, s: Listened())
    monkeypatch.setenv("GO2_AES_128_KEY", "nothex")
    with pytest.raises(SystemExit):
        main([], listen_fn=lambda c, s: Listened())
    monkeypatch.setenv("GO2_AES_128_KEY", "b" * 32)
    assert main([], listen_fn=lambda c, s: Listened(error="x")) == 1
