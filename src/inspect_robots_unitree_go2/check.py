"""Read-only Go2 check: connect, listen, report, never command motion.

Run it on the laptop that will drive the robot, with the robot standing and
the Unitree phone app closed::

    GO2_AES_128_KEY=... inspect-robots-go2-check -E robot_ip=192.168.x.y

It opens the WebRTC link, subscribes to the sport state, turns on the video
stream, and asks the motion switcher which mode it is in (a query that
changes nothing). It sends nothing on the sport topic: no BalanceStand, no
Move, no StopMove. It then reports whether the state arrives, how often and
how fresh, whether the robot is standing, the motion mode, and one camera
frame. The exit code is 1 if the adapter would refuse the robot.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from inspect_robots_unitree_go2.config import Go2Config
from inspect_robots_unitree_go2.go2 import (
    MOTION_MODE_QUERY,
    MOTION_SWITCHER_TOPIC,
    Go2WebRTCDriver,
    WebRTCLink,
    read_aes_key,
)


@dataclass(frozen=True)
class Check:
    """One line of the report. ``level`` is ``ok``, ``warn`` or ``fail``."""

    name: str
    level: str
    detail: str


@dataclass
class Listened:
    """What one listening window saw. ``error`` is set when the link never came up."""

    error: str | None = None
    driver: Go2WebRTCDriver | None = None
    sample_times: list[float] = field(default_factory=list)
    last_sample: Mapping[str, Any] | None = None
    motion_mode: str | None = None
    motion_error: str | None = None
    window_s: float = 0.0
    close: Callable[[], None] | None = None


def motion_mode_name(reply: Any) -> str:
    """``json.loads(reply['data']['data'])['name']``, as the library's example reads it."""
    payload = reply["data"]["data"]
    if isinstance(payload, str):
        payload = json.loads(payload)
    return str(payload["name"])


def listen(
    cfg: Go2Config,
    conn: Any,
    *,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    seconds: float,
) -> Listened:
    """Open the link on ``conn``, listen for ``seconds``, query the mode, close."""
    seen = Listened(window_s=seconds)
    link = WebRTCLink(conn, clock=clock, request_timeout_s=cfg.request_timeout_s)
    driver = Go2WebRTCDriver(cfg, link=link, clock=clock)

    def on_state(data: Mapping[str, Any]) -> None:
        seen.sample_times.append(clock())
        seen.last_sample = data
        driver.on_sport_state(data)

    try:
        link.start(on_state, connect_timeout_s=cfg.connect_timeout_s, video=True)
    except Exception as exc:
        seen.error = f"could not connect: {exc!r}"
        _close_quietly(link)
        return seen
    seen.driver = driver
    seen.close = lambda: _close_quietly(link)
    try:
        seen.motion_mode = motion_mode_name(link.request(MOTION_SWITCHER_TOPIC, MOTION_MODE_QUERY))
    except Exception as exc:
        seen.motion_error = repr(exc)
    sleep(seconds)
    return seen


def _close_quietly(link: WebRTCLink) -> None:
    try:
        link.close()
    except Exception as exc:
        print(f"warning: cleanup failed: {exc!r}")


def assess(cfg: Go2Config, seen: Listened) -> list[Check]:
    """Every check, in report order."""
    if seen.error is not None or seen.driver is None:
        return [Check("link", "fail", seen.error or "not connected")]
    driver = seen.driver
    checks = [Check("link", "ok", f"WebRTC data channel open ({cfg.connection})")]

    count = len(seen.sample_times)
    rate = count / seen.window_s if seen.window_s > 0 else 0.0
    try:
        state = driver.read_state()
    except RuntimeError as exc:
        checks.append(Check("sport state", "fail", f"{count} samples; {exc}"))
    else:
        level = "ok" if rate * cfg.state_stale_s >= 2 else "warn"
        detail = f"{count} samples in {seen.window_s:g} s (~{rate:.1f} Hz)"
        if level == "warn":
            detail += f": slow for state_stale_s={cfg.state_stale_s}; raise it if this holds"
        checks.append(Check("sport state", level, detail))
        pose = np.round(state.pose, 3).tolist()
        checks.append(Check("odometry", "ok", f"x, y, yaw = {pose}"))

    sample = seen.last_sample or {}
    height = sample.get("body_height")
    if isinstance(height, int | float) and math.isfinite(height):
        standing = height >= cfg.min_body_height_m
        checks.append(
            Check(
                "standing",
                "ok" if standing else "fail",
                f"body height {height:.2f} m, mode {sample.get('mode')}"
                + ("" if standing else f": below {cfg.min_body_height_m} m, stand it up first"),
            )
        )
    else:
        checks.append(Check("standing", "fail", "no body_height in the sport state"))

    if seen.motion_mode is not None:
        checks.append(Check("motion mode", "ok", seen.motion_mode))
    else:
        checks.append(Check("motion mode", "warn", f"query failed: {seen.motion_error}"))

    try:
        frame = driver.read_frame()
    except RuntimeError as exc:
        checks.append(Check("camera", "fail", str(exc)))
    else:
        checks.append(Check("camera", "ok", f"frame resized to {tuple(frame.shape)}"))
    return checks


def format_report(checks: list[Check]) -> str:
    """Human-readable report, one line per check, with a verdict last."""
    lines = [f"{check.level.upper():<5} {check.name:<12} {check.detail}" for check in checks]
    failed = [check.name for check in checks if check.level == "fail"]
    verdict = (
        "NOT READY: " + ", ".join(failed)
        if failed
        else "READY: the adapter will accept this robot (nothing was commanded)"
    )
    return "\n".join([*lines, verdict])


ListenFn = Callable[[Go2Config, float], Listened]


def _parse_options(items: list[str]) -> dict[str, str]:
    options: dict[str, str] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"-E expects key=value; got {item!r}")
        options[key] = value
    return options


def main(argv: list[str] | None = None, *, listen_fn: ListenFn | None = None) -> int:
    """Listen to the Go2, print the report, return 1 if not ready."""
    parser = argparse.ArgumentParser(
        prog="inspect-robots-go2-check",
        description="Read-only check of a real Go2 before the first run. Commands no motion.",
    )
    parser.add_argument(
        "-E", dest="options", action="append", default=[], help="go2_base option key=value"
    )
    parser.add_argument("--seconds", type=float, default=3.0, help="listening window")
    parser.add_argument("--json", action="store_true", help="print the checks as JSON")
    args = parser.parse_args(argv)
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be positive")
    options = _parse_options(args.options)
    options["backend"] = "sdk"
    cfg = Go2Config.from_kwargs(**options)
    try:
        aes = "set" if read_aes_key(cfg) else "not set (fine before firmware 1.1.15)"
    except ValueError as exc:
        parser.error(str(exc))
    seen = (listen_fn or listen_real)(cfg, args.seconds)
    try:
        checks = [Check("aes key", "ok", f"{cfg.aes_key_env} {aes}"), *assess(cfg, seen)]
    finally:
        if seen.close is not None:
            seen.close()
    if args.json:
        payload = [{"name": c.name, "level": c.level, "detail": c.detail} for c in checks]
        print(json.dumps(payload, indent=2))
    else:
        print(format_report(checks))
    return 1 if any(check.level == "fail" for check in checks) else 0


def listen_real(cfg: Go2Config, seconds: float) -> Listened:  # pragma: no cover - hardware
    """Listen to a real Go2."""
    import time

    from inspect_robots_unitree_go2.go2 import make_connection

    try:
        conn = make_connection(cfg)
    except Exception as exc:
        return Listened(error=f"could not build the connection: {exc!r}")
    return listen(cfg, conn, clock=time.monotonic, sleep=time.sleep, seconds=seconds)
