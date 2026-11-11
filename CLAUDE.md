# inspect-robots-unitree-go2 agent guide

Inspect Robots embodiment `go2_base`: a Unitree Go2 (Air) steered by per-step
planar body displacement `(base_forward, base_left, base_turn)` through the
robot's own sport controller over WebRTC.

## Layout

- `config.py`: frozen config and spaces (`LeggedConfig`, `Go2Config`).
- `driver.py`: the `LeggedDriver` protocol, the kinematic stand-in, the
  backend factory.
- `go2.py`: `WebRTCLink` (asyncio loop thread around `unitree_webrtc_connect`)
  and `Go2WebRTCDriver`.
- `check.py`: `inspect-robots-go2-check`, the read-only pre-motion check. It
  must never send anything on the sport topic.
- `embodiment.py`: command streamer with deadline, geofence, teardown.
- `_isaaclab.py`: Isaac Lab backend (needs a GPU machine; excluded from
  coverage).

## Working here

- Set `UV_CACHE_DIR=$PWD/.uv-cache` for uv commands.
- Gates: ruff check, ruff format check, strict mypy, pytest at 100 percent
  statement and branch coverage.
- Keep `unitree_webrtc_connect`, aiortc and Isaac imports lazy. Construction
  is inert.

## Safety invariants

- Never stand the robot up, lie it down or damp it. Refuse a robot that is not
  already standing.
- The streamer sends zero velocity once a step's deadline passes; the robot is
  not assumed to stop by itself.
- Every send checks the data channel is open (the library drops messages on a
  closed channel silently).
- Stale state or video ends the trial and stops the robot.
- The AES key comes only from the environment and is never printed.

## CI and releases

- CI installs from `uv.lock`; `ci-ok` needs every blocking job, including
  import hygiene (the package imports with only inspect-robots and numpy).
- Versions come from git tags through hatch-vcs. Design notes live in
  `plans/`.

## Writing style

- No em dashes in prose. No decorative emoji. Plain headers.
