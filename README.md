# inspect-robots-unitree-go2

Run [Inspect Robots](https://github.com/robocurve/inspect-robots) evals on a
Unitree Go2 quadruped, including the Go2 Air. The package registers one
embodiment, `go2_base`: the LLM steers the body and the Go2's own walking
controller handles the legs.

> [!NOTE]
> Alpha, written before we had the robot. The `kinematic` backend, the Isaac
> Lab backend and the fake-connection tests are verified. The real backend
> follows the `unitree_webrtc_connect` source (2.2.0) but has not run on a Go2.

## Action and observation contract

Each action is how far the body should move during one control step
(`joint_delta`, robot frame at the start of the step):

| Label | Unit | Meaning |
|---|---|---|
| `base_forward` | m | toward the head (negative walks backward) |
| `base_left` | m | toward the robot's left |
| `base_turn` | rad | counter-clockwise seen from above |

Bounds are the speed caps divided by `control_hz` (defaults 0.4 m/s forward,
0.25 m/s back and sideways, 0.8 rad/s, at 10 Hz). With the `agent` policy the
LLM calls `move_by`, for example `{"base_forward": 0.5, "base_turn": 0.3}`;
longer moves are split into steps at full speed.

Observations: the front camera (`front_cam`, resized to 640x480) and state
`base_pose` (x, y, yaw odometry from the reset point), `base_vel` (body
frame) and `body_rpy`.

## How it drives the robot

- The Go2 Air has no official SDK. The real backend uses
  [`unitree_webrtc_connect`](https://github.com/legion1581/unitree_webrtc_connect),
  the WebRTC link the Unitree phone app uses. Only one WebRTC client can be
  connected, so close the phone app first.
- The robot must already be standing. Nothing here stands it up or lies it
  down. On connect the adapter refuses a robot whose body height is below
  `min_body_height_m`, then sends `BalanceStand` once so walking commands are
  accepted.
- Each step streams `Move(vx, vy, yaw rate)` at `stream_hz` until the step's
  deadline, then `Move(0, 0, 0)`. That host-side watchdog is the stop: no
  source says the Go2 stops by itself when `Move` stops arriving. Closing
  sends `StopMove`, then disconnects.
- Stale sport state (`state_stale_s`) or camera frames (`frame_stale_s`) end
  the trial and stop the robot. So does leaving the `geofence_m` circle.
- If the laptop dies or the link drops mid-step, the robot may keep its last
  velocity for a while. Keep the area clear and a way to stop it at hand
  (its remote, if it has one).

## Firmware and the AES key

Go2 firmware 1.0.19 to 1.1.14 connects without setup. Firmware 1.1.15 and
later need the robot's own AES key (32 hex characters), fetched once with
`unitree-fetch-aes-key` and the owner's Unitree account. Put it in the
environment as `GO2_AES_128_KEY` (see `.env.example`); it is never taken on
the command line. Turn off automatic firmware updates before an event.

## First run on a real Go2

1. Install on the laptop (Python 3.11 or 3.12; on Linux first
   `apt install portaudio19-dev`):

   ```bash
   git clone https://github.com/robocurve/inspect-robots-unitree-go2 && cd inspect-robots-unitree-go2
   uv venv && source .venv/bin/activate
   uv pip install inspect-robots inspect-robots-agent -e ".[hardware]"
   ```

2. **Read-only check.** Robot standing, phone app closed, laptop on the same
   network as the robot (or on its hotspot with `-E connection=local_ap`):

   ```bash
   inspect-robots-go2-check -E robot_ip=<robot ip>
   ```

   It connects, listens for three seconds and reports whether the sport state
   arrives and how often, the body height, the motion mode and one camera
   frame, then READY or NOT READY. It sends no motion command. If the state
   is slower than expected, raise `state_stale_s`.

3. **First walk, slowly:** low caps, operator prompts on:

   ```bash
   inspect-robots run --policy agent --embodiment go2_base \
       -E robot_ip=<robot ip> -E max_forward_speed=0.2 -E max_turn_rate=0.4 \
       -P model=anthropic/claude-opus-5-5 \
       --instruction "walk forward half a metre, then turn left 90 degrees"
   ```

## Simulation

- `-E backend=kinematic` runs anywhere: a point robot with a top-down camera.
- `-E backend=isaaclab -E isaac_checkpoint=PATH/policy.pt` needs an Isaac
  Lab 2.3 machine with a GPU. It runs `Isaac-Velocity-Rough-Unitree-Go2-Play-v0`
  on a flat floor with a camera on the body. `policy.pt` is the TorchScript
  walking policy that Isaac Lab's `rsl_rl/play.py --use_pretrained_checkpoint`
  exports for `Isaac-Velocity-Rough-Unitree-Go2-v0`. The pretrained flat-terrain
  policy is not used because it barely turns in place. The rough policy walks
  forward accurately but turns about 60 percent of what is commanded.

## Develop

```bash
export UV_CACHE_DIR=$PWD/.uv-cache
uv sync --locked --extra dev
uv run pre-commit install    # lint, format and types on commit; tests on push
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy && .venv/bin/pytest --cov
```

## License

MIT
