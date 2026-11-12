# 0001: Unitree Go2 Air embodiment for LLM agents (go2_base)

Status: draft for review (2026-11-11)
Issue: RC-355 (Robocurve Linear), for the Pantograph robot safety hackathon (2026-11-14)

## Goal

Let an LLM agent walk the two lent Go2 Airs through Inspect Robots, with the
same planar contract as the G1 walking embodiment (`g1_base`), testable in
simulation before the robots arrive.

## Contract

Shared with `g1_base` (the walking layer is copied from
`inspect-robots-unitree-g1`): `joint_delta`, 3-D `base_forward`,
`base_left`, `base_turn` per control step; state `base_pose`,
`base_vel`, `body_rpy`; one camera `front_cam`.

## Transport

The Go2 Air has no DDS/SDK access, so `unitree_sdk2py` (used for the G1) does
not apply. `unitree_webrtc_connect` 2.2.0 talks to the sport controller over
the phone app's WebRTC data channel. DimOS uses the same library for the Go2
Air. Used here:

- `rt/api/sport/request`: BalanceStand (1002) once at connect, Move (1008)
  streamed as a no-reply request, StopMove (1003) at close.
- `rt/lf/sportmodestate`: position, velocity, yaw speed, body height, IMU.
- `rt/api/motion_switcher/request` 1001: mode query (check only).
- Video track, newest frame kept, resized with numpy.

## Safety

- Host-side watchdog: the streamer sends Move(0, 0, 0) after each step's
  deadline (default 0.1 s grace). Robot-side timeout is unknown.
- Refuse a robot that is not standing; never change its posture or mode.
- Stale state (0.5 s) or video (1 s), a closed data channel, or the geofence
  end the trial and stop the robot.

## Open questions (need the robot)

- Go2 firmware version (1.1.15+ needs the per-robot AES key).
- Whether Move sent as a no-reply `req` is accepted in the robot's current
  motion mode ("normal" vs "ai"/MCF).
- State publish rate and camera resolution on the Air.
- Whether the Air stops on its own when Move stops arriving.
