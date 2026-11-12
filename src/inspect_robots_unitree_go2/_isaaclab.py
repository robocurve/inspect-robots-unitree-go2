"""Isaac Lab backend: a simulated Go2 walking under a trained velocity policy.

Runs the configured Isaac Lab velocity-tracking task with one environment. The task's
own command generator is frozen, and its base-velocity command is overwritten
with whatever the embodiment asks for. A trained locomotion policy (an
exported TorchScript ``policy.pt`` from rsl_rl) turns that command into joint
targets, so the body walks the way the real controller would.

Pacing: the simulator keeps its own clock. The driver exposes ``sim_time``
and ``advance``, and the embodiment uses them in place of the wall clock and
``time.sleep``, so each control step runs exactly one control period of
physics and the world pauses while the policy (for example an LLM) thinks.
The streamer thread only latches commands; physics advances on the caller's
thread inside ``advance``, because Isaac Sim must stay on one thread.

This module needs Isaac Lab and an NVIDIA RTX GPU; it is exercised on a GPU
host, not in the unit tests.
"""

from __future__ import annotations  # pragma: no cover

import threading  # pragma: no cover
from typing import Any  # pragma: no cover

import numpy as np  # pragma: no cover
import numpy.typing as npt  # pragma: no cover

from inspect_robots_unitree_go2.config import LeggedConfig  # pragma: no cover
from inspect_robots_unitree_go2.driver import LeggedState  # pragma: no cover

_APP: Any = None  # pragma: no cover


def _launch_app(headless: bool) -> Any:  # pragma: no cover - needs Isaac Sim
    global _APP
    if _APP is None:
        from isaaclab.app import AppLauncher

        _APP = AppLauncher(headless=headless, enable_cameras=True).app
    return _APP


#: Coloured boxes placed around the start pose, robot frame (x forward, y left):
#: name, (x, y) in metres, RGB. All sit inside the default 3 m geofence.
PROPS: tuple[
    tuple[str, tuple[float, float], tuple[float, float, float]], ...
] = (  # pragma: no cover
    ("RedBox", (2.0, 1.0), (0.85, 0.1, 0.1)),
    ("BlueBox", (0.8, -1.8), (0.1, 0.25, 0.85)),
    ("GreenBox", (-1.8, 0.4), (0.1, 0.7, 0.2)),
)


def _add_props(env_cfg: Any, sim_utils: Any) -> None:  # pragma: no cover - needs Isaac Sim
    """Add static, collidable coloured boxes (0.4 x 0.4 x 0.8 m) to the scene config."""
    from isaaclab.assets import AssetBaseCfg

    for name, (x, y), rgb in PROPS:
        setattr(
            env_cfg.scene,
            name.lower(),
            AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/" + name,
                spawn=sim_utils.CuboidCfg(
                    size=(0.4, 0.4, 0.8),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(x, y, 0.4)),
            ),
        )


class IsaacLabVelocityDriver:  # pragma: no cover - needs Isaac Sim and a GPU
    """LeggedDriver backed by one Isaac Lab velocity-task environment."""

    def __init__(self, cfg: LeggedConfig) -> None:
        if not cfg.isaac_checkpoint:
            raise ValueError(
                "the isaaclab backend needs isaac_checkpoint: a TorchScript policy.pt "
                "exported by rsl_rl for this robot's velocity task (-E isaac_checkpoint=PATH)"
            )
        _launch_app(cfg.headless)
        import gymnasium as gym
        import isaaclab.sim as sim_utils
        import isaaclab_tasks  # noqa: F401 - registers the gym ids
        import torch
        from isaaclab.sensors import TiledCameraCfg
        from isaaclab_tasks.utils import parse_env_cfg

        self._torch = torch
        self._cfg = cfg
        env_cfg = parse_env_cfg(cfg.isaac_task, device="cuda:0", num_envs=1)
        command_cfg = env_cfg.commands.base_velocity
        command_cfg.resampling_time_range = (1.0e9, 1.0e9)
        command_cfg.heading_command = False
        command_cfg.rel_standing_envs = 0.0
        # The command's debug arrows float over the robot and fill the head camera.
        command_cfg.debug_vis = False
        env_cfg.episode_length_s = 1.0e6
        # Always walk on a flat floor. A rough-terrain task keeps its height
        # scanner, which reads the plane at /World/ground as level ground.
        terrain = env_cfg.scene.terrain
        if getattr(terrain, "terrain_type", "plane") != "plane":
            terrain.terrain_type = "plane"
            terrain.terrain_generator = None
            curriculum = getattr(env_cfg, "curriculum", None)
            if curriculum is not None and hasattr(curriculum, "terrain_levels"):
                curriculum.terrain_levels = None
        # Start at the env origin facing +x, at rest, so props sit where PROPS says
        # relative to the robot. The task randomises the reset pose by default.
        reset_base = getattr(env_cfg.events, "reset_base", None)
        if reset_base is not None:
            reset_base.params = {
                "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
                "velocity_range": {},
            }
        env_cfg.scene.front_cam = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + cfg.ISAAC_CAMERA_LINK + "/front_cam",
            offset=TiledCameraCfg.OffsetCfg(
                pos=cfg.ISAAC_CAMERA_POS, rot=(0.5, -0.5, 0.5, -0.5), convention="ros"
            ),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=12.0, clipping_range=(0.05, 50.0)),
            width=cfg.camera_width,
            height=cfg.camera_height,
        )
        if cfg.isaac_props:
            _add_props(env_cfg, sim_utils)
        self._env = gym.make(cfg.isaac_task, cfg=env_cfg)
        self._policy = torch.jit.load(cfg.isaac_checkpoint, map_location="cuda:0").eval()
        obs, _ = self._env.reset()
        self._obs = obs
        self._dt = float(self._env.unwrapped.step_dt)
        self._lock = threading.Lock()
        self._command = (0.0, 0.0, 0.0)
        self.resets = 0
        self._sim_time = 0.0

    def _velocity_term(self) -> Any:
        return self._env.unwrapped.command_manager.get_term("base_velocity")

    def sim_time(self) -> float:
        """Seconds of simulated time since the environment was created."""
        return self._sim_time

    def advance(self, seconds: float) -> None:
        """Run physics for ``seconds`` of sim time under the latest command."""
        torch = self._torch
        target = self._sim_time + float(seconds)
        term = self._velocity_term()
        while self._sim_time + 0.5 * self._dt < target:
            with self._lock:
                command = self._command
            term.vel_command_b[:] = torch.tensor(command, device=term.vel_command_b.device)
            # The observation still holds last tick's command: one physics tick of lag.
            with torch.inference_mode():
                action = self._policy(self._obs["policy"])
            self._obs, _, terminated, truncated, _ = self._env.step(action)
            self._sim_time += self._dt
            if bool(terminated.any()) or bool(truncated.any()):
                # Isaac Lab has already reset the scene: this trial cannot go on.
                self.resets += 1
                raise RuntimeError(
                    "the simulated robot fell or its episode ended; Isaac Lab reset the scene"
                )

    def send_velocity(self, forward: float, left: float, turn: float) -> None:
        """Latch the commanded body velocity; physics steps on the reading thread."""
        with self._lock:
            self._command = (float(forward), float(left), float(turn))

    def stand(self) -> None:
        """Latch a zero velocity command."""
        self.send_velocity(0.0, 0.0, 0.0)

    def read_state(self) -> LeggedState:
        """Read root pose, body velocity and attitude at the current sim time."""
        robot = self._env.unwrapped.scene["robot"].data
        pos = robot.root_pos_w[0].cpu().numpy()
        quat = robot.root_quat_w[0].cpu().numpy()  # w, x, y, z
        w, x, y, z = (float(v) for v in quat)
        roll = float(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
        pitch = float(np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0)))
        yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        lin = robot.root_lin_vel_b[0].cpu().numpy()
        ang = robot.root_ang_vel_b[0].cpu().numpy()
        return LeggedState(
            pose=np.asarray((pos[0], pos[1], yaw), dtype=np.float64),
            velocity=np.asarray((lin[0], lin[1], ang[2]), dtype=np.float64),
            rpy=np.asarray((roll, pitch, yaw), dtype=np.float64),
        )

    def read_frame(self) -> npt.NDArray[np.uint8]:
        """Return the camera's RGB frame at the current sim time."""
        rgb = self._env.unwrapped.scene["front_cam"].data.output["rgb"][0]
        return np.ascontiguousarray(rgb.cpu().numpy()[..., :3], dtype=np.uint8)

    def disconnect(self) -> None:
        """Close the environment. The SimulationApp stays up for the process."""
        self._env.close()
