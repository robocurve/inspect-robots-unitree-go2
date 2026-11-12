"""Inspect Robots embodiment for walking a Unitree Go2 (Air) quadruped.

The package registers embodiment ``go2_base``: per-step planar body
displacement handed to the Go2's own sport controller. Construction is inert.
"""

from __future__ import annotations

from inspect_robots_unitree_go2.config import ACTION_DIM, DIM_LABELS, Go2Config, LeggedConfig
from inspect_robots_unitree_go2.driver import KinematicLeggedDriver, LeggedDriver, LeggedState
from inspect_robots_unitree_go2.embodiment import Go2Embodiment, LeggedEmbodiment
from inspect_robots_unitree_go2.operator import OperatorIO

try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("inspect-robots-unitree-go2")
except PackageNotFoundError:  # pragma: no cover - non-installed source tree
    __version__ = "0.0.0+unknown"

__all__ = [
    "ACTION_DIM",
    "DIM_LABELS",
    "Go2Config",
    "Go2Embodiment",
    "KinematicLeggedDriver",
    "LeggedConfig",
    "LeggedDriver",
    "LeggedEmbodiment",
    "LeggedState",
    "OperatorIO",
    "__version__",
]
