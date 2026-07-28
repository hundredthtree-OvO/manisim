from __future__ import annotations

import numpy as np


PANDA_SAFE_QPOS = np.array(
    [
        0.0,
        0.17212291,
        0.0,
        -1.53357124,
        0.0,
        1.70569408,
        np.pi / 4,
        0.04,
        0.04,
    ],
    dtype=np.float32,
)


def initialize_panda(base_env: object) -> None:
    """Reset Panda to the fixed downward pose with TCP at about z=0.45 m."""

    base_env.agent.reset(PANDA_SAFE_QPOS)
