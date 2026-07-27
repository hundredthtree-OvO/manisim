from __future__ import annotations

import numpy as np


PANDA_SAFE_QPOS = np.array(
    [
        0.0,
        np.pi / 8,
        0.0,
        -np.pi * 5 / 8,
        0.0,
        np.pi * 3 / 4,
        np.pi / 4,
        0.04,
        0.04,
    ],
    dtype=np.float32,
)


def initialize_panda(base_env: object) -> None:
    """Reset Panda to ManiSkill's standard tabletop-safe joint configuration."""

    base_env.agent.reset(PANDA_SAFE_QPOS)
