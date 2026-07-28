from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoxObjectSpec:
    """Declarative specification for a box-shaped scene entity."""

    name: str
    size_m: tuple[float, float, float]
    position_m: tuple[float, float, float]
    color_rgba: tuple[float, float, float, float]
    body_type: str = "dynamic"
    add_collision: bool = True
