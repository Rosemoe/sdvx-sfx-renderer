"""Timed chart events used by the renderer."""
from __future__ import annotations

from dataclasses import dataclass

from sdvxparser.classes.effects import Effect


@dataclass(frozen=True)
class FXRenderEvent:
    """A timed FX event derived from an FX button note."""

    start_sample: int
    end_sample: int
    bpm: float
    effect: Effect
    label: str = ""
