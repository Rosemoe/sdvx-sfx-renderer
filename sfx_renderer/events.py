"""Timed chart events used by the renderer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from vox_parser.classes.effects import Effect

EffectT = TypeVar("EffectT", bound=Effect)


@dataclass(frozen=True)
class FXRenderEvent(Generic[EffectT]):
    """A timed FX event derived from an FX button note."""

    start_sample: int
    end_sample: int
    bpm: float
    effect: EffectT
    label: str = ""
