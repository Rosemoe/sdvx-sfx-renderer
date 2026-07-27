"""Helpers for VOL filter parameters."""
from dataclasses import dataclass

from .base import VoxEntity
from .effects import Bitcrush, Effect, HighpassFilter, LowpassFilter

__all__ = [
    "AutoTabParam",
    "get_default_filters",
]


@dataclass
class AutoTabParam(VoxEntity):
    """One parameter assignment used by a VOX auto-tab effect."""

    effect_index: int
    param_index: int = 0
    min_value: float = 0.00
    max_value: float = 0.00

    def to_vox_string(self) -> str:
        return ",\t".join(
            [f"{self.effect_index}", f"{self.param_index}", f"{self.min_value:.2f}", f"{self.max_value:.2f}"]
        )


def get_default_filters() -> list[Effect]:
    """Get the default VOL filter settings."""
    return [
        LowpassFilter(mix=90.00, vol_cutoff_bound=400.00, cutoff=18000.00, q=0.70),
        LowpassFilter(mix=90.00, vol_cutoff_bound=600.00, cutoff=15000.00, q=5.00),
        HighpassFilter(mix=90.00, cutoff=40.00, vol_cutoff_bound=5000.00, q=0.70),
        HighpassFilter(mix=90.00, cutoff=40.00, vol_cutoff_bound=2000.00, q=3.00),
        Bitcrush(mix=100.00, hold_samples=30),
    ]
