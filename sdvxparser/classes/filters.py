"""
Classes and functions that represent and handle filters.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from .base import VoxEntity

__all__ = [
    "KSHFilterType",
    "Filter",
    "LowpassFilter",
    "HighpassFilter",
    "BitcrushFilter",
    "AutoTabParam",
    "get_default_filters",
]


class KSHFilterType(Enum):
    """Enumeration for KSH filter types."""

    PEAK = 0
    LPF = 1
    HPF = 2
    BITCRUSH = 3


@dataclass
class Filter(VoxEntity, ABC):
    """Abstract base class for laser filters."""

    @property
    @abstractmethod
    def filter_index(self) -> KSHFilterType:
        """Return the enumeration value corresponding to this filter."""
        pass

    @abstractmethod
    def to_vox_string(self) -> str:
        pass


@dataclass
class LowpassFilter(Filter):
    """A class representing a low-pass filter on lasers."""

    mix: float = 90.00
    min_cutoff: float = 400.00
    max_cutoff: float = 18000.00
    bandwidth: float = 0.70

    @property
    def filter_index(self) -> KSHFilterType:
        return KSHFilterType.LPF

    def to_vox_string(self) -> str:
        return ",\t".join(
            [
                f"{self.filter_index.value}",
                f"{self.mix:.2f}",
                f"{self.min_cutoff:.2f}",
                f"{self.max_cutoff:.2f}",
                f"{self.bandwidth:.2f}",
            ]
        )


@dataclass
class HighpassFilter(Filter):
    """A class representing a high-pass filter on lasers."""

    mix: float = 90.00
    min_cutoff: float = 40.00
    max_cutoff: float = 5000.00
    bandwidth: float = 0.70

    @property
    def filter_index(self) -> KSHFilterType:
        return KSHFilterType.HPF

    def to_vox_string(self) -> str:
        return ",\t".join(
            [
                f"{self.filter_index.value}",
                f"{self.mix:.2f}",
                f"{self.min_cutoff:.2f}",
                f"{self.max_cutoff:.2f}",
                f"{self.bandwidth:.2f}",
            ]
        )


@dataclass
class BitcrushFilter(Filter):
    """A class representing a bitcrush filter on lasers."""

    mix: float = 100.00
    max_amount: int = 30

    @property
    def filter_index(self) -> KSHFilterType:
        return KSHFilterType.BITCRUSH

    def to_vox_string(self) -> str:
        return ",\t".join([f"{self.filter_index.value}", f"{self.mix:.2f}", f"{self.max_amount}"])


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


def get_default_filters() -> list[Filter]:
    """Get the default filter settings."""
    return [
        LowpassFilter(),
        LowpassFilter(min_cutoff=600.00, max_cutoff=15000.00, bandwidth=5.00),
        HighpassFilter(),
        HighpassFilter(max_cutoff=2000.00, bandwidth=3.00),
        BitcrushFilter(),
    ]
