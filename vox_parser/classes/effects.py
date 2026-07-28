"""
Classes and functions that represent and handle audio effects.
"""
import logging

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from enum import Enum

from .base import VoxEntity

__all__ = [
    "FXType",
    "PassFilterType",
    "WaveShape",
    "Effect",
    "NoEffect",
    "Retrigger",
    "Gate",
    "Flanger",
    "Tapestop",
    "Sidechain",
    "Wobble",
    "Bitcrush",
    "RetriggerEx",
    "PitchShift",
    "PitchShiftEx",
    "Tapescratch",
    "LowpassFilter",
    "HighpassFilter",
    "EffectEntry",
    "enum_to_effect",
    "from_vox_params",
    "get_default_effects",
]

logger = logging.getLogger(__name__)
_enumToEffect: dict = {}


class _StringifiableEnum(Enum):
    def __str__(self) -> str:
        name_parts = [s.capitalize() for s in self.name.split("_")]
        return "".join(name_parts)


class FXType(_StringifiableEnum):
    """Enumeration for effect types."""

    NO_EFFECT = 0
    RETRIGGER = 1
    GATE = 2
    FLANGER = 3
    TAPESTOP = 4
    SIDECHAIN = 5
    WOBBLE = 6
    BITCRUSH = 7
    RETRIGGER_EX = 8
    PITCH_SHIFT = 9
    TAPESCRATCH = 10
    LOW_PASS_FILTER = 11
    HIGH_PASS_FILTER = 12
    PITCH_SHIFT_EX = 13


class PassFilterType(_StringifiableEnum):
    """Enumeration for pass filter type."""

    LOW_PASS = 0
    HIGH_PASS = 1
    BAND_PASS = 2


class WaveShape(_StringifiableEnum):
    """Raw VOX Wobble cutoff-modulation shapes."""

    SAW_UP = 0
    SAW_DOWN = 1
    SINE = 2
    TRIANGLE = 3
    SQUARE = 4


@dataclass
class Effect(VoxEntity, ABC):
    """Abstract base class for effects."""

    @property
    def effect_name(self) -> str:
        """Return the effect name."""
        return str(self.effect_index)

    @property
    @abstractmethod
    def effect_index(self) -> FXType:
        """Return the enumeration value corresponding to this effect."""
        pass

    @abstractmethod
    def to_vox_string(self) -> str:
        pass

    def duplicate(self):
        """Create a copy of this object."""
        return replace(self)

    def get_vox_param_field(self, param_order: int) -> str | None:
        """Return the dataclass field at a 1-based FXBUTTON VOX parameter order."""

        fields_by_type: dict[FXType, tuple[str, ...]] = {
            FXType.RETRIGGER: ("wave_length", "mix", "update_period", "feedback", "active_ratio", "fade_ratio"),
            FXType.GATE: ("mix", "wave_length", "length"),
            FXType.FLANGER: ("mix", "period", "feedback", "stereo_width", "hicut_gain"),
            FXType.TAPESTOP: ("mix", "speed", "duration_seconds"),
            FXType.SIDECHAIN: ("mix", "frequency", "attack", "hold", "release"),
            FXType.WOBBLE: ("filter_type", "wave_shape", "mix", "low_cutoff", "hi_cutoff", "frequency", "q"),
            FXType.BITCRUSH: ("mix", "hold_samples"),
            FXType.RETRIGGER_EX: ("wave_length", "mix", "update_period", "feedback", "active_ratio", "fade_ratio"),
            FXType.PITCH_SHIFT: ("mix", "semitones"),
            FXType.TAPESCRATCH: ("mix", "curve_slope", "attack", "hold", "release"),
            FXType.LOW_PASS_FILTER: ("mix", "vol_cutoff_bound", "cutoff", "q"),
            FXType.HIGH_PASS_FILTER: ("mix", "cutoff", "vol_cutoff_bound", "q"),
            FXType.PITCH_SHIFT_EX: ("mix", "semitones", "ex_param"),
        }
        fields = fields_by_type.get(self.effect_index)
        if fields is None or not 1 <= param_order <= len(fields):
            return None
        return fields[param_order - 1]


def _register_effect(cls):
    global _enumToEffect
    _enumToEffect[cls().effect_index] = cls

    return cls


@_register_effect
@dataclass
class NoEffect(Effect):
    """A class representing a null effect."""

    @property
    def effect_index(self) -> FXType:
        return FXType.NO_EFFECT

    def to_vox_string(self) -> str:
        return ",\t".join([f"{self.effect_index.value}", "0", "0", "0", "0", "0", "0"])


@_register_effect
@dataclass
class Retrigger(Effect):
    """A class representing a retrigger effect."""

    mix: float = 95.00
    wave_length: int = 4
    update_period: float = 2.00
    feedback: float = 1.00
    active_ratio: float = 0.85
    fade_ratio: float = 0.15

    @property
    def effect_index(self) -> FXType:
        return FXType.RETRIGGER

    def to_vox_string(self) -> str:
        return ",\t".join(
            [
                f"{self.effect_index.value}",
                f"{self.wave_length}",
                f"{self.mix:.2f}",
                f"{self.update_period:.2f}",
                f"{self.feedback:.2f}",
                f"{self.active_ratio:.2f}",
                f"{self.fade_ratio:.2f}",
            ]
        )


@_register_effect
@dataclass
class Gate(Effect):
    """A class representing a gate effect."""

    mix: float = 98.00
    wave_length: int = 16
    length: float = 2.00

    @property
    def effect_index(self) -> FXType:
        return FXType.GATE

    def to_vox_string(self) -> str:
        return ",\t".join([f"{self.effect_index.value}", f"{self.mix:.2f}", f"{self.wave_length}", f"{self.length:.2f}"])


@_register_effect
@dataclass
class Flanger(Effect):
    """A class representing a flanger effect."""

    # Parameter names yoinked off VoxCharger lol
    mix: float = 75.00
    period: float = 2.00
    feedback: float = 0.50
    stereo_width: int = 90
    hicut_gain: float = 2.00

    @property
    def effect_index(self) -> FXType:
        return FXType.FLANGER

    def to_vox_string(self) -> str:
        return ",\t".join(
            [
                f"{self.effect_index.value}",
                f"{self.mix:.2f}",
                f"{self.period:.2f}",
                f"{self.feedback:.2f}",
                f"{self.stereo_width}",
                f"{self.hicut_gain:.2f}",
            ]
        )


@_register_effect
@dataclass
class Tapestop(Effect):
    """A class representing a tapestop effect."""

    mix: float = 100.00
    speed: float = 8.00
    duration_seconds: float = 0.40

    @property
    def effect_index(self) -> FXType:
        return FXType.TAPESTOP

    def to_vox_string(self) -> str:
        return ",\t".join(
            [f"{self.effect_index.value}", f"{self.mix:.2f}", f"{self.speed:.2f}", f"{self.duration_seconds:.2f}"]
        )


@_register_effect
@dataclass
class Sidechain(Effect):
    """A class representing a sidechain effect."""

    mix: float = 90.00
    frequency: float = 1.00
    attack: int = 45
    hold: int = 50
    release: int = 60

    @property
    def effect_index(self) -> FXType:
        return FXType.SIDECHAIN

    def to_vox_string(self) -> str:
        return ",\t".join(
            [
                f"{self.effect_index.value}",
                f"{self.mix:.2f}",
                f"{self.frequency:.2f}",
                f"{self.attack}",
                f"{self.hold}",
                f"{self.release}",
            ]
        )


@_register_effect
@dataclass
class Wobble(Effect):
    """A class representing a wobble effect."""

    mix: float = 80.00
    filter_type: PassFilterType = PassFilterType.LOW_PASS
    wave_shape: WaveShape = WaveShape.TRIANGLE
    low_cutoff: float = 500.00
    hi_cutoff: float = 18000.00
    frequency: float = 4.00
    q: float = 1.40

    @property
    def effect_index(self) -> FXType:
        return FXType.WOBBLE

    def to_vox_string(self) -> str:
        return ",\t".join(
            [
                f"{self.effect_index.value}",
                f"{self.filter_type.value}",
                f"{self.wave_shape.value}",
                f"{self.mix:.2f}",
                f"{self.low_cutoff:.2f}",
                f"{self.hi_cutoff:.2f}",
                f"{self.frequency:.2f}",
                f"{self.q:.2f}",
            ]
        )


@_register_effect
@dataclass
class Bitcrush(Effect):
    """A class representing a bitcrush effect."""

    mix: float = 100.00
    hold_samples: int = 12

    @property
    def effect_index(self) -> FXType:
        return FXType.BITCRUSH

    def to_vox_string(self) -> str:
        return ",\t".join([f"{self.effect_index.value}", f"{self.mix:.2f}", f"{self.hold_samples}"])


@_register_effect
@dataclass
class RetriggerEx(Effect):
    """
    A class representing a retrigger effect.

    This effect samples from the start of the effect, instead of at the beginning of the update period.
    """

    mix: float = 95.00
    wave_length: int = 8
    update_period: float = 2.00
    feedback: float = 1.00
    active_ratio: float = 0.85
    fade_ratio: float = 0.15

    @property
    def effect_index(self) -> FXType:
        return FXType.RETRIGGER_EX

    def to_vox_string(self) -> str:
        return ",\t".join(
            [
                f"{self.effect_index.value}",
                f"{self.wave_length}",
                f"{self.mix:.2f}",
                f"{self.update_period:.2f}",
                f"{self.feedback:.2f}",
                f"{self.active_ratio:.2f}",
                f"{self.fade_ratio:.2f}",
            ]
        )


@_register_effect
@dataclass
class PitchShift(Effect):
    """A class representing a pitch shift effect."""

    mix: float = 100.00
    semitones: float = 12.00

    @property
    def effect_index(self) -> FXType:
        return FXType.PITCH_SHIFT

    def to_vox_string(self) -> str:
        return ",\t".join([f"{self.effect_index.value}", f"{self.mix:.2f}", f"{self.semitones:.2f}"])


@_register_effect
@dataclass
class PitchShiftEx(PitchShift):
    """A pitch shift variant with one currently unknown VOX parameter."""

    ex_param: float = 1.00

    @property
    def effect_index(self) -> FXType:
        return FXType.PITCH_SHIFT_EX

    def to_vox_string(self) -> str:
        return ",\t".join(
            [
                f"{self.effect_index.value}",
                f"{self.mix:.2f}",
                f"{self.semitones:.2f}",
                f"{self.ex_param:.2f}",
            ]
        )


@_register_effect
@dataclass
class Tapescratch(Effect):
    """A class representing a tapescratch effect."""

    mix: float = 100.00
    curve_slope: float = 5.00
    attack: float = 1.00
    hold: float = 0.10
    release: float = 1.00

    @property
    def effect_index(self) -> FXType:
        return FXType.TAPESCRATCH

    def to_vox_string(self) -> str:
        return ",\t".join(
            [
                f"{self.effect_index.value}",
                f"{self.mix:.2f}",
                f"{self.curve_slope:.2f}",
                f"{self.attack:.2f}",
                f"{self.hold:.2f}",
                f"{self.release:.2f}",
            ]
        )


@_register_effect
@dataclass
class LowpassFilter(Effect):
    """A class representing a low-pass filter effect."""

    mix: float = 75.00
    vol_cutoff_bound: float = 400.00
    cutoff: float = 900.00
    q: float = 2.00

    @property
    def effect_index(self) -> FXType:
        return FXType.LOW_PASS_FILTER

    def to_vox_string(self) -> str:
        return ",\t".join(
            [
                f"{self.effect_index.value}",
                f"{self.mix:.2f}",
                f"{self.vol_cutoff_bound:.2f}",
                f"{self.cutoff:.2f}",
                f"{self.q:.2f}",
            ]
        )


@_register_effect
@dataclass
class HighpassFilter(Effect):
    """A class representing a high-pass filter effect."""

    mix: float = 100.00
    cutoff: float = 2000.00
    vol_cutoff_bound: float = 5.00
    q: float = 1.40

    @property
    def effect_index(self) -> FXType:
        return FXType.HIGH_PASS_FILTER

    def to_vox_string(self) -> str:
        return ",\t".join(
            [
                f"{self.effect_index.value}",
                f"{self.mix:.2f}",
                f"{self.cutoff:.2f}",
                f"{self.vol_cutoff_bound:.2f}",
                f"{self.q:.2f}",
            ]
        )


@dataclass
class EffectEntry(VoxEntity):
    """
    A class representing a single effect setting.

    A single effect setting consists of two effects rendered together.
    """

    effect1: Effect = field(default_factory=NoEffect)
    effect2: Effect = field(default_factory=NoEffect)

    def __str__(self) -> str:
        return f"{self.effect1.effect_name}, {self.effect2.effect_name}"

    def to_vox_string(self) -> str:
        return f"{self.effect1.to_vox_string()}\n" f"{self.effect2.to_vox_string()}\n"


def enum_to_effect(val: FXType) -> type[Effect]:
    """Return the class corresponding to an enumeration member."""
    return _enumToEffect[val]


def from_vox_params(effect_index: int, params: Sequence[float]) -> Effect:
    """Construct an effect object from a VOX FXBUTTON EFFECT INFO row."""
    try:
        effect_type = FXType(effect_index)
    except ValueError:
        logger.warning(f"unknown effect index in VOX params: {effect_index}")
        return NoEffect()

    def get(index: int, default: float = 0.0) -> float:
        return params[index] if index < len(params) else default

    match effect_type:
        case FXType.NO_EFFECT:
            return NoEffect()
        case FXType.RETRIGGER:
            return Retrigger(
                wave_length=int(get(0, 4)),
                mix=get(1, 95.0),
                update_period=get(2, 2.0),
                feedback=get(3, 1.0),
                active_ratio=get(4, 0.85),
                fade_ratio=get(5, 0.15),
            )
        case FXType.GATE:
            return Gate(mix=get(0, 98.0), wave_length=int(get(1, 16)), length=get(2, 2.0))
        case FXType.FLANGER:
            return Flanger(
                mix=get(0, 75.0),
                period=get(1, 2.0),
                feedback=get(2, 0.5),
                stereo_width=int(get(3, 90)),
                hicut_gain=get(4, 2.0),
            )
        case FXType.TAPESTOP:
            return Tapestop(mix=get(0, 100.0), speed=get(1, 8.0), duration_seconds=get(2, 0.4))
        case FXType.SIDECHAIN:
            return Sidechain(
                mix=get(0, 90.0),
                frequency=get(1, 1.0),
                attack=int(get(2, 45)),
                hold=int(get(3, 50)),
                release=int(get(4, 60)),
            )
        case FXType.WOBBLE:
            return Wobble(
                filter_type=PassFilterType(int(get(0, PassFilterType.LOW_PASS.value))),
                wave_shape=WaveShape(int(get(1, WaveShape.TRIANGLE.value))),
                mix=get(2, 80.0),
                low_cutoff=get(3, 500.0),
                hi_cutoff=get(4, 18000.0),
                frequency=get(5, 4.0),
                q=get(6, 1.4),
            )
        case FXType.BITCRUSH:
            return Bitcrush(mix=get(0, 100.0), hold_samples=int(get(1, 12)))
        case FXType.RETRIGGER_EX:
            return RetriggerEx(
                wave_length=int(get(0, 8)),
                mix=get(1, 95.0),
                update_period=get(2, 2.0),
                feedback=get(3, 1.0),
                active_ratio=get(4, 0.85),
                fade_ratio=get(5, 0.15),
            )
        case FXType.PITCH_SHIFT:
            return PitchShift(mix=get(0, 100.0), semitones=get(1, 12.0))
        case FXType.PITCH_SHIFT_EX:
            return PitchShiftEx(
                mix=get(0, 100.0),
                semitones=get(1, 12.0),
                ex_param=get(2, 1.0),
            )
        case FXType.TAPESCRATCH:
            return Tapescratch(
                mix=get(0, 100.0),
                curve_slope=get(1, 5.0),
                attack=get(2, 1.0),
                hold=get(3, 0.1),
                release=get(4, 1.0),
            )
        case FXType.LOW_PASS_FILTER:
            return LowpassFilter(mix=get(0, 75.0), vol_cutoff_bound=get(1, 400.0), cutoff=get(2, 900.0), q=get(3, 2.0))
        case FXType.HIGH_PASS_FILTER:
            return HighpassFilter(mix=get(0, 100.0), cutoff=get(1, 2000.0), vol_cutoff_bound=get(2, 5.0), q=get(3, 1.4))


def get_default_effects() -> list[EffectEntry]:
    """Get the default effect settings."""
    return [
        # Re8
        EffectEntry(Retrigger()),
        # Re16
        EffectEntry(Retrigger(wave_length=8, fade_ratio=0.1)),
        # Ga16
        EffectEntry(Gate()),
        # Flanger
        EffectEntry(Flanger()),
        # Re32
        EffectEntry(Retrigger(wave_length=16, active_ratio=0.87, fade_ratio=0.13)),
        # Ga8
        EffectEntry(Gate(wave_length=4)),
        # Echo4
        EffectEntry(RetriggerEx(mix=100, wave_length=4, update_period=4, feedback=0.6, active_ratio=1, fade_ratio=0.8)),
        # Tapestop
        EffectEntry(Tapestop()),
        # Sidechain
        EffectEntry(Sidechain()),
        # Wo12
        EffectEntry(Wobble()),
        # Re12
        EffectEntry(Retrigger(wave_length=6)),
        # Bitcrush
        EffectEntry(Bitcrush()),
    ]
