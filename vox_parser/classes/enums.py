"""Enumerations used by SOUND VOLTEX VOX charts."""
from enum import Enum, Flag, auto

__all__ = [
    "SpinType",
    "EasingType",
    "TiltType",
    "SegmentFlag",
    "VOXSection",
    "NoteType",
]


class SpinType(Enum):
    """Lane spin type occurring with laser slams."""

    NO_SPIN = 0
    SINGLE_SPIN = 1
    SINGLE_SPIN_2 = 2
    SINGLE_SPIN_3 = 3
    TRIPLE_SPIN = 4
    HALF_SPIN = 5


class EasingType(Enum):
    """Laser segment easing type."""

    NO_EASING = 0
    LINEAR = 2
    EASE_IN_SINE = 4
    EASE_OUT_SINE = 5


class TiltType(Enum):
    """Lane tilt type."""

    NORMAL = 0
    BIGGER = 1
    KEEP = 2


class SegmentFlag(Flag):
    """Laser segment marker."""

    MIDDLE = 0
    START = 1
    END = 2
    POINT = 3


class VOXSection(Enum):
    """VOX file sections understood by the parser."""

    NONE = 0
    VERSION = auto()
    BEAT_RESOLUTION = auto()
    TIME_SIGNATURE = auto()
    BPM = auto()
    TILT = auto()
    LYRICS = auto()
    END_POSITION = auto()
    FILTER_PARAMS = auto()
    EFFECT_PARAMS = auto()
    TAB_PARAM_ASSIGN = auto()
    REVERB = auto()
    POST_EFFECT = auto()
    TRACK_VOL_L = auto()
    TRACK_FX_L = auto()
    TRACK_BT_A = auto()
    TRACK_BT_B = auto()
    TRACK_BT_C = auto()
    TRACK_BT_D = auto()
    TRACK_FX_R = auto()
    TRACK_VOL_R = auto()
    AUTOTAB_INFO = auto()
    TRACK_VOL_L_ORIG = auto()
    TRACK_VOL_R_ORIG = auto()
    SPCONTROLLER = auto()
    LOCKED_SPCONTROLLER = auto()
    SCRIPT = auto()
    SCRIPTED_TRACK = auto()


class NoteType(Flag):
    """VOX note lane types."""

    VOL_R = auto()
    FX_R = auto()
    BT_D = auto()
    BT_C = auto()
    BT_B = auto()
    BT_A = auto()
    FX_L = auto()
    VOL_L = auto()
    DUMMY = auto()

    def __str__(self) -> str:
        if self.name is None:
            return super().__str__()
        return self.name.replace("_", "-")
