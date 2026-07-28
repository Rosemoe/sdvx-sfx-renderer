"""
Classes that represent chart-related entities.
"""
import itertools
import logging
import math

from collections.abc import Iterable
from dataclasses import dataclass, field, InitVar
from decimal import Decimal
from fractions import Fraction
from typing import Any

from .base import (
    Validateable,
)
from .effects import (
    Effect,
    EffectEntry,
    get_default_effects,
)
from .enums import (
    EasingType,
    NoteType,
    SegmentFlag,
    SpinType,
    TiltType,
)
from .filters import (
    TabParamAssignEntry,
    get_default_filters,
)
from .time import (
    TimePoint,
    TimeSignature,
)

__all__ = [
    "BTInfo",
    "FXInfo",
    "VolInfo",
    "OriginalVolData",
    "NoteData",
    "SPControllerInfo",
    "SPControllerData",
    "parse_spcontroller_param",
    "AutoTabInfo",
    "LyricInfo",
    "PostEffectInfo",
    "ChartInfo",
]

TICKS_PER_BAR = 192
"""Number of ticks in a single 4/4 bar."""

HALF_TICK_BPM_THRESHOLD = Decimal("255")
"""Threshold for the BPM at which the tick rate halves."""

def parse_spcontroller_param(value: str) -> float | int | str:
    """Parse one SPCONTROLER parameter, preserving unrecognized values as strings."""

    try:
        if value.startswith("d"):
            return float(value[1:]) * math.pi / 180
        if value.startswith("f"):
            return float(value[1:])
        if value.startswith("i"):
            return int(value[1:])
        if value.startswith("x"):
            return int(value[1:], 16)
        return float(value)
    except ValueError:
        return value

logger = logging.getLogger(__name__)


@dataclass
class BTInfo(Validateable):
    """A class that represents a BT object."""

    _duration: InitVar[Fraction | int]
    duration: Fraction = field(init=False)
    special: int = 0
    param_ex: int = 0

    def _setattrhook(self, __name: str, __value: Any):
        super().__setattr__(__name, __value)
        self.validate()

    def __post_init__(self, _duration):
        self.duration = Fraction(_duration)
        self.validate()
        self.__setattr__ = self._setattrhook

    def validate(self):
        if self.duration < 0:
            raise ValueError(f"duration cannot be negative (got {self.duration})")
        if self.special < 0:
            raise ValueError(f"special must be positive (got {self.special})")

    def duration_as_tick(self) -> int:
        """Convert the button duration to tick count."""
        return round(TICKS_PER_BAR * self.duration)


@dataclass
class FXInfo(Validateable):
    """A class that represents an FX object."""

    _duration: InitVar[Fraction | int]
    duration: Fraction = field(init=False)
    special: int
    param_ex: int = 0

    def _setattrhook(self, __name: str, __value: Any):
        super().__setattr__(__name, __value)
        self.validate()

    def __post_init__(self, _duration):
        self.duration = Fraction(_duration)
        self.validate()
        self.__setattr__ = self._setattrhook

    def validate(self):
        if self.duration < 0:
            raise ValueError(f"duration cannot be negative (got {self.duration})")
        if self.special < 0:
            raise ValueError(f"special must be positive (got {self.special})")

    def duration_as_tick(self) -> int:
        """Convert the button duration to tick count."""
        return round(TICKS_PER_BAR * self.duration)


@dataclass
class VolInfo(Validateable):
    """A class that represents a singular point on a VOL segment."""

    start: Fraction
    end: Fraction
    spin_type: SpinType = SpinType.NO_SPIN
    spin_duration: int = 0
    ease_type: EasingType = EasingType.NO_EASING
    filter_index: int = 0
    point_type: SegmentFlag = SegmentFlag.START
    wide_laser: int = 0
    param_ex_1: int = 0
    param_ex_2: int = 0
    interpolated: bool = False

    def _setattrhook(self, __name: str, __value: Any):
        super().__setattr__(__name, __value)
        self.validate()

    def __post_init__(self):
        self.validate()
        self.__setattr__ = self._setattrhook

    def validate(self):
        if not 0 <= self.start <= 1:
            raise ValueError(f"start value out of range (got {self.start})")
        if not 0 <= self.end <= 1:
            raise ValueError(f"end value out of range (got {self.end})")
        if self.spin_duration < 0:
            raise ValueError(f"spin_duration cannot be negative (got {self.spin_duration})")


@dataclass
class OriginalVolData:
    """VOL points from the VOX ``TRACK ORIGINAL L/R`` sections."""

    vol_l: dict[TimePoint, VolInfo] = field(default_factory=dict)
    vol_r: dict[TimePoint, VolInfo] = field(default_factory=dict)


@dataclass
class NoteData:
    """A class encapsulating all the note data in a chart."""

    # BT
    bt_a: dict[TimePoint, BTInfo] = field(default_factory=dict)
    bt_b: dict[TimePoint, BTInfo] = field(default_factory=dict)
    bt_c: dict[TimePoint, BTInfo] = field(default_factory=dict)
    bt_d: dict[TimePoint, BTInfo] = field(default_factory=dict)

    # FX
    fx_l: dict[TimePoint, FXInfo] = field(default_factory=dict)
    fx_r: dict[TimePoint, FXInfo] = field(default_factory=dict)

    # VOL
    vol_l: dict[TimePoint, VolInfo] = field(default_factory=dict)
    vol_r: dict[TimePoint, VolInfo] = field(default_factory=dict)

    def iter_bts(self) -> Iterable[tuple[NoteType, TimePoint, BTInfo]]:
        """
        Iterate through every BT object.

        :returns: A generator that emits a 3-tuple of: note type, time point of the note object, and note object
            containing the note's data.
        """
        dicts: list[tuple[NoteType, dict]] = [
            (NoteType.BT_A, self.bt_a),
            (NoteType.BT_B, self.bt_b),
            (NoteType.BT_C, self.bt_c),
            (NoteType.BT_D, self.bt_d),
        ]
        for note_type, note_dict in dicts:
            for key, value in note_dict.items():
                yield note_type, key, value

    def iter_fxs(self) -> Iterable[tuple[NoteType, TimePoint, FXInfo]]:
        """
        Iterate through every FX object.

        :returns: A generator that emits a 3-tuple of: note type, time point of the note object, and note object
            containing the note's data.
        """
        dicts: list[tuple[NoteType, dict]] = [
            (NoteType.FX_L, self.fx_l),
            (NoteType.FX_R, self.fx_r),
        ]
        for note_type, note_dict in dicts:
            for key, value in note_dict.items():
                yield note_type, key, value

    def iter_vols(self, *, add_dummy=False) -> Iterable[tuple[NoteType, TimePoint, VolInfo]]:
        """
        Iterate through every VOL point.

        :param add_dummy: If `True`, this method will emit a dummy point at the end. This is used for pairwise
            iteration.
        :returns: A generator that emits a 3-tuple of: note type, time point of the note object, and note object
            containing the note's data.
        """
        dicts: list[tuple[NoteType, dict]] = [
            (NoteType.VOL_L, self.vol_l),
            (NoteType.VOL_R, self.vol_r),
        ]
        is_empty_loop = True
        key, value = TimePoint(), VolInfo(Fraction(), Fraction())
        for note_type, note_dict in dicts:
            for key, value in note_dict.items():
                is_empty_loop = False
                yield note_type, key, value
        if not is_empty_loop and add_dummy:
            yield NoteType.DUMMY, key, value

    def iter_buttons(self) -> Iterable[tuple[NoteType, TimePoint, BTInfo | FXInfo]]:
        """Iterate through every BT and FX object.

        :returns: A generator that emits a 3-tuple of: note type, time point of the note object, and note object
            containing the note's data."""
        dicts: list[tuple[NoteType, dict]] = [
            (NoteType.BT_A, self.bt_a),
            (NoteType.BT_B, self.bt_b),
            (NoteType.BT_C, self.bt_c),
            (NoteType.BT_D, self.bt_d),
            (NoteType.FX_L, self.fx_l),
            (NoteType.FX_R, self.fx_r),
        ]
        for note_type, note_dict in dicts:
            for key, value in note_dict.items():
                yield note_type, key, value

    def iter_notes(self) -> Iterable[tuple[NoteType, TimePoint, BTInfo | FXInfo | VolInfo]]:
        """
        Iterate through every note object and VOL point.

        :returns: A generator that emits a 3-tuple of: note type, time point of the note object, and note object
            containing the note's data.
        """
        dicts: list[tuple[NoteType, dict]] = [
            (NoteType.BT_A, self.bt_a),
            (NoteType.BT_B, self.bt_b),
            (NoteType.BT_C, self.bt_c),
            (NoteType.BT_D, self.bt_d),
            (NoteType.FX_L, self.fx_l),
            (NoteType.FX_R, self.fx_r),
            (NoteType.VOL_L, self.vol_l),
            (NoteType.VOL_R, self.vol_r),
        ]
        for note_type, note_dict in dicts:
            for key, value in note_dict.items():
                yield note_type, key, value


@dataclass
class SPControllerInfo:
    """A raw entry from the VOX ``SPCONTROLER`` section."""

    timepoint: TimePoint
    sp_type: str
    sp_subtype: str
    duration: int
    params: tuple[str, str, str, str]


@dataclass
class SPControllerData:
    """Raw SPCONTROLER entries."""

    entries: list[SPControllerInfo] = field(default_factory=list)


@dataclass
class AutoTabInfo:
    """A timed assignment of one tab effect to a laser segment."""

    duration: Fraction
    effect_index: int


@dataclass
class LyricInfo:
    """Timed lyric text from the VOX ``LYRIC INFO`` section."""

    duration: Fraction
    text: str


@dataclass
class PostEffectInfo:
    """One raw entry from the VOX ``POSTEFFECT`` section.

    ``start`` is a grid :class:`TimePoint` or an integer millisecond offset.
    ``duration`` is a beat fraction converted from VOX ticks or an integer
    millisecond duration.
    """

    start: TimePoint | int
    unknown_1: int
    duration: Fraction | int
    effect_name: str
    unknown_2: int
    unknown_3: int
    property_name: str
    start_value: float
    end_value: float


@dataclass
class ChartInfo:
    """Parsed data from one SOUND VOLTEX VOX chart."""

    # VOX header
    format_version: int = 0
    beat_resolution: int | None = None

    # Chart bounds
    end_measure: int = 0
    end_position: TimePoint | None = None

    # Calculated data
    _chip_notecount: int = -1
    _long_notecount: int = -1
    _vol_notecount: int = -1

    # Song data that may change mid-song
    bpms: dict[TimePoint, Decimal] = field(default_factory=dict)
    timesigs: dict[TimePoint, TimeSignature] = field(default_factory=dict)
    stops: dict[TimePoint, bool] = field(default_factory=dict)
    tilt_type: dict[TimePoint, TiltType] = field(default_factory=dict)
    lyrics: dict[TimePoint, LyricInfo] = field(default_factory=dict)

    # Effect info
    effect_list: list[EffectEntry] = field(default_factory=list)
    filter_list: list[Effect] = field(default_factory=list)
    autotab_params: list[TabParamAssignEntry] = field(default_factory=list)

    autotab_infos: dict[TimePoint, AutoTabInfo] = field(default_factory=dict)
    post_effect_infos: list[PostEffectInfo] = field(default_factory=list)

    # Actual chart data
    note_data: NoteData = field(default_factory=NoteData)
    original_vol_data: OriginalVolData = field(default_factory=OriginalVolData)

    # SPController data
    spcontroller_data: SPControllerData = field(default_factory=SPControllerData)
    locked_spcontroller_data: SPControllerData = field(default_factory=SPControllerData)

    # Scripting assist
    script_definitions: dict[int, str] = field(default_factory=dict)
    script_ids: dict[NoteType, dict[TimePoint, list[int]]] = field(default_factory=dict)

    # Private data
    # Name to effect object mapping
    _custom_effect: dict[str, Effect] = field(default_factory=dict, init=False, repr=False)
    _custom_filter: dict[str, Effect] = field(default_factory=dict, init=False, repr=False)

    # Timing cache
    _elapsed_time: dict[TimePoint, Decimal] = field(default_factory=dict, init=False, repr=False)
    _elapsed_time_bpm: dict[TimePoint, Decimal] = field(default_factory=dict, init=False, repr=False)
    _bpm_durations: dict[Decimal, Decimal] = field(default_factory=dict, init=False, repr=False)

    # Cached values
    _timesig_cache: dict[int, TimeSignature] = field(default_factory=dict, init=False, repr=False)
    _bpm_cache: dict[TimePoint, Decimal] = field(default_factory=dict, init=False, repr=False)
    _tickrate_cache: dict[TimePoint, Fraction] = field(default_factory=dict, init=False, repr=False)
    _time_to_frac_cache: dict[TimePoint, Fraction] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        # Default values
        self.bpms[TimePoint()] = Decimal("120")
        self.timesigs[TimePoint()] = TimeSignature()
        self.tilt_type[TimePoint()] = TiltType.NORMAL

        # Populate filter list
        self.filter_list = get_default_filters()

        # Populate effect list
        self.effect_list = get_default_effects()

    # TODO: Look into why sometimes this predicts wrong long/tsumami counts
    def _calculate_notecounts(self) -> None:
        """
        Calculate the chart's note counts.

        This is called automatically when getting the note counts for the first time.
        """
        self._chip_notecount = 0
        self._long_notecount = 0
        self._vol_notecount = 0

        # Chip and long notes
        for note_type, timept, note in self.note_data.iter_buttons():
            if note.duration == 0:
                self._chip_notecount += 1
            else:
                tick_rate = self.get_tick_rate(timept)
                tick_start = self.timepoint_to_fraction(timept)
                hold_end = self.add_duration(timept, note.duration)
                cur_hold_ticks = 0
                # Round up to next tick
                if tick_start % tick_rate != 0:
                    cur_hold_ticks += 1
                    timept = self.add_duration(timept, tick_rate - (tick_start % tick_rate))
                # Add ticks
                while timept < hold_end:
                    cur_hold_ticks += 1
                    tick_rate = self.get_tick_rate(timept)
                    timept = self.add_duration(timept, tick_rate)
                # Long enough holds become lenient at the end
                if cur_hold_ticks > 5:
                    cur_hold_ticks -= 1
                if cur_hold_ticks > 6:
                    cur_hold_ticks -= 1
                self._long_notecount += cur_hold_ticks

        # Lasers
        cur_note_type: NoteType | None = None
        cur_note_type = NoteType.DUMMY
        laser_start, laser_end = TimePoint(), TimePoint()
        slam_locations: list[TimePoint] = []
        for note_type, timept, laser in self.note_data.iter_vols():
            # Reset state variables when changing tracks
            if note_type != cur_note_type:
                cur_note_type = note_type
                laser_start, laser_end = TimePoint(), TimePoint()
                slam_locations: list[TimePoint] = []
            # This really should only be slams
            if laser.point_type == SegmentFlag.POINT:
                self._vol_notecount += 1
            elif laser.point_type in [SegmentFlag.START, SegmentFlag.END]:
                if laser.start != laser.end:
                    slam_locations.append(timept)
                if laser.point_type == SegmentFlag.START:
                    laser_start = timept
                elif laser.point_type == SegmentFlag.END:
                    laser_end = timept
                    logger.debug(
                        f"laser segment: {self.timepoint_to_vox(laser_start)} => {self.timepoint_to_vox(laser_end)}"
                    )
                    # Process ticks
                    tick_rate = self.get_tick_rate(laser_start)
                    tick_start = self.timepoint_to_fraction(laser_start)
                    # Round up to next tick
                    if tick_start % tick_rate != 0:
                        laser_start = self.add_duration(laser_start, tick_rate - (tick_start % tick_rate))
                    timept = laser_start
                    # Get tick locations
                    tick_locations: dict[TimePoint, bool] = {}
                    tick_keys: list[TimePoint] = []
                    while timept < laser_end:
                        tick_locations[timept] = True
                        tick_keys.append(timept)
                        tick_rate = self.get_tick_rate(timept)
                        timept = self.add_duration(timept, tick_rate)
                    # Mark ticks as "occupied" by slams
                    tick_index = -1
                    for slam in slam_locations:
                        if not tick_keys:
                            break
                        while tick_index < len(tick_keys) - 1 and tick_keys[tick_index + 1] < slam:
                            tick_index += 1
                        if tick_index == -1:
                            tick_locations[tick_keys[0]] = False
                        elif tick_index == len(tick_keys) - 1:
                            tick_rate = self.get_tick_rate(tick_keys[tick_index])
                            next_tick_timept = self.add_duration(tick_keys[tick_index], tick_rate)
                            if slam < next_tick_timept:
                                tick_locations[tick_keys[tick_index]] = False
                        else:
                            tick_rate = self.get_tick_rate(tick_keys[tick_index])
                            halfway_timept = self.add_duration(tick_keys[tick_index], tick_rate / 2)
                            if slam <= halfway_timept:
                                tick_locations[tick_keys[tick_index]] = False
                            if slam >= halfway_timept:
                                tick_locations[tick_keys[tick_index + 1]] = False
                    disabled_ticks = [k for k, v in tick_locations.items() if not v]
                    if disabled_ticks:
                        logger.debug(f"disabled tick: {[self.timepoint_to_vox(t) for t in disabled_ticks]}")
                    self._vol_notecount += len(slam_locations) + sum(tick_locations.values())
                    slam_locations = []
            else:
                if laser.start != laser.end:
                    slam_locations.append(timept)

    # Figure out how long each particular BPM lasts
    # Helpful to figure out time elapsed for a particular note
    def _calculate_bpm_durations(self, endpoint: TimePoint) -> None:
        """
        Calculate the time elapsed between each BPM change.

        This function is a prerequisite to :meth:`~vox_parser.classes.chart.ChartInfo._get_elapsed_time`.

        :param endpoint: The time point for the end of the chart.
        """
        running_total = Decimal()
        for timept_i, timept_f in itertools.pairwise([*self.bpms.keys(), endpoint]):
            # First BPM point should be at 001,01,00, which means elapsed time is 0 sec.
            if not self._elapsed_time_bpm:
                self._elapsed_time_bpm[timept_i] = Decimal()
            cur_bpm = self.bpms[timept_i]
            # Distance is in fractions of 4 beats -- i.e. 1 distance = 4 beats
            bpm_distance = self.get_distance(timept_i, timept_f)
            # Inverse of BPM is in minutes/beat
            # Multiply that with distance to get duration in minutes
            # So we need to multiply with 60 sec/min
            # tl;dr: 1 / bpm (min/beat) * 60 (sec/min) * distance (dist) * 4 (beats/dist)
            bpm_duration = 1 / cur_bpm * 4 * 60 * bpm_distance.numerator / bpm_distance.denominator
            if cur_bpm not in self._bpm_durations:
                self._bpm_durations[cur_bpm] = Decimal()
            self._bpm_durations[cur_bpm] += bpm_duration
            running_total += bpm_duration
            self._elapsed_time_bpm[timept_f] = running_total

        self._elapsed_time = dict(self._elapsed_time_bpm)

    def _get_elapsed_time(self, timept: TimePoint) -> Decimal:
        """Convert timepoint into seconds."""
        if timept not in self._elapsed_time:
            prev_elapsed_time = Decimal()
            prev_timept = TimePoint()
            for cur_timept, elapsed_time in self._elapsed_time.items():
                if cur_timept > timept:
                    break
                prev_elapsed_time = elapsed_time
                prev_timept = cur_timept
            # Similar calculation as in _populate_bpm_durations
            cur_bpm = self.get_bpm(prev_timept)
            note_distance_frac = self.get_distance(timept, prev_timept)
            note_distance = 1 / cur_bpm * 4 * 60 * note_distance_frac.numerator / note_distance_frac.denominator
            self._elapsed_time[timept] = prev_elapsed_time + note_distance

        return self._elapsed_time[timept]

    def get_timesig(self, measure: int) -> TimeSignature:
        """
        Fetch the prevailing time signature at the given measure.

        :param measure: The measure number (measure starts from 1).
        :returns: The measure's time signature.
        """
        if measure not in self._timesig_cache:
            prev_timesig = TimeSignature()
            for timept, timesig in self.timesigs.items():
                if timept.measure > measure:
                    break
                prev_timesig = timesig
            self._timesig_cache[measure] = prev_timesig

        return self._timesig_cache[measure]

    def get_bpm(self, timepoint: TimePoint) -> Decimal:
        """
        Fetch the prevailing BPM at the given time point.

        :param timepoint: The time point to query.
        :returns: The chart's BPM at that time point.
        """
        if timepoint not in self._bpm_cache:
            prev_bpm = Decimal()
            for timept, bpm in self.bpms.items():
                if timept > timepoint:
                    break
                prev_bpm = bpm
            self._bpm_cache[timepoint] = prev_bpm

        return self._bpm_cache[timepoint]

    def get_tick_rate(self, timepoint: TimePoint) -> Fraction:
        """
        Fetch the prevailing tick rate at the given time point.

        :param measure: The point to query.
        :returns: The active tick rate at that time point. Holds and lasers tick at this rate.
        """
        if timepoint not in self._tickrate_cache:
            self._tickrate_cache[timepoint] = (
                Fraction(1, 16) if self.get_bpm(timepoint) < HALF_TICK_BPM_THRESHOLD else Fraction(1, 8)
            )

        return self._tickrate_cache[timepoint]

    def get_distance(self, a: TimePoint, b: TimePoint) -> Fraction:
        """
        Calculate the distance between two timepoints as a fraction.

        :param a: The first time point.
        :param b: The second time point.
        :returns: The distance between two timepoints. This is always non-negative.
        """
        if a == b:
            return Fraction()
        if b < a:
            a, b = b, a

        distance = Fraction()
        for m_no in range(a.measure, b.measure):
            distance += self.get_timesig(m_no).as_fraction()
        distance += b.position - a.position

        return distance

    def add_duration(self, a: TimePoint, b: Fraction | int) -> TimePoint:
        """
        Calculate the resulting timepoint after adding an amount of time to a timepoint.

        If the second argument is an integer, it is assumed to be the tick count.

        :param a: The starting time point.
        :param b: The amount to add to the time point.
        :returns: The resulting time point.
        """
        if isinstance(b, Fraction):
            modified_length = a.position + b
        else:
            modified_length = a.position + Fraction(b, TICKS_PER_BAR)

        m_no = a.measure
        while modified_length >= (m_len := self.get_timesig(m_no).as_fraction()):
            modified_length -= m_len
            m_no += 1

        return TimePoint(m_no, modified_length.numerator, modified_length.denominator)

    def timepoint_to_vox(self, timepoint: TimePoint) -> str:
        """
        Convert a timepoint to string in VOX format.

        This requires the time signature data.

        :param timepoint: The time point to convert.
        :returns: A string representing the time point in VOX format.
        """
        timesig = self.get_timesig(timepoint.measure)

        note_val = Fraction(1, timesig.lower)
        div = timepoint.position // note_val
        subdiv = round(TICKS_PER_BAR * (timepoint.position % note_val))

        return f"{timepoint.measure:03},{div + 1:02},{subdiv:02}"

    def timepoint_to_fraction(self, timepoint: TimePoint) -> Fraction:
        """
        Convert a timepoint to a fraction representation.

        This requires the time signature data.

        :param timepoint: The time point to convert.
        :returns: A fraction representing the time point.
        """
        if timepoint not in self._time_to_frac_cache:
            if timepoint == TimePoint():
                self._time_to_frac_cache[timepoint] = Fraction()
            elif timepoint.position == 0:
                prev_timepoint = TimePoint(timepoint.measure - 1, 0, 1)
                prev_timesig = self.get_timesig(timepoint.measure - 1)
                self._time_to_frac_cache[timepoint] = (
                    self.timepoint_to_fraction(prev_timepoint) + prev_timesig.as_fraction()
                )
            else:
                prev_timepoint = TimePoint(timepoint.measure, 0, 1)
                self._time_to_frac_cache[timepoint] = self.timepoint_to_fraction(prev_timepoint) + timepoint.position

        return self._time_to_frac_cache[timepoint]

    @property
    def chip_notecount(self) -> int:
        """The number of chip notes in the chart."""
        if self._chip_notecount == -1:
            self._calculate_notecounts()
        return self._chip_notecount

    @property
    def long_notecount(self) -> int:
        """The number of long notes in the chart."""
        if self._long_notecount == -1:
            self._calculate_notecounts()
        return self._long_notecount

    @property
    def vol_notecount(self) -> int:
        """The number of laser notes in the chart."""
        if self._vol_notecount == -1:
            self._calculate_notecounts()
        return self._vol_notecount

    @property
    def max_chain(self) -> int:
        """The total chain of the chart."""
        return self.chip_notecount + self.long_notecount + self.vol_notecount

    @property
    def max_ex_score(self) -> int:
        """The total ex score of the chart."""
        return 5 * self.chip_notecount + 2 * (self.long_notecount + self.vol_notecount)
