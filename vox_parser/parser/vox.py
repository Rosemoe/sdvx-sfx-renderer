"""Parser for SOUND VOLTEX VOX chart files."""
import logging
import re

from decimal import Decimal
from fractions import Fraction
from typing import TextIO

from ..classes.chart import (
    AutoTabInfo,
    BTInfo,
    ChartInfo,
    FXInfo,
    LyricInfo,
    PostEffectInfo,
    SPControllerInfo,
    VolInfo,
    TICKS_PER_BAR,
)
from ..classes.effects import (
    Bitcrush,
    Effect,
    EffectEntry,
    HighpassFilter,
    LowpassFilter,
    from_vox_params,
)
from ..classes.filters import (
    AutoTabParam,
)
from ..classes.enums import (
    EasingType,
    NoteType,
    SegmentFlag,
    SpinType,
    TiltType,
    VOXSection,
)
from ..classes.time import (
    TimePoint,
    TimeSignature,
)

__all__ = [
    "VOXParser",
]

SECTION_MAP: dict[str, VOXSection] = {
    "END": VOXSection.NONE,
    "FORMAT VERSION": VOXSection.VERSION,
    "BEAT RESOLUTION": VOXSection.BEAT_RESOLUTION,
    "BEAT INFO": VOXSection.TIME_SIGNATURE,
    "BPM INFO": VOXSection.BPM,
    "TILT MODE INFO": VOXSection.TILT,
    "LYRIC INFO": VOXSection.LYRICS,
    "END POSITION": VOXSection.END_POSITION,
    "TAB EFFECT INFO": VOXSection.FILTER_PARAMS,
    "FXBUTTON EFFECT INFO": VOXSection.EFFECT_PARAMS,
    "TAB PARAM ASSIGN INFO": VOXSection.AUTOTAB_PARAMS,
    "REVERB EFFECT PARAM": VOXSection.REVERB,
    "POSTEFFECT": VOXSection.POST_EFFECT,
    "TRACK1": VOXSection.TRACK_VOL_L,
    "TRACK2": VOXSection.TRACK_FX_L,
    "TRACK3": VOXSection.TRACK_BT_A,
    "TRACK4": VOXSection.TRACK_BT_B,
    "TRACK5": VOXSection.TRACK_BT_C,
    "TRACK6": VOXSection.TRACK_BT_D,
    "TRACK7": VOXSection.TRACK_FX_R,
    "TRACK8": VOXSection.TRACK_VOL_R,
    "TRACK AUTO TAB": VOXSection.AUTOTAB_INFO,
    "TRACK ORIGINAL L": VOXSection.TRACK_VOL_L_ORIG,
    "TRACK ORIGINAL R": VOXSection.TRACK_VOL_R_ORIG,
    "SPCONTROLER": VOXSection.SPCONTROLLER,
    "LOCKED_SPCONTROLER": VOXSection.LOCKED_SPCONTROLLER,
    "SCRIPT_DEFINE": VOXSection.SCRIPT,
    "SCRIPTED_TRACK1": VOXSection.SCRIPTED_TRACK,
    "SCRIPTED_TRACK2": VOXSection.SCRIPTED_TRACK,
    "SCRIPTED_TRACK3": VOXSection.SCRIPTED_TRACK,
    "SCRIPTED_TRACK4": VOXSection.SCRIPTED_TRACK,
    "SCRIPTED_TRACK5": VOXSection.SCRIPTED_TRACK,
    "SCRIPTED_TRACK6": VOXSection.SCRIPTED_TRACK,
    "SCRIPTED_TRACK7": VOXSection.SCRIPTED_TRACK,
    "SCRIPTED_TRACK8": VOXSection.SCRIPTED_TRACK,
}
SCRIPTED_TRACK_MAP: dict[str, NoteType] = {
    "SCRIPTED_TRACK1": NoteType.VOL_L,
    "SCRIPTED_TRACK2": NoteType.FX_L,
    "SCRIPTED_TRACK3": NoteType.BT_A,
    "SCRIPTED_TRACK4": NoteType.BT_B,
    "SCRIPTED_TRACK5": NoteType.BT_C,
    "SCRIPTED_TRACK6": NoteType.BT_D,
    "SCRIPTED_TRACK7": NoteType.FX_R,
    "SCRIPTED_TRACK8": NoteType.VOL_R,
}
VOL_TRACK_REGEX = re.compile(
    r"^(?P<timepoint>\d+,\d+,\d+)\s+"
    r"(?P<position>-?(?:\d+(?:\.\d*)?|\.\d+))\s+"
    r"(?P<segment_type>\d+)\s+(?P<spin_type>\d+)\s+(?P<filter_type>\d+)"
    r"(?P<extra_params>(?:\s+\S+)*)\s*$"
)
STOF_PREFIX_REGEX = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
SPCONTROLLER_REGEX = re.compile(
    r"(?P<timepoint>\d+,\d+,\d+)\s+"
    r"(?P<sp_type>\S+)\s+(?P<sp_subtype>\S+)\s+"
    r"(?P<duration>\d+)\s+"
    r"(?P<param_1>\S+)\s+(?P<param_2>\S+)\s+"
    r"(?P<param_3>\S+)\s+(?P<param_4>\S+)"
)

# fmt: off
SECTION_REGEX: dict[VOXSection, re.Pattern] = {
    VOXSection.NONE            : re.compile(r"(?!)"),
    VOXSection.VERSION         : re.compile(r"(?P<version>\d+)"),
    VOXSection.BEAT_RESOLUTION : re.compile(r"(?P<resolution>\d+)"),
    VOXSection.TIME_SIGNATURE  : re.compile(r"(?P<timepoint>\d+,\d+,\d+)\s+(?P<upper>\d+)\s+(?P<lower>\d+)"),
    VOXSection.BPM             : re.compile(r"(?P<timepoint>\d+,\d+,\d+)\s+(?P<bpm>\d+\.\d+)\s+(?P<unknown>\d+-?)"),
    VOXSection.TILT            : re.compile(r"(?P<timepoint>\d+,\d+,\d+)(\s+(?P<tilt_type>\d))?"),
    VOXSection.LYRICS          : re.compile(r"(?P<timepoint>\d+,\d+,\d+)\s+(?P<duration>\d+)\s+(?P<text>.+)"),
    VOXSection.END_POSITION    : re.compile(r"(?P<timepoint>\d+,\d+,\d+)"),
    VOXSection.FILTER_PARAMS   : re.compile(r"(?P<filter_index>\d+)(?P<content>(,\s+\d+(\.\d+)?))+"),
    VOXSection.EFFECT_PARAMS   : re.compile(r"(?P<effect_index>\d+)(?P<content>(,\s+\d+(\.\d+)?))+"),
    VOXSection.AUTOTAB_PARAMS  : re.compile(r"(?P<index>\d+),\s+(?P<param_index>\d+),\s+"
                                            r"(?P<param_start>\d+(\.\d+)?),\s+(?P<param_end>\d+(\.\d+)?)"),
    VOXSection.REVERB          : re.compile(r"(?P<timepoint>\d+,\d+,\d+)\s+"
                                            r"(?P<param_1>-?\d+(?:\.\d+)?)\s+"
                                            r"(?P<param_2>-?\d+(?:\.\d+)?)\s+"
                                            r"(?P<param_3>-?\d+(?:\.\d+)?)\s+"
                                            r"(?P<param_4>-?\d+(?:\.\d+)?)"),
    VOXSection.POST_EFFECT      : re.compile(r"(?P<start>\d+,\d+,\d+|\d+ms)\t"
                                            r"(?P<unknown_1>-?\d+)\t"
                                            r"(?P<duration>\d+|\d+ms)\t"
                                            r"(?P<effect_name>[^\t]+)\t"
                                            r"(?P<unknown_2>-?\d+)\t"
                                            r"(?P<unknown_3>-?\d+)\t"
                                            r"(?P<property_name>[^\t]+)\t"
                                            r"(?P<value_1>[^\t]+)\t"
                                            r"(?P<value_2>[^\t]+)"),
    VOXSection.TRACK_VOL_L     : VOL_TRACK_REGEX,
    VOXSection.TRACK_FX_L      : re.compile(r"(?P<timepoint>\d+,\d+,\d+)(\s+(?P<duration>\d+))?(\s+(?P<special>-?\d+))?(\s+(?P<param_ex>-?\d+))?"),
    VOXSection.TRACK_BT_A      : re.compile(r"(?P<timepoint>\d+,\d+,\d+)(\s+(?P<duration>\d+))?(\s+(?P<special>-?\d+))?(\s+(?P<param_ex>-?\d+))?"),
    VOXSection.TRACK_BT_B      : re.compile(r"(?P<timepoint>\d+,\d+,\d+)(\s+(?P<duration>\d+))?(\s+(?P<special>-?\d+))?(\s+(?P<param_ex>-?\d+))?"),
    VOXSection.TRACK_BT_C      : re.compile(r"(?P<timepoint>\d+,\d+,\d+)(\s+(?P<duration>\d+))?(\s+(?P<special>-?\d+))?(\s+(?P<param_ex>-?\d+))?"),
    VOXSection.TRACK_BT_D      : re.compile(r"(?P<timepoint>\d+,\d+,\d+)(\s+(?P<duration>\d+))?(\s+(?P<special>-?\d+))?(\s+(?P<param_ex>-?\d+))?"),
    VOXSection.TRACK_FX_R      : re.compile(r"(?P<timepoint>\d+,\d+,\d+)(\s+(?P<duration>\d+))?(\s+(?P<special>-?\d+))?(\s+(?P<param_ex>-?\d+))?"),
    VOXSection.TRACK_VOL_R     : VOL_TRACK_REGEX,
    VOXSection.AUTOTAB_INFO    : re.compile(r"(?P<timepoint>\d+,\d+,\d+)\s+(?P<duration>\d+)\s+(?P<effect_index>\d+)"),
    VOXSection.TRACK_VOL_L_ORIG: VOL_TRACK_REGEX,
    VOXSection.TRACK_VOL_R_ORIG: VOL_TRACK_REGEX,
    VOXSection.SPCONTROLLER    : SPCONTROLLER_REGEX,
    VOXSection.LOCKED_SPCONTROLLER: SPCONTROLLER_REGEX,
    VOXSection.SCRIPT          : re.compile(r".*"),
    VOXSection.SCRIPTED_TRACK  : re.compile(r"(?P<timepoint>\d+,\d+,\d+)(?P<script_ids>(?:\s+\d+)*)"),
}
# fmt: on
SEGMENT_TYPE_MAP = [
    SegmentFlag.MIDDLE,
    SegmentFlag.POINT,
    SegmentFlag.START,
    SegmentFlag.END,
]
SCRIPT_START_REGEX = re.compile(r"@SCRIPTSTART\s+(?P<script_id>\d+)")
SCRIPT_END_REGEX = re.compile(r"@SCRIPTEND")

logger = logging.getLogger(__name__)


class VOXParser:
    """A parser for the VOX file format."""

    _chart: ChartInfo

    # Intrinsic data
    _vox_version: int

    # Stateful data
    _current_section: VOXSection
    _effect_param_buffer: list[Effect]
    _parsed_effect_params: bool
    _parsed_tab_effect_params: bool
    _parsed_autotab_params: bool
    _parse_original_vols: bool
    _scripted_track: NoteType | None
    _current_script_id: int | None
    _current_script_lines: list[str]

    def __init__(self, *, parse_original_vols: bool = False) -> None:
        self._parse_original_vols = parse_original_vols
        self._reset_parse_state()

    def _reset_parse_state(self) -> None:
        self._chart = ChartInfo()
        self._vox_version = 0

        self._current_section = VOXSection.NONE
        self._effect_param_buffer = []
        self._parsed_effect_params = False
        self._parsed_tab_effect_params = False
        self._parsed_autotab_params = False
        self._scripted_track = None
        self._current_script_id = None
        self._current_script_lines = []

    def parse(self, file: TextIO) -> ChartInfo:
        """Parse one VOX stream into its chart data."""
        self._reset_parse_state()

        for lineno, line in enumerate(file):
            # Remove comments.
            if "//" in line:
                index = line.find("//")
                line = line[:index]
            # Remove whitespace from the end
            line = line.rstrip()
            # Section markers start with '#'
            if line.startswith("#"):
                section_name = line[1:]
                self._current_section = SECTION_MAP[section_name]
                if self._current_section in (VOXSection.TRACK_VOL_L_ORIG, VOXSection.TRACK_VOL_R_ORIG):
                    if not self._parse_original_vols:
                        self._current_section = VOXSection.NONE
                self._scripted_track = SCRIPTED_TRACK_MAP.get(section_name)
            # Content
            else:
                # Ignore everything between sections
                if self._current_section == VOXSection.NONE:
                    continue
                # Ignore empty lines
                if not line:
                    continue
                # Parse lines
                try:
                    self._parse_line(line)
                except ValueError:
                    logger.warning(f'unrecognized line at line {lineno + 1}: "{line}"')

        self._post_process()

        return self._chart

    def _convert_vox_timepoint(self, s: str) -> TimePoint:
        # This assumes there is no need to normalize the timepoint
        m, c, d = map(int, s.split(",", maxsplit=3))
        timesig = self._chart.get_timesig(m)
        position = Fraction(c - 1, timesig.lower) + Fraction(d, TICKS_PER_BAR)
        t = TimePoint(m, position.numerator, position.denominator)
        return t

    def _parse_vol_position(self, value: str) -> Fraction:
        """Parse a laser position using the VOX version-dependent game behavior."""

        if value in {"63", "64"}:
            return Fraction(1, 2)
        if self._vox_version >= 12:
            return Fraction(value)
        return Fraction(int(value), 127)

    def _parse_vol_extra_params(self, value: str) -> tuple[int, int, int, int, int]:
        """Decode version-specific trailing fields of a VOL track record."""

        params = value.split()
        if self._vox_version <= 5:
            if params:
                logger.warning(
                    "ignoring %d extra VOL field(s) for VOX version %d",
                    len(params),
                    self._vox_version,
                )
            return 1, 0, 0, 0, 0

        if self._vox_version <= 9:
            if not params:
                raise ValueError("VOX versions 6 through 9 VOL records require wide_laser only")
            if len(params) > 1:
                logger.warning(
                    "ignoring %d extra VOL field(s) for VOX version %d",
                    len(params) - 1,
                    self._vox_version,
                )
            return int(params[0]), 0, 0, 0, 0

        if len(params) < 2:
            raise ValueError("VOX version 10 or later VOL records require wide_laser and ease_type")
        if len(params) == 3:
            raise ValueError("VOX version 10 or later VOL records require param_ex_1 with spin_length")
        if len(params) > 5:
            logger.warning(
                "ignoring %d extra VOL field(s) for VOX version %d",
                len(params) - 5,
                self._vox_version,
            )

        wide_laser, ease_type = map(int, params[:2])
        spin_length = int(params[2]) if len(params) >= 4 else 0
        param_ex_1 = int(params[3]) if len(params) >= 4 else 0
        param_ex_2 = int(params[4]) if len(params) >= 5 else 0
        return wide_laser, ease_type, spin_length, param_ex_1, param_ex_2

    def _normalize_button_effect(self, duration: int, effect: int) -> int:
        """Apply the VOX version 6 button-effect compatibility mapping."""

        if self._vox_version >= 6:
            if effect in {-1, 255}:
                return 0
            if duration > 0 and effect == 254:
                return 14
        return effect

    @staticmethod
    def _parse_stof(value: str) -> float:
        """Parse the numeric prefix accepted by the game's ``std::stof`` call."""
        match = STOF_PREFIX_REGEX.match(value)
        if match is None:
            raise ValueError
        return float(match.group())

    def _parse_line(self, line: str) -> None:
        # Ignore invalid lines
        match = SECTION_REGEX[self._current_section].match(line)
        if not match:
            raise ValueError

        if self._current_section == VOXSection.VERSION:
            self._vox_version = int(match["version"])
            self._chart.format_version = self._vox_version
        elif self._current_section == VOXSection.BEAT_RESOLUTION:
            self._chart.beat_resolution = int(match["resolution"])
        elif self._current_section == VOXSection.TIME_SIGNATURE:
            timepoint = match["timepoint"]
            upper = int(match["upper"])
            lower = int(match["lower"])
            # Not gonna bother checking multiple measure overflow
            m, c, d = map(int, timepoint.split(",", maxsplit=3))
            if (c, d) != (1, 0):
                m += 1
            self._chart.timesigs[TimePoint(m, 0, 1)] = TimeSignature(upper, lower)
        elif self._current_section == VOXSection.BPM:
            timepoint = self._convert_vox_timepoint(match["timepoint"])
            bpm = Decimal(match["bpm"])
            # Ignoring stops because it's unnecessary (for now)
            self._chart.bpms[timepoint] = bpm
        elif self._current_section == VOXSection.TILT:
            timepoint = self._convert_vox_timepoint(match["timepoint"])
            tilt_type = TiltType(int(match["tilt_type"] or TiltType.NORMAL.value))
            self._chart.tilt_type[timepoint] = tilt_type
        elif self._current_section == VOXSection.LYRICS:
            timepoint = self._convert_vox_timepoint(match["timepoint"])
            duration = Fraction(int(match["duration"]), TICKS_PER_BAR)
            self._chart.lyrics[timepoint] = LyricInfo(duration, match["text"])
        elif self._current_section == VOXSection.END_POSITION:
            end_position = self._convert_vox_timepoint(match["timepoint"])
            self._chart.end_position = end_position
            self._chart.end_measure = end_position.measure
        elif self._current_section == VOXSection.FILTER_PARAMS:
            if not self._parsed_tab_effect_params:
                self._chart.filter_list = []
                self._parsed_tab_effect_params = True

            values = [v.strip() for v in line.split(",") if v.strip()]
            effect_type = int(values[0])
            params = [float(v) for v in values[1:]]
            tab_effect = None
            if effect_type == 1 and len(params) >= 4:
                tab_effect = LowpassFilter(
                    mix=params[0],
                    vol_cutoff_bound=params[1],
                    cutoff=params[2],
                    q=params[3],
                )
            elif effect_type == 2 and len(params) >= 4:
                tab_effect = HighpassFilter(
                    mix=params[0],
                    cutoff=params[1],
                    vol_cutoff_bound=params[2],
                    q=params[3],
                )
            elif effect_type == 3 and len(params) >= 2:
                tab_effect = Bitcrush(mix=params[0], hold_samples=int(params[1]))

            if tab_effect is not None:
                self._chart.filter_list.append(tab_effect)
        elif self._current_section == VOXSection.EFFECT_PARAMS:
            if not self._parsed_effect_params:
                self._chart.effect_list = []
                self._effect_param_buffer = []
                self._parsed_effect_params = True
            values = [v.strip() for v in line.split(",") if v.strip()]
            effect_index = int(values[0])
            params = [float(v) for v in values[1:]]
            self._effect_param_buffer.append(from_vox_params(effect_index, params))
            if len(self._effect_param_buffer) == 2:
                effect1, effect2 = self._effect_param_buffer
                self._chart.effect_list.append(EffectEntry(effect1, effect2))
                self._effect_param_buffer = []
        elif self._current_section == VOXSection.AUTOTAB_PARAMS:
            if not self._parsed_autotab_params:
                self._chart.autotab_params = []
                self._parsed_autotab_params = True

            self._chart.autotab_params.append(
                AutoTabParam(
                    effect_index=int(match["index"]),
                    param_index=int(match["param_index"]),
                    min_value=float(match["param_start"]),
                    max_value=float(match["param_end"]),
                )
            )
        elif self._current_section == VOXSection.REVERB:
            pass
        elif self._current_section == VOXSection.POST_EFFECT:
            start_value = match["start"]
            start = int(start_value[:-2]) if start_value.endswith("ms") else self._convert_vox_timepoint(start_value)
            duration_value = match["duration"]
            duration = (
                int(duration_value[:-2])
                if duration_value.endswith("ms")
                else Fraction(int(duration_value), TICKS_PER_BAR)
            )
            self._chart.post_effect_infos.append(
                PostEffectInfo(
                    start=start,
                    unknown_1=int(match["unknown_1"]),
                    duration=duration,
                    effect_name=match["effect_name"],
                    unknown_2=int(match["unknown_2"]),
                    unknown_3=int(match["unknown_3"]),
                    property_name=match["property_name"],
                    start_value=self._parse_stof(match["value_1"]),
                    end_value=self._parse_stof(match["value_2"]),
                )
            )
        elif self._current_section in [
            VOXSection.TRACK_VOL_L,
            VOXSection.TRACK_VOL_R,
            VOXSection.TRACK_VOL_L_ORIG,
            VOXSection.TRACK_VOL_R_ORIG,
        ]:
            # Parse all parameters
            timepoint = self._convert_vox_timepoint(match["timepoint"])
            position = self._parse_vol_position(match["position"])
            segment_type_str = match["segment_type"]
            segment_type = (
                SegmentFlag.START
                if segment_type_str == "1"
                else SegmentFlag.END
                if segment_type_str == "2"
                else SegmentFlag.MIDDLE
            )
            spin_type_str = match["spin_type"]
            spin_type = SpinType(int(spin_type_str)) if "1" <= spin_type_str <= "5" else SpinType.NO_SPIN
            filter_type_str = match["filter_type"]
            filter_type = int(filter_type_str)
            wide_laser, ease_type_value, spin_length, param_ex_1, param_ex_2 = self._parse_vol_extra_params(
                match["extra_params"]
            )
            ease_type_str = str(ease_type_value)
            ease_type = (
                EasingType.LINEAR
                if ease_type_str == "2"
                else EasingType.EASE_IN_SINE
                if ease_type_str == "4"
                else EasingType.EASE_OUT_SINE
                if ease_type_str == "5"
                else EasingType.NO_EASING
            )
            # Insert into the right dictionary
            vol_dict: dict[TimePoint, VolInfo]
            if self._current_section == VOXSection.TRACK_VOL_L:
                vol_dict = self._chart.note_data.vol_l
            elif self._current_section == VOXSection.TRACK_VOL_R:
                vol_dict = self._chart.note_data.vol_r
            elif self._current_section == VOXSection.TRACK_VOL_L_ORIG:
                vol_dict = self._chart.original_vol_data.vol_l
            else:
                vol_dict = self._chart.original_vol_data.vol_r
            # Become slam if timepoint already exists
            if timepoint in vol_dict:
                vol_dict[timepoint].point_type |= segment_type
                vol_dict[timepoint].end = position
            else:
                vol_dict[timepoint] = VolInfo(
                    start=position,
                    end=position,
                    spin_type=spin_type,
                    spin_duration=spin_length,
                    ease_type=ease_type,
                    filter_index=filter_type,
                    point_type=segment_type,
                    wide_laser=wide_laser,
                    param_ex_1=param_ex_1,
                    param_ex_2=param_ex_2,
                )
        elif self._current_section in [VOXSection.TRACK_FX_L, VOXSection.TRACK_FX_R]:
            timepoint = self._convert_vox_timepoint(match["timepoint"])
            duration = int(match["duration"] or 0)
            special = self._normalize_button_effect(duration, int(match["special"] or 0))
            param_ex = int(match["param_ex"] or 0) if self._vox_version >= 10 else 0
            fx_dict: dict[TimePoint, FXInfo]
            if self._current_section == VOXSection.TRACK_FX_L:
                fx_dict = self._chart.note_data.fx_l
            else:
                fx_dict = self._chart.note_data.fx_r
            fx_dict[timepoint] = FXInfo(Fraction(duration, TICKS_PER_BAR), special, param_ex)
        elif self._current_section in [
            VOXSection.TRACK_BT_A,
            VOXSection.TRACK_BT_B,
            VOXSection.TRACK_BT_C,
            VOXSection.TRACK_BT_D,
        ]:
            timepoint = self._convert_vox_timepoint(match["timepoint"])
            duration = int(match["duration"] or 0)
            special = self._normalize_button_effect(duration, int(match["special"] or 0))
            param_ex = int(match["param_ex"] or 0) if self._vox_version >= 10 else 0
            bt_dict: dict[TimePoint, BTInfo]
            if self._current_section == VOXSection.TRACK_BT_A:
                bt_dict = self._chart.note_data.bt_a
            elif self._current_section == VOXSection.TRACK_BT_B:
                bt_dict = self._chart.note_data.bt_b
            elif self._current_section == VOXSection.TRACK_BT_C:
                bt_dict = self._chart.note_data.bt_c
            else:
                bt_dict = self._chart.note_data.bt_d
            bt_dict[timepoint] = BTInfo(Fraction(duration, TICKS_PER_BAR), special, param_ex)
        elif self._current_section == VOXSection.AUTOTAB_INFO:
            timepoint = self._convert_vox_timepoint(match["timepoint"])
            duration = Fraction(int(match["duration"]), TICKS_PER_BAR)
            # This is the same raw effect index used by FXInfo.special for
            # sustained FX buttons, including the VOX format's offset.
            effect_index = int(match["effect_index"])
            self._chart.autotab_infos[timepoint] = AutoTabInfo(
                duration=duration,
                effect_index=effect_index,
            )
        elif self._current_section in (VOXSection.SPCONTROLLER, VOXSection.LOCKED_SPCONTROLLER):
            timepoint = self._convert_vox_timepoint(match["timepoint"])
            sp_type = match["sp_type"]
            info = SPControllerInfo(
                timepoint=timepoint,
                sp_type=sp_type,
                sp_subtype=match["sp_subtype"],
                duration=int(match["duration"]),
                params=(match["param_1"], match["param_2"], match["param_3"], match["param_4"]),
            )
            target = (
                self._chart.spcontroller_data
                if self._current_section == VOXSection.SPCONTROLLER
                else self._chart.locked_spcontroller_data
            )
            target.entries.append(info)
        elif self._current_section == VOXSection.SCRIPT:
            script_start = SCRIPT_START_REGEX.fullmatch(line)
            if script_start is not None:
                self._current_script_id = int(script_start["script_id"])
                self._current_script_lines = []
            elif SCRIPT_END_REGEX.fullmatch(line) is not None:
                if self._current_script_id is None:
                    raise ValueError
                self._chart.script_definitions[self._current_script_id] = "\n".join(
                    self._current_script_lines
                )
                self._current_script_id = None
                self._current_script_lines = []
            elif self._current_script_id is not None:
                self._current_script_lines.append(line)
        elif self._current_section == VOXSection.SCRIPTED_TRACK:
            if self._scripted_track is None:
                raise ValueError
            timepoint = self._convert_vox_timepoint(match["timepoint"])
            script_ids = [int(script_id) for script_id in match["script_ids"].split()]
            track_scripts = self._chart.script_ids.setdefault(self._scripted_track, {})
            track_scripts[timepoint] = script_ids
        else:
            pass

    def _post_process(self) -> None:
        # Get final measure
        final_note_timept = TimePoint()
        for _, timept, _ in self._chart.note_data.iter_notes():
            final_note_timept = max(timept, final_note_timept)
        chart = self._chart
        if chart.end_position is None:
            chart.end_measure = final_note_timept.measure + 2

        # Fix when last vol segment isn't properly indicated
        for vol_data in [
            self._chart.note_data.vol_l,
            self._chart.note_data.vol_r,
        ]:
            vol_keys = list(vol_data.keys())
            if not vol_keys:
                continue
            vol_keys.sort()

            last_timept = vol_keys[-1]
            vol_data[last_timept].point_type |= SegmentFlag.END
