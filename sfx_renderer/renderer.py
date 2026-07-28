"""Chart-level SFX render orchestration and command-line entry point."""
from __future__ import annotations

import argparse
import os
from dataclasses import replace
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import cast

import numpy as np

from vox_parser import parse_vox
from vox_parser.classes.chart import ChartInfo
from vox_parser.classes.effects import Effect, Flanger, PitchShift, ProvisionalSampler, Retrigger, RetriggerEx
from vox_parser.classes.enums import FILTER_TYPE_PARAM_ASSIGN, NoteType, SegmentFlag
from vox_parser.classes.time import TimePoint

from .audio import decode_audio as _decode_audio
from .audio import encode_audio as _encode_audio
from .events import FXRenderEvent
from .fx_dsp import FXDSP
from .note_sfx import NoteHitSFX
from .shot_sfx import ShotSFX
from .vol_dsp import VolDSP

DEFAULT_SAMPLE_RATE = 44100
DEFAULT_CHANNELS = 2
FX_PARAM_ASSIGN_BLOCK_SIZE = 512
RESOURCE_DIR = Path(__file__).with_name("resources")
DEFAULT_KNOB_PATH = RESOURCE_DIR / "knob.wav"
DEFAULT_CLICK_PATH = RESOURCE_DIR / "click.wav"
DEFAULT_SHOT_DIR = RESOURCE_DIR / "shot"


class FXEffects(FXDSP, VolDSP, NoteHitSFX, ShotSFX):
    """Apply SDVX FX button and VOL effects to full-song audio."""

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE, channels: int = DEFAULT_CHANNELS) -> None:
        self.sample_rate = sample_rate
        self.channels = channels

    def render_file(
        self,
        vox_path: str | Path,
        audio_path: str | Path,
        output_path: str | Path,
        *,
        offset_ms: float = 0.0,
        knob_path: str | Path | None = None,
        knob_volume: float = 1.0,
        click_path: str | Path | None = None,
        click_volume: float = 1.0,
        shot_dir: str | Path | None = DEFAULT_SHOT_DIR,
        shot_volume: float = 1.0,
    ) -> list[FXRenderEvent[Effect]]:
        """Render a chart's FX button effects and write the processed full-song audio."""
        vox_path = Path(vox_path)
        audio_path = Path(audio_path)
        output_path = Path(output_path)
        if knob_path is None:
            knob_path = DEFAULT_KNOB_PATH
        knob_path = Path(knob_path) if knob_path else None
        click_path = Path(click_path) if click_path else None
        shot_dir = Path(shot_dir) if shot_dir else None

        chart = parse_vox(vox_path)

        audio = _decode_audio(audio_path, self.sample_rate, self.channels)
        knob_audio = _decode_audio(knob_path, self.sample_rate, self.channels) if knob_path and knob_path.exists() else None
        click_audio = _decode_audio(click_path, self.sample_rate, self.channels) if click_path and click_path.exists() else None
        shots = self._load_shots(shot_dir)
        rendered, events = self.render_chart(
            chart,
            audio,
            offset_ms=offset_ms,
            knob_audio=knob_audio,
            knob_volume=knob_volume,
            click_audio=click_audio,
            click_volume=click_volume,
            shots=shots,
            shot_volume=shot_volume,
        )
        rendered = np.clip(rendered, -1.0, 1.0)
        _encode_audio(output_path, rendered, self.sample_rate, self.channels)
        return events

    @staticmethod
    def _merge_duplicate_fx_events(events: list[FXRenderEvent[Effect]]) -> list[FXRenderEvent[Effect]]:
        merged: list[FXRenderEvent[Effect]] = []
        for event in events:
            duplicate_index = next(
                (
                    index
                    for index, existing in enumerate(merged)
                    if existing.start_sample == event.start_sample
                    and existing.end_sample == event.end_sample
                    and existing.chain_index == event.chain_index
                    and existing.effect_entry_index == event.effect_entry_index
                    and type(existing.effect) is type(event.effect)
                    and existing.effect == event.effect
                ),
                None,
            )
            if duplicate_index is None:
                merged.append(event)
                continue

            existing = merged[duplicate_index]
            merged[duplicate_index] = FXRenderEvent(
                start_sample=existing.start_sample,
                end_sample=existing.end_sample,
                bpm=existing.bpm,
                effect=existing.effect,
                chain_index=existing.chain_index,
                effect_entry_index=existing.effect_entry_index,
                label=f"{existing.label} | {event.label}",
            )
        return merged

    def render_chart(
        self,
        chart: ChartInfo,
        audio: np.ndarray,
        *,
        offset_ms: float = 0.0,
        knob_audio: np.ndarray | None = None,
        knob_volume: float = 1.0,
        click_audio: np.ndarray | None = None,
        click_volume: float = 1.0,
        shots: dict[int, np.ndarray] | None = None,
        shot_volume: float = 1.0,
    ) -> tuple[np.ndarray, list[FXRenderEvent[Effect]]]:
        """Render all active FX button holds into a copy of ``audio``."""
        events = self._collect_events(chart, len(audio), offset_ms)
        param_assign_values, param_assign_active = self._collect_param_assign_values(
            chart,
            len(audio),
            offset_ms,
        )
        output = audio.astype(np.float32, copy=True)
        source_audio = output.copy()
        source_offset_samples = int(round(offset_ms * self.sample_rate / 1000.0))
        for event in events:
            self._render_effect_event_with_param_assign(
                chart,
                output,
                event,
                param_assign_values,
                param_assign_active,
                source_audio=source_audio,
                source_offset_samples=source_offset_samples,
            )
        self._render_vol_effects(chart, output, offset_ms=offset_ms)
        if knob_audio is not None and len(knob_audio) > 0 and knob_volume > 0:
            self._render_knob_sounds(chart, output, knob_audio, offset_ms=offset_ms, volume=knob_volume)
        if click_audio is not None and len(click_audio) > 0 and click_volume > 0:
            self._render_note_hit_sounds(chart, output, click_audio, offset_ms=offset_ms, volume=click_volume)
        if shots and shot_volume > 0:
            self._render_fx_shots(chart, output, shots, offset_ms=offset_ms, volume=shot_volume)
        return output, events

    def _collect_param_assign_values(
        self,
        chart: ChartInfo,
        audio_samples: int,
        offset_ms: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the combined laser control curve for ``FILTER_TYPE_PARAM_ASSIGN``."""

        values = np.zeros(audio_samples, dtype=np.float32)
        active = np.zeros(audio_samples, dtype=bool)
        offset_seconds = Decimal(str(offset_ms)) / Decimal(1000)
        for note_type, vol_dict in ((NoteType.VOL_L, chart.note_data.vol_l), (NoteType.VOL_R, chart.note_data.vol_r)):
            points = sorted(vol_dict.items())
            for (time_i, vol_i), (time_f, vol_f) in zip(points, points[1:]):
                if SegmentFlag.END in vol_i.point_type or vol_i.filter_index != FILTER_TYPE_PARAM_ASSIGN:
                    continue

                start = chart._get_elapsed_time(time_i) + offset_seconds
                end = chart._get_elapsed_time(time_f) + offset_seconds
                start_sample = int(round(float(start) * self.sample_rate))
                end_sample = int(round(float(end) * self.sample_rate))
                clip_start = max(0, start_sample)
                clip_end = min(audio_samples, end_sample)
                if clip_end <= clip_start:
                    continue

                full_count = end_sample - start_sample
                sample_offset = clip_start - start_sample
                phase = np.arange(sample_offset, sample_offset + clip_end - clip_start, dtype=np.float32) / max(full_count, 1)
                phase = self._apply_laser_easing(phase, vol_i.ease_type)
                lane_curve = float(vol_i.end) + (float(vol_f.start) - float(vol_i.end)) * phase
                value_curve = self._laser_value_for_lane(note_type, lane_curve)

                value_view = values[clip_start:clip_end]
                active_view = active[clip_start:clip_end]
                update = ~active_view | (value_curve > value_view)
                value_view[update] = value_curve[update]
                active_view[update] = True
        return values, active

    def _render_effect_event_with_param_assign(
        self,
        chart: ChartInfo,
        output: np.ndarray,
        event: FXRenderEvent[Effect],
        values: np.ndarray,
        active: np.ndarray,
        *,
        source_audio: np.ndarray,
        source_offset_samples: int,
    ) -> None:
        """Render one effect event, refreshing an assigned parameter every 512 samples."""

        if not np.any(active[event.start_sample : event.end_sample]) or not self._has_param_assign(chart, event):
            self._render_effect_event(
                output,
                event,
                sample_offset=0,
                chart=chart,
                source_audio=source_audio,
                source_offset_samples=source_offset_samples,
            )
            return

        source_segment = output[event.start_sample : event.end_sample].copy()
        for start in range(event.start_sample, event.end_sample, FX_PARAM_ASSIGN_BLOCK_SIZE):
            end = min(start + FX_PARAM_ASSIGN_BLOCK_SIZE, event.end_sample)
            effect = event.effect
            block_active = active[start:end]
            if np.any(block_active):
                value = float(np.mean(values[start:end][block_active]))
                effect = self._effect_with_param_assign(chart, event, value)
            block_event = replace(event, start_sample=start, end_sample=end, effect=effect)
            self._render_effect_event(
                output,
                block_event,
                sample_offset=start - event.start_sample,
                source_segment=source_segment,
                chart=chart,
                source_audio=source_audio,
                source_offset_samples=source_offset_samples,
            )

    @staticmethod
    def _has_param_assign(chart: ChartInfo, event: FXRenderEvent[Effect]) -> bool:
        if event.effect_entry_index is None or event.effect_entry_index >= len(chart.tab_param_assignments):
            return False
        entry = chart.tab_param_assignments[event.effect_entry_index]
        assign = entry.param1 if event.chain_index == 0 else entry.param2
        return event.effect.get_vox_param_field(assign.param_index) is not None

    def _effect_with_param_assign(
        self,
        chart: ChartInfo,
        event: FXRenderEvent[Effect],
        vol_value: float,
    ) -> Effect:
        """Return an effect copy with this event's assigned VOL parameter interpolated."""

        if event.effect_entry_index is None or event.effect_entry_index >= len(chart.tab_param_assignments):
            return event.effect
        assign_entry = chart.tab_param_assignments[event.effect_entry_index]
        assign = assign_entry.param1 if event.chain_index == 0 else assign_entry.param2
        field = event.effect.get_vox_param_field(assign.param_index)
        if field is None:
            return event.effect

        value = assign.min_value + (assign.max_value - assign.min_value) * float(np.clip(vol_value, 0.0, 1.0))
        current = getattr(event.effect, field)
        if isinstance(current, Enum):
            value = type(current)(round(value))
        elif isinstance(current, int) and not isinstance(current, bool):
            value = int(round(value))
        return replace(event.effect, **{field: value})

    def _render_effect_event(
        self,
        output: np.ndarray,
        event: FXRenderEvent[Effect],
        *,
        sample_offset: int,
        source_segment: np.ndarray | None = None,
        chart: ChartInfo | None = None,
        source_audio: np.ndarray | None = None,
        source_offset_samples: int = 0,
    ) -> None:
        """Render one already-parameterized event stage into ``output``."""

        segment = output[event.start_sample : event.end_sample]
        if len(segment) == 0:
            return
        if os.environ.get("SDVX_FX_DEBUG") and isinstance(event.effect, (Retrigger, RetriggerEx)):
            print(
                "FX event: "
                f"{event.label} "
                f"start_sample={event.start_sample} "
                f"end_sample={event.end_sample} "
                f"duration_samples={event.end_sample - event.start_sample}"
            )
        if isinstance(event.effect, PitchShift):
            self._render_pitch_shift_event(output, cast(FXRenderEvent[PitchShift], event))
        elif isinstance(event.effect, Flanger):
            self._render_flanger_event(output, cast(FXRenderEvent[Flanger], event))
        elif isinstance(event.effect, ProvisionalSampler) and source_audio is not None:
            source_start = self._provisional_sampler_source_start(
                chart,
                event,
                event.effect,
                source_offset_samples,
            )
            if sample_offset and event.effect.audio_offset >= 0.0:
                source_start += -sample_offset if event.effect.mode_control // 10 % 10 == 1 else sample_offset
            output[event.start_sample : event.end_sample] = self.apply_provisional_sampler(
                event.effect,
                segment,
                source_audio,
                source_start,
            )
        else:
            output[event.start_sample : event.end_sample] = self.apply(
                event.effect,
                segment,
                event.bpm,
                sample_offset=sample_offset,
                source_segment=source_segment,
            )

    def _provisional_sampler_source_start(
        self,
        chart: ChartInfo | None,
        event: FXRenderEvent[Effect],
        effect: ProvisionalSampler,
        source_offset_samples: int,
    ) -> int:
        """Resolve the source sample, snapping absolute offsets to the chart BPM grid."""

        if effect.audio_offset < 0.0 or chart is None:
            return event.start_sample

        grid = int(effect.mode_control) % 10
        source_seconds = effect.audio_offset
        if grid > 0:
            source_seconds = self._snap_provisional_sampler_time(chart, source_seconds, grid)
        return int(round(source_seconds * self.sample_rate)) + source_offset_samples

    def _snap_provisional_sampler_time(self, chart: ChartInfo, seconds: float, grid: int) -> float:
        """Snap a song time to the nearest beat subdivision across BPM segments."""

        bpm_points = sorted(chart.bpms)
        if not bpm_points:
            return seconds
        target = Decimal(str(max(seconds, 0.0)))
        for index, timepoint in enumerate(bpm_points):
            start = chart._get_elapsed_time(timepoint)
            end = chart._get_elapsed_time(bpm_points[index + 1]) if index + 1 < len(bpm_points) else None
            if end is not None and target >= end:
                continue
            bpm = float(chart.bpms[timepoint])
            beat_seconds = 60.0 / max(bpm, 1.0)
            local_beats = (float(target - start) / beat_seconds)
            snapped_beats = round(local_beats * grid) / grid
            return float(start) + snapped_beats * beat_seconds
        return seconds

    def _load_shots(self, shot_dir: Path | None) -> dict[int, np.ndarray]:
        """Decode numbered shot resources keyed by their 1-based FX slot."""
        if shot_dir is None or not shot_dir.is_dir():
            return {}

        shots: dict[int, np.ndarray] = {}
        for path in shot_dir.glob("*.wav"):
            try:
                slot = int(path.stem)
            except ValueError:
                continue
            if slot > 0:
                shots[slot] = _decode_audio(path, self.sample_rate, self.channels)
        return shots

    def _collect_events(self, chart: ChartInfo, audio_samples: int, offset_ms: float) -> list[FXRenderEvent[Effect]]:
        fx_notes = sorted(chart.note_data.iter_fxs(), key=lambda item: item[1])
        latest = TimePoint()
        for _, timepoint, fx in fx_notes:
            latest = max(latest, chart.add_duration(timepoint, fx.duration))
        for timepoint, autotab in chart.autotab_infos.items():
            latest = max(latest, chart.add_duration(timepoint, autotab.duration))
        chart_endpoint = chart.end_position or TimePoint(chart.end_measure, 0, 1)
        endpoint = max(chart_endpoint, latest)
        chart._elapsed_time.clear()
        chart._elapsed_time_bpm.clear()
        chart._bpm_durations.clear()
        chart._calculate_bpm_durations(endpoint)

        # A zero-duration FX note is a tap/shot trigger, not the beginning of
        # a sustained effect.  It must not truncate an active FX hold.
        fx_start_times = sorted({timepoint for _, timepoint, fx in fx_notes if fx.duration > 0})
        next_fx_start = {
            timepoint: fx_start_times[index + 1] if index + 1 < len(fx_start_times) else None
            for index, timepoint in enumerate(fx_start_times)
        }

        offset_seconds = Decimal(str(offset_ms)) / Decimal(1000)
        events: list[FXRenderEvent[Effect]] = []
        for note_type, timepoint, fx in fx_notes:
            if fx.duration <= 0 or fx.special <= 0:
                continue
            effect_index = fx.special - 2
            if not 0 <= effect_index < len(chart.effect_list):
                continue
            nominal_end_timepoint = chart.add_duration(timepoint, fx.duration)
            following_start = next_fx_start[timepoint]
            end_timepoint = (
                min(nominal_end_timepoint, following_start)
                if following_start is not None
                else nominal_end_timepoint
            )
            cut_label = (
                f" cut@{chart.timepoint_to_vox(end_timepoint)}"
                if end_timepoint < nominal_end_timepoint
                else ""
            )
            start = chart._get_elapsed_time(timepoint) + offset_seconds
            end = chart._get_elapsed_time(end_timepoint) + offset_seconds
            start_sample = max(0, int(round(float(start) * self.sample_rate)))
            end_sample = min(audio_samples, int(round(float(end) * self.sample_rate)))
            if end_sample <= start_sample:
                continue
            slot = chart.effect_list[effect_index]
            for chain_index, effect in enumerate((slot.effect1, slot.effect2)):
                events.append(
                    FXRenderEvent(
                        start_sample=start_sample,
                        end_sample=end_sample,
                        bpm=float(chart.get_bpm(timepoint)),
                        effect=effect,
                        chain_index=chain_index,
                        effect_entry_index=effect_index,
                        label=(
                            f"{note_type} {chart.timepoint_to_vox(timepoint)} "
                            f"slot={fx.special} effect{chain_index + 1}{cut_label}"
                        ),
                    )
                )

        for timepoint, autotab in sorted(chart.autotab_infos.items()):
            if autotab.duration <= 0 or autotab.effect_index <= 0:
                continue
            effect_index = autotab.effect_index - 2
            if not 0 <= effect_index < len(chart.effect_list):
                continue
            end_timepoint = chart.add_duration(timepoint, autotab.duration)
            start = chart._get_elapsed_time(timepoint) + offset_seconds
            end = chart._get_elapsed_time(end_timepoint) + offset_seconds
            start_sample = max(0, int(round(float(start) * self.sample_rate)))
            end_sample = min(audio_samples, int(round(float(end) * self.sample_rate)))
            if end_sample <= start_sample:
                continue
            slot = chart.effect_list[effect_index]
            for chain_index, effect in enumerate((slot.effect1, slot.effect2)):
                events.append(
                    FXRenderEvent(
                        start_sample=start_sample,
                        end_sample=end_sample,
                        bpm=float(chart.get_bpm(timepoint)),
                        effect=effect,
                        chain_index=chain_index,
                        effect_entry_index=effect_index,
                        label=(
                            f"AUTO TAB {chart.timepoint_to_vox(timepoint)} "
                            f"slot={autotab.effect_index} effect{chain_index + 1}"
                        ),
                    )
                )

        events.sort(key=lambda event: event.start_sample)
        return self._merge_duplicate_fx_events(events)

def main() -> None:
    parser = argparse.ArgumentParser(description="Render SDVX FX button effects into full-song audio.")
    parser.add_argument("vox", type=Path, help="Path to the VOX chart file.")
    parser.add_argument("audio", type=Path, help="Path to the source chart audio, e.g. .s3v/.wma/.asf.")
    parser.add_argument("-o", "--output", type=Path, default=Path("output/fx_render.wav"), help="Output audio path.")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE, help="Render sample rate.")
    parser.add_argument("--offset-ms", type=float, default=0.0, help="Audio offset applied to chart events.")
    parser.add_argument("--knob-sound", type=Path, default=DEFAULT_KNOB_PATH, help="VOL side-to-side knob sound path.")
    parser.add_argument("--knob-volume", type=float, default=1.0, help="VOL knob sound gain.")
    parser.add_argument("--no-knob", action="store_true", help="Disable VOL side-to-side knob sounds.")
    parser.add_argument("--note-hit", action="store_true", help="Overlay click sounds at BT/FX note starts.")
    parser.add_argument("--click-sound", type=Path, default=DEFAULT_CLICK_PATH, help="BT/FX note-start click sound path.")
    parser.add_argument("--click-volume", type=float, default=1.0, help="BT/FX note-start click sound gain.")
    parser.add_argument("--shot-dir", type=Path, default=DEFAULT_SHOT_DIR, help="Directory of numbered zero-duration FX shot sounds.")
    parser.add_argument("--shot-volume", type=float, default=0.4, help="Zero-duration FX shot sound gain.")
    parser.add_argument("--no-shot", action="store_true", help="Disable zero-duration FX shot sounds.")
    args = parser.parse_args()

    renderer = FXEffects(sample_rate=args.sample_rate)
    events = renderer.render_file(
        args.vox,
        args.audio,
        args.output,
        offset_ms=args.offset_ms,
        knob_path=None if args.no_knob else args.knob_sound,
        knob_volume=args.knob_volume,
        click_path=args.click_sound if args.note_hit else None,
        click_volume=args.click_volume,
        shot_dir=None if args.no_shot else args.shot_dir,
        shot_volume=args.shot_volume,
    )
    print(f"Rendered {len(events)} FX events to {args.output}")


if __name__ == "__main__":
    main()
