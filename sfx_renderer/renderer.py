"""Chart-level SFX render orchestration and command-line entry point."""
from __future__ import annotations

import argparse
import os
from decimal import Decimal
from pathlib import Path
from typing import cast

import numpy as np

from sdvxparser import parse_vox
from sdvxparser.classes.chart import ChartInfo
from sdvxparser.classes.effects import Effect, Flanger, PitchShift, Retrigger, RetriggerEx
from sdvxparser.classes.time import TimePoint

from .audio import decode_audio as _decode_audio
from .audio import encode_audio as _encode_audio
from .events import FXRenderEvent
from .fx_dsp import FXDSP
from .note_sfx import NoteHitSFX
from .shot_sfx import ShotSFX
from .vol_dsp import VolDSP

DEFAULT_SAMPLE_RATE = 44100
DEFAULT_CHANNELS = 2
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
        output = audio.astype(np.float32, copy=True)
        for event in events:
            segment = output[event.start_sample : event.end_sample]
            if len(segment) == 0:
                continue
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
            else:
                output[event.start_sample : event.end_sample] = self.apply(event.effect, segment, event.bpm)
        self._render_vol_effects(chart, output, offset_ms=offset_ms)
        if knob_audio is not None and len(knob_audio) > 0 and knob_volume > 0:
            self._render_knob_sounds(chart, output, knob_audio, offset_ms=offset_ms, volume=knob_volume)
        if click_audio is not None and len(click_audio) > 0 and click_volume > 0:
            self._render_note_hit_sounds(chart, output, click_audio, offset_ms=offset_ms, volume=click_volume)
        if shots and shot_volume > 0:
            self._render_fx_shots(chart, output, shots, offset_ms=offset_ms, volume=shot_volume)
        return output, events

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
            effect = chart.effect_list[effect_index].effect1
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
            events.append(
                FXRenderEvent(
                    start_sample=start_sample,
                    end_sample=end_sample,
                    bpm=float(chart.get_bpm(timepoint)),
                    effect=effect,
                    label=(
                        f"{note_type} {chart.timepoint_to_vox(timepoint)} "
                        f"slot={fx.special}{cut_label}"
                    ),
                )
            )

        for timepoint, autotab in sorted(chart.autotab_infos.items()):
            if autotab.duration <= 0 or autotab.effect_index <= 0:
                continue
            effect_index = autotab.effect_index - 2
            if not 0 <= effect_index < len(chart.effect_list):
                continue
            effect = chart.effect_list[effect_index].effect1
            end_timepoint = chart.add_duration(timepoint, autotab.duration)
            start = chart._get_elapsed_time(timepoint) + offset_seconds
            end = chart._get_elapsed_time(end_timepoint) + offset_seconds
            start_sample = max(0, int(round(float(start) * self.sample_rate)))
            end_sample = min(audio_samples, int(round(float(end) * self.sample_rate)))
            if end_sample <= start_sample:
                continue
            events.append(
                FXRenderEvent(
                    start_sample=start_sample,
                    end_sample=end_sample,
                    bpm=float(chart.get_bpm(timepoint)),
                    effect=effect,
                    label=f"AUTO TAB {chart.timepoint_to_vox(timepoint)} slot={autotab.effect_index}",
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
