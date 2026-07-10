"""Render SDVX FX button effects over a chart audio file."""
from __future__ import annotations

import argparse
import os
import subprocess

from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy import signal

from .classes.chart import ChartInfo
from .classes.effects import (
    Bitcrush,
    Effect,
    Flanger,
    Gate,
    HighpassFilter,
    LowpassFilter,
    NoEffect,
    PassFilterType,
    PitchShift,
    Retrigger,
    RetriggerEx,
    Sidechain,
    Tapestop,
    Tapescratch,
    Wobble,
)
from .classes.enums import EasingType, FilterIndex, NoteType, SegmentFlag
from .classes.time import TimePoint
from .parser.vox import VOXParser


DEFAULT_SAMPLE_RATE = 44100
DEFAULT_CHANNELS = 2
SIDE_VOL_EDGE_THRESHOLD = 1 / 127
LASER_FILTER_BLOCK_SIZE = 256
LASER_PASS_FILTER_BLOCK_SIZE = 64
LASER_V_EASING_PER_44100_FRAME = 0.01
PITCH_SHIFT_CHUNK_SIZE_44100 = 700
PITCH_SHIFT_OVERLAP = 0.4
PITCH_SHIFT_PREROLL_CHUNKS = 4
FLANGER_DELAY_SAMPLES_44100 = 30.0
FLANGER_DEPTH_SAMPLES_44100 = 45.0
FLANGER_OUTPUT_VOLUME = 0.75
FLANGER_LOW_SHELF_FREQ = 250.0
FLANGER_LOW_SHELF_Q = 0.5
FLANGER_LOW_SHELF_GAIN_DB = -20.0
FLANGER_PREROLL_SAMPLES_44100 = 128


@dataclass(frozen=True)
class FXRenderEvent:
    """A timed FX event derived from an FX button note."""

    start_sample: int
    end_sample: int
    bpm: float
    effect: Effect
    label: str = ""


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _mix(dry: np.ndarray, wet: np.ndarray, wet_percent: float) -> np.ndarray:
    """Blend dry and wet signals where the VOX mix value is the wet percentage."""
    wet_ratio = _clamp(wet_percent / 100.0, 0.0, 1.0)
    return dry * (1.0 - wet_ratio) + wet * wet_ratio


def _decode_audio(path: Path, sample_rate: int, channels: int) -> np.ndarray:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-",
    ]
    result = subprocess.run(command, check=True, capture_output=True)
    audio = np.frombuffer(result.stdout, dtype=np.float32)
    return audio.reshape((-1, channels)).copy()


def _encode_audio(path: Path, audio: np.ndarray, sample_rate: int, channels: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-i",
        "-",
        str(path),
    ]
    subprocess.run(command, input=np.asarray(audio, dtype=np.float32).tobytes(), check=True)


class FXEffects:
    """Apply SDVX FX button effects to full-song audio."""

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
    ) -> list[FXRenderEvent]:
        """Render a chart's FX button effects and write the processed full-song audio."""
        vox_path = Path(vox_path)
        audio_path = Path(audio_path)
        output_path = Path(output_path)
        if knob_path is None:
            knob_path = Path(__file__).with_name("knob.wav")
        knob_path = Path(knob_path) if knob_path else None

        with vox_path.open("r", encoding="utf-8-sig") as file:
            container = VOXParser().parse(file)

        audio = _decode_audio(audio_path, self.sample_rate, self.channels)
        knob_audio = _decode_audio(knob_path, self.sample_rate, self.channels) if knob_path and knob_path.exists() else None
        rendered, events = self.render_chart(
            container.chart_info,
            audio,
            offset_ms=offset_ms,
            knob_audio=knob_audio,
            knob_volume=knob_volume,
        )
        rendered = np.clip(rendered, -1.0, 1.0)
        _encode_audio(output_path, rendered, self.sample_rate, self.channels)
        return events

    @staticmethod
    def _merge_duplicate_fx_events(events: list[FXRenderEvent]) -> list[FXRenderEvent]:
        merged: list[FXRenderEvent] = []
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
    ) -> tuple[np.ndarray, list[FXRenderEvent]]:
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
                self._render_pitch_shift_event(output, event)
            elif isinstance(event.effect, Flanger):
                self._render_flanger_event(output, event)
            else:
                output[event.start_sample : event.end_sample] = self.apply(event.effect, segment, event.bpm)
        self._render_vol_effects(chart, output, offset_ms=offset_ms)
        if knob_audio is not None and len(knob_audio) > 0 and knob_volume > 0:
            self._render_knob_sounds(chart, output, knob_audio, offset_ms=offset_ms, volume=knob_volume)
        return output, events

    def apply(self, effect: Effect, segment: np.ndarray, bpm: float) -> np.ndarray:
        """Apply one FX effect to an audio segment."""
        if isinstance(effect, NoEffect):
            return segment
        if isinstance(effect, (Retrigger, RetriggerEx)):
            return self._apply_retrigger(effect, segment, bpm)
        if isinstance(effect, Gate):
            return self._apply_gate(effect, segment, bpm)
        if isinstance(effect, Flanger):
            return self._apply_isolated_flanger(effect, segment, bpm)
        if isinstance(effect, Tapestop):
            return self._apply_tapestop(effect, segment)
        if isinstance(effect, Sidechain):
            return self._apply_sidechain(effect, segment, bpm)
        if isinstance(effect, Wobble):
            return self._apply_wobble(effect, segment, bpm)
        if isinstance(effect, Bitcrush):
            return self._apply_bitcrush(effect, segment)
        if isinstance(effect, PitchShift):
            return self._apply_isolated_pitch_shift(effect, segment)
        if isinstance(effect, Tapescratch):
            return self._apply_tapescratch(effect, segment)
        if isinstance(effect, LowpassFilter):
            return self._apply_static_filter(segment, "lowpass", effect.low_cutoff, effect.mix, max(effect.bandwidth, 0.1))
        if isinstance(effect, HighpassFilter):
            return self._apply_static_filter(segment, "highpass", effect.cutoff, effect.mix, max(effect.bandwidth, 0.1))
        return segment

    def _collect_events(self, chart: ChartInfo, audio_samples: int, offset_ms: float) -> list[FXRenderEvent]:
        fx_notes = sorted(chart.note_data.iter_fxs(), key=lambda item: item[1])
        latest = TimePoint()
        for _, timepoint, fx in fx_notes:
            latest = max(latest, chart.add_duration(timepoint, fx.duration))
        endpoint = max(TimePoint(chart.end_measure, 0, 1), latest)
        chart._elapsed_time.clear()
        chart._elapsed_time_bpm.clear()
        chart._bpm_durations.clear()
        chart._calculate_bpm_durations(endpoint)

        fx_start_times = sorted({timepoint for _, timepoint, _ in fx_notes})
        next_fx_start = {
            timepoint: fx_start_times[index + 1] if index + 1 < len(fx_start_times) else None
            for index, timepoint in enumerate(fx_start_times)
        }

        offset_seconds = Decimal(str(offset_ms)) / Decimal(1000)
        replacement_flanger = next(
            (entry.effect1 for entry in chart.effect_list if isinstance(entry.effect1, Flanger)),
            Flanger(),
        )
        events: list[FXRenderEvent] = []
        for note_type, timepoint, fx in fx_notes:
            if fx.duration <= 0 or fx.special <= 0:
                continue
            effect_index = fx.special - 1
            if effect_index >= len(chart.effect_list):
                continue
            effect = chart.effect_list[effect_index].effect1
            nominal_end_timepoint = chart.add_duration(timepoint, fx.duration)
            following_start = next_fx_start[timepoint]
            end_timepoint = (
                min(nominal_end_timepoint, following_start)
                if following_start is not None
                else nominal_end_timepoint
            )
            effective_duration = chart.get_distance(timepoint, end_timepoint)
            replacement_label = ""
            if isinstance(effect, (Retrigger, RetriggerEx)) and effective_duration >= Fraction(1, 2):
                effect = replacement_flanger
                replacement_label = " Retrigger->Flanger"
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
                        f"slot={fx.special}{replacement_label}{cut_label}"
                    ),
                )
            )
        return self._merge_duplicate_fx_events(events)

    def _beats_to_samples(self, beats: float, bpm: float) -> int:
        return max(1, int(round((60.0 / max(bpm, 1.0)) * beats * self.sample_rate)))

    def _render_pitch_shift_event(self, output: np.ndarray, event: FXRenderEvent) -> None:
        chunk_size = self._pitch_shift_chunk_size()
        preroll = PITCH_SHIFT_PREROLL_CHUNKS * chunk_size
        context_start = max(0, event.start_sample - preroll)
        context = output[context_start : event.end_sample].copy()
        active_start = event.start_sample - context_start
        processed = self._apply_pitch_shift(event.effect, context, active_start_sample=active_start)
        output[event.start_sample : event.end_sample] = processed[active_start:]

    def _apply_isolated_pitch_shift(self, effect: PitchShift, segment: np.ndarray) -> np.ndarray:
        if len(segment) < 2:
            return segment.copy()
        preroll = min(PITCH_SHIFT_PREROLL_CHUNKS * self._pitch_shift_chunk_size(), len(segment) - 1)
        context = np.concatenate((segment[1 : preroll + 1][::-1], segment), axis=0)
        processed = self._apply_pitch_shift(effect, context, active_start_sample=preroll)
        return processed[preroll:]

    def _pitch_shift_chunk_size(self) -> int:
        return max(2, int(round(PITCH_SHIFT_CHUNK_SIZE_44100 * self.sample_rate / 44100)))

    def _render_flanger_event(self, output: np.ndarray, event: FXRenderEvent) -> None:
        preroll = max(1, int(round(FLANGER_PREROLL_SAMPLES_44100 * self.sample_rate / 44100)))
        context_start = max(0, event.start_sample - preroll)
        context = output[context_start : event.end_sample].copy()
        active_start = event.start_sample - context_start
        processed = self._apply_flanger(event.effect, context, event.bpm, active_start_sample=active_start)
        output[event.start_sample : event.end_sample] = processed[active_start:]

    def _apply_isolated_flanger(self, effect: Flanger, segment: np.ndarray, bpm: float) -> np.ndarray:
        if len(segment) < 2:
            return segment.copy()
        requested = max(1, int(round(FLANGER_PREROLL_SAMPLES_44100 * self.sample_rate / 44100)))
        preroll = min(requested, len(segment) - 1)
        context = np.concatenate((segment[1 : preroll + 1][::-1], segment), axis=0)
        processed = self._apply_flanger(effect, context, bpm, active_start_sample=preroll)
        return processed[preroll:]

    def _render_vol_effects(self, chart: ChartInfo, output: np.ndarray, *, offset_ms: float) -> None:
        offset_seconds = Decimal(str(offset_ms)) / Decimal(1000)
        laser_value = np.zeros(len(output), dtype=np.float32)
        laser_filter = np.full(len(output), -1, dtype=np.int16)

        for note_type, vol_dict in ((NoteType.VOL_L, chart.note_data.vol_l), (NoteType.VOL_R, chart.note_data.vol_r)):
            points = sorted(vol_dict.items())
            for (time_i, vol_i), (time_f, vol_f) in zip(points, points[1:]):
                if SegmentFlag.END in vol_i.point_type:
                    continue

                start_time = chart._get_elapsed_time(time_i) + offset_seconds
                end_time = chart._get_elapsed_time(time_f) + offset_seconds
                start_sample = int(round(float(start_time) * self.sample_rate))
                end_sample = int(round(float(end_time) * self.sample_rate))
                if end_sample <= start_sample:
                    continue

                clip_start = max(0, start_sample)
                clip_end = min(len(output), end_sample)
                if clip_end <= clip_start:
                    continue

                full_count = end_sample - start_sample
                sample_offset = clip_start - start_sample
                count = clip_end - clip_start
                phase = (np.arange(sample_offset, sample_offset + count, dtype=np.float32) / max(full_count, 1))
                phase = self._apply_laser_easing(phase, vol_i.ease_type)
                lane_curve = float(vol_i.end) + (float(vol_f.start) - float(vol_i.end)) * phase
                value_curve = self._laser_value_for_lane(note_type, lane_curve)

                current = laser_value[clip_start:clip_end]
                update_mask = value_curve > current
                if np.any(update_mask):
                    current[update_mask] = value_curve[update_mask]
                    filter_view = laser_filter[clip_start:clip_end]
                    filter_view[update_mask] = vol_i.filter_index.value

        active = laser_filter >= 0
        if not np.any(active):
            return

        edges = np.flatnonzero(
            np.r_[True, (active[1:] != active[:-1]) | (laser_filter[1:] != laser_filter[:-1]), True]
        )
        rendered_ranges = 0
        for start, end in zip(edges[:-1], edges[1:]):
            if not active[start]:
                continue
            filter_index = self._coerce_filter_index(int(laser_filter[start]))
            output[start:end] = self._apply_laser_filter(output[start:end], laser_value[start:end], filter_index)
            rendered_ranges += 1

        if os.environ.get("SDVX_FX_DEBUG"):
            print(f"VOL laser filter ranges: {rendered_ranges}")

    def _apply_laser_easing(self, phase: np.ndarray, ease_type: EasingType) -> np.ndarray:
        phase = np.clip(phase, 0.0, 1.0)
        if ease_type == EasingType.EASE_IN_SINE:
            return np.sin(phase * np.pi / 2).astype(np.float32)
        if ease_type == EasingType.EASE_OUT_SINE:
            return (np.sin((phase - 1.0) * np.pi / 2) + 1.0).astype(np.float32)
        return phase

    def _laser_value_for_lane(self, note_type: NoteType, lane_curve: np.ndarray) -> np.ndarray:
        if note_type == NoteType.VOL_R:
            return (1.0 - lane_curve).astype(np.float32)
        return lane_curve.astype(np.float32)

    def _coerce_filter_index(self, value: int) -> FilterIndex:
        try:
            return FilterIndex(value)
        except ValueError:
            return FilterIndex.PEAK

    def _apply_laser_filter(self, segment: np.ndarray, values: np.ndarray, filter_index: FilterIndex) -> np.ndarray:
        if filter_index in (FilterIndex.LPF_ALT, FilterIndex.LPF):
            return self._apply_laser_pass_filter(segment, values, "lowpass")
        if filter_index in (FilterIndex.HPF_ALT, FilterIndex.HPF):
            return self._apply_laser_pass_filter(segment, values, "highpass")
        if filter_index == FilterIndex.BITCRUSH:
            return self._apply_laser_bitcrusher(segment, values)
        return self._apply_laser_peaking_filter(segment, values)

    def _apply_laser_peaking_filter(self, segment: np.ndarray, values: np.ndarray) -> np.ndarray:
        wet = np.empty_like(segment)
        for start in range(0, len(segment), LASER_FILTER_BLOCK_SIZE):
            end = min(start + LASER_FILTER_BLOCK_SIZE, len(segment))
            v = float(values[(start + end - 1) // 2])
            freq = self._geom_lerp(50.0, 9000.0, v)
            if freq < 100.0:
                wet[start:end] = segment[start:end]
                continue
            base_gain_db = 34.0 * min(v / 0.35, 1.0) if v < 0.35 else 34.0 - (34.0 - 15.85) * ((v - 0.35) / 0.65)
            b, a = self._biquad_peaking(freq, bandwidth=1.2, gain_db=base_gain_db * 0.5)
            wet[start:end] = signal.lfilter(b, a, segment[start:end], axis=0)
        return wet

    def _apply_laser_pass_filter(self, segment: np.ndarray, values: np.ndarray, filter_type: str) -> np.ndarray:
        if len(segment) == 0:
            return segment.copy()

        channels = segment.shape[1]
        wet = np.empty_like(segment)
        input_history = np.zeros((channels, 2), dtype=np.float64)
        output_history = np.zeros((channels, 2), dtype=np.float64)
        smoothed_v = float(values[0])
        easing_per_frame = LASER_V_EASING_PER_44100_FRAME * 44100.0 / self.sample_rate

        for start in range(0, len(segment), LASER_PASS_FILTER_BLOCK_SIZE):
            end = min(start + LASER_PASS_FILTER_BLOCK_SIZE, len(segment))
            target_v = float(values[end - 1])
            max_change = easing_per_frame * (end - start)
            smoothed_v += _clamp(target_v - smoothed_v, -max_change, max_change)
            if filter_type == "lowpass":
                freq = self._geom_lerp(15000.0, 800.0, smoothed_v)
                mix_skipped = freq > 14800.0
                b, a = self._biquad_pass(freq, q=3.6, filter_type="lowpass")
            else:
                freq = self._geom_lerp(100.0, 2200.0, smoothed_v)
                mix_skipped = freq < 200.0
                b, a = self._biquad_pass(freq, q=5.0, filter_type="highpass")

            block = segment[start:end].astype(np.float64, copy=False)
            filtered = np.empty_like(block)
            for channel in range(channels):
                zi = signal.lfiltic(
                    b,
                    a,
                    output_history[channel],
                    input_history[channel],
                )
                filtered[:, channel], _ = signal.lfilter(
                    b,
                    a,
                    block[:, channel],
                    zi=zi,
                )

            if len(block) >= 2:
                input_history[:, 0] = block[-1]
                input_history[:, 1] = block[-2]
                output_history[:, 0] = filtered[-1]
                output_history[:, 1] = filtered[-2]
            else:
                input_history[:, 1] = input_history[:, 0]
                input_history[:, 0] = block[-1]
                output_history[:, 1] = output_history[:, 0]
                output_history[:, 0] = filtered[-1]

            wet[start:end] = block if mix_skipped else filtered
        return wet

    def _apply_laser_bitcrusher(self, segment: np.ndarray, values: np.ndarray) -> np.ndarray:
        wet = segment.copy()
        for start in range(0, len(segment), LASER_FILTER_BLOCK_SIZE):
            end = min(start + LASER_FILTER_BLOCK_SIZE, len(segment))
            v = float(values[(start + end - 1) // 2])
            hold = max(1, int(round(v * 30.0)))
            for sample in range(start, end, hold):
                wet[sample : min(sample + hold, end)] = wet[sample]
        return wet

    def _geom_lerp(self, start: float, end: float, value: float) -> float:
        value = _clamp(value, 0.0, 1.0)
        return float(np.exp(np.log(start) + (np.log(end) - np.log(start)) * value))

    def _biquad_peaking(self, freq: float, *, bandwidth: float, gain_db: float) -> tuple[np.ndarray, np.ndarray]:
        freq = _clamp(freq, 20.0, self.sample_rate / 2.0 - 100.0)
        omega = 2 * np.pi * freq / self.sample_rate
        sin_omega = np.sin(omega)
        cos_omega = np.cos(omega)
        a_gain = 10 ** (gain_db / 40.0)
        alpha = sin_omega * np.sinh(np.log(2.0) / 2.0 * bandwidth * omega / max(sin_omega, 1e-8))
        b = np.array([1 + alpha * a_gain, -2 * cos_omega, 1 - alpha * a_gain], dtype=np.float64)
        a = np.array([1 + alpha / a_gain, -2 * cos_omega, 1 - alpha / a_gain], dtype=np.float64)
        return b / a[0], a / a[0]

    def _biquad_pass(self, freq: float, *, q: float, filter_type: str) -> tuple[np.ndarray, np.ndarray]:
        freq = _clamp(freq, 20.0, self.sample_rate / 2.0 - 100.0)
        omega = 2 * np.pi * freq / self.sample_rate
        sin_omega = np.sin(omega)
        cos_omega = np.cos(omega)
        alpha = sin_omega / (2.0 * max(q, 0.1))
        if filter_type == "highpass":
            b = np.array([(1 + cos_omega) / 2, -(1 + cos_omega), (1 + cos_omega) / 2], dtype=np.float64)
        else:
            b = np.array([(1 - cos_omega) / 2, 1 - cos_omega, (1 - cos_omega) / 2], dtype=np.float64)
        a = np.array([1 + alpha, -2 * cos_omega, 1 - alpha], dtype=np.float64)
        return b / a[0], a / a[0]

    def _biquad_low_shelf(self, freq: float, *, q: float, gain_db: float) -> tuple[np.ndarray, np.ndarray]:
        freq = _clamp(freq, 20.0, self.sample_rate / 2.0 - 100.0)
        omega = 2 * np.pi * freq / self.sample_rate
        a_gain = 10 ** (gain_db / 40.0)
        beta = np.sqrt(a_gain) / max(q, 0.1)
        sin_omega = np.sin(omega)
        cos_omega = np.cos(omega)
        a = np.array(
            [
                (a_gain + 1) + (a_gain - 1) * cos_omega + beta * sin_omega,
                -2 * ((a_gain - 1) + (a_gain + 1) * cos_omega),
                (a_gain + 1) + (a_gain - 1) * cos_omega - beta * sin_omega,
            ],
            dtype=np.float64,
        )
        b = np.array(
            [
                a_gain * ((a_gain + 1) - (a_gain - 1) * cos_omega + beta * sin_omega),
                2 * a_gain * ((a_gain - 1) - (a_gain + 1) * cos_omega),
                a_gain * ((a_gain + 1) - (a_gain - 1) * cos_omega - beta * sin_omega),
            ],
            dtype=np.float64,
        )
        return b / a[0], a / a[0]

    def _render_knob_sounds(
        self,
        chart: ChartInfo,
        output: np.ndarray,
        knob_audio: np.ndarray,
        *,
        offset_ms: float,
        volume: float,
    ) -> None:
        offset_seconds = Decimal(str(offset_ms)) / Decimal(1000)
        for note_type, timepoint, vol in sorted(chart.note_data.iter_vols(), key=lambda item: item[1]):
            if not self._is_side_to_side_vol(vol.start, vol.end):
                continue
            start = chart._get_elapsed_time(timepoint) + offset_seconds
            start_sample = int(round(float(start) * self.sample_rate))
            if os.environ.get("SDVX_FX_DEBUG"):
                print(
                    "Knob event: "
                    f"{note_type} {chart.timepoint_to_vox(timepoint)} "
                    f"start={float(vol.start):.6f} "
                    f"end={float(vol.end):.6f} "
                    f"start_sample={start_sample}"
                )
            self._overlay_audio(output, knob_audio, start_sample, volume)

    def _is_side_to_side_vol(self, start, end) -> bool:
        start_value = float(start)
        end_value = float(end)
        if start_value == end_value:
            return False
        start_left = start_value <= SIDE_VOL_EDGE_THRESHOLD
        start_right = start_value >= 1.0 - SIDE_VOL_EDGE_THRESHOLD
        end_left = end_value <= SIDE_VOL_EDGE_THRESHOLD
        end_right = end_value >= 1.0 - SIDE_VOL_EDGE_THRESHOLD
        return (start_left and end_right) or (start_right and end_left)

    def _overlay_audio(self, output: np.ndarray, overlay: np.ndarray, start_sample: int, volume: float) -> None:
        output_start = max(0, start_sample)
        overlay_start = max(0, -start_sample)
        if output_start >= len(output) or overlay_start >= len(overlay):
            return
        sample_count = min(len(output) - output_start, len(overlay) - overlay_start)
        if sample_count <= 0:
            return
        output[output_start : output_start + sample_count] += overlay[overlay_start : overlay_start + sample_count] * volume

    def _bar_subdivision_samples(self, wavelength: int, bpm: float) -> int:
        wavelength = max(1, wavelength)
        return max(1, self._beats_to_samples(4.0, bpm) // (wavelength * 2))

    def _apply_retrigger(self, effect: Retrigger | RetriggerEx, segment: np.ndarray, bpm: float) -> np.ndarray:
        # Both values are denominators of a 4-beat measure: waveLength=16
        # loops a 1/16-measure sample, while updatePeriod=2 resamples every
        # 1/2 measure.
        wavelength = max(float(effect.wavelength), 1.0)
        update_period = max(float(effect.update_period), 1.0)
        chunk_samples = self._beats_to_samples(4.0 / wavelength, bpm)
        update_samples = self._beats_to_samples(4.0 / update_period, bpm)
        chunks_per_update = update_samples / chunk_samples
        if os.environ.get("SDVX_FX_DEBUG"):
            print(
                "Retrigger debug: "
                f"segment_samples={len(segment)} "
                f"bpm={bpm:.3f} "
                f"waveLength={effect.wavelength} "
                f"updatePeriod={effect.update_period:.3f} "
                f"chunk_samples={chunk_samples} "
                f"chunk_sec={chunk_samples / self.sample_rate:.6f} "
                f"chunks_per_update={chunks_per_update:.3f} "
                f"update_samples={update_samples} "
                f"update_sec={update_samples / self.sample_rate:.6f}"
            )
        indices = np.arange(len(segment))
        if isinstance(effect, RetriggerEx):
            source_indices = indices % chunk_samples
            repeats = indices // chunk_samples
        else:
            local = indices % update_samples
            source_indices = indices - local + (local % chunk_samples)
            repeats = local // chunk_samples
        source_indices = np.minimum(source_indices, len(segment) - 1)
        gain = np.power(max(effect.feedback, 0.0), repeats, dtype=np.float64)
        gain = np.maximum(gain, max(effect.decay, 0.0)) * max(effect.amount, 0.0)
        wet = segment[source_indices] * gain[:, None]
        return _mix(segment, wet.astype(np.float32), effect.mix)

    def _apply_gate(self, effect: Gate, segment: np.ndarray, bpm: float) -> np.ndarray:
        period = self._bar_subdivision_samples(effect.wavelength, bpm)
        duty = _clamp(effect.length / 2.0, 0.05, 1.0)
        phase = (np.arange(len(segment)) % period) / period
        envelope = (phase < duty).astype(np.float32)
        fade = min(period // 12, max(1, int(0.004 * self.sample_rate)))
        if fade > 1:
            edge = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            for start in range(0, len(envelope), period):
                end = min(start + fade, len(envelope))
                envelope[start:end] *= edge[: end - start]
        return _mix(segment, segment * envelope[:, None], effect.mix)

    def _apply_flanger(
        self,
        effect: Flanger,
        segment: np.ndarray,
        bpm: float,
        *,
        active_start_sample: int = 0,
    ) -> np.ndarray:
        if len(segment) == 0:
            return segment.copy()

        mix = _clamp(effect.mix / 100.0, 0.0, 1.0)
        if mix == 0.0:
            return segment.copy()

        sample_rate_scale = self.sample_rate / 44100.0
        base_delay = FLANGER_DELAY_SAMPLES_44100 * sample_rate_scale
        depth = FLANGER_DEPTH_SAMPLES_44100 * sample_rate_scale
        max_delay = int(np.ceil(base_delay + depth)) + 2
        ring_size = max(self.sample_rate * 3, max_delay + 2)
        ring = np.zeros((ring_size, segment.shape[1]), dtype=np.float64)
        output = segment.astype(np.float64, copy=True)
        feedback = _clamp(effect.feedback, 0.0, 1.0)
        stereo_width = _clamp(effect.stereo_width / 100.0, 0.0, 1.0)
        gain = (1.0 - mix) + FLANGER_OUTPUT_VOLUME * mix
        period_samples = self._beats_to_samples(max(effect.period, 0.01), bpm)
        lfo_step = 1.0 / period_samples
        lfo_phase = 0.0
        cursor = 0
        active_start_sample = max(0, min(active_start_sample, len(segment)))

        b, a = self._biquad_low_shelf(
            FLANGER_LOW_SHELF_FREQ,
            q=FLANGER_LOW_SHELF_Q,
            gain_db=FLANGER_LOW_SHELF_GAIN_DB,
        )
        input1 = np.zeros(segment.shape[1], dtype=np.float64)
        input2 = np.zeros(segment.shape[1], dtype=np.float64)
        output1 = np.zeros(segment.shape[1], dtype=np.float64)
        output2 = np.zeros(segment.shape[1], dtype=np.float64)

        for frame in range(len(segment)):
            dry = segment[frame].astype(np.float64, copy=False)
            if frame < active_start_sample:
                ring[cursor] = dry
                cursor = (cursor + 1) % ring_size
                continue

            delayed = np.zeros(segment.shape[1], dtype=np.float64)
            for channel in range(segment.shape[1]):
                phase = lfo_phase if channel == 0 else (lfo_phase + stereo_width / 2.0) % 1.0
                lfo = phase * 2.0 if phase < 0.5 else 2.0 - phase * 2.0
                delay_frames = max(1.0, base_delay + lfo * depth)
                delay_int = int(delay_frames)
                fraction = delay_frames - delay_int
                first = ring[(cursor - delay_int) % ring_size, channel]
                second = ring[(cursor - delay_int - 1) % ring_size, channel]
                delayed[channel] = first + (second - first) * fraction

            feedback_input = (dry + delayed * feedback) * gain
            filtered = (
                b[0] * feedback_input
                + b[1] * input1
                + b[2] * input2
                - a[1] * output1
                - a[2] * output2
            )
            input2 = input1.copy()
            input1 = feedback_input.copy()
            output2 = output1.copy()
            output1 = filtered.copy()
            ring[cursor] = filtered
            output[frame] = (dry + delayed * mix) * gain

            cursor = (cursor + 1) % ring_size
            lfo_phase = (lfo_phase + lfo_step) % 1.0

        return output.astype(np.float32)

    def _apply_tapestop(self, effect: Tapestop, segment: np.ndarray) -> np.ndarray:
        n = len(segment)
        speed_floor = _clamp(effect.rate, 0.0, 1.0)
        t = np.linspace(0.0, 1.0, n, dtype=np.float64)
        speed = speed_floor + (1.0 - speed_floor) * np.power(1.0 - t, max(effect.speed / 8.0, 0.1))
        read_positions = np.cumsum(speed)
        read_positions -= read_positions[0]
        read_positions = np.clip(read_positions, 0, n - 1)
        wet = np.zeros_like(segment)
        indices = np.arange(n)
        for channel in range(segment.shape[1]):
            wet[:, channel] = np.interp(read_positions, indices, segment[:, channel])
        return _mix(segment, wet, effect.mix)

    def _apply_sidechain(self, effect: Sidechain, segment: np.ndarray, bpm: float) -> np.ndarray:
        period = self._beats_to_samples(1.0 / max(effect.frequency, 0.01), bpm)
        attack = int(effect.attack / 1000 * self.sample_rate)
        hold = int(effect.hold / 1000 * self.sample_rate)
        release = int(effect.release / 1000 * self.sample_rate)
        phase = np.arange(len(segment)) % period
        envelope = np.ones(len(segment), dtype=np.float32)
        low = 0.35
        if attack > 0:
            attack_mask = phase < attack
            envelope[attack_mask] = 1.0 - (1.0 - low) * (phase[attack_mask] / attack)
        hold_mask = (phase >= attack) & (phase < attack + hold)
        envelope[hold_mask] = low
        if release > 0:
            release_mask = (phase >= attack + hold) & (phase < attack + hold + release)
            rel_phase = (phase[release_mask] - attack - hold) / release
            envelope[release_mask] = low + (1.0 - low) * rel_phase
        return _mix(segment, segment * envelope[:, None], effect.mix)

    def _apply_wobble(self, effect: Wobble, segment: np.ndarray, bpm: float) -> np.ndarray:
        block_size = 1024
        wet = np.zeros_like(segment)
        beat_seconds = 60.0 / max(bpm, 1.0)
        lfo_hz = max(effect.frequency / beat_seconds, 0.01)
        nyquist = self.sample_rate / 2.0
        for start in range(0, len(segment), block_size):
            end = min(start + block_size, len(segment))
            center = (start + end) / 2 / self.sample_rate
            lfo = 0.5 + 0.5 * np.sin(2 * np.pi * lfo_hz * center)
            cutoff = _clamp(effect.low_cutoff + (effect.hi_cutoff - effect.low_cutoff) * lfo, 20.0, nyquist - 100.0)
            if effect.filter_type == PassFilterType.HIGH_PASS:
                sos = signal.butter(2, cutoff, btype="highpass", fs=self.sample_rate, output="sos")
            elif effect.filter_type == PassFilterType.BAND_PASS:
                low = _clamp(cutoff / max(effect.bandwidth, 1.0), 20.0, nyquist - 200.0)
                high = _clamp(cutoff * max(effect.bandwidth, 1.0), low + 20.0, nyquist - 100.0)
                sos = signal.butter(2, [low, high], btype="bandpass", fs=self.sample_rate, output="sos")
            else:
                sos = signal.butter(2, cutoff, btype="lowpass", fs=self.sample_rate, output="sos")
            wet[start:end] = signal.sosfilt(sos, segment[start:end], axis=0)
        return _mix(segment, wet, effect.mix)

    def _apply_bitcrush(self, effect: Bitcrush, segment: np.ndarray) -> np.ndarray:
        hold = max(1, effect.amount)
        wet = segment.copy()
        for start in range(0, len(wet), hold):
            wet[start : start + hold] = wet[start]
        levels = 2**8
        wet = np.round(wet * levels) / levels
        return _mix(segment, wet, effect.mix)

    def _apply_pitch_shift(
        self,
        effect: PitchShift,
        segment: np.ndarray,
        *,
        active_start_sample: int = 0,
    ) -> np.ndarray:
        if len(segment) == 0 or effect.amount == 0:
            return segment.copy()

        chunk_size = self._pitch_shift_chunk_size()
        overlap_samples = max(1, int(PITCH_SHIFT_OVERLAP * chunk_size))
        hop_samples = chunk_size - overlap_samples
        if hop_samples <= 0:
            return segment.copy()

        play_speed = 2 ** (effect.amount / 12.0)
        filtered = segment.astype(np.float64, copy=True)
        if play_speed > 1.0:
            cutoff = self.sample_rate / (2.0 * play_speed) * 0.95
            b, a = self._biquad_pass(cutoff, q=0.707, filter_type="lowpass")
            for _ in range(3):
                filtered = signal.lfilter(b, a, filtered, axis=0)

        ring_size = max(self.sample_rate * 4, chunk_size * 4)
        delay = np.zeros((ring_size, segment.shape[1]), dtype=np.float64)
        wet = np.zeros_like(filtered)
        cursor = 0
        count = 0
        start = ring_size - chunk_size
        prev_start = ring_size - chunk_size - overlap_samples
        prev_prev_start = prev_start
        third_chunk_blend_step: int | None = None
        wet_ratio = _clamp(effect.mix / 100.0, 0.0, 1.0)
        active_start_sample = max(0, min(active_start_sample, len(segment)))

        for frame in range(len(segment)):
            delay[cursor] = filtered[frame]
            frame_mix = wet_ratio if frame >= active_start_sample else 0.0
            count_times_speed = int(count * play_speed)
            step = count_times_speed % hop_samples

            if frame_mix > 0.0:
                if count_times_speed <= overlap_samples:
                    rate = step / overlap_samples
                    current_idx = (start - chunk_size + step) % ring_size
                    prev_idx = (prev_start + step) % ring_size
                    if third_chunk_blend_step is None:
                        wet[frame] = delay[current_idx] * rate + delay[prev_idx] * (1.0 - rate)
                    else:
                        prev_prev_idx = (prev_prev_start + step) % ring_size
                        rate2 = min(third_chunk_blend_step / overlap_samples, 1.0)
                        previous = delay[prev_idx] * rate2 + delay[prev_prev_idx] * (1.0 - rate2)
                        wet[frame] = delay[current_idx] * rate + previous * (1.0 - rate)
                        third_chunk_blend_step += 1
                elif step <= overlap_samples:
                    rate = step / overlap_samples
                    current_idx = (start - chunk_size + step) % ring_size
                    overlap_idx = (start - overlap_samples + step) % ring_size
                    wet[frame] = delay[current_idx] * rate + delay[overlap_idx] * (1.0 - rate)
                else:
                    current_idx = (start - chunk_size + step) % ring_size
                    wet[frame] = delay[current_idx]

            count += 1
            if count > hop_samples:
                previous_step = int((count - 1) * play_speed) % hop_samples
                if previous_step <= overlap_samples:
                    prev_prev_start = (start - overlap_samples + previous_step) % ring_size
                    prev_start = (start - chunk_size + previous_step) % ring_size
                    third_chunk_blend_step = previous_step + 1
                else:
                    prev_start = (start - chunk_size + previous_step) % ring_size
                    prev_prev_start = prev_start
                    third_chunk_blend_step = None
                count = 0
                start = cursor

            cursor = (cursor + 1) % ring_size

        output = segment.astype(np.float64, copy=True)
        output[active_start_sample:] = (
            wet[active_start_sample:] * wet_ratio
            + output[active_start_sample:] * (1.0 - wet_ratio)
        )
        return output.astype(np.float32)

    def _apply_tapescratch(self, effect: Tapescratch, segment: np.ndarray) -> np.ndarray:
        n = len(segment)
        t = np.linspace(0.0, 1.0, n, dtype=np.float64)
        wobble = np.sin(2 * np.pi * effect.curve_slope * t) * 0.12
        read_positions = np.clip(np.arange(n) * (1.0 + wobble), 0, n - 1)
        wet = np.zeros_like(segment)
        indices = np.arange(n)
        for channel in range(segment.shape[1]):
            wet[:, channel] = np.interp(read_positions, indices, segment[:, channel])
        return _mix(segment, wet, effect.mix)

    def _apply_static_filter(self, segment: np.ndarray, filter_type: str, cutoff: float, mix: float, q: float) -> np.ndarray:
        nyquist = self.sample_rate / 2.0
        cutoff = _clamp(cutoff, 20.0, nyquist - 100.0)
        sos = signal.butter(max(1, min(4, int(round(q)))), cutoff, btype=filter_type, fs=self.sample_rate, output="sos")
        wet = signal.sosfilt(sos, segment, axis=0)
        return _mix(segment, wet, mix)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render SDVX FX button effects into full-song audio.")
    parser.add_argument("vox", type=Path, help="Path to the VOX chart file.")
    parser.add_argument("audio", type=Path, help="Path to the source chart audio, e.g. .s3v/.wma/.asf.")
    parser.add_argument("-o", "--output", type=Path, default=Path("output/fx_render.wav"), help="Output audio path.")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE, help="Render sample rate.")
    parser.add_argument("--offset-ms", type=float, default=0.0, help="Audio offset applied to chart events.")
    parser.add_argument("--knob-sound", type=Path, default=Path(__file__).with_name("knob.wav"), help="VOL side-to-side knob sound path.")
    parser.add_argument("--knob-volume", type=float, default=1.0, help="VOL knob sound gain.")
    parser.add_argument("--no-knob", action="store_true", help="Disable VOL side-to-side knob sounds.")
    args = parser.parse_args()

    renderer = FXEffects(sample_rate=args.sample_rate)
    events = renderer.render_file(
        args.vox,
        args.audio,
        args.output,
        offset_ms=args.offset_ms,
        knob_path=None if args.no_knob else args.knob_sound,
        knob_volume=args.knob_volume,
    )
    print(f"Rendered {len(events)} FX events to {args.output}")


if __name__ == "__main__":
    main()
