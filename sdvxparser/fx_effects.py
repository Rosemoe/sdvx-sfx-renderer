"""Render SDVX FX button effects over a chart audio file."""
from __future__ import annotations

import argparse
import os
import subprocess

from dataclasses import dataclass
from decimal import Decimal
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
from .classes.time import TimePoint
from .parser.vox import VOXParser


DEFAULT_SAMPLE_RATE = 44100
DEFAULT_CHANNELS = 2
SIDE_VOL_EDGE_THRESHOLD = 1 / 127


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
            output[event.start_sample : event.end_sample] = self.apply(event.effect, segment, event.bpm)
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
            return self._apply_flanger(effect, segment, bpm)
        if isinstance(effect, Tapestop):
            return self._apply_tapestop(effect, segment)
        if isinstance(effect, Sidechain):
            return self._apply_sidechain(effect, segment, bpm)
        if isinstance(effect, Wobble):
            return self._apply_wobble(effect, segment, bpm)
        if isinstance(effect, Bitcrush):
            return self._apply_bitcrush(effect, segment)
        if isinstance(effect, PitchShift):
            return self._apply_pitch_shift(effect, segment)
        if isinstance(effect, Tapescratch):
            return self._apply_tapescratch(effect, segment)
        if isinstance(effect, LowpassFilter):
            return self._apply_static_filter(segment, "lowpass", effect.low_cutoff, effect.mix, max(effect.bandwidth, 0.1))
        if isinstance(effect, HighpassFilter):
            return self._apply_static_filter(segment, "highpass", effect.cutoff, effect.mix, max(effect.bandwidth, 0.1))
        return segment

    def _collect_events(self, chart: ChartInfo, audio_samples: int, offset_ms: float) -> list[FXRenderEvent]:
        latest = TimePoint()
        for _, timepoint, fx in chart.note_data.iter_fxs():
            latest = max(latest, chart.add_duration(timepoint, fx.duration))
        endpoint = max(TimePoint(chart.end_measure, 0, 1), latest)
        chart._elapsed_time.clear()
        chart._elapsed_time_bpm.clear()
        chart._bpm_durations.clear()
        chart._calculate_bpm_durations(endpoint)

        offset_seconds = Decimal(str(offset_ms)) / Decimal(1000)
        events: list[FXRenderEvent] = []
        for note_type, timepoint, fx in sorted(chart.note_data.iter_fxs(), key=lambda item: item[1]):
            if fx.duration <= 0 or fx.special <= 0:
                continue
            effect_index = fx.special - 1
            if effect_index >= len(chart.effect_list):
                continue
            effect = chart.effect_list[effect_index].effect1
            start = chart._get_elapsed_time(timepoint) + offset_seconds
            end_timepoint = chart.add_duration(timepoint, fx.duration)
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
                    label=f"{note_type} {chart.timepoint_to_vox(timepoint)} slot={fx.special}",
                )
            )
        return events

    def _beats_to_samples(self, beats: float, bpm: float) -> int:
        return max(1, int(round((60.0 / max(bpm, 1.0)) * beats * self.sample_rate)))

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
        # waveLength controls how much of this FX segment is treated as one
        # sampled fragment: 16 means the whole segment, 8 means half, etc.
        chunk_samples = max(1, int(round(len(segment) * max(effect.wavelength, 1) / 16.0)))
        sample_update_repeats = max(1, int(round(effect.update_period)))
        update_samples = chunk_samples * sample_update_repeats
        if os.environ.get("SDVX_FX_DEBUG"):
            print(
                "Retrigger debug: "
                f"segment_samples={len(segment)} "
                f"bpm={bpm:.3f} "
                f"waveLength={effect.wavelength} "
                f"updatePeriod={effect.update_period:.3f} "
                f"chunk_samples={chunk_samples} "
                f"chunk_sec={chunk_samples / self.sample_rate:.6f} "
                f"sample_update_repeats={sample_update_repeats} "
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

    def _apply_flanger(self, effect: Flanger, segment: np.ndarray, bpm: float) -> np.ndarray:
        n = len(segment)
        lfo_period = self._beats_to_samples(max(effect.period, 0.01), bpm)
        indices = np.arange(n)
        lfo = 0.5 + 0.5 * np.sin(2 * np.pi * indices / lfo_period)
        max_delay = max(1, int(0.006 * self.sample_rate))
        delay = 1 + lfo * max_delay
        wet = np.zeros_like(segment)
        for channel in range(segment.shape[1]):
            read_positions = np.maximum(indices - delay, 0.0)
            wet[:, channel] = np.interp(read_positions, indices, segment[:, channel])
        wet = wet + segment * _clamp(effect.feedback, -0.95, 0.95) * 0.35
        return _mix(segment, wet, effect.mix)

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

    def _apply_pitch_shift(self, effect: PitchShift, segment: np.ndarray) -> np.ndarray:
        n = len(segment)
        factor = 2 ** (effect.amount / 12.0)
        read_positions = np.clip(np.arange(n, dtype=np.float64) * factor, 0, n - 1)
        wet = np.zeros_like(segment)
        indices = np.arange(n)
        for channel in range(segment.shape[1]):
            wet[:, channel] = np.interp(read_positions, indices, segment[:, channel])
        return _mix(segment, wet, effect.mix)

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
