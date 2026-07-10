"""DSP implementations for FX button effects."""
from __future__ import annotations

import os

import numpy as np
from scipy import signal

from sdvxparser.classes.effects import (
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

from .audio import clamp, mix
from .events import FXRenderEvent
from .filters import FilterDSP

_clamp = clamp
_mix = mix

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


class FXDSP(FilterDSP):
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


