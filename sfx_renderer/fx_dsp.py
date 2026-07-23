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
FLANGER_MIN_DELAY_MS = 0.1
FLANGER_MAX_DELAY_MS = 3.0
FLANGER_MAX_STAGE = 4.0


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

    def _render_pitch_shift_event(self, output: np.ndarray, event: FXRenderEvent[PitchShift]) -> None:
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

    def _render_flanger_event(self, output: np.ndarray, event: FXRenderEvent[Flanger]) -> None:
        preroll = self._flanger_preroll_samples(event.effect)
        context_start = max(0, event.start_sample - preroll)
        context = output[context_start : event.end_sample].copy()
        active_start = event.start_sample - context_start
        processed = self._apply_flanger(event.effect, context, event.bpm, active_start_sample=active_start)
        output[event.start_sample : event.end_sample] = processed[active_start:]

    def _apply_isolated_flanger(self, effect: Flanger, segment: np.ndarray, bpm: float) -> np.ndarray:
        if len(segment) < 2:
            return segment.copy()
        preroll = min(self._flanger_preroll_samples(effect), len(segment) - 1)
        context = np.concatenate((segment[1 : preroll + 1][::-1], segment), axis=0)
        processed = self._apply_flanger(effect, context, bpm, active_start_sample=preroll)
        return processed[preroll:]

    def _flanger_preroll_samples(self, effect: Flanger) -> int:
        base_delay = _clamp(effect.period, FLANGER_MIN_DELAY_MS, FLANGER_MAX_DELAY_MS) * self.sample_rate / 1000.0
        depth = _clamp(effect.stereo_width, 0, 100) * 0.01 * base_delay
        stages = int(np.ceil(_clamp(effect.hicut_gain, 0.0, FLANGER_MAX_STAGE))) + 1
        return max(1, int(np.ceil(base_delay + depth)) * stages)

    def _bar_subdivision_samples(self, wavelength: int, bpm: float) -> int:
        wavelength = max(1, wavelength)
        return max(1, self._beats_to_samples(4.0, bpm) // (wavelength * 2))

    def _apply_retrigger(self, effect: Retrigger | RetriggerEx, segment: np.ndarray, bpm: float) -> np.ndarray:
        if len(segment) == 0:
            return segment.copy()

        mix = _clamp(effect.mix, 0.0, 100.0)
        calculated_update_period = (60.0 / max(bpm, 1.0)) * effect.update_period
        final_update_period = _clamp(calculated_update_period, 0.1, 8.0)
        feedback = _clamp(effect.feedback, 0.1, 1.0)
        wavelength = max(1, min(int(effect.wavelength), 32))
        amount = _clamp(effect.amount, 0.1, 1.0)
        decay = _clamp(effect.decay, 0.0, 1.0)

        period_samples = max(1, int(final_update_period * self.sample_rate) // wavelength)
        amount_samples = int(period_samples * amount)
        decay_samples = int(amount_samples * decay)
        if os.environ.get("SDVX_FX_DEBUG"):
            print(
                "Retrigger debug: "
                f"segment_samples={len(segment)} "
                f"bpm={bpm:.3f} "
                f"waveLength={effect.wavelength} "
                f"updatePeriod={effect.update_period:.3f} "
                f"calculatedUpdatePeriod={calculated_update_period:.6f} "
                f"period_samples={period_samples} "
                f"period_sec={period_samples / self.sample_rate:.6f} "
                f"amount_samples={amount_samples} "
                f"decay_samples={decay_samples}"
            )
        indices = np.arange(len(segment))
        phase = indices % period_samples
        repeats = (indices // period_samples) % wavelength
        if isinstance(effect, RetriggerEx):
            source_indices = phase
        else:
            source_indices = indices - period_samples * repeats

        wet = np.zeros_like(segment)
        active = phase <= amount_samples
        if np.any(active):
            gain = np.power(feedback, repeats[active], dtype=np.float64)
            if decay_samples > 0:
                active_phase = phase[active]
                fading = active_phase > amount_samples - decay_samples
                if np.any(fading):
                    gain[fading] *= (amount_samples - active_phase[fading]) / decay_samples
            wet[active] = segment[source_indices[active]] * gain[:, None]

        return _mix(segment, wet, mix)

    def _apply_gate(self, effect: Gate, segment: np.ndarray, bpm: float) -> np.ndarray:
        if len(segment) == 0:
            return segment.copy()

        mix = _clamp(effect.mix / 100.0, 0.0, 1.0)
        wavelength = max(1, min(int(effect.wavelength), 32))
        calculated_length = (60.0 / max(bpm, 1.0)) * effect.length
        final_length = _clamp(calculated_length, 0.1, 4.0)
        length_samples = max(1, int(final_length * self.sample_rate))
        block_samples = max(1, length_samples // wavelength)

        positions = np.arange(len(segment)) % length_samples
        step_indices = positions // block_samples
        gate_gain = (step_indices % 2 == 0).astype(np.float32)
        envelope = (1.0 - mix) + mix * gate_gain
        return segment * envelope[:, None]

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
        active_start_sample = max(0, min(active_start_sample, len(segment)))
        base_delay = _clamp(effect.period, FLANGER_MIN_DELAY_MS, FLANGER_MAX_DELAY_MS) * self.sample_rate / 1000.0
        modulation_depth = _clamp(effect.stereo_width, 0, 100) * 0.01 * base_delay

        # The legacy VOX parameter names are misleading: feedback is the LFO
        # frequency control (feedback / 2 Hz), while hiCutGain is a cascade
        # stage count with an optional fractional initial stage.
        lfo_angular_step = max(effect.feedback, 0.0) * np.pi / self.sample_rate
        stage_amount = _clamp(effect.hicut_gain, 0.0, FLANGER_MAX_STAGE)
        first_stage = int(np.ceil(stage_amount))
        fractional_stage = first_stage - stage_amount
        history_samples = max(1, int(np.ceil(base_delay + modulation_depth)))
        processed = segment.astype(np.float64, copy=True)

        for stage_index in range(first_stage, -1, -1):
            output = processed.copy()
            start = max(0, active_start_sample - stage_index * history_samples)
            if stage_index == first_stage:
                wet_gain = mix - (1.0 - mix) * fractional_stage
                dry_gain = (1.0 - mix) + mix * fractional_stage
            else:
                wet_gain = mix
                dry_gain = 1.0 - mix

            for frame in range(start, len(processed)):
                phase = (frame - stage_index * history_samples) * lfo_angular_step
                left_delay = base_delay + np.sin(phase) * modulation_depth
                right_delay = base_delay - np.sin(phase) * modulation_depth
                delayed = np.zeros(processed.shape[1], dtype=np.float64)

                for channel, delay in enumerate((left_delay, right_delay)):
                    if channel >= processed.shape[1]:
                        break
                    read_position = frame - delay
                    read_index = int(np.floor(read_position))
                    if read_index < 0:
                        continue
                    fraction = read_position - read_index
                    first = processed[read_index, channel]
                    second = processed[min(read_index + 1, len(processed) - 1), channel]
                    delayed[channel] = first + (second - first) * fraction

                output[frame] = dry_gain * processed[frame] + wet_gain * delayed
                if stage_amount >= 1.0 and stage_index == 0:
                    output[frame] *= 1.5
            processed = output

        return np.clip(processed, -1.0, 1.0).astype(np.float32)

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
        if len(segment) == 0:
            return segment.copy()

        mix = _clamp(effect.mix / 100.0, 0.0, 1.0)
        cycles_per_beat = max(effect.frequency, 0.01)
        calculated_period = (60.0 / max(bpm, 1.0)) / cycles_per_beat
        period_seconds = max(calculated_period, 0.1)
        period_samples = max(1, int(period_seconds * self.sample_rate))
        attack = _clamp(effect.attack, 0, 100)
        hold = _clamp(effect.hold, 0, 100)
        release = _clamp(effect.release, 0, 100)
        attack_samples = int(period_samples * attack * 0.002)
        hold_samples = int(period_samples * hold * 0.003)
        release_samples = int(period_samples * release * 0.005)

        phase = np.arange(len(segment)) % period_samples
        envelope = np.ones(len(segment), dtype=np.float32)
        if attack_samples > 0:
            attack_mask = phase < attack_samples
            envelope[attack_mask] = 1.0 - phase[attack_mask] / attack_samples
        hold_start = attack_samples
        hold_end = hold_start + hold_samples
        envelope[(phase >= hold_start) & (phase < hold_end)] = 0.0
        if release_samples > 0:
            release_end = hold_end + release_samples
            release_mask = (phase >= hold_end) & (phase < release_end)
            envelope[release_mask] = (phase[release_mask] - hold_end) / release_samples

        return segment * ((1.0 - mix) + mix * envelope[:, None])

    def _apply_wobble(self, effect: Wobble, segment: np.ndarray, bpm: float) -> np.ndarray:
        if effect.filter_type == PassFilterType.LOW_PASS:
            return self._apply_wobble_lowpass(effect, segment, bpm)
        if effect.filter_type == PassFilterType.HIGH_PASS:
            return self._apply_wobble_highpass(effect, segment, bpm)
        if effect.filter_type == PassFilterType.BAND_PASS:
            return self._apply_wobble_bandpass(effect, segment, bpm)
        return segment.copy()

    def _wobble_cutoff(self, effect: Wobble, phase: float) -> float:
        """Calculate the game's Wobble cutoff for one LFO phase."""
        low_cutoff = _clamp(min(effect.low_cutoff, effect.hi_cutoff), 20.0, self.sample_rate / 2.0 - 100.0)
        high_cutoff = _clamp(max(effect.low_cutoff, effect.hi_cutoff), low_cutoff, self.sample_rate / 2.0 - 100.0)
        phase = phase % 1.0
        shape = effect.wave_shape.value

        if shape == 0:
            return low_cutoff + phase * (high_cutoff - low_cutoff)
        if shape == 1:
            return high_cutoff - phase * (high_cutoff - low_cutoff)
        if shape == 2:
            exponent = (np.sin(phase * 2.0 * np.pi) + 1.0) * 0.5
            return low_cutoff * (high_cutoff / low_cutoff) ** exponent
        if shape == 3:
            exponent = phase * 2.0 if phase < 0.5 else 2.0 - phase * 2.0
            return low_cutoff * (high_cutoff / low_cutoff) ** exponent
        return high_cutoff if shape == 4 and phase >= 0.5 else low_cutoff

    def _apply_wobble_lowpass(self, effect: Wobble, segment: np.ndarray, bpm: float) -> np.ndarray:
        """Render ProcessWobbleInternal's low-pass filter branch."""
        if len(segment) == 0:
            return segment.copy()

        block_size = 1024
        channels = segment.shape[1]
        output = np.empty_like(segment)
        input_history = np.zeros((channels, 2), dtype=np.float64)
        output_history = np.zeros((channels, 2), dtype=np.float64)
        mix = _clamp(effect.mix / 100.0, 0.0, 1.0)
        q = max(effect.bandwidth, 0.1)
        output_gain = 1.0 - q * 0.04
        period_samples = max(
            1,
            int(max((60.0 / max(bpm, 1.0)) / max(effect.frequency, 0.01), 0.1) * self.sample_rate),
        )

        for start in range(0, len(segment), block_size):
            end = min(start + block_size, len(segment))
            phase = (start % period_samples) / period_samples
            cutoff = self._wobble_cutoff(effect, phase)
            b, a = self._biquad_pass(cutoff, q=q, filter_type="lowpass")
            block = segment[start:end].astype(np.float64, copy=False)
            filtered = np.empty_like(block)
            for channel in range(channels):
                zi = signal.lfiltic(b, a, output_history[channel], input_history[channel])
                filtered[:, channel], _ = signal.lfilter(b, a, block[:, channel], zi=zi)

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

            output[start:end] = ((1.0 - mix) * block + mix * filtered) * output_gain
        return output

    def _apply_wobble_highpass(self, effect: Wobble, segment: np.ndarray, bpm: float) -> np.ndarray:
        """Render ProcessWobbleInternal's high-pass filter branch."""
        if len(segment) == 0:
            return segment.copy()

        block_size = 1024
        channels = segment.shape[1]
        output = np.empty_like(segment)
        input_history = np.zeros((channels, 2), dtype=np.float64)
        output_history = np.zeros((channels, 2), dtype=np.float64)
        mix = _clamp(effect.mix / 100.0, 0.0, 1.0)
        q = max(effect.bandwidth, 0.1)
        output_gain = 1.0 - q * 0.04
        period_samples = max(
            1,
            int(max((60.0 / max(bpm, 1.0)) / max(effect.frequency, 0.01), 0.1) * self.sample_rate),
        )

        for start in range(0, len(segment), block_size):
            end = min(start + block_size, len(segment))
            phase = (start % period_samples) / period_samples
            cutoff = self._wobble_cutoff(effect, phase)
            b, a = self._biquad_pass(cutoff, q=q, filter_type="highpass")
            block = segment[start:end].astype(np.float64, copy=False)
            filtered = np.empty_like(block)
            for channel in range(channels):
                zi = signal.lfiltic(b, a, output_history[channel], input_history[channel])
                filtered[:, channel], _ = signal.lfilter(b, a, block[:, channel], zi=zi)

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

            output[start:end] = ((1.0 - mix) * block + mix * filtered) * output_gain
        return output

    def _apply_wobble_bandpass(self, effect: Wobble, segment: np.ndarray, bpm: float) -> np.ndarray:
        """Render ProcessWobbleInternal's band-pass filter branch."""
        if len(segment) == 0:
            return segment.copy()

        block_size = 1024
        channels = segment.shape[1]
        output = np.empty_like(segment)
        input_history = np.zeros((channels, 2), dtype=np.float64)
        output_history = np.zeros((channels, 2), dtype=np.float64)
        mix = _clamp(effect.mix / 100.0, 0.0, 1.0)
        q = max(effect.bandwidth, 0.1)
        if q <= 1.0:
            band_gain = q + 0.9
        else:
            band_gain = q * 0.2 + 2.0
            if band_gain > 4.0:
                band_gain = 3.0
        period_samples = max(
            1,
            int(max((60.0 / max(bpm, 1.0)) / max(effect.frequency, 0.01), 0.1) * self.sample_rate),
        )

        for start in range(0, len(segment), block_size):
            end = min(start + block_size, len(segment))
            phase = (start % period_samples) / period_samples
            cutoff = self._wobble_cutoff(effect, phase)
            b, a = self._biquad_pass(cutoff, q=q, filter_type="bandpass")
            block = segment[start:end].astype(np.float64, copy=False)
            filtered = np.empty_like(block)
            for channel in range(channels):
                zi = signal.lfiltic(b, a, output_history[channel], input_history[channel])
                filtered[:, channel], _ = signal.lfilter(b, a, block[:, channel], zi=zi)

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

            output[start:end] = (1.0 - mix) * block + mix * filtered * band_gain
        return output

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


