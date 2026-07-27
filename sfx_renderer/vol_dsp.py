"""DSP implementations for VOL laser filters and knob sounds."""
from __future__ import annotations

import os
from decimal import Decimal

import numpy as np
from scipy import signal

from sdvxparser.classes.chart import ChartInfo
from sdvxparser.classes.effects import Bitcrush, Effect, HighpassFilter, LowpassFilter
from sdvxparser.classes.enums import EasingType, FilterIndex, NoteType, SegmentFlag

from .audio import clamp, overlay_audio
from .filters import FilterDSP
from .vol_peaking import get_peak_parameters

_clamp = clamp

LASER_FILTER_UPDATE_HZ = 20.0
VOL_FILTER_BLOCK_SIZE = 512

class VolDSP(FilterDSP):
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
            filter_effect = self._get_laser_filter(chart, filter_index)
            output[start:end] = self._apply_laser_filter(
                output[start:end],
                laser_value[start:end],
                filter_effect,
            )
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

    def _get_laser_filter(self, chart: ChartInfo, filter_index: FilterIndex) -> Effect | None:
        index = filter_index.value - 1
        if 0 <= index < len(chart.filter_list):
            return chart.filter_list[index]
        return None

    def _apply_laser_filter(
        self,
        segment: np.ndarray,
        values: np.ndarray,
        filter_effect: Effect | None,
    ) -> np.ndarray:
        if isinstance(filter_effect, LowpassFilter):
            return self._apply_laser_pass_filter(segment, values, filter_effect)
        if isinstance(filter_effect, HighpassFilter):
            return self._apply_laser_pass_filter(segment, values, filter_effect)
        if isinstance(filter_effect, Bitcrush):
            return self._apply_laser_bitcrusher(segment, values)
        return self._apply_laser_peaking_filter(segment, values)

    def _apply_laser_peaking_filter(self, segment: np.ndarray, values: np.ndarray) -> np.ndarray:
        wet = np.empty_like(segment)
        block_size = max(1, round(self.sample_rate / LASER_FILTER_UPDATE_HZ))
        for start in range(0, len(segment), block_size):
            end = min(start + block_size, len(segment))
            v = float(values[(start + end - 1) // 2])
            center_frequency_hz, bandwidth_semitones, gain_db = get_peak_parameters(v * 127.0)
            bandwidth_octaves = bandwidth_semitones / 12.0
            b, a = self._biquad_peaking(
                center_frequency_hz,
                bandwidth_octaves=bandwidth_octaves,
                gain_db=gain_db,
            )
            wet[start:end] = signal.lfilter(b, a, segment[start:end], axis=0)
        return wet

    def _apply_laser_pass_filter(
        self,
        segment: np.ndarray,
        values: np.ndarray,
        filter_effect: LowpassFilter | HighpassFilter,
    ) -> np.ndarray:
        if len(segment) == 0:
            return segment.copy()

        channels = segment.shape[1]
        wet = np.empty_like(segment)
        input_history = np.zeros((channels, 2), dtype=np.float64)
        output_history = np.zeros((channels, 2), dtype=np.float64)
        block_size = VOL_FILTER_BLOCK_SIZE

        for start in range(0, len(segment), block_size):
            end = min(start + block_size, len(segment))
            t = float(values[(start + end - 1) // 2])
            if isinstance(filter_effect, LowpassFilter):
                freq = self._geom_lerp(filter_effect.cutoff, filter_effect.vol_cutoff_bound, 1.0 - t)
                b, a = self._biquad_pass(freq, q=max(filter_effect.q, 0.1), filter_type="lowpass")
            else:
                freq = self._geom_lerp(filter_effect.cutoff, filter_effect.vol_cutoff_bound, t)
                b, a = self._biquad_pass(freq, q=max(filter_effect.q, 0.1), filter_type="highpass")

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

            wet[start:end] = filtered
        return wet

    @staticmethod
    def _bitcrush_amount_for_value(value: float) -> int:
        return int(_clamp(round(1.0 + _clamp(value, 0.0, 1.0) * 29.0), 1, 30))

    def _apply_laser_bitcrusher(self, segment: np.ndarray, values: np.ndarray) -> np.ndarray:
        wet = segment.copy()
        block_size = VOL_FILTER_BLOCK_SIZE
        for start in range(0, len(segment), block_size):
            end = min(start + block_size, len(segment))
            v = float(values[(start + end - 1) // 2])
            hold = self._bitcrush_amount_for_value(v)
            for sample in range(start, end, hold):
                wet[sample : min(sample + hold, end)] = wet[sample]
        return wet

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
            if not self._is_vol_slam(vol.start, vol.end):
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

    def _is_vol_slam(self, start, end) -> bool:
        return float(start) != float(end)

    def _overlay_audio(self, output: np.ndarray, overlay: np.ndarray, start_sample: int, volume: float) -> None:
        overlay_audio(output, overlay, start_sample, volume)

