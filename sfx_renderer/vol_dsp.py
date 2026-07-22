"""DSP implementations for VOL laser filters and knob sounds."""
from __future__ import annotations

import os
from decimal import Decimal

import numpy as np
from scipy import signal

from sdvxparser.classes.chart import ChartInfo
from sdvxparser.classes.enums import EasingType, FilterIndex, NoteType, SegmentFlag

from .audio import clamp, overlay_audio
from .filters import FilterDSP

_clamp = clamp

LASER_FILTER_BLOCK_SIZE = 256
LASER_PASS_FILTER_BLOCK_SIZE = 64
LASER_V_EASING_PER_44100_FRAME = 0.01

# Sampled from KSM's PeakingFilterDSP lookup tables. The final point is index
# 255; the preceding values are every eighth table entry.
KSM_PEAK_TABLE_POSITIONS = np.array([*(index / 255 for index in range(0, 256, 8)), 1.0], dtype=np.float32)
KSM_PEAK_FREQUENCIES = np.array(
    [
        50.0000, 57.6169, 80.4440, 118.4107, 171.3997, 239.2472, 321.7439, 418.6352,
        529.6227, 654.3651, 792.4791, 943.5414, 1107.0899, 1282.6252, 1469.6130,
        1667.4855, 1875.6437, 2093.4597, 2320.2787, 2555.4216, 2798.1873, 3047.8554,
        3303.6888, 3564.9362, 3830.8348, 4100.6133, 4610.3561, 5356.2081, 6103.6017,
        6851.7587, 7599.9059, 8347.2783, 9000.0000,
    ],
    dtype=np.float32,
)
KSM_PEAK_GAINS_DB = np.array(
    [
        0.0000, 6.3948, 12.7897, 19.1845, 25.5794, 30.6894, 31.2273, 31.7652,
        32.3031, 32.8410, 33.3788, 33.9167, 33.5454, 33.0075, 32.4696, 31.9317,
        31.3939, 30.8560, 30.3181, 29.7802, 29.2423, 28.7044, 28.1666, 27.6287,
        27.0908, 26.5529, 25.4623, 23.8263, 22.1902, 20.5541, 18.9181, 17.2820,
        15.8505,
    ],
    dtype=np.float32,
)


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
            freq = float(np.interp(v, KSM_PEAK_TABLE_POSITIONS, KSM_PEAK_FREQUENCIES))
            if freq < 100.0:
                wet[start:end] = segment[start:end]
                continue
            base_gain_db = float(np.interp(v, KSM_PEAK_TABLE_POSITIONS, KSM_PEAK_GAINS_DB))
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

