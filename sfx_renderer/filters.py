"""Reusable biquad coefficient helpers."""
from __future__ import annotations

import numpy as np

from .audio import clamp


class FilterDSP:
    sample_rate: int

    def _geom_lerp(self, start: float, end: float, value: float) -> float:
        value = clamp(value, 0.0, 1.0)
        return float(np.exp(np.log(start) + (np.log(end) - np.log(start)) * value))

    def _biquad_peaking(
        self,
        freq: float,
        *,
        bandwidth_octaves: float,
        gain_db: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        freq = clamp(freq, 20.0, self.sample_rate / 2.0 - 100.0)
        omega = 2 * np.pi * freq / self.sample_rate
        sin_omega = np.sin(omega)
        cos_omega = np.cos(omega)
        a_gain = 10 ** (gain_db / 40.0)
        alpha = sin_omega * np.sinh(
            np.log(2.0) / 2.0 * bandwidth_octaves * omega / max(sin_omega, 1e-8)
        )
        b = np.array([1 + alpha * a_gain, -2 * cos_omega, 1 - alpha * a_gain], dtype=np.float64)
        a = np.array([1 + alpha / a_gain, -2 * cos_omega, 1 - alpha / a_gain], dtype=np.float64)
        return b / a[0], a / a[0]

    def _biquad_pass(self, freq: float, *, q: float, filter_type: str) -> tuple[np.ndarray, np.ndarray]:
        freq = clamp(freq, 20.0, self.sample_rate / 2.0 - 100.0)
        omega = 2 * np.pi * freq / self.sample_rate
        sin_omega = np.sin(omega)
        cos_omega = np.cos(omega)
        alpha = sin_omega / (2.0 * max(q, 0.1))
        if filter_type == "highpass":
            b = np.array([(1 + cos_omega) / 2, -(1 + cos_omega), (1 + cos_omega) / 2], dtype=np.float64)
        elif filter_type == "bandpass":
            b = np.array([alpha, 0.0, -alpha], dtype=np.float64)
        else:
            b = np.array([(1 - cos_omega) / 2, 1 - cos_omega, (1 - cos_omega) / 2], dtype=np.float64)
        a = np.array([1 + alpha, -2 * cos_omega, 1 - alpha], dtype=np.float64)
        return b / a[0], a / a[0]

    def _biquad_low_shelf(self, freq: float, *, q: float, gain_db: float) -> tuple[np.ndarray, np.ndarray]:
        freq = clamp(freq, 20.0, self.sample_rate / 2.0 - 100.0)
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
