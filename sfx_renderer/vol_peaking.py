"""Game-derived parameter mapping for the VOL Peak filter."""
from __future__ import annotations


PEAK_FREQUENCIES_HZ = [
    0.0, 6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0, 48.0,
    54.0, 100.0, 106.0, 112.0, 118.0, 124.0, 130.0, 136.0,
    142.0, 148.0, 154.0, 160.0, 166.0, 172.0, 178.0, 184.0,
    190.0, 196.0, 202.0, 232.0, 262.0, 292.0, 322.0, 352.0,
    382.0, 412.0, 442.0, 472.0, 522.0, 572.0, 622.0, 672.0,
    722.0, 772.0, 822.0, 872.0, 922.0, 972.0, 1022.0, 1072.0,
    1122.0, 1172.0, 1222.0, 1272.0, 1322.0, 1372.0, 1422.0,
    1472.0, 1522.0, 1572.0, 1622.0, 1672.0, 1722.0, 1772.0,
    1822.0, 1872.0, 1922.0, 1972.0, 2022.0, 2072.0, 2122.0,
    2172.0, 2222.0, 2272.0, 2322.0, 2372.0, 2422.0, 2472.0,
    2522.0, 2572.0, 2622.0, 2672.0, 2722.0, 2772.0, 2822.0,
    2872.0, 2922.0, 2972.0, 3022.0, 3072.0, 3122.0, 3172.0,
    3222.0, 3272.0, 3322.0, 3372.0, 3422.0, 3472.0, 3522.0,
    3572.0, 3622.0, 3672.0, 3852.0, 4032.0, 4212.0, 4392.0,
    4572.0, 4752.0, 4932.0, 5112.0, 5292.0, 5472.0, 5652.0,
    5832.0, 6012.0, 6192.0, 6372.0, 6552.0, 6732.0, 6912.0,
    7400.0, 7700.0, 8000.0, 8400.0, 8800.0, 9270.0, 9750.0,
    10240.0, 10800.0
]

def get_peak_parameters(raw_position: float) -> tuple[float, float, float]:
    """Map a game laser coordinate to DirectSound ParamEq parameters.

    Returns ``(center_frequency_hz, bandwidth_semitones, gain_db)`` for one
    stereo ``IDirectSoundFXParamEq`` instance.
    """
    index = max(0, min(int(raw_position), len(PEAK_FREQUENCIES_HZ) - 1))
    frequency_hz = max(80.0, min(PEAK_FREQUENCIES_HZ[index], 16000.0))

    if frequency_hz < 200.0:
        bandwidth_semitones = frequency_hz * 0.075
        gain_db = bandwidth_semitones
    elif frequency_hz < 1000.0:
        bandwidth_semitones = 15.0
        gain_db = 15.0
    else:
        bandwidth_semitones = 15.0 - (frequency_hz - 1000.0) * 0.0003
        gain_db = 15.0 - (frequency_hz - 1000.0) * 0.0005

    if index < 4:
        gain_db = 0.0

    return frequency_hz, bandwidth_semitones, gain_db
