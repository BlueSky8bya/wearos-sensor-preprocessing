"""PPG 밴드패스 필터 (전처리)
원본: ppg_data_visualization.ipynb
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

def bandpass_filter(data, fs, lowcut=0.5, highcut=4.0, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)
