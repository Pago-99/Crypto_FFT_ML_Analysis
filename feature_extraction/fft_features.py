"""
Module: feature_extraction.fft_features

Functions to compute frequency-domain features from time-series data.
"""
import numpy as np
import pandas as pd


def compute_fft_features(
    series: pd.Series,
    n_features: int = 10,
    sample_rate: float = 1.0
) -> np.ndarray:
    """
    Compute FFT on a numeric series and return the top n_features frequencies and magnitudes.

    Parameters:
    - series: pandas Series of numeric values (regularly sampled).
    - n_features: number of top frequency components by magnitude to return.
    - sample_rate: sampling frequency in Hz (samples per unit time).

    Returns:
    - features: 1D numpy array of length 2*n_features: [freq1, mag1, ..., freqN, magN]
    """
    # Convert to numpy and detrend
    y = series.values.astype(float)
    y = y - np.mean(y)

    # Compute FFT
    fft_vals = np.fft.rfft(y)
    fft_freqs = np.fft.rfftfreq(len(y), d=1.0/sample_rate)
    magnitudes = np.abs(fft_vals)

    # Identify top n_features by magnitude
    idx = np.argsort(magnitudes)[-n_features:]
    top_freqs = fft_freqs[idx]
    top_mags = magnitudes[idx]

    # Combine into feature vector sorted by frequency
    sorted_idx = np.argsort(top_freqs)
    freqs_sorted = top_freqs[sorted_idx]
    mags_sorted = top_mags[sorted_idx]

    features = np.empty(2 * n_features)
    features[0::2] = freqs_sorted
    features[1::2] = mags_sorted
    return features