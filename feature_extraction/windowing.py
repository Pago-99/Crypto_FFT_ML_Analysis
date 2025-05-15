"""
Module: feature_extraction.windowing

Utilities to generate sliding windows over time-series data.
"""
import pandas as pd
from typing import Iterator, Tuple


def sliding_window(
    df: pd.DataFrame,
    window_size: int,
    step_size: int = None
) -> Iterator[Tuple[int, pd.DataFrame]]:
    """
    Generate sliding windows over DataFrame rows.

    Parameters:
    - df: DataFrame indexed by time or sequential index.
    - window_size: number of rows per window.
    - step_size: number of rows to advance the window each iteration (defaults to window_size).

    Yields:
    - (start_idx, window_df): Tuple of starting index and the window as a DataFrame slice.
    """
    if step_size is None:
        step_size = window_size
    total = len(df)
    for start in range(0, total - window_size + 1, step_size):
        yield start, df.iloc[start:start + window_size]
