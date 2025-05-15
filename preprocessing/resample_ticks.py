import pandas as pd


def resample_tick_data(
    df: pd.DataFrame,
    freq: str = "100ms",
    price_col: str = "price",
    volume_col: str = "volume"
) -> pd.DataFrame:
    """
    Resample uneven tick data to a regular frequency with filling missing intervals.

    Parameters:
    - df: DataFrame with a DateTimeIndex or 'timestamp' column,
          and columns for price and volume.
    - freq: pandas offset alias (e.g., '100ms', '1s', '1min').
    - price_col: name of the column containing price values.
    - volume_col: name of the column containing volume values.

    Returns:
    - DataFrame with columns ['open','high','low','close','volume']
      indexed by the resampled timestamp, with missing price values forward-filled
      and missing volumes set to zero.
    """
    data = df.copy()
    # Ensure timestamp index
    if 'timestamp' in data.columns:
        data = data.set_index('timestamp')
    # Resample price to OHLC
    ohlc = data[price_col].resample(freq).ohlc()
    # Sum volumes per interval
    vol = data[volume_col].resample(freq).sum().rename('volume')
    # Merge OHLC and volume
    bars = pd.concat([ohlc, vol], axis=1)
    # Forward-fill price columns to carry last known price
    bars[['open', 'high', 'low', 'close']] = bars[['open', 'high', 'low', 'close']].ffill()
    # Replace missing volumes with zero (no trades)
    bars['volume'] = bars['volume'].fillna(0)
    return bars
