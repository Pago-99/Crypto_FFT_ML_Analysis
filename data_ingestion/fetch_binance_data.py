import requests
import pandas as pd

BASE_URL = "https://api.binance.com/api/v3"


def fetch_recent_trades(symbol: str, limit: int = 1000) -> pd.DataFrame:
    """
    Fetch the most recent `limit` public trades for `symbol`.
    Returns a DataFrame with columns ['price', 'volume'] indexed by timestamp.
    """
    url = f"{BASE_URL}/trades"
    params = {"symbol": symbol.upper(), "limit": limit}
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data)
    df = df.rename(columns={'time': 'timestamp', 'price': 'price', 'qty': 'volume'})
    df['price'] = df['price'].astype(float)
    df['volume'] = df['volume'].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df.set_index('timestamp')[['price', 'volume']]


def fetch_historical_agg_trades(symbol: str,
                                start_time: int,
                                end_time: int = None,
                                limit: int = 1000) -> pd.DataFrame:
    """
    Fetch historical aggregated trades for `symbol` between start_time and end_time (ms).
    Uses the public /aggTrades endpoint; no API key required.
    Implements pagination using fromId when necessary.
    Returns a DataFrame with ['price', 'volume'] indexed by timestamp.
    """
    all_trades = []

    # Initial fetch by time range
    params = {"symbol": symbol.upper(), "startTime": start_time, "limit": limit}
    if end_time is not None:
        params["endTime"] = end_time
    resp = requests.get(f"{BASE_URL}/aggTrades", params=params)
    resp.raise_for_status()
    data = resp.json()
    all_trades.extend(data)

    # Paginate if exactly 'limit' records returned
    if len(data) == limit:
        last_id = data[-1]["a"]
        while True:
            page_params = {"symbol": symbol.upper(), "fromId": last_id + 1, "limit": limit}
            resp = requests.get(f"{BASE_URL}/aggTrades", params=page_params)
            resp.raise_for_status()
            page_data = resp.json()
            if not page_data:
                break
            all_trades.extend(page_data)
            if len(page_data) < limit:
                break
            last_id = page_data[-1]["a"]

    # Construct DataFrame
    df = pd.DataFrame(all_trades)
    df = df.rename(columns={'p': 'price', 'q': 'volume', 'T': 'timestamp'})
    df['price'] = df['price'].astype(float)
    df['volume'] = df['volume'].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df.set_index('timestamp')[['price', 'volume']]