"""
Module: preprocessing.orderbook_features

Functions to derive features from order book snapshots, such as imbalance.
"""
import pandas as pd
from typing import Dict, Any


def compute_orderbook_imbalance(
    snapshot: Dict[str, Any],
    depth: int = 5
) -> float:
    """
    Compute normalized order book imbalance:
      (sum of top N bid volumes - sum of top N ask volumes)
      divided by total volume at top N levels.

    Parameters:
    - snapshot: dictionary with keys 'bids' and 'asks', each a list of [price, volume].
    - depth: number of top levels to include.

    Returns:
    - imbalance: float in [-1, 1], where positive means bid-side dominates.
    """
    bids = pd.DataFrame(snapshot.get('bids', [])[:depth], columns=['price', 'volume'])
    asks = pd.DataFrame(snapshot.get('asks', [])[:depth], columns=['price', 'volume'])
    bid_vol = bids['volume'].astype(float).sum()
    ask_vol = asks['volume'].astype(float).sum()
    total = bid_vol + ask_vol
    if total == 0:
        return 0.0
    return (bid_vol - ask_vol) / total
