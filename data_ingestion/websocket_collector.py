import json
import csv
import time
import os
from websocket import create_connection


class LiveTradeStreamer:
    """
    Streams live trades for a symbol and writes them to CSV.
    """
    def __init__(self, symbol: str, output_path: str = 'data/live_trades.csv'):
        self.symbol = symbol.lower()
        self.output_path = output_path
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

    def run(self):
        ws_endpoint = f"wss://stream.binance.com:9443/ws/{self.symbol}@trade"
        ws = create_connection(ws_endpoint)
        # Write header if file does not exist
        write_header = not os.path.isfile(self.output_path)
        with open(self.output_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if write_header:
                writer.writerow(['timestamp', 'price', 'volume'])
            while True:
                msg = ws.recv()
                data = json.loads(msg)
                timestamp = int(data['T'])
                price = float(data['p'])
                volume = float(data['q'])
                ts_str = time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(timestamp/1000))
                writer.writerow([ts_str, price, volume])
