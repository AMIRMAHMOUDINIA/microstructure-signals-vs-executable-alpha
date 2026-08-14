from __future__ import annotations
import argparse
import asyncio
import csv
import json
from pathlib import Path
import time
import websockets


BASE = "wss://fstream.binance.com/public/stream"


async def collect(symbol: str, minutes: int, output: Path):
    symbol = symbol.lower()
    streams = f"{symbol}@bookTicker/{symbol}@aggTrade"
    url = f"{BASE}?streams={streams}"

    output.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + minutes * 60

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["received_ns", "stream", "payload_json"])

        async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
            while time.time() < deadline:
                msg = await ws.recv()
                obj = json.loads(msg)
                writer.writerow([
                    time.time_ns(),
                    obj.get("stream", ""),
                    json.dumps(obj.get("data", {}), separators=(",", ":")),
                ])

    print(f"Saved {output}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--minutes", type=int, default=10)
    p.add_argument("--output", type=Path, default=Path("data/raw/live_capture.csv"))
    args = p.parse_args()
    asyncio.run(collect(args.symbol, args.minutes, args.output))


if __name__ == "__main__":
    main()
