"""
Backfill historical 1d candles from Yahoo Finance and Binance.

- Yahoo: stocks, indices, forex, commodities, ETFs (~80 symbols)
- Binance: crypto (~20 symbols)
- Skips symbols already having >= 50 candles in DB
- Rate limiting: 0.5s between Yahoo requests, ccxt built-in for Binance

Usage:
    python -m src.db.backfill
    python -m src.db.backfill --force   # re-fetch even if data exists
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config.assets import MONITORED_ASSETS
from src.data.yahoo import convert_to_yahoo_symbol
from src.db.database import TradingDatabase

TIMEFRAME = "1d"
MIN_CANDLES_SKIP = 50


def backfill_yahoo(db: TradingDatabase, symbols: list[str], force: bool = False):
    """Fetch daily candles from Yahoo Finance for non-crypto symbols."""
    import yfinance as yf

    fetched = 0
    skipped = 0
    errors = 0

    for symbol in symbols:
        if not force and db.count_candles(symbol, TIMEFRAME) >= MIN_CANDLES_SKIP:
            skipped += 1
            continue

        yahoo_sym = convert_to_yahoo_symbol(symbol)
        try:
            ticker = yf.Ticker(yahoo_sym)
            data = ticker.history(period="max", interval="1d")

            if data is None or data.empty:
                print(f"  WARN {symbol} ({yahoo_sym}): no data")
                errors += 1
                continue

            data.columns = [col.lower() for col in data.columns]
            required_cols = ["open", "high", "low", "close"]
            if not all(col in data.columns for col in required_cols):
                print(f"  WARN {symbol}: missing columns {data.columns.tolist()}")
                errors += 1
                continue

            keep_cols = [col for col in ["open", "high", "low", "close", "volume"]
                         if col in data.columns]
            data = data[keep_cols]

            db.upsert_candles(symbol, TIMEFRAME, data, source="yahoo")
            fetched += 1
            print(f"  OK {symbol}: {len(data)} candles")

        except Exception as exc:
            print(f"  ERR {symbol}: {exc}")
            errors += 1

        time.sleep(0.5)

    return fetched, skipped, errors


def backfill_binance(db: TradingDatabase, symbols: list[str], force: bool = False):
    """Fetch daily candles from Binance for crypto symbols."""
    import ccxt

    exchange = ccxt.binance({"enableRateLimit": True})

    fetched = 0
    skipped = 0
    errors = 0

    for symbol in symbols:
        if not force and db.count_candles(symbol, TIMEFRAME) >= MIN_CANDLES_SKIP:
            skipped += 1
            continue

        try:
            all_candles = []
            since = exchange.parse8601("2023-01-01T00:00:00Z")

            while True:
                ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, since=since, limit=1000)
                if not ohlcv:
                    break
                all_candles.extend(ohlcv)
                last_ts = ohlcv[-1][0]
                if last_ts == since or len(ohlcv) < 1000:
                    break
                since = last_ts + 1

            if not all_candles:
                print(f"  WARN {symbol}: no data from Binance")
                errors += 1
                continue

            dataframe = pd.DataFrame(
                all_candles,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"], unit="ms")
            dataframe.set_index("timestamp", inplace=True)

            db.upsert_candles(symbol, TIMEFRAME, dataframe, source="binance")
            fetched += 1
            print(f"  OK {symbol}: {len(dataframe)} candles")

        except Exception as exc:
            print(f"  ERR {symbol}: {exc}")
            errors += 1

    return fetched, skipped, errors


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Backfill historical candles")
    parser.add_argument("--force", action="store_true",
                        help="Re-fetch even if data exists")
    parser.add_argument("--db", default=None)
    args = parser.parse_args()

    db = TradingDatabase(args.db) if args.db else TradingDatabase()

    # Split symbols into crypto vs everything else
    crypto_symbols = []
    yahoo_symbols = []

    for category, assets in MONITORED_ASSETS.items():
        for asset in assets:
            symbol = asset["symbol"]
            if category == "crypto":
                crypto_symbols.append(symbol)
            else:
                yahoo_symbols.append(symbol)

    print("=" * 60)
    print("BACKFILL: Historical candles → SQLite")
    print(f"  DB: {db.db_path}")
    print(f"  Yahoo symbols: {len(yahoo_symbols)}")
    print(f"  Binance symbols: {len(crypto_symbols)}")
    print(f"  Force: {args.force}")
    print("=" * 60)

    print(f"\n[1/2] Yahoo Finance ({len(yahoo_symbols)} symbols)")
    yahoo_fetched, yahoo_skipped, yahoo_errors = backfill_yahoo(
        db, yahoo_symbols, force=args.force
    )

    print(f"\n[2/2] Binance ({len(crypto_symbols)} symbols)")
    binance_fetched, binance_skipped, binance_errors = backfill_binance(
        db, crypto_symbols, force=args.force
    )

    total_fetched = yahoo_fetched + binance_fetched
    total_skipped = yahoo_skipped + binance_skipped
    total_errors = yahoo_errors + binance_errors

    print("\n" + "=" * 60)
    print("BACKFILL COMPLETE")
    print(f"  Fetched: {total_fetched}")
    print(f"  Skipped: {total_skipped} (already had >= {MIN_CANDLES_SKIP} candles)")
    print(f"  Errors:  {total_errors}")

    import os
    db_size = os.path.getsize(db.db_path) / (1024 * 1024)
    print(f"  DB size: {db_size:.1f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
