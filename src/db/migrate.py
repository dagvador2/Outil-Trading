"""
Migrate seed_data/ (SCP'd from live server) into SQLite.

Reads:
  - 20 × auto_state.json  → portfolio_state + positions + pending_orders
  - 12 × auto_trades.csv  → closed_trades
  - consolidated_state.json → name mapping reference
  - signals_tous_1d.json  → signal_cache
  - test_cache.json        → macro_signals

Usage:
    python -m src.db.migrate
    python -m src.db.migrate --seed-dir seed_data --db data/trading.db
"""

import csv
import json
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.db.database import TradingDatabase


SEED_DIR = PROJECT_ROOT / "seed_data"
PORTFOLIO_BASE = SEED_DIR / "paper_trading_state" / "paper_trading_state"

# Map directory name → portfolio display name used in consolidated_state
PORTFOLIO_NAME_MAP = {
    "portfolio_01_conservative": "Conservative",
    "portfolio_02_balanced": "Balanced",
    "portfolio_03_aggressive": "Aggressive",
    "portfolio_04_high_sharpe": "High_Sharpe",
    "portfolio_05_low_drawdown": "Low_Drawdown",
    "portfolio_06_equal_weight": "Equal_Weight",
    "portfolio_07_risk_parity": "Risk_Parity",
    "portfolio_08_concentrated": "Concentrated",
    "portfolio_09_diversified": "Diversified",
    "portfolio_10_crypto_commodities": "Crypto_Commodities",
    "portfolio_11_conservative_macro": "Conservative_Macro",
    "portfolio_12_balanced_macro": "Balanced_Macro",
    "portfolio_13_aggressive_macro": "Aggressive_Macro",
    "portfolio_14_high_sharpe_macro": "High_Sharpe_Macro",
    "portfolio_15_low_drawdown_macro": "Low_Drawdown_Macro",
    "portfolio_16_equal_weight_macro": "Equal_Weight_Macro",
    "portfolio_17_risk_parity_macro": "Risk_Parity_Macro",
    "portfolio_18_concentrated_macro": "Concentrated_Macro",
    "portfolio_19_diversified_macro": "Diversified_Macro",
    "portfolio_20_crypto_commodities_macro": "Crypto_Commodities_Macro",
}


def migrate_portfolios(db: TradingDatabase, base_path: Path):
    """Import 20 portfolio auto_state.json files."""
    imported = 0
    for dir_name, portfolio_name in PORTFOLIO_NAME_MAP.items():
        state_file = base_path / dir_name / "auto_state.json"
        if not state_file.exists():
            print(f"  SKIP {portfolio_name}: no auto_state.json")
            continue

        with open(state_file) as fh:
            state = json.load(fh)

        # Portfolio state
        db.save_portfolio_state(portfolio_name, {
            "cash": state.get("cash", 0),
            "total_capital": state.get("total_capital", 0),
            "cycle_count": state.get("cycle_count", 0),
            "last_update": state.get("last_update"),
        })

        # Positions
        positions = state.get("positions", {})
        if positions:
            db.save_positions(portfolio_name, positions)

        # Pending orders
        pending = state.get("pending_orders", {})
        if pending:
            db.save_pending_orders(portfolio_name, pending)

        imported += 1
        pos_count = len(positions)
        pending_count = len(pending)
        print(f"  OK {portfolio_name}: cash={state.get('cash', 0):,.2f}, "
              f"positions={pos_count}, pending={pending_count}")

    return imported


def migrate_trades(db: TradingDatabase, base_path: Path):
    """Import auto_trades.csv from each portfolio that has one."""
    total_trades = 0
    for dir_name, portfolio_name in PORTFOLIO_NAME_MAP.items():
        trades_file = base_path / dir_name / "auto_trades.csv"
        if not trades_file.exists():
            continue

        trades = []
        with open(trades_file, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                trades.append({
                    "symbol": row.get("symbol"),
                    "side": row.get("side"),
                    "entry_price": float(row.get("entry_price", 0)),
                    "exit_price": float(row.get("exit_price", 0)),
                    "quantity": float(row.get("quantity", 0)),
                    "entry_time": row.get("entry_time"),
                    "exit_time": row.get("exit_time"),
                    "pnl": float(row.get("pnl", 0)),
                    "pnl_pct": float(row.get("pnl_pct", 0)),
                    "exit_reason": row.get("exit_reason"),
                    "strategy": row.get("strategy"),
                })

        if trades:
            db.save_closed_trades(portfolio_name, trades)
            total_trades += len(trades)
            print(f"  OK {portfolio_name}: {len(trades)} trades")

    return total_trades


def migrate_signal_cache(db: TradingDatabase, seed_dir: Path):
    """Import signals_tous_1d.json into signal_cache table."""
    cache_file = seed_dir / "signals_tous_1d.json"
    if not cache_file.exists():
        print("  SKIP signals_tous_1d.json: not found")
        return 0

    with open(cache_file) as fh:
        signals = json.load(fh)

    if isinstance(signals, list):
        db.set_signal_cache("tous", "1d", signals)
        print(f"  OK signal_cache: {len(signals)} signals cached")
        return len(signals)
    return 0


def migrate_macro_signals(db: TradingDatabase, seed_dir: Path):
    """Import test_cache.json into macro_signals table."""
    cache_file = seed_dir / "test_cache.json"
    if not cache_file.exists():
        print("  SKIP test_cache.json: not found")
        return 0

    with open(cache_file) as fh:
        signals = json.load(fh)

    if isinstance(signals, list):
        # Convert to the format expected by insert_macro_signals
        macro_dicts = []
        for signal in signals:
            macro_dicts.append({
                "timestamp": signal.get("timestamp"),
                "source": signal.get("source"),
                "category": signal.get("category"),
                "title": signal.get("title"),
                "description": signal.get("description"),
                "impact_score": signal.get("impact_score"),
                "confidence": signal.get("confidence"),
                "affected_assets": signal.get("affected_assets", []),
                "url": signal.get("url"),
                "sentiment": signal.get("sentiment"),
            })
        db.insert_macro_signals(macro_dicts)
        print(f"  OK macro_signals: {len(macro_dicts)} signals imported")
        return len(macro_dicts)
    return 0


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Migrate seed_data → SQLite")
    parser.add_argument("--seed-dir", default=str(SEED_DIR))
    parser.add_argument("--db", default=None)
    args = parser.parse_args()

    seed_dir = Path(args.seed_dir)
    db = TradingDatabase(args.db) if args.db else TradingDatabase()

    print("=" * 60)
    print("MIGRATION: seed_data → SQLite")
    print(f"  Source: {seed_dir}")
    print(f"  Target: {db.db_path}")
    print("=" * 60)

    portfolio_base = seed_dir / "paper_trading_state" / "paper_trading_state"

    print("\n[1/4] Portfolios (auto_state.json)")
    portfolio_count = migrate_portfolios(db, portfolio_base)

    print(f"\n[2/4] Closed trades (auto_trades.csv)")
    trade_count = migrate_trades(db, portfolio_base)

    print(f"\n[3/4] Signal cache (signals_tous_1d.json)")
    signal_count = migrate_signal_cache(db, seed_dir)

    print(f"\n[4/4] Macro signals (test_cache.json)")
    macro_count = migrate_macro_signals(db, seed_dir)

    print("\n" + "=" * 60)
    print("MIGRATION COMPLETE")
    print(f"  Portfolios: {portfolio_count}")
    print(f"  Trades:     {trade_count}")
    print(f"  Signals:    {signal_count}")
    print(f"  Macro:      {macro_count}")
    print(f"  DB size:    {os.path.getsize(db.db_path) / 1024:.1f} KB")
    print("=" * 60)


if __name__ == "__main__":
    main()
