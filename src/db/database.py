"""
TradingDatabase — single SQLite persistence layer.

Thread-safe: one connection per thread via threading.local().
All writes use `with connection:` for atomicity (WAL mode).
"""

import json
import os
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

from src.db.schema import PRAGMAS, TABLES, INDEXES, SCHEMA_VERSION

DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "trading.db",
)


class TradingDatabase:
    """Consolidated SQLite store for the entire trading system."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._local = threading.local()
        # Ensure schema exists on the creating thread
        self._init_schema()

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        """Return a per-thread connection (created lazily)."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, timeout=10)
            conn.row_factory = sqlite3.Row
            for line in PRAGMAS.strip().splitlines():
                line = line.strip()
                if line:
                    conn.execute(line)
            self._local.conn = conn
        return conn

    def _init_schema(self):
        conn = self._get_conn()
        for statement in TABLES.strip().split(";"):
            statement = statement.strip()
            if statement:
                conn.execute(statement)
        for statement in INDEXES.strip().split(";"):
            statement = statement.strip()
            if statement:
                conn.execute(statement)
        # Track schema version
        row = conn.execute(
            "SELECT version FROM schema_version LIMIT 1"
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)",
                (SCHEMA_VERSION,),
            )
        conn.commit()

    # ==================================================================
    # Candles
    # ==================================================================

    def upsert_candles(
        self,
        symbol: str,
        timeframe: str,
        dataframe: pd.DataFrame,
        source: str = "yahoo",
    ):
        """Bulk-upsert OHLCV rows from a DataFrame (index = datetime)."""
        if dataframe is None or dataframe.empty:
            return
        conn = self._get_conn()
        rows = []
        for ts, row in dataframe.iterrows():
            timestamp_str = str(ts)
            rows.append((
                symbol,
                timeframe,
                timestamp_str,
                float(row.get("open", 0)),
                float(row.get("high", 0)),
                float(row.get("low", 0)),
                float(row.get("close", 0)),
                float(row.get("volume", 0)) if "volume" in row.index else None,
                source,
            ))
        with conn:
            conn.executemany(
                """INSERT OR REPLACE INTO candles
                   (symbol, timeframe, timestamp, open, high, low, close, volume, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )

    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
    ) -> pd.DataFrame:
        """Return OHLCV DataFrame ordered by timestamp ascending."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT timestamp, open, high, low, close, volume
               FROM candles
               WHERE symbol = ? AND timeframe = ?
               ORDER BY timestamp DESC LIMIT ?""",
            (symbol, timeframe, limit),
        ).fetchall()
        if not rows:
            return pd.DataFrame()
        data = [dict(row) for row in rows]
        dataframe = pd.DataFrame(data)
        dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"])
        dataframe.set_index("timestamp", inplace=True)
        return dataframe.sort_index()

    def get_latest_timestamp(self, symbol: str, timeframe: str) -> Optional[str]:
        conn = self._get_conn()
        row = conn.execute(
            """SELECT MAX(timestamp) FROM candles
               WHERE symbol = ? AND timeframe = ?""",
            (symbol, timeframe),
        ).fetchone()
        return row[0] if row and row[0] else None

    def count_candles(self, symbol: str, timeframe: str) -> int:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT COUNT(*) FROM candles WHERE symbol = ? AND timeframe = ?",
            (symbol, timeframe),
        ).fetchone()
        return row[0] if row else 0

    # ==================================================================
    # Signals
    # ==================================================================

    def insert_signal(self, signal_dict: dict):
        conn = self._get_conn()
        indicators = signal_dict.get("indicators")
        indicators_json = json.dumps(indicators) if indicators else None
        with conn:
            conn.execute(
                """INSERT INTO signals
                   (symbol, signal, confidence, price, timestamp,
                    strategy, asset_type, consensus, macro_impact, indicators_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    signal_dict.get("symbol"),
                    signal_dict.get("signal"),
                    signal_dict.get("confidence"),
                    signal_dict.get("current_price") or signal_dict.get("price"),
                    signal_dict.get("timestamp", datetime.now().isoformat()),
                    signal_dict.get("strategy"),
                    signal_dict.get("asset_type"),
                    signal_dict.get("consensus"),
                    signal_dict.get("macro_impact"),
                    indicators_json,
                ),
            )

    def get_signals(
        self,
        symbol: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 200,
    ) -> List[dict]:
        conn = self._get_conn()
        clauses = []
        params: list = []
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol)
        if since:
            clauses.append("timestamp >= ?")
            params.append(since)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = conn.execute(
            f"SELECT * FROM signals {where} ORDER BY timestamp DESC LIMIT ?",
            params + [limit],
        ).fetchall()
        return [dict(row) for row in rows]

    # ==================================================================
    # Macro signals
    # ==================================================================

    def insert_macro_signals(self, signals: list):
        """Insert a batch of macro signal dicts."""
        if not signals:
            return
        conn = self._get_conn()
        rows = []
        for signal in signals:
            affected = signal.get("affected_assets", [])
            if isinstance(affected, list):
                affected = json.dumps(affected)
            ts = signal.get("timestamp", datetime.now().isoformat())
            if hasattr(ts, "isoformat"):
                ts = ts.isoformat()
            rows.append((
                ts,
                signal.get("source"),
                signal.get("category"),
                signal.get("title"),
                signal.get("description"),
                signal.get("impact_score"),
                signal.get("confidence"),
                affected,
                signal.get("url"),
                signal.get("sentiment"),
            ))
        with conn:
            conn.executemany(
                """INSERT INTO macro_signals
                   (timestamp, source, category, title, description,
                    impact_score, confidence, affected_assets, url, sentiment)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )

    def get_macro_signals(self, hours: int = 24) -> List[dict]:
        conn = self._get_conn()
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        rows = conn.execute(
            """SELECT * FROM macro_signals
               WHERE timestamp >= ?
               ORDER BY timestamp DESC""",
            (cutoff,),
        ).fetchall()
        results = []
        for row in rows:
            record = dict(row)
            try:
                record["affected_assets"] = json.loads(
                    record.get("affected_assets", "[]")
                )
            except (json.JSONDecodeError, TypeError):
                record["affected_assets"] = []
            results.append(record)
        return results

    def prune_macro_signals(self, days: int = 30):
        conn = self._get_conn()
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        with conn:
            conn.execute(
                "DELETE FROM macro_signals WHERE timestamp < ?", (cutoff,)
            )

    # ==================================================================
    # Signal cache
    # ==================================================================

    def get_signal_cache(
        self, asset_filter: str, timeframe: str, ttl_seconds: float = 3600
    ) -> Optional[List[dict]]:
        conn = self._get_conn()
        row = conn.execute(
            """SELECT cached_at, signals_json FROM signal_cache
               WHERE asset_filter = ? AND timeframe = ?""",
            (asset_filter, timeframe),
        ).fetchone()
        if not row:
            return None
        if time.time() - row["cached_at"] > ttl_seconds:
            return None
        try:
            return json.loads(row["signals_json"])
        except (json.JSONDecodeError, TypeError):
            return None

    def set_signal_cache(
        self, asset_filter: str, timeframe: str, signals: list
    ):
        conn = self._get_conn()
        with conn:
            conn.execute(
                """INSERT OR REPLACE INTO signal_cache
                   (asset_filter, timeframe, cached_at, signals_json)
                   VALUES (?, ?, ?, ?)""",
                (asset_filter, timeframe, time.time(), json.dumps(signals, default=str)),
            )

    # ==================================================================
    # Portfolio state
    # ==================================================================

    def save_portfolio_state(self, portfolio_name: str, state: dict):
        conn = self._get_conn()
        with conn:
            conn.execute(
                """INSERT OR REPLACE INTO portfolio_state
                   (portfolio_name, cash, total_capital, cycle_count, last_update)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    portfolio_name,
                    state.get("cash", 0),
                    state.get("total_capital", 0),
                    state.get("cycle_count", 0),
                    state.get("last_update", datetime.now().isoformat()),
                ),
            )

    def load_portfolio_state(self, portfolio_name: str) -> Optional[dict]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM portfolio_state WHERE portfolio_name = ?",
            (portfolio_name,),
        ).fetchone()
        return dict(row) if row else None

    # ==================================================================
    # Positions
    # ==================================================================

    def save_positions(self, portfolio_name: str, positions: dict):
        """Replace all positions for a portfolio (dict keyed by symbol)."""
        conn = self._get_conn()
        with conn:
            conn.execute(
                "DELETE FROM positions WHERE portfolio_name = ?",
                (portfolio_name,),
            )
            for symbol, pos in positions.items():
                if hasattr(pos, "__dict__"):
                    pos = pos.__dict__
                conn.execute(
                    """INSERT INTO positions
                       (portfolio_name, symbol, side, entry_price, quantity,
                        entry_time, stop_loss, take_profit, strategy,
                        allocation_pct, confidence)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        portfolio_name,
                        pos.get("symbol", symbol),
                        pos.get("side"),
                        pos.get("entry_price"),
                        pos.get("quantity"),
                        pos.get("entry_time"),
                        pos.get("stop_loss"),
                        pos.get("take_profit"),
                        pos.get("strategy"),
                        pos.get("allocation_pct"),
                        pos.get("confidence"),
                    ),
                )

    def load_positions(self, portfolio_name: str) -> dict:
        """Return positions as {symbol: dict}."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM positions WHERE portfolio_name = ?",
            (portfolio_name,),
        ).fetchall()
        return {row["symbol"]: dict(row) for row in rows}

    # ==================================================================
    # Pending orders
    # ==================================================================

    def save_pending_orders(self, portfolio_name: str, orders: dict):
        """Replace all pending orders for a portfolio."""
        conn = self._get_conn()
        with conn:
            conn.execute(
                "DELETE FROM pending_orders WHERE portfolio_name = ?",
                (portfolio_name,),
            )
            for symbol, order in orders.items():
                if hasattr(order, "__dict__"):
                    order = order.__dict__
                conn.execute(
                    """INSERT INTO pending_orders
                       (portfolio_name, symbol, side, signal_price,
                        target_entry_price, strategy, allocation_pct,
                        stop_loss_pct, take_profit_pct, created_time, confidence)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        portfolio_name,
                        order.get("symbol", symbol),
                        order.get("side"),
                        order.get("signal_price"),
                        order.get("target_entry_price"),
                        order.get("strategy"),
                        order.get("allocation_pct"),
                        order.get("stop_loss_pct"),
                        order.get("take_profit_pct"),
                        order.get("created_time"),
                        order.get("confidence"),
                    ),
                )

    def load_pending_orders(self, portfolio_name: str) -> dict:
        """Return pending orders as {symbol: dict}."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM pending_orders WHERE portfolio_name = ?",
            (portfolio_name,),
        ).fetchall()
        return {row["symbol"]: dict(row) for row in rows}

    # ==================================================================
    # Closed trades
    # ==================================================================

    def insert_closed_trade(self, portfolio_name: str, trade: dict):
        if hasattr(trade, "__dict__"):
            trade = trade.__dict__
        conn = self._get_conn()
        with conn:
            conn.execute(
                """INSERT INTO closed_trades
                   (portfolio_name, symbol, side, entry_price, exit_price,
                    quantity, entry_time, exit_time, pnl, pnl_pct,
                    exit_reason, strategy)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    portfolio_name,
                    trade.get("symbol"),
                    trade.get("side"),
                    trade.get("entry_price"),
                    trade.get("exit_price"),
                    trade.get("quantity"),
                    trade.get("entry_time"),
                    trade.get("exit_time"),
                    trade.get("pnl"),
                    trade.get("pnl_pct"),
                    trade.get("exit_reason"),
                    trade.get("strategy"),
                ),
            )

    def get_closed_trades(self, portfolio_name: str) -> List[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT * FROM closed_trades
               WHERE portfolio_name = ?
               ORDER BY exit_time ASC""",
            (portfolio_name,),
        ).fetchall()
        return [dict(row) for row in rows]

    def save_closed_trades(self, portfolio_name: str, trades: list):
        """Replace all closed trades for a portfolio (used by migration)."""
        conn = self._get_conn()
        with conn:
            conn.execute(
                "DELETE FROM closed_trades WHERE portfolio_name = ?",
                (portfolio_name,),
            )
            for trade in trades:
                if hasattr(trade, "__dict__"):
                    trade = trade.__dict__
                conn.execute(
                    """INSERT INTO closed_trades
                       (portfolio_name, symbol, side, entry_price, exit_price,
                        quantity, entry_time, exit_time, pnl, pnl_pct,
                        exit_reason, strategy)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        portfolio_name,
                        trade.get("symbol"),
                        trade.get("side"),
                        trade.get("entry_price"),
                        trade.get("exit_price"),
                        trade.get("quantity"),
                        trade.get("entry_time"),
                        trade.get("exit_time"),
                        trade.get("pnl"),
                        trade.get("pnl_pct"),
                        trade.get("exit_reason"),
                        trade.get("strategy"),
                    ),
                )

    # ==================================================================
    # Equity curve
    # ==================================================================

    def record_equity(
        self,
        portfolio_name: str,
        total_value: float,
        cash: float,
        positions_value: float,
    ):
        conn = self._get_conn()
        with conn:
            conn.execute(
                """INSERT INTO equity_curve
                   (portfolio_name, timestamp, total_value, cash, positions_value)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    portfolio_name,
                    datetime.now().isoformat(),
                    total_value,
                    cash,
                    positions_value,
                ),
            )

    def get_equity_curve(self, portfolio_name: str) -> pd.DataFrame:
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT timestamp, total_value, cash, positions_value
               FROM equity_curve
               WHERE portfolio_name = ?
               ORDER BY timestamp ASC""",
            (portfolio_name,),
        ).fetchall()
        if not rows:
            return pd.DataFrame()
        data = [dict(row) for row in rows]
        dataframe = pd.DataFrame(data)
        dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"])
        dataframe.set_index("timestamp", inplace=True)
        return dataframe
