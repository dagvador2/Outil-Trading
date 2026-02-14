"""DDL statements for the trading database."""

SCHEMA_VERSION = 1

PRAGMAS = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA busy_timeout=5000;
PRAGMA foreign_keys=ON;
"""

TABLES = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS candles (
    symbol    TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    open      REAL NOT NULL,
    high      REAL NOT NULL,
    low       REAL NOT NULL,
    close     REAL NOT NULL,
    volume    REAL,
    source    TEXT DEFAULT 'yahoo',
    PRIMARY KEY (symbol, timeframe, timestamp)
);

CREATE TABLE IF NOT EXISTS signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT NOT NULL,
    signal          TEXT NOT NULL,
    confidence      REAL,
    price           REAL,
    timestamp       TEXT NOT NULL,
    strategy        TEXT,
    asset_type      TEXT,
    consensus       TEXT,
    macro_impact    REAL,
    indicators_json TEXT
);

CREATE TABLE IF NOT EXISTS macro_signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    source          TEXT,
    category        TEXT,
    title           TEXT,
    description     TEXT,
    impact_score    REAL,
    confidence      REAL,
    affected_assets TEXT,
    url             TEXT,
    sentiment       TEXT
);

CREATE TABLE IF NOT EXISTS signal_cache (
    asset_filter TEXT NOT NULL,
    timeframe    TEXT NOT NULL,
    cached_at    REAL NOT NULL,
    signals_json TEXT NOT NULL,
    PRIMARY KEY (asset_filter, timeframe)
);

CREATE TABLE IF NOT EXISTS portfolio_state (
    portfolio_name TEXT PRIMARY KEY,
    cash           REAL NOT NULL,
    total_capital  REAL NOT NULL,
    cycle_count    INTEGER DEFAULT 0,
    last_update    TEXT
);

CREATE TABLE IF NOT EXISTS positions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    portfolio_name  TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL,
    entry_price     REAL NOT NULL,
    quantity        REAL NOT NULL,
    entry_time      TEXT,
    stop_loss       REAL,
    take_profit     REAL,
    strategy        TEXT,
    allocation_pct  REAL,
    confidence      REAL,
    UNIQUE (portfolio_name, symbol)
);

CREATE TABLE IF NOT EXISTS pending_orders (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    portfolio_name     TEXT NOT NULL,
    symbol             TEXT NOT NULL,
    side               TEXT NOT NULL,
    signal_price       REAL,
    target_entry_price REAL,
    strategy           TEXT,
    allocation_pct     REAL,
    stop_loss_pct      REAL,
    take_profit_pct    REAL,
    created_time       TEXT,
    confidence         REAL,
    UNIQUE (portfolio_name, symbol)
);

CREATE TABLE IF NOT EXISTS closed_trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    portfolio_name  TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL,
    entry_price     REAL NOT NULL,
    exit_price      REAL NOT NULL,
    quantity        REAL NOT NULL,
    entry_time      TEXT,
    exit_time       TEXT,
    pnl             REAL,
    pnl_pct         REAL,
    exit_reason     TEXT,
    strategy        TEXT
);

CREATE TABLE IF NOT EXISTS equity_curve (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    portfolio_name  TEXT NOT NULL,
    timestamp       TEXT NOT NULL,
    total_value     REAL NOT NULL,
    cash            REAL,
    positions_value REAL
);
"""

INDEXES = """
CREATE INDEX IF NOT EXISTS idx_candles_symbol_tf
    ON candles (symbol, timeframe);
CREATE INDEX IF NOT EXISTS idx_signals_symbol
    ON signals (symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_macro_signals_ts
    ON macro_signals (timestamp);
CREATE INDEX IF NOT EXISTS idx_positions_portfolio
    ON positions (portfolio_name);
CREATE INDEX IF NOT EXISTS idx_pending_portfolio
    ON pending_orders (portfolio_name);
CREATE INDEX IF NOT EXISTS idx_closed_trades_portfolio
    ON closed_trades (portfolio_name);
CREATE INDEX IF NOT EXISTS idx_equity_portfolio
    ON equity_curve (portfolio_name, timestamp);
"""
