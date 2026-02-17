"""
Backtest avance : 200 actifs x 25 strategies x 6 modes risk management
Compare : sans filtre / macro / SL-TP / macro+SL-TP / trailing / macro+trailing
Inclut regime ADX tagging sur chaque trade
"""

import sys
import os
import time
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.backtest.engine import BacktestEngine
from src.strategies.base import (
    MovingAverageCrossover, RSIStrategy, MACDStrategy,
    BollingerBandsStrategy, CombinedStrategy
)
from src.strategies.extended import (
    ADXTrendStrategy, VWAPStrategy, IchimokuStrategy
)
from src.strategies.long_only import (
    MovingAverageCrossoverLongOnly, RSIStrategyLongOnly,
    TrendFollowingLongOnly, BollingerLongOnly,
    HybridTrendMomentumLongOnly
)
from src.strategies.ema_strategies import EMACrossover
from src.strategies.stochastic_strategy import StochasticStrategy
from src.strategies.atr_strategy import ATRBreakoutStrategy
from src.strategies.momentum_strategies import (
    MACDHistogramStrategy, RSIStochasticConvergence
)
from src.strategies.filters import (
    RegimeAwareStrategy, MultiTimeframeStrategy
)
from src.config.assets import MONITORED_ASSETS
from src.data.yahoo import convert_to_yahoo_symbol
from src.indicators.technical import TechnicalIndicators
from backtest_macro_filter import BacktestMacroFilter
from src.backtest.allocator import RISK_RANGES


# ============================================================================
# Helper : strategie pre-calculee
# ============================================================================
class PrecomputedStrategy:
    """Wrapper pour passer des signaux pre-generes au BacktestEngine"""
    def __init__(self, name, signals_df):
        self.name = name
        self._signals = signals_df
    def generate_signals(self, data):
        return self._signals


# ============================================================================
# 25 strategies
# ============================================================================
STRATEGY_MAP = {
    # --- Originales (10) ---
    'MA_Cross_20_50': lambda: MovingAverageCrossover(20, 50),
    'MA_Cross_10_30': lambda: MovingAverageCrossover(10, 30),
    'RSI_14_30_70': lambda: RSIStrategy(14, 30, 70),
    'RSI_14_35_80': lambda: RSIStrategy(14, 35, 80),
    'MACD_Standard': lambda: MACDStrategy(12, 26, 9),
    'Bollinger_20_2': lambda: BollingerBandsStrategy(20, 2),
    'Combined': lambda: CombinedStrategy(),
    'ADX_Trend_14_25': lambda: ADXTrendStrategy(14, 25),
    'VWAP_20': lambda: VWAPStrategy(20),
    'Ichimoku_9_26_52': lambda: IchimokuStrategy(9, 26, 52),
    # --- Long-only (5) ---
    'MA_Cross_LO': lambda: MovingAverageCrossoverLongOnly(20, 50),
    'RSI_LO': lambda: RSIStrategyLongOnly(14, 35, 80),
    'TrendFollow_LO': lambda: TrendFollowingLongOnly(20, 50, 200),
    'Bollinger_LO': lambda: BollingerLongOnly(20, 2),
    'HybridMom_LO': lambda: HybridTrendMomentumLongOnly(),
    # --- Nouvelles (6) ---
    'EMA_Cross_12_26': lambda: EMACrossover(12, 26),
    'EMA_Cross_9_21': lambda: EMACrossover(9, 21),
    'Stochastic_14': lambda: StochasticStrategy(14, 3, 20, 80),
    'ATR_Breakout': lambda: ATRBreakoutStrategy(20, 14, 2.0),
    'MACD_Histogram': lambda: MACDHistogramStrategy(12, 26, 9),
    'RSI_Stoch_Conv': lambda: RSIStochasticConvergence(),
    # --- Regime-aware (2) ---
    'Regime_MA_Boll': lambda: RegimeAwareStrategy(
        MovingAverageCrossover(20, 50), BollingerBandsStrategy(20, 2)
    ),
    'Regime_EMA_Stoch': lambda: RegimeAwareStrategy(
        EMACrossover(12, 26), StochasticStrategy(14, 3, 20, 80)
    ),
    # --- Multi-timeframe (2) ---
    'MA_Cross_MTF200': lambda: MultiTimeframeStrategy(
        MovingAverageCrossover(20, 50), 200
    ),
    'EMA_Cross_MTF200': lambda: MultiTimeframeStrategy(
        EMACrossover(12, 26), 200
    ),
}

START_DATE = '2024-01-01'
END_DATE = '2025-12-31'
MARKET_LOOKBACK_START = '2023-01-01'
OUTPUT_FILE = 'RESULTS_200_ASSETS_2024_2025.csv'

# Risk management modes
RISK_MODES = {
    'none': {'sl': None, 'tp': None, 'trail': None},
    'sltp': 'dynamic',
    'trailing': 'dynamic',
}

MACRO_MODES = [False, True]


def get_risk_params(mode, category):
    """Get SL/TP/trailing params for a risk mode + asset category"""
    if mode == 'none':
        return None, None, None

    ranges = RISK_RANGES.get(category, {'sl': (3, 6), 'tp': (5, 10)})
    sl = (ranges['sl'][0] + ranges['sl'][1]) / 2
    tp = (ranges['tp'][0] + ranges['tp'][1]) / 2

    if mode == 'sltp':
        return sl, tp, None
    elif mode == 'trailing':
        return sl, None, sl * 0.8

    return None, None, None


def run_single_backtest(data, strategy_name, category, risk_mode='none',
                        macro_filter=None):
    """Execute un backtest avec toutes les options"""
    strategy = STRATEGY_MAP[strategy_name]()
    signals = strategy.generate_signals(data)

    macro_info = None
    if macro_filter is not None:
        signals, macro_info = macro_filter.filter_signals(signals, data)

    wrapper = PrecomputedStrategy(strategy_name, signals)

    sl, tp, trail = get_risk_params(risk_mode, category)

    engine = BacktestEngine(
        initial_capital=10000,
        commission=0.001,
        slippage=0.0005,
        stop_loss_pct=sl,
        take_profit_pct=tp,
        trailing_stop_pct=trail,
    )
    metrics = engine.run_backtest(data, wrapper)
    return metrics, macro_info


def compute_adx_regime(data):
    """Pre-compute ADX for regime tagging"""
    try:
        return TechnicalIndicators.adx(data['high'], data['low'], data['close'], 14)
    except Exception:
        return pd.Series(0, index=data.index)


def run_all_backtests():
    """Execute tous les backtests et sauvegarde"""

    print("=" * 80)
    print("BACKTEST AVANCE : 25 STRATEGIES x 6 MODES RISK x 200 ACTIFS")
    print(f"Periode: {START_DATE} -> {END_DATE}")
    print("=" * 80)

    # Macro filter
    macro_filter = BacktestMacroFilter(strong_threshold=60.0)
    print("\nChargement donnees de marche (VIX + S&P 500)...")
    macro_filter.load_market_data(MARKET_LOOKBACK_START, END_DATE)
    print("  -> OK")

    all_assets = []
    for category, assets in MONITORED_ASSETS.items():
        for asset in assets:
            all_assets.append((category, asset))

    strategy_names = list(STRATEGY_MAP.keys())
    risk_modes = list(RISK_MODES.keys())
    combos_per_asset = len(strategy_names) * len(risk_modes) * len(MACRO_MODES)
    total_combos = len(all_assets) * combos_per_asset

    print(f"\nActifs: {len(all_assets)}")
    print(f"Strategies: {len(strategy_names)}")
    print(f"Risk modes: {len(risk_modes)} ({', '.join(risk_modes)})")
    print(f"Macro: sans + avec")
    print(f"Total: {total_combos} combinaisons")
    print("=" * 80)

    results = []
    current = 0
    errors = 0
    skipped = 0
    start_time = time.time()

    for category, asset in all_assets:
        symbol = asset['symbol']
        asset_name = asset['name']
        yahoo_symbol = convert_to_yahoo_symbol(symbol)

        try:
            ticker = yf.Ticker(yahoo_symbol)
            data = ticker.history(start=START_DATE, end=END_DATE)

            if data is None or len(data) < 50:
                current += combos_per_asset
                skipped += 1
                elapsed = time.time() - start_time
                pct = current / total_combos * 100
                n_c = len(data) if data is not None else 0
                print(f"  [{current}/{total_combos}] ({pct:.0f}%) SKIP {symbol} - {n_c} bougies [{elapsed:.0f}s]")
                continue

            data.columns = [col.lower() for col in data.columns]
            if len([c for c in ['open','high','low','close','volume'] if c in data.columns]) < 4:
                current += combos_per_asset
                skipped += 1
                continue

            num_candles = len(data)
            adx_series = compute_adx_regime(data)

        except Exception as e:
            current += combos_per_asset
            errors += 1
            elapsed = time.time() - start_time
            pct = current / total_combos * 100
            print(f"  [{current}/{total_combos}] ({pct:.0f}%) ERROR {symbol}: {str(e)[:60]} [{elapsed:.0f}s]")
            continue

        for strategy_name in strategy_names:
            for risk_mode in risk_modes:
                for use_macro in MACRO_MODES:
                    current += 1
                    mf = macro_filter if use_macro else None
                    macro_label = 'macro' if use_macro else 'none'

                    try:
                        metrics, macro_info = run_single_backtest(
                            data, strategy_name, category, risk_mode, mf
                        )

                        num_trades = metrics.get('total_trades', 0)
                        total_return = metrics.get('total_return_pct', 0)

                        avg_trade_return = 0
                        exit_dist = ''
                        regime_trades = {'trending': 0, 'ranging': 0, 'transition': 0}
                        regime_pnl = {'trending': [], 'ranging': [], 'transition': []}

                        if num_trades > 0:
                            trades_df = metrics.get('trades_df', None)
                            if trades_df is not None and len(trades_df) > 0:
                                avg_trade_return = trades_df['pnl_pct'].mean()

                                if 'exit_reason' in trades_df.columns:
                                    dist = trades_df['exit_reason'].value_counts(normalize=True)
                                    exit_dist = ','.join(f"{k}:{v:.0%}" for k, v in dist.items())

                                for _, trade in trades_df.iterrows():
                                    adx_val = adx_series.get(trade['entry_date'], 0)
                                    if pd.isna(adx_val):
                                        adx_val = 0
                                    regime = 'trending' if adx_val > 25 else ('ranging' if adx_val < 20 else 'transition')
                                    regime_trades[regime] += 1
                                    regime_pnl[regime].append(trade.get('pnl_pct', 0))

                        sl, tp, trail = get_risk_params(risk_mode, category)

                        row = {
                            'asset': symbol, 'asset_type': category,
                            'asset_name': asset_name, 'strategy': strategy_name,
                            'macro_filter': macro_label, 'risk_mode': risk_mode,
                            'stop_loss_pct': sl or 0, 'take_profit_pct': tp or 0,
                            'trailing_stop_pct': trail or 0,
                            'total_return_pct': round(total_return, 4),
                            'sharpe_ratio': round(metrics.get('sharpe_ratio', 0), 4),
                            'max_drawdown_pct': round(abs(metrics.get('max_drawdown', 0)), 4),
                            'win_rate': round(metrics.get('win_rate', 0), 2),
                            'num_trades': num_trades,
                            'avg_trade_return_pct': round(avg_trade_return, 4),
                            'exit_reasons': exit_dist,
                            'trades_trending': regime_trades['trending'],
                            'trades_ranging': regime_trades['ranging'],
                            'trades_transition': regime_trades['transition'],
                            'pnl_trending': round(np.mean(regime_pnl['trending']), 4) if regime_pnl['trending'] else 0,
                            'pnl_ranging': round(np.mean(regime_pnl['ranging']), 4) if regime_pnl['ranging'] else 0,
                            'data_source': 'yahoo_finance',
                            'num_candles': num_candles,
                            'period': f"{START_DATE} - {END_DATE}",
                        }
                        if macro_info:
                            row['macro_avg_score'] = macro_info['avg_score']
                            row['macro_filtered_trades'] = macro_info['total_filtered']

                        results.append(row)

                    except Exception as e:
                        results.append({
                            'asset': symbol, 'asset_type': category,
                            'asset_name': asset_name, 'strategy': strategy_name,
                            'macro_filter': macro_label, 'risk_mode': risk_mode,
                            'stop_loss_pct': 0, 'take_profit_pct': 0,
                            'trailing_stop_pct': 0,
                            'total_return_pct': 0, 'sharpe_ratio': 0,
                            'max_drawdown_pct': 0, 'win_rate': 0,
                            'num_trades': 0, 'avg_trade_return_pct': 0,
                            'exit_reasons': '', 'trades_trending': 0,
                            'trades_ranging': 0, 'trades_transition': 0,
                            'pnl_trending': 0, 'pnl_ranging': 0,
                            'data_source': f'error: {str(e)[:50]}',
                            'num_candles': 0,
                            'period': f"{START_DATE} - {END_DATE}",
                        })

        elapsed = time.time() - start_time
        pct = current / total_combos * 100
        eta = (elapsed / current * (total_combos - current)) if current > 0 else 0
        print(f"  [{current}/{total_combos}] ({pct:.0f}%) {symbol:15} ({asset_name:25}) - "
              f"{num_candles} bougies [{elapsed:.0f}s, ETA {eta:.0f}s]")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)
    elapsed = time.time() - start_time

    # ========================================================================
    # Report
    # ========================================================================
    print("\n" + "=" * 80)
    print("RESULTATS DU BACKTEST AVANCE")
    print("=" * 80)
    print(f"\nDuree: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Combinaisons: {len(results)}")
    print(f"Actifs: {results_df['asset'].nunique()} | Strategies: {results_df['strategy'].nunique()}")
    print(f"Skip: {skipped} | Erreurs: {errors}")

    valid = results_df[results_df['data_source'] == 'yahoo_finance']
    if len(valid) == 0:
        print("Aucun resultat valide.")
        return results_df

    # --- Par risk mode ---
    print("\n" + "-" * 80)
    print("COMPARAISON PAR MODE DE RISK MANAGEMENT")
    print("-" * 80)
    print(f"  {'Mode':<25} {'Return%':>10} {'Sharpe':>10} {'WR%':>8} {'DD%':>8} {'Trades':>8}")
    print("  " + "-" * 70)

    for rm in risk_modes:
        for macro_lbl in ['none', 'macro']:
            subset = valid[(valid['risk_mode'] == rm) & (valid['macro_filter'] == macro_lbl)]
            if len(subset) == 0:
                continue
            label = f"{rm}{'_macro' if macro_lbl == 'macro' else ''}"
            print(f"  {label:<25} {subset['total_return_pct'].mean():>+9.2f}% "
                  f"{subset['sharpe_ratio'].mean():>10.3f} "
                  f"{subset['win_rate'].mean():>7.1f}% "
                  f"{subset['max_drawdown_pct'].mean():>7.1f}% "
                  f"{subset['num_trades'].mean():>7.0f}")

    # --- Par strategie (mode sltp + macro) ---
    best_mode = valid[(valid['risk_mode'] == 'sltp') & (valid['macro_filter'] == 'macro')]
    if len(best_mode) > 0:
        print("\n" + "-" * 80)
        print("PERFORMANCE PAR STRATEGIE (mode: SL/TP + Macro)")
        print("-" * 80)
        strat_perf = best_mode.groupby('strategy').agg({
            'total_return_pct': 'mean', 'sharpe_ratio': 'mean',
            'win_rate': 'mean', 'max_drawdown_pct': 'mean',
            'num_trades': 'mean',
        }).sort_values('sharpe_ratio', ascending=False)

        for sn, row in strat_perf.iterrows():
            print(f"  {sn:<25} Return={row['total_return_pct']:>+7.2f}%  "
                  f"Sharpe={row['sharpe_ratio']:.3f}  WR={row['win_rate']:.0f}%  "
                  f"DD={row['max_drawdown_pct']:.1f}%  Trades={row['num_trades']:.0f}")

    # --- Top 20 par Sharpe ---
    if len(best_mode) > 0:
        print("\n" + "-" * 80)
        print("TOP 20 (SL/TP + Macro, par Sharpe)")
        print("-" * 80)
        top20 = best_mode.nlargest(20, 'sharpe_ratio')
        for _, row in top20.iterrows():
            print(f"  {row['asset']:15} x {row['strategy']:22} "
                  f"Sharpe={row['sharpe_ratio']:.2f}  Return={row['total_return_pct']:+.1f}%  "
                  f"WR={row['win_rate']:.0f}%  DD={row['max_drawdown_pct']:.1f}%")

    # --- Impact SL/TP ---
    no_risk = valid[(valid['risk_mode'] == 'none') & (valid['macro_filter'] == 'none')]
    sltp_risk = valid[(valid['risk_mode'] == 'sltp') & (valid['macro_filter'] == 'none')]
    if len(no_risk) > 0 and len(sltp_risk) > 0:
        print("\n" + "-" * 80)
        print("IMPACT DU SL/TP (sans macro)")
        print("-" * 80)
        for col, label in [('total_return_pct', 'Return%'), ('sharpe_ratio', 'Sharpe'),
                           ('win_rate', 'Win Rate%'), ('max_drawdown_pct', 'Max DD%')]:
            v1, v2 = no_risk[col].mean(), sltp_risk[col].mean()
            delta = v2 - v1
            sign = "+" if delta > 0 else ""
            print(f"  {label:<15} Sans: {v1:>8.3f}  Avec: {v2:>8.3f}  Delta: {sign}{delta:.3f}")

    # --- Regime ---
    if len(best_mode) > 0:
        print("\n" + "-" * 80)
        print("PERFORMANCE PAR REGIME (sltp+macro)")
        print("-" * 80)
        print(f"  Trending (ADX>25): PnL/trade = {best_mode['pnl_trending'].mean():+.3f}%  "
              f"({best_mode['trades_trending'].sum():.0f} trades)")
        print(f"  Ranging  (ADX<20): PnL/trade = {best_mode['pnl_ranging'].mean():+.3f}%  "
              f"({best_mode['trades_ranging'].sum():.0f} trades)")

    # --- Par categorie ---
    if len(best_mode) > 0:
        print("\n" + "-" * 80)
        print("PAR CATEGORIE (sltp+macro)")
        print("-" * 80)
        cat_perf = best_mode.groupby('asset_type').agg({
            'total_return_pct': 'mean', 'sharpe_ratio': 'mean',
        }).sort_values('sharpe_ratio', ascending=False)
        for cat, row in cat_perf.iterrows():
            print(f"  {cat:<25} Return={row['total_return_pct']:>+7.2f}%  Sharpe={row['sharpe_ratio']:.3f}")

    print(f"\nResultats: {OUTPUT_FILE}")
    results_df.to_csv('RESULTS_2024_2025_COMPLETE.csv', index=False)
    print(f"Cache: RESULTS_2024_2025_COMPLETE.csv")
    print("=" * 80)

    return results_df


if __name__ == '__main__':
    run_all_backtests()
