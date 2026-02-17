"""
Holdback Out-of-Sample Validation
Train sur 2024-01-01 -> 2025-06-30, test sur 2025-07-01 -> 2025-12-31
Mesure la degradation des strategies entre train et test
"""

import sys
import os
import time
import pandas as pd
import numpy as np
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.backtest.engine import BacktestEngine
from src.backtest.library import STRATEGY_MAP, instantiate_strategy
from src.config.assets import MONITORED_ASSETS
from src.data.yahoo import convert_to_yahoo_symbol
from src.backtest.allocator import RISK_RANGES

TRAIN_START = '2024-01-01'
TRAIN_END = '2025-06-30'
TEST_START = '2025-07-01'
TEST_END = '2025-12-31'
OUTPUT_FILE = 'HOLDBACK_VALIDATION.csv'


def get_midpoint_sl_tp(category):
    """Get midpoint SL/TP for a category"""
    ranges = RISK_RANGES.get(category, {'sl': (3, 6), 'tp': (5, 10)})
    sl = (ranges['sl'][0] + ranges['sl'][1]) / 2
    tp = (ranges['tp'][0] + ranges['tp'][1]) / 2
    return sl, tp


def run_backtest_period(data, strategy_name, sl=None, tp=None):
    """Run a single backtest on a data slice"""
    strategy = instantiate_strategy(strategy_name)
    if strategy is None:
        return None

    engine = BacktestEngine(
        initial_capital=10000,
        commission=0.001,
        slippage=0.0005,
        stop_loss_pct=sl,
        take_profit_pct=tp,
    )
    return engine.run_backtest(data, strategy)


def run_holdback():
    print("=" * 80)
    print("HOLDBACK OUT-OF-SAMPLE VALIDATION")
    print(f"Train: {TRAIN_START} -> {TRAIN_END}")
    print(f"Test:  {TEST_START} -> {TEST_END}")
    print("=" * 80)

    all_assets = []
    for category, assets in MONITORED_ASSETS.items():
        for asset in assets:
            all_assets.append((category, asset))

    strategy_names = list(STRATEGY_MAP.keys())
    total = len(all_assets) * len(strategy_names)
    current = 0
    results = []
    start_time = time.time()

    for category, asset in all_assets:
        symbol = asset['symbol']
        asset_name = asset['name']
        yahoo_symbol = convert_to_yahoo_symbol(symbol)

        try:
            ticker = yf.Ticker(yahoo_symbol)
            full_data = ticker.history(start=TRAIN_START, end=TEST_END)

            if full_data is None or len(full_data) < 100:
                current += len(strategy_names)
                continue

            full_data.columns = [c.lower() for c in full_data.columns]

            # Split train/test
            train_data = full_data[full_data.index <= TRAIN_END]
            test_data = full_data[full_data.index >= TEST_START]

            if len(train_data) < 50 or len(test_data) < 20:
                current += len(strategy_names)
                continue

            sl, tp = get_midpoint_sl_tp(category)

        except Exception:
            current += len(strategy_names)
            continue

        for strategy_name in strategy_names:
            current += 1

            try:
                # Train
                train_metrics = run_backtest_period(train_data, strategy_name, sl, tp)
                if train_metrics is None:
                    continue

                # Test
                test_metrics = run_backtest_period(test_data, strategy_name, sl, tp)
                if test_metrics is None:
                    continue

                train_return = train_metrics.get('total_return_pct', 0)
                test_return = test_metrics.get('total_return_pct', 0)
                train_sharpe = train_metrics.get('sharpe_ratio', 0)
                test_sharpe = test_metrics.get('sharpe_ratio', 0)
                train_wr = train_metrics.get('win_rate', 0)
                test_wr = test_metrics.get('win_rate', 0)

                # Degradation scores
                if abs(train_sharpe) > 0.01:
                    sharpe_degradation = (train_sharpe - test_sharpe) / abs(train_sharpe)
                else:
                    sharpe_degradation = 1.0 if test_sharpe < 0 else 0.0

                holdback_score = np.clip(1 - sharpe_degradation, 0, 2)

                results.append({
                    'asset': symbol,
                    'asset_type': category,
                    'asset_name': asset_name,
                    'strategy': strategy_name,
                    'train_return_pct': round(train_return, 4),
                    'test_return_pct': round(test_return, 4),
                    'train_sharpe': round(train_sharpe, 4),
                    'test_sharpe': round(test_sharpe, 4),
                    'train_win_rate': round(train_wr, 2),
                    'test_win_rate': round(test_wr, 2),
                    'train_trades': train_metrics.get('total_trades', 0),
                    'test_trades': test_metrics.get('total_trades', 0),
                    'sharpe_degradation': round(sharpe_degradation, 4),
                    'holdback_score': round(holdback_score, 4),
                })

            except Exception:
                continue

        if current % (len(strategy_names) * 10) == 0:
            elapsed = time.time() - start_time
            pct = current / total * 100
            print(f"  [{current}/{total}] ({pct:.0f}%) {symbol} [{elapsed:.0f}s]")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)
    elapsed = time.time() - start_time

    # Report
    print("\n" + "=" * 80)
    print("HOLDBACK VALIDATION RESULTS")
    print("=" * 80)
    print(f"\nDuree: {elapsed:.0f}s")
    print(f"Combinaisons valides: {len(results_df)}")

    if len(results_df) == 0:
        return results_df

    print(f"\nHoldback score moyen: {results_df['holdback_score'].mean():.3f}")
    print(f"Holdback score median: {results_df['holdback_score'].median():.3f}")

    # Strategies les plus robustes
    print("\n" + "-" * 80)
    print("ROBUSTESSE PAR STRATEGIE (holdback score moyen)")
    print("-" * 80)
    strat_robust = results_df.groupby('strategy').agg({
        'holdback_score': 'mean',
        'train_sharpe': 'mean',
        'test_sharpe': 'mean',
        'sharpe_degradation': 'mean',
    }).sort_values('holdback_score', ascending=False)

    for sn, row in strat_robust.iterrows():
        print(f"  {sn:<25} HB={row['holdback_score']:.3f}  "
              f"Train Sharpe={row['train_sharpe']:.3f}  "
              f"Test Sharpe={row['test_sharpe']:.3f}  "
              f"Degrad={row['sharpe_degradation']:.3f}")

    # Categorie
    print("\n" + "-" * 80)
    print("ROBUSTESSE PAR CATEGORIE")
    print("-" * 80)
    cat_robust = results_df.groupby('asset_type').agg({
        'holdback_score': 'mean',
        'test_sharpe': 'mean',
    }).sort_values('holdback_score', ascending=False)

    for cat, row in cat_robust.iterrows():
        print(f"  {cat:<25} HB={row['holdback_score']:.3f}  Test Sharpe={row['test_sharpe']:.3f}")

    # Top 20 most robust combos
    print("\n" + "-" * 80)
    print("TOP 20 COMBOS LES PLUS ROBUSTES")
    print("-" * 80)
    robust_combos = results_df[results_df['train_trades'] >= 10]
    if len(robust_combos) > 0:
        top20 = robust_combos.nlargest(20, 'holdback_score')
        for _, row in top20.iterrows():
            print(f"  {row['asset']:15} x {row['strategy']:22} "
                  f"HB={row['holdback_score']:.2f}  "
                  f"Train={row['train_return_pct']:+.1f}%  Test={row['test_return_pct']:+.1f}%")

    print(f"\nResultats: {OUTPUT_FILE}")
    return results_df


if __name__ == '__main__':
    run_holdback()
