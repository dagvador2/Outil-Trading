"""
Parameter Sensitivity Analysis
Pour chaque strategie, teste une grille de parametres et mesure la stabilite du Sharpe.
robustness_score = 1 - (std_sharpe / mean_sharpe) sur la grille
Un score > 0.5 = robuste, < 0.3 = fragile (overfit probable)
"""

import sys
import os
import time
import pandas as pd
import numpy as np
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.backtest.engine import BacktestEngine
from src.strategies.base import (
    MovingAverageCrossover, RSIStrategy, MACDStrategy,
    BollingerBandsStrategy
)
from src.strategies.extended import ADXTrendStrategy, IchimokuStrategy
from src.strategies.ema_strategies import EMACrossover
from src.strategies.stochastic_strategy import StochasticStrategy
from src.strategies.atr_strategy import ATRBreakoutStrategy
from src.config.assets import MONITORED_ASSETS
from src.data.yahoo import convert_to_yahoo_symbol
from src.backtest.allocator import RISK_RANGES

START_DATE = '2024-01-01'
END_DATE = '2025-12-31'
OUTPUT_FILE = 'SENSITIVITY_ANALYSIS.csv'

# Grilles de parametres par strategie
PARAM_GRIDS = {
    'MA_Crossover': [
        {'fast': f, 'slow': s}
        for f in [8, 10, 15, 20, 25, 30]
        for s in [30, 40, 50, 60, 75]
        if f < s
    ],
    'RSI': [
        {'period': p, 'oversold': os, 'overbought': ob}
        for p in [10, 12, 14, 16, 18]
        for os in [25, 30, 35]
        for ob in [65, 70, 75, 80]
    ],
    'EMA_Crossover': [
        {'fast': f, 'slow': s}
        for f in [5, 9, 12, 15, 20]
        for s in [15, 21, 26, 30, 40]
        if f < s
    ],
    'Bollinger': [
        {'period': p, 'std': s}
        for p in [15, 20, 25, 30]
        for s in [1.5, 2.0, 2.5, 3.0]
    ],
    'Stochastic': [
        {'k': k, 'oversold': os, 'overbought': ob}
        for k in [10, 14, 18, 21]
        for os in [15, 20, 25]
        for ob in [75, 80, 85]
    ],
    'ATR_Breakout': [
        {'sma': p, 'mult': m}
        for p in [15, 20, 25, 30]
        for m in [1.5, 2.0, 2.5, 3.0]
    ],
    'ADX_Trend': [
        {'period': p, 'threshold': t}
        for p in [10, 14, 18, 21]
        for t in [20, 25, 30, 35]
    ],
    'MACD': [
        {'fast': f, 'slow': s, 'signal': sig}
        for f in [8, 10, 12, 15]
        for s in [20, 24, 26, 30]
        for sig in [7, 9, 11]
        if f < s
    ],
}


def create_strategy(name, params):
    """Create a strategy instance from name and params"""
    if name == 'MA_Crossover':
        return MovingAverageCrossover(params['fast'], params['slow'])
    elif name == 'RSI':
        return RSIStrategy(params['period'], params['oversold'], params['overbought'])
    elif name == 'EMA_Crossover':
        return EMACrossover(params['fast'], params['slow'])
    elif name == 'Bollinger':
        return BollingerBandsStrategy(params['period'], params['std'])
    elif name == 'Stochastic':
        return StochasticStrategy(params['k'], 3, params['oversold'], params['overbought'])
    elif name == 'ATR_Breakout':
        return ATRBreakoutStrategy(params['sma'], 14, params['mult'])
    elif name == 'ADX_Trend':
        return ADXTrendStrategy(params['period'], params['threshold'])
    elif name == 'MACD':
        return MACDStrategy(params['fast'], params['slow'], params['signal'])
    return None


def run_sensitivity():
    print("=" * 80)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print(f"Periode: {START_DATE} -> {END_DATE}")
    print("=" * 80)

    # Select a representative sample of assets (1 per category)
    sample_assets = []
    seen_cats = set()
    for category, assets in MONITORED_ASSETS.items():
        base_cat = category.replace('_extra', '')
        if base_cat in seen_cats:
            continue
        seen_cats.add(base_cat)
        if assets:
            sample_assets.append((category, assets[0]))

    print(f"\nActifs echantillon: {len(sample_assets)}")
    print(f"Strategies: {len(PARAM_GRIDS)}")
    total_params = sum(len(g) for g in PARAM_GRIDS.values())
    print(f"Total parametres: {total_params}")
    total_combos = len(sample_assets) * total_params
    print(f"Total combinaisons: {total_combos}")
    print("=" * 80)

    results = []
    current = 0
    start_time = time.time()

    for category, asset in sample_assets:
        symbol = asset['symbol']
        yahoo_symbol = convert_to_yahoo_symbol(symbol)

        try:
            ticker = yf.Ticker(yahoo_symbol)
            data = ticker.history(start=START_DATE, end=END_DATE)
            if data is None or len(data) < 100:
                current += total_params
                continue
            data.columns = [c.lower() for c in data.columns]
        except Exception:
            current += total_params
            continue

        ranges = RISK_RANGES.get(category, {'sl': (3, 6), 'tp': (5, 10)})
        sl = (ranges['sl'][0] + ranges['sl'][1]) / 2
        tp = (ranges['tp'][0] + ranges['tp'][1]) / 2

        for strat_name, param_grid in PARAM_GRIDS.items():
            sharpe_values = []
            return_values = []

            for params in param_grid:
                current += 1
                try:
                    strategy = create_strategy(strat_name, params)
                    if strategy is None:
                        continue

                    engine = BacktestEngine(
                        initial_capital=10000, commission=0.001,
                        slippage=0.0005, stop_loss_pct=sl, take_profit_pct=tp,
                    )
                    metrics = engine.run_backtest(data, strategy)
                    sharpe = metrics.get('sharpe_ratio', 0)
                    ret = metrics.get('total_return_pct', 0)
                    sharpe_values.append(sharpe)
                    return_values.append(ret)

                except Exception:
                    continue

            if len(sharpe_values) >= 3:
                mean_sharpe = np.mean(sharpe_values)
                std_sharpe = np.std(sharpe_values)

                if abs(mean_sharpe) > 0.01:
                    robustness = max(0, 1 - (std_sharpe / abs(mean_sharpe)))
                else:
                    robustness = 0.5

                results.append({
                    'asset': symbol,
                    'asset_type': category,
                    'strategy_family': strat_name,
                    'num_param_combos': len(sharpe_values),
                    'mean_sharpe': round(mean_sharpe, 4),
                    'std_sharpe': round(std_sharpe, 4),
                    'mean_return': round(np.mean(return_values), 4),
                    'std_return': round(np.std(return_values), 4),
                    'best_sharpe': round(max(sharpe_values), 4),
                    'worst_sharpe': round(min(sharpe_values), 4),
                    'robustness_score': round(robustness, 4),
                    'pct_positive_sharpe': round(
                        len([s for s in sharpe_values if s > 0]) / len(sharpe_values) * 100, 1
                    ),
                })

        elapsed = time.time() - start_time
        pct = current / total_combos * 100
        print(f"  [{current}/{total_combos}] ({pct:.0f}%) {symbol} [{elapsed:.0f}s]")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)
    elapsed = time.time() - start_time

    # Report
    print("\n" + "=" * 80)
    print("SENSITIVITY ANALYSIS RESULTS")
    print("=" * 80)
    print(f"\nDuree: {elapsed:.0f}s")

    if len(results_df) == 0:
        print("Aucun resultat.")
        return results_df

    # Par strategie
    print("\n" + "-" * 80)
    print("ROBUSTESSE PAR FAMILLE DE STRATEGIE")
    print("-" * 80)
    strat_robust = results_df.groupby('strategy_family').agg({
        'robustness_score': 'mean',
        'mean_sharpe': 'mean',
        'std_sharpe': 'mean',
        'pct_positive_sharpe': 'mean',
    }).sort_values('robustness_score', ascending=False)

    for sn, row in strat_robust.iterrows():
        label = "ROBUSTE" if row['robustness_score'] > 0.5 else "FRAGILE" if row['robustness_score'] < 0.3 else "MOYEN"
        print(f"  {sn:<20} Robustness={row['robustness_score']:.3f} [{label}]  "
              f"Sharpe={row['mean_sharpe']:.3f} +/- {row['std_sharpe']:.3f}  "
              f"{row['pct_positive_sharpe']:.0f}% positif")

    # Strategies fragiles (warning)
    fragile = results_df[results_df['robustness_score'] < 0.3]
    if len(fragile) > 0:
        print(f"\n  WARNING: {len(fragile)} combos fragiles (robustness < 0.3)")
        for _, row in fragile.iterrows():
            print(f"    {row['asset']:15} x {row['strategy_family']:20} "
                  f"Robustness={row['robustness_score']:.3f}")

    print(f"\nResultats: {OUTPUT_FILE}")
    return results_df


if __name__ == '__main__':
    run_sensitivity()
