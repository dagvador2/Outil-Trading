"""
Backtest complet 2024-2025 pour 200 actifs x 10 strategies
Utilise Yahoo Finance pour les donnees historiques
Sauvegarde les resultats dans RESULTS_200_ASSETS_2024_2025.csv
"""

import sys
import os
import time
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# S'assurer qu'on importe depuis le bon repertoire
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtesting_engine import BacktestEngine
from strategies import (
    MovingAverageCrossover, RSIStrategy, MACDStrategy,
    BollingerBandsStrategy, CombinedStrategy
)
from strategies_extended import (
    ADXTrendStrategy, VWAPStrategy, IchimokuStrategy
)
from assets_config import MONITORED_ASSETS
from yahoo_data_feed import convert_to_yahoo_symbol


# ============================================================================
# 10 strategies a tester
# ============================================================================
STRATEGY_MAP = {
    'MA_Crossover_20_50': lambda: MovingAverageCrossover(20, 50),
    'MA_Crossover_10_30': lambda: MovingAverageCrossover(10, 30),
    'RSI_14_30_70': lambda: RSIStrategy(14, 30, 70),
    'RSI_14_35_80': lambda: RSIStrategy(14, 35, 80),
    'MACD_Standard': lambda: MACDStrategy(12, 26, 9),
    'Bollinger_20_2': lambda: BollingerBandsStrategy(20, 2),
    'Combined': lambda: CombinedStrategy(),
    'ADX_Trend_14_25': lambda: ADXTrendStrategy(14, 25),
    'VWAP_20': lambda: VWAPStrategy(20),
    'Ichimoku_9_26_52': lambda: IchimokuStrategy(9, 26, 52),
}

START_DATE = '2024-01-01'
END_DATE = '2025-12-31'
OUTPUT_FILE = 'RESULTS_200_ASSETS_2024_2025.csv'


def run_all_backtests():
    """Execute tous les backtests et sauvegarde les resultats"""

    print("=" * 80)
    print(f"BACKTEST COMPLET : {START_DATE} -> {END_DATE}")
    print(f"Strategies: {len(STRATEGY_MAP)}")
    print("=" * 80)

    # Compter les actifs
    all_assets = []
    for category, assets in MONITORED_ASSETS.items():
        for asset in assets:
            all_assets.append((category, asset))

    strategy_names = list(STRATEGY_MAP.keys())
    total_combos = len(all_assets) * len(strategy_names)

    print(f"Actifs: {len(all_assets)}")
    print(f"Total combinaisons: {total_combos}")
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

        # Telecharger donnees UNE seule fois par actif
        try:
            ticker = yf.Ticker(yahoo_symbol)
            data = ticker.history(start=START_DATE, end=END_DATE)

            if data is None or len(data) < 50:
                for sn in strategy_names:
                    current += 1
                skipped += 1
                elapsed = time.time() - start_time
                pct = current / total_combos * 100
                print(f"  [{current}/{total_combos}] ({pct:.0f}%) SKIP {symbol} ({yahoo_symbol}) - pas assez de donnees ({len(data) if data is not None else 0} bougies) [{elapsed:.0f}s]")
                continue

            # Normaliser colonnes
            data.columns = [col.lower() for col in data.columns]

            # S'assurer qu'on a les colonnes OHLCV
            required = ['open', 'high', 'low', 'close', 'volume']
            available = [c for c in required if c in data.columns]
            if len(available) < 4:
                for sn in strategy_names:
                    current += 1
                skipped += 1
                print(f"  [{current}/{total_combos}] SKIP {symbol} - colonnes manquantes")
                continue

            num_candles = len(data)

        except Exception as e:
            for sn in strategy_names:
                current += 1
            errors += 1
            elapsed = time.time() - start_time
            pct = current / total_combos * 100
            print(f"  [{current}/{total_combos}] ({pct:.0f}%) ERROR {symbol}: {str(e)[:60]} [{elapsed:.0f}s]")
            continue

        # Tester chaque strategie
        for strategy_name in strategy_names:
            current += 1

            try:
                strategy = STRATEGY_MAP[strategy_name]()
                engine = BacktestEngine(
                    initial_capital=10000,
                    commission=0.001,
                    slippage=0.0005
                )

                metrics = engine.run_backtest(data, strategy)

                num_trades = metrics.get('total_trades', 0)
                total_return = metrics.get('total_return_pct', 0)

                avg_trade_return = 0
                if num_trades > 0:
                    trades_df = metrics.get('trades_df', None)
                    if trades_df is not None and len(trades_df) > 0:
                        avg_trade_return = trades_df['pnl_pct'].mean()

                results.append({
                    'asset': symbol,
                    'asset_type': category,
                    'asset_name': asset_name,
                    'strategy': strategy_name,
                    'total_return_pct': round(total_return, 4),
                    'sharpe_ratio': round(metrics.get('sharpe_ratio', 0), 4),
                    'max_drawdown_pct': round(abs(metrics.get('max_drawdown', 0)), 4),
                    'win_rate': round(metrics.get('win_rate', 0), 2),
                    'num_trades': num_trades,
                    'avg_trade_return_pct': round(avg_trade_return, 4),
                    'data_source': 'yahoo_finance',
                    'num_candles': num_candles,
                    'period': f"{START_DATE} - {END_DATE}"
                })

            except Exception as e:
                results.append({
                    'asset': symbol,
                    'asset_type': category,
                    'asset_name': asset_name,
                    'strategy': strategy_name,
                    'total_return_pct': 0,
                    'sharpe_ratio': 0,
                    'max_drawdown_pct': 0,
                    'win_rate': 0,
                    'num_trades': 0,
                    'avg_trade_return_pct': 0,
                    'data_source': f'error: {str(e)[:50]}',
                    'num_candles': 0,
                    'period': f"{START_DATE} - {END_DATE}"
                })

        # Afficher progression par actif
        elapsed = time.time() - start_time
        pct = current / total_combos * 100
        eta = (elapsed / current * (total_combos - current)) if current > 0 else 0
        print(f"  [{current}/{total_combos}] ({pct:.0f}%) {symbol:15} ({asset_name:25}) - {num_candles} bougies - OK [{elapsed:.0f}s, ETA {eta:.0f}s]")

    # Sauvegarder
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)

    elapsed = time.time() - start_time

    # ========================================================================
    # Resume
    # ========================================================================
    print("\n" + "=" * 80)
    print("RESULTATS DU BACKTEST")
    print("=" * 80)

    print(f"\nPeriode         : {START_DATE} -> {END_DATE}")
    print(f"Duree           : {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Combinaisons    : {len(results)}")
    print(f"Actifs testes   : {results_df['asset'].nunique()}")
    print(f"Strategies      : {results_df['strategy'].nunique()}")
    print(f"Actifs skip/err : {skipped} skip, {errors} erreurs")

    if len(results_df) > 0:
        valid = results_df[results_df['data_source'] == 'yahoo_finance']
        if len(valid) > 0:
            print(f"\nRendement moyen     : {valid['total_return_pct'].mean():.2f}%")
            print(f"Rendement median    : {valid['total_return_pct'].median():.2f}%")
            print(f"Meilleur rendement  : {valid['total_return_pct'].max():.2f}%")
            print(f"Pire rendement      : {valid['total_return_pct'].min():.2f}%")

            num_positive = len(valid[valid['total_return_pct'] > 0])
            print(f"Rendements positifs : {num_positive}/{len(valid)} ({num_positive/len(valid)*100:.1f}%)")

            print(f"\nSharpe moyen        : {valid['sharpe_ratio'].mean():.3f}")
            print(f"Win rate moyen      : {valid['win_rate'].mean():.1f}%")
            print(f"Drawdown max moyen  : {valid['max_drawdown_pct'].mean():.2f}%")

            # Top 20 meilleures combinaisons
            print("\n" + "-" * 80)
            print("TOP 20 MEILLEURES COMBINAISONS (par rendement)")
            print("-" * 80)
            top20 = valid.nlargest(20, 'total_return_pct')
            for i, row in top20.iterrows():
                print(f"  {row['asset']:15} x {row['strategy']:22} -> {row['total_return_pct']:+8.2f}%  "
                      f"Sharpe={row['sharpe_ratio']:.2f}  WR={row['win_rate']:.0f}%  "
                      f"DD={row['max_drawdown_pct']:.1f}%  Trades={row['num_trades']}")

            # Top par categorie
            print("\n" + "-" * 80)
            print("MEILLEURE COMBINAISON PAR CATEGORIE")
            print("-" * 80)
            for cat in valid['asset_type'].unique():
                cat_data = valid[valid['asset_type'] == cat]
                best = cat_data.nlargest(1, 'total_return_pct').iloc[0]
                print(f"  {cat:25} -> {best['asset']:15} x {best['strategy']:22} = {best['total_return_pct']:+.2f}%")

            # Performance par strategie (moyenne)
            print("\n" + "-" * 80)
            print("PERFORMANCE MOYENNE PAR STRATEGIE")
            print("-" * 80)
            strat_perf = valid.groupby('strategy').agg({
                'total_return_pct': 'mean',
                'sharpe_ratio': 'mean',
                'win_rate': 'mean',
                'max_drawdown_pct': 'mean'
            }).sort_values('total_return_pct', ascending=False)

            for sn, row in strat_perf.iterrows():
                print(f"  {sn:25} -> Return={row['total_return_pct']:+6.2f}%  "
                      f"Sharpe={row['sharpe_ratio']:.3f}  WR={row['win_rate']:.1f}%  "
                      f"DD={row['max_drawdown_pct']:.1f}%")

            # Performance par region
            print("\n" + "-" * 80)
            print("PERFORMANCE MOYENNE PAR CATEGORIE D'ACTIFS")
            print("-" * 80)
            cat_perf = valid.groupby('asset_type').agg({
                'total_return_pct': 'mean',
                'sharpe_ratio': 'mean',
                'win_rate': 'mean'
            }).sort_values('total_return_pct', ascending=False)

            for cat, row in cat_perf.iterrows():
                print(f"  {cat:25} -> Return={row['total_return_pct']:+6.2f}%  "
                      f"Sharpe={row['sharpe_ratio']:.3f}  WR={row['win_rate']:.1f}%")

    print(f"\nResultats sauvegardes dans: {OUTPUT_FILE}")
    print("=" * 80)

    # Sauvegarder aussi en tant que cache pour backtest_library
    results_df.to_csv('RESULTS_2024_2025_COMPLETE.csv', index=False)
    print(f"Cache mis a jour: RESULTS_2024_2025_COMPLETE.csv")

    return results_df


if __name__ == '__main__':
    run_all_backtests()
