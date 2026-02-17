"""
Backtest complet 2024-2025 pour 200 actifs x 10 strategies x 2 modes (avec/sans macro filter)
Utilise Yahoo Finance pour les donnees historiques
Compare les resultats SANS filtre macro vs AVEC filtre macro (comme en live)
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

from src.backtest.engine import BacktestEngine
from src.strategies.base import (
    MovingAverageCrossover, RSIStrategy, MACDStrategy,
    BollingerBandsStrategy, CombinedStrategy
)
from src.strategies.extended import (
    ADXTrendStrategy, VWAPStrategy, IchimokuStrategy
)
from src.config.assets import MONITORED_ASSETS
from src.data.yahoo import convert_to_yahoo_symbol
from backtest_macro_filter import BacktestMacroFilter


# ============================================================================
# Helper : strategie pre-calculee (pour injecter des signaux filtres)
# ============================================================================
class PrecomputedStrategy:
    """Wrapper pour passer des signaux deja generes/filtres au BacktestEngine"""

    def __init__(self, name: str, signals_df: pd.DataFrame):
        self.name = name
        self._signals = signals_df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        return self._signals


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
MARKET_LOOKBACK_START = '2023-01-01'  # 1 an de lookback pour SMA200 du marche
OUTPUT_FILE = 'RESULTS_200_ASSETS_2024_2025.csv'


def run_single_backtest(data, strategy_name, macro_filter=None):
    """
    Execute un backtest pour un actif + strategie, avec ou sans filtre macro.

    Args:
        data: DataFrame OHLCV
        strategy_name: nom de la strategie
        macro_filter: BacktestMacroFilter ou None

    Returns:
        dict avec metriques, ou None si erreur
    """
    strategy = STRATEGY_MAP[strategy_name]()

    # Generer les signaux bruts
    signals = strategy.generate_signals(data)

    # Appliquer le filtre macro si demande
    macro_info = None
    if macro_filter is not None:
        signals, macro_info = macro_filter.filter_signals(signals, data)

    # Creer une strategie wrapper avec les signaux (potentiellement filtres)
    wrapper = PrecomputedStrategy(strategy_name, signals)

    # Executer le backtest
    engine = BacktestEngine(
        initial_capital=10000,
        commission=0.001,
        slippage=0.0005
    )
    metrics = engine.run_backtest(data, wrapper)

    return metrics, macro_info


def run_all_backtests():
    """Execute tous les backtests (avec et sans macro) et sauvegarde les resultats"""

    print("=" * 80)
    print(f"BACKTEST COMPLET AVEC COMPARAISON MACRO FILTER")
    print(f"Periode: {START_DATE} -> {END_DATE}")
    print(f"Strategies: {len(STRATEGY_MAP)}")
    print(f"Modes: SANS macro filter + AVEC macro filter")
    print("=" * 80)

    # --- Initialiser le filtre macro ---
    macro_filter = BacktestMacroFilter(strong_threshold=60.0)
    print("\nChargement des donnees de marche (VIX + S&P 500)...")
    macro_filter.load_market_data(MARKET_LOOKBACK_START, END_DATE)
    print("  -> Donnees de marche chargees (avec lookback depuis 2023)")

    # Compter les actifs
    all_assets = []
    for category, assets in MONITORED_ASSETS.items():
        for asset in assets:
            all_assets.append((category, asset))

    strategy_names = list(STRATEGY_MAP.keys())
    # 2 modes: sans macro + avec macro
    total_combos = len(all_assets) * len(strategy_names) * 2

    print(f"\nActifs: {len(all_assets)}")
    print(f"Strategies: {len(strategy_names)}")
    print(f"Total combinaisons: {total_combos} ({len(all_assets)} x {len(strategy_names)} x 2 modes)")
    print("=" * 80)

    results = []
    current = 0
    errors = 0
    skipped = 0
    macro_stats = {'filtered_longs': 0, 'filtered_shorts': 0}
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
                    current += 2  # 2 modes
                skipped += 1
                elapsed = time.time() - start_time
                pct = current / total_combos * 100
                print(f"  [{current}/{total_combos}] ({pct:.0f}%) SKIP {symbol} ({yahoo_symbol}) - "
                      f"pas assez de donnees ({len(data) if data is not None else 0} bougies) [{elapsed:.0f}s]")
                continue

            # Normaliser colonnes
            data.columns = [col.lower() for col in data.columns]

            # S'assurer qu'on a les colonnes OHLCV
            required = ['open', 'high', 'low', 'close', 'volume']
            available = [c for c in required if c in data.columns]
            if len(available) < 4:
                for sn in strategy_names:
                    current += 2
                skipped += 1
                print(f"  [{current}/{total_combos}] SKIP {symbol} - colonnes manquantes")
                continue

            num_candles = len(data)

        except Exception as e:
            for sn in strategy_names:
                current += 2
            errors += 1
            elapsed = time.time() - start_time
            pct = current / total_combos * 100
            print(f"  [{current}/{total_combos}] ({pct:.0f}%) ERROR {symbol}: {str(e)[:60]} [{elapsed:.0f}s]")
            continue

        # Tester chaque strategie dans les 2 modes
        for strategy_name in strategy_names:

            for mode, mf in [('none', None), ('macro_filtered', macro_filter)]:
                current += 1

                try:
                    metrics, macro_info = run_single_backtest(data, strategy_name, mf)

                    num_trades = metrics.get('total_trades', 0)
                    total_return = metrics.get('total_return_pct', 0)

                    avg_trade_return = 0
                    if num_trades > 0:
                        trades_df = metrics.get('trades_df', None)
                        if trades_df is not None and len(trades_df) > 0:
                            avg_trade_return = trades_df['pnl_pct'].mean()

                    row_data = {
                        'asset': symbol,
                        'asset_type': category,
                        'asset_name': asset_name,
                        'strategy': strategy_name,
                        'macro_filter': mode,
                        'total_return_pct': round(total_return, 4),
                        'sharpe_ratio': round(metrics.get('sharpe_ratio', 0), 4),
                        'max_drawdown_pct': round(abs(metrics.get('max_drawdown', 0)), 4),
                        'win_rate': round(metrics.get('win_rate', 0), 2),
                        'num_trades': num_trades,
                        'avg_trade_return_pct': round(avg_trade_return, 4),
                        'data_source': 'yahoo_finance',
                        'num_candles': num_candles,
                        'period': f"{START_DATE} - {END_DATE}"
                    }

                    # Ajouter les infos macro si disponibles
                    if macro_info is not None:
                        row_data['macro_avg_score'] = macro_info['avg_score']
                        row_data['macro_filtered_trades'] = macro_info['total_filtered']
                        row_data['macro_filter_active_pct'] = macro_info['filter_active_pct']
                        macro_stats['filtered_longs'] += macro_info['filtered_longs']
                        macro_stats['filtered_shorts'] += macro_info['filtered_shorts']

                    results.append(row_data)

                except Exception as e:
                    results.append({
                        'asset': symbol,
                        'asset_type': category,
                        'asset_name': asset_name,
                        'strategy': strategy_name,
                        'macro_filter': mode,
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
        print(f"  [{current}/{total_combos}] ({pct:.0f}%) {symbol:15} ({asset_name:25}) - "
              f"{num_candles} bougies - OK [{elapsed:.0f}s, ETA {eta:.0f}s]")

    # Sauvegarder
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)

    elapsed = time.time() - start_time

    # ========================================================================
    # Resume global
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

    valid = results_df[results_df['data_source'] == 'yahoo_finance']
    if len(valid) == 0:
        print("\nAucun resultat valide.")
        return results_df

    # ========================================================================
    # Comparaison SANS vs AVEC macro filter
    # ========================================================================
    no_macro = valid[valid['macro_filter'] == 'none']
    with_macro = valid[valid['macro_filter'] == 'macro_filtered']

    print("\n" + "=" * 80)
    print("COMPARAISON: SANS MACRO FILTER vs AVEC MACRO FILTER")
    print("=" * 80)

    print(f"\n{'Metrique':<30} {'Sans Macro':>15} {'Avec Macro':>15} {'Delta':>12}")
    print("-" * 75)

    metrics_compare = [
        ('Rendement moyen (%)', 'total_return_pct', 'mean'),
        ('Rendement median (%)', 'total_return_pct', 'median'),
        ('Sharpe moyen', 'sharpe_ratio', 'mean'),
        ('Win Rate moyen (%)', 'win_rate', 'mean'),
        ('Drawdown max moyen (%)', 'max_drawdown_pct', 'mean'),
        ('Nb trades moyen', 'num_trades', 'mean'),
        ('Rendement/trade moyen (%)', 'avg_trade_return_pct', 'mean'),
    ]

    for label, col, agg_fn in metrics_compare:
        val_no = getattr(no_macro[col], agg_fn)()
        val_with = getattr(with_macro[col], agg_fn)()
        delta = val_with - val_no
        sign = "+" if delta > 0 else ""
        print(f"  {label:<28} {val_no:>15.3f} {val_with:>15.3f} {sign}{delta:>11.3f}")

    # Pourcentage de combos positives
    pct_pos_no = (no_macro['total_return_pct'] > 0).mean() * 100
    pct_pos_with = (with_macro['total_return_pct'] > 0).mean() * 100
    delta_pos = pct_pos_with - pct_pos_no
    sign = "+" if delta_pos > 0 else ""
    print(f"  {'Combos positives (%)':28} {pct_pos_no:>15.1f} {pct_pos_with:>15.1f} {sign}{delta_pos:>11.1f}")

    # Stats du filtre macro
    print(f"\n  Signaux LONG annules par macro  : {macro_stats['filtered_longs']}")
    print(f"  Signaux SHORT annules par macro : {macro_stats['filtered_shorts']}")
    print(f"  Total signaux filtres           : {macro_stats['filtered_longs'] + macro_stats['filtered_shorts']}")

    # ========================================================================
    # Comparaison par strategie
    # ========================================================================
    print("\n" + "-" * 80)
    print("IMPACT MACRO PAR STRATEGIE")
    print("-" * 80)
    print(f"  {'Strategie':<25} {'Return SANS':>12} {'Return AVEC':>12} {'Delta':>8} {'Sharpe SANS':>12} {'Sharpe AVEC':>12}")
    print("  " + "-" * 83)

    for sn in strategy_names:
        no_s = no_macro[no_macro['strategy'] == sn]
        with_s = with_macro[with_macro['strategy'] == sn]
        r_no = no_s['total_return_pct'].mean()
        r_with = with_s['total_return_pct'].mean()
        delta = r_with - r_no
        sh_no = no_s['sharpe_ratio'].mean()
        sh_with = with_s['sharpe_ratio'].mean()
        sign = "+" if delta > 0 else ""
        print(f"  {sn:<25} {r_no:>+11.2f}% {r_with:>+11.2f}% {sign}{delta:>7.2f} {sh_no:>12.3f} {sh_with:>12.3f}")

    # ========================================================================
    # Comparaison par categorie d'actifs
    # ========================================================================
    print("\n" + "-" * 80)
    print("IMPACT MACRO PAR CATEGORIE D'ACTIFS")
    print("-" * 80)
    print(f"  {'Categorie':<25} {'Return SANS':>12} {'Return AVEC':>12} {'Delta':>8}")
    print("  " + "-" * 60)

    for cat in sorted(valid['asset_type'].unique()):
        no_c = no_macro[no_macro['asset_type'] == cat]
        with_c = with_macro[with_macro['asset_type'] == cat]
        r_no = no_c['total_return_pct'].mean()
        r_with = with_c['total_return_pct'].mean()
        delta = r_with - r_no
        sign = "+" if delta > 0 else ""
        print(f"  {cat:<25} {r_no:>+11.2f}% {r_with:>+11.2f}% {sign}{delta:>7.2f}")

    # ========================================================================
    # Top 20 meilleures combinaisons (avec macro)
    # ========================================================================
    print("\n" + "-" * 80)
    print("TOP 20 MEILLEURES COMBINAISONS AVEC MACRO FILTER")
    print("-" * 80)
    top20_macro = with_macro.nlargest(20, 'total_return_pct')
    for _, row in top20_macro.iterrows():
        print(f"  {row['asset']:15} x {row['strategy']:22} -> {row['total_return_pct']:+8.2f}%  "
              f"Sharpe={row['sharpe_ratio']:.2f}  WR={row['win_rate']:.0f}%  "
              f"DD={row['max_drawdown_pct']:.1f}%  Trades={row['num_trades']}")

    # ========================================================================
    # Top 20 sans macro (pour comparaison)
    # ========================================================================
    print("\n" + "-" * 80)
    print("TOP 20 MEILLEURES COMBINAISONS SANS MACRO FILTER")
    print("-" * 80)
    top20_no = no_macro.nlargest(20, 'total_return_pct')
    for _, row in top20_no.iterrows():
        print(f"  {row['asset']:15} x {row['strategy']:22} -> {row['total_return_pct']:+8.2f}%  "
              f"Sharpe={row['sharpe_ratio']:.2f}  WR={row['win_rate']:.0f}%  "
              f"DD={row['max_drawdown_pct']:.1f}%  Trades={row['num_trades']}")

    # ========================================================================
    # Actifs ou le macro filter a le PLUS d'impact
    # ========================================================================
    print("\n" + "-" * 80)
    print("TOP 10 ACTIFS OU LE MACRO FILTER A LE PLUS D'IMPACT")
    print("-" * 80)

    # Comparer rendement moyen par actif
    asset_no = no_macro.groupby('asset')['total_return_pct'].mean()
    asset_with = with_macro.groupby('asset')['total_return_pct'].mean()
    asset_delta = (asset_with - asset_no).sort_values(ascending=False)

    print("\n  Plus gros gain avec macro:")
    for symbol, delta in asset_delta.head(5).items():
        print(f"    {symbol:15} : {delta:+.2f}% de rendement additionnel")

    print("\n  Plus grosse perte avec macro:")
    for symbol, delta in asset_delta.tail(5).items():
        print(f"    {symbol:15} : {delta:+.2f}% de rendement")

    print(f"\nResultats sauvegardes dans: {OUTPUT_FILE}")
    print("=" * 80)

    # Sauvegarder aussi en tant que cache
    results_df.to_csv('RESULTS_2024_2025_COMPLETE.csv', index=False)
    print(f"Cache mis a jour: RESULTS_2024_2025_COMPLETE.csv")

    return results_df


if __name__ == '__main__':
    run_all_backtests()
