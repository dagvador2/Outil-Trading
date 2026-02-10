"""
Bibliothèque des résultats de backtesting 2024-2025
Calcule de vrais backtests via Yahoo Finance + BacktestEngine
pour garantir la cohérence avec l'explorateur détaillé
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional
import yfinance as yf

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
# Correspondance nom de stratégie → instance
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


def instantiate_strategy(strategy_name: str):
    """
    Instancie une stratégie à partir de son nom.
    Utilisé par la bibliothèque ET l'explorateur détaillé pour garantir la cohérence.

    Args:
        strategy_name: Nom de la stratégie (ex: 'MA_Crossover_20_50')

    Returns:
        Instance de la stratégie, ou None si non trouvée
    """
    # Recherche exacte
    if strategy_name in STRATEGY_MAP:
        return STRATEGY_MAP[strategy_name]()

    # Recherche partielle (pour compatibilité avec noms affichés)
    name_lower = strategy_name.lower().replace(' ', '_')
    for key, factory in STRATEGY_MAP.items():
        if key.lower() in name_lower or name_lower in key.lower():
            return factory()

    # Fallback par mots-clés
    if 'ma' in name_lower and ('crossover' in name_lower or '20' in name_lower and '50' in name_lower):
        return MovingAverageCrossover(20, 50)
    if 'ma' in name_lower and '10' in name_lower and '30' in name_lower:
        return MovingAverageCrossover(10, 30)
    if 'rsi' in name_lower and '35' in name_lower:
        return RSIStrategy(14, 35, 80)
    if 'rsi' in name_lower:
        return RSIStrategy(14, 30, 70)
    if 'bollinger' in name_lower:
        return BollingerBandsStrategy(20, 2)
    if 'macd' in name_lower:
        return MACDStrategy(12, 26, 9)
    if 'adx' in name_lower:
        return ADXTrendStrategy(14, 25)
    if 'vwap' in name_lower:
        return VWAPStrategy(20)
    if 'ichimoku' in name_lower:
        return IchimokuStrategy(9, 26, 52)
    if 'combined' in name_lower:
        return CombinedStrategy()

    return None


class BacktestLibrary:
    """
    Gestionnaire de la bibliothèque de backtests historiques.
    Calcule de vrais backtests via Yahoo Finance + BacktestEngine.
    """

    CACHE_FILE = 'RESULTS_2024_2025_COMPLETE.csv'

    def __init__(self):
        """Charge les résultats depuis le cache ou lance le calcul"""
        self.results_df = None
        self.loaded = False
        self._load_cached_results()

    def _load_cached_results(self):
        """Charge les résultats depuis le CSV cache s'il existe et est valide"""
        if os.path.exists(self.CACHE_FILE):
            try:
                df = pd.read_csv(self.CACHE_FILE)
                required_cols = ['asset', 'strategy', 'total_return_pct']
                if all(col in df.columns for col in required_cols):
                    # Vérifier que c'est un vrai résultat (pas des données fictives)
                    # Les vrais résultats ont la colonne 'data_source'
                    if 'data_source' in df.columns:
                        self.results_df = df
                        self.loaded = True
                        return
            except Exception:
                pass

        # Pas de cache valide → DataFrame vide
        self.results_df = pd.DataFrame(columns=[
            'asset', 'asset_type', 'asset_name', 'strategy',
            'total_return_pct', 'sharpe_ratio', 'max_drawdown_pct',
            'win_rate', 'num_trades', 'avg_trade_return_pct', 'data_source'
        ])
        self.loaded = False

    def compute_all_backtests(self, start_date: str = '2024-01-01',
                               end_date: str = '2025-12-31',
                               progress_callback=None) -> pd.DataFrame:
        """
        Calcule TOUS les backtests réels via Yahoo Finance.

        Args:
            start_date: Date de début
            end_date: Date de fin
            progress_callback: Fonction callback(current, total, message) pour le progrès

        Returns:
            DataFrame avec tous les résultats
        """
        results = []
        all_assets = []

        for category, assets in MONITORED_ASSETS.items():
            for asset in assets:
                all_assets.append((category, asset))

        strategy_names = list(STRATEGY_MAP.keys())
        total = len(all_assets) * len(strategy_names)
        current = 0

        for category, asset in all_assets:
            symbol = asset['symbol']
            yahoo_symbol = convert_to_yahoo_symbol(symbol)

            # Télécharger données une seule fois par actif
            try:
                ticker = yf.Ticker(yahoo_symbol)
                data = ticker.history(start=start_date, end=end_date)

                if len(data) < 50:
                    # Pas assez de données pour un backtest fiable
                    for strategy_name in strategy_names:
                        current += 1
                        if progress_callback:
                            progress_callback(current, total, f"Pas assez de données pour {symbol}")
                    continue

                data.columns = [col.lower() for col in data.columns]

            except Exception as e:
                for strategy_name in strategy_names:
                    current += 1
                    if progress_callback:
                        progress_callback(current, total, f"Erreur données {symbol}: {e}")
                continue

            # Tester chaque stratégie sur cet actif
            for strategy_name in strategy_names:
                current += 1
                if progress_callback:
                    progress_callback(current, total, f"{symbol} × {strategy_name}")

                try:
                    strategy = instantiate_strategy(strategy_name)
                    if strategy is None:
                        continue

                    engine = BacktestEngine(
                        initial_capital=10000,
                        commission=0.001,
                        slippage=0.0005
                    )

                    metrics = engine.run_backtest(data, strategy)

                    num_trades = metrics.get('total_trades', 0)
                    total_return = metrics.get('total_return_pct', 0)

                    # Calculer rendement moyen par trade
                    avg_trade_return = 0
                    if num_trades > 0:
                        trades_df = metrics.get('trades_df', None)
                        if trades_df is not None and len(trades_df) > 0:
                            avg_trade_return = trades_df['pnl_pct'].mean()

                    results.append({
                        'asset': symbol,
                        'asset_type': category,
                        'asset_name': asset['name'],
                        'strategy': strategy_name,
                        'total_return_pct': round(total_return, 4),
                        'sharpe_ratio': round(metrics.get('sharpe_ratio', 0), 4),
                        'max_drawdown_pct': round(abs(metrics.get('max_drawdown', 0)), 4),
                        'win_rate': round(metrics.get('win_rate', 0), 2),
                        'num_trades': num_trades,
                        'avg_trade_return_pct': round(avg_trade_return, 4),
                        'data_source': 'yahoo_finance',
                        'num_candles': len(data),
                        'period': f"{start_date} - {end_date}"
                    })

                except Exception as e:
                    # Enregistrer l'erreur mais continuer
                    results.append({
                        'asset': symbol,
                        'asset_type': category,
                        'asset_name': asset['name'],
                        'strategy': strategy_name,
                        'total_return_pct': 0,
                        'sharpe_ratio': 0,
                        'max_drawdown_pct': 0,
                        'win_rate': 0,
                        'num_trades': 0,
                        'avg_trade_return_pct': 0,
                        'data_source': f'error: {str(e)[:50]}',
                        'num_candles': 0,
                        'period': f"{start_date} - {end_date}"
                    })

        # Créer DataFrame et sauvegarder
        self.results_df = pd.DataFrame(results)
        self.loaded = True

        # Sauvegarder en cache
        self.results_df.to_csv(self.CACHE_FILE, index=False)

        return self.results_df

    def get_all_results(self) -> pd.DataFrame:
        """Retourne tous les résultats"""
        return self.results_df.copy() if self.results_df is not None else pd.DataFrame()

    def get_top_strategies(self, n: int = 10, metric: str = 'total_return_pct') -> pd.DataFrame:
        """Retourne les N meilleures stratégies"""
        if self.results_df is None or len(self.results_df) == 0:
            return pd.DataFrame()
        return self.results_df.nlargest(n, metric)

    def get_results_by_asset(self, asset: str) -> pd.DataFrame:
        """Résultats pour un actif spécifique"""
        if self.results_df is None:
            return pd.DataFrame()
        return self.results_df[self.results_df['asset'] == asset].copy()

    def get_results_by_strategy(self, strategy: str) -> pd.DataFrame:
        """Résultats pour une stratégie spécifique"""
        if self.results_df is None:
            return pd.DataFrame()
        return self.results_df[self.results_df['strategy'] == strategy].copy()

    def get_best_strategy_for_asset(self, asset: str, metric: str = 'total_return_pct') -> Optional[Dict]:
        """Trouve la meilleure stratégie pour un actif"""
        asset_results = self.get_results_by_asset(asset)
        if len(asset_results) == 0:
            return None
        best = asset_results.nlargest(1, metric).iloc[0]
        return {
            'asset': best['asset'],
            'strategy': best['strategy'],
            'return': best.get('total_return_pct', 0),
            'sharpe': best.get('sharpe_ratio', 0),
            'drawdown': best.get('max_drawdown_pct', 0),
            'win_rate': best.get('win_rate', 0),
            'trades': best.get('num_trades', 0)
        }

    def search_strategies(self, min_return: float = 0, min_sharpe: float = 0,
                          min_win_rate: float = 0) -> pd.DataFrame:
        """Recherche de stratégies selon critères"""
        if self.results_df is None or len(self.results_df) == 0:
            return pd.DataFrame()

        df = self.results_df
        filtered = df[
            (df['total_return_pct'] >= min_return) &
            (df['sharpe_ratio'] >= min_sharpe) &
            (df['win_rate'] >= min_win_rate)
        ].copy()

        return filtered.sort_values('total_return_pct', ascending=False)

    def get_heatmap_data(self) -> pd.DataFrame:
        """Prépare données pour heatmap (actifs x stratégies)"""
        if self.results_df is None or len(self.results_df) == 0:
            return pd.DataFrame()

        return self.results_df.pivot_table(
            index='asset',
            columns='strategy',
            values='total_return_pct',
            aggfunc='mean'
        )

    def get_statistics(self) -> Dict:
        """Statistiques globales de la bibliothèque"""
        if self.results_df is None or len(self.results_df) == 0:
            return {
                'total_combinations': 0,
                'num_assets': 0,
                'num_strategies': 0,
                'avg_return': 0,
                'best_return': 0,
                'worst_return': 0,
                'positive_returns': 0,
                'positive_rate': 0
            }

        num_positive = len(self.results_df[self.results_df['total_return_pct'] > 0])
        return {
            'total_combinations': len(self.results_df),
            'num_assets': self.results_df['asset'].nunique(),
            'num_strategies': self.results_df['strategy'].nunique(),
            'avg_return': self.results_df['total_return_pct'].mean(),
            'best_return': self.results_df['total_return_pct'].max(),
            'worst_return': self.results_df['total_return_pct'].min(),
            'positive_returns': num_positive,
            'positive_rate': (num_positive / len(self.results_df)) * 100
        }


# ============================================================================
# Script de test / génération
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("Calcul des backtests reels via Yahoo Finance")
    print("=" * 80)

    library = BacktestLibrary()

    def progress(current, total, msg):
        pct = current / total * 100
        print(f"  [{current}/{total}] ({pct:.0f}%) {msg}")

    library.compute_all_backtests(progress_callback=progress)

    if library.loaded:
        stats = library.get_statistics()
        print(f"\nTotal combinaisons  : {stats['total_combinations']}")
        print(f"Actifs testes       : {stats['num_assets']}")
        print(f"Strategies testees  : {stats['num_strategies']}")
        print(f"Rendement moyen     : {stats['avg_return']:.2f}%")
        print(f"Meilleur rendement  : {stats['best_return']:.2f}%")
        print(f"Pire rendement      : {stats['worst_return']:.2f}%")
        print(f"Rendements positifs : {stats['positive_returns']} ({stats['positive_rate']:.1f}%)")

        print(f"\nSauvegarde dans: {BacktestLibrary.CACHE_FILE}")
