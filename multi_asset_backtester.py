"""
Système de backtesting multi-asset
Teste plusieurs stratégies sur plusieurs actifs et trouve les meilleurs setups
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from backtesting_engine import BacktestEngine
from optimizer import GridSearchOptimizer
from strategies import (MovingAverageCrossover, RSIStrategy, MACDStrategy,
                       BollingerBandsStrategy, CombinedStrategy)
from data_fetcher import get_data


# ============================================================================
# DÉFINITION DES ACTIFS PAR CATÉGORIE
# ============================================================================

ASSET_UNIVERSE = {
    'crypto': [
        {'symbol': 'BTC/USDT', 'name': 'Bitcoin'},
        {'symbol': 'ETH/USDT', 'name': 'Ethereum'},
        {'symbol': 'BNB/USDT', 'name': 'Binance Coin'},
        {'symbol': 'SOL/USDT', 'name': 'Solana'},
        {'symbol': 'XRP/USDT', 'name': 'Ripple'},
    ],

    'indices': [
        {'symbol': '^GSPC', 'name': 'S&P 500'},
        {'symbol': '^IXIC', 'name': 'Nasdaq'},
        {'symbol': '^DJI', 'name': 'Dow Jones'},
        {'symbol': '^GDAXI', 'name': 'DAX'},
        {'symbol': '^FCHI', 'name': 'CAC 40'},
    ],

    'forex': [
        {'symbol': 'EURUSD=X', 'name': 'EUR/USD'},
        {'symbol': 'GBPUSD=X', 'name': 'GBP/USD'},
        {'symbol': 'USDJPY=X', 'name': 'USD/JPY'},
        {'symbol': 'USDCHF=X', 'name': 'USD/CHF'},
        {'symbol': 'AUDUSD=X', 'name': 'AUD/USD'},
    ],

    'commodities': [
        {'symbol': 'GC=F', 'name': 'Gold'},
        {'symbol': 'SI=F', 'name': 'Silver'},
        {'symbol': 'CL=F', 'name': 'Crude Oil'},
        {'symbol': 'NG=F', 'name': 'Natural Gas'},
    ],

    'stocks': [
        {'symbol': 'AAPL', 'name': 'Apple'},
        {'symbol': 'MSFT', 'name': 'Microsoft'},
        {'symbol': 'GOOGL', 'name': 'Google'},
        {'symbol': 'AMZN', 'name': 'Amazon'},
        {'symbol': 'TSLA', 'name': 'Tesla'},
        {'symbol': 'NVDA', 'name': 'Nvidia'},
    ]
}


# ============================================================================
# CONFIGURATIONS DES STRATÉGIES
# ============================================================================

STRATEGY_CONFIGS = {
    'MA_Crossover_20_50': {
        'class': MovingAverageCrossover,
        'params': {'fast_period': 20, 'slow_period': 50}
    },
    'MA_Crossover_10_30': {
        'class': MovingAverageCrossover,
        'params': {'fast_period': 10, 'slow_period': 30}
    },
    'RSI_14_30_70': {
        'class': RSIStrategy,
        'params': {'period': 14, 'oversold': 30, 'overbought': 70}
    },
    'RSI_14_35_80': {
        'class': RSIStrategy,
        'params': {'period': 14, 'oversold': 35, 'overbought': 80}
    },
    'MACD_Standard': {
        'class': MACDStrategy,
        'params': {'fast': 12, 'slow': 26, 'signal': 9}
    },
    'Bollinger_20_2': {
        'class': BollingerBandsStrategy,
        'params': {'period': 20, 'num_std': 2}
    },
    'Combined': {
        'class': CombinedStrategy,
        'params': {}
    }
}


class MultiAssetBacktester:
    """
    Classe pour tester plusieurs stratégies sur plusieurs actifs
    """

    def __init__(self,
                 start_date: str = '2024-01-01',
                 end_date: str = '2025-12-31',
                 engine_config: Optional[Dict] = None):
        """
        Args:
            start_date: Date de début du backtest
            end_date: Date de fin du backtest
            engine_config: Configuration du BacktestEngine
        """
        self.start_date = start_date
        self.end_date = end_date
        self.engine_config = engine_config or {
            'initial_capital': 10000,
            'commission': 0.001,
            'slippage': 0.0005,
            'stop_loss_pct': 3.0,
            'take_profit_pct': 6.0,
            'position_size_pct': 100.0
        }

        self.data_cache = {}
        self.results = []

    def fetch_asset_data(self,
                        asset_type: str,
                        symbol: str,
                        use_sample: bool = False) -> Optional[pd.DataFrame]:
        """
        Récupère les données pour un actif

        Args:
            asset_type: Type d'actif ('crypto', 'stock', 'forex', etc.)
            symbol: Symbole de l'actif
            use_sample: Si True, utilise des données synthétiques

        Returns:
            DataFrame OHLCV ou None si erreur
        """
        cache_key = f"{asset_type}_{symbol}"

        if cache_key in self.data_cache:
            return self.data_cache[cache_key]

        try:
            if use_sample:
                # Données synthétiques
                data = get_data('sample', symbol, self.start_date, self.end_date,
                              trend=0.0003, volatility=0.02)
            else:
                # Données réelles
                if asset_type == 'crypto':
                    data = get_data('crypto', symbol, self.start_date, self.end_date,
                                  exchange='binance', timeframe='1d')
                elif asset_type in ['indices', 'forex', 'commodities', 'stocks']:
                    data = get_data('stock', symbol, self.start_date, self.end_date,
                                  interval='1d')
                else:
                    print(f"⚠️  Type d'actif non supporté: {asset_type}")
                    return None

            if data is None or len(data) == 0:
                print(f"⚠️  Pas de données pour {symbol}")
                return None

            # Cache
            self.data_cache[cache_key] = data
            return data

        except Exception as e:
            print(f"❌ Erreur lors de la récupération de {symbol}: {e}")
            return None

    def test_strategy_on_asset(self,
                               asset_type: str,
                               asset_symbol: str,
                               asset_name: str,
                               strategy_name: str,
                               strategy_config: Dict,
                               use_sample: bool = False) -> Optional[Dict]:
        """
        Teste une stratégie sur un actif

        Args:
            asset_type: Type d'actif
            asset_symbol: Symbole
            asset_name: Nom de l'actif
            strategy_name: Nom de la stratégie
            strategy_config: Config de la stratégie
            use_sample: Utiliser des données synthétiques

        Returns:
            Dictionnaire de résultats ou None
        """
        # Récupérer les données
        data = self.fetch_asset_data(asset_type, asset_symbol, use_sample)

        if data is None:
            return None

        try:
            # Créer la stratégie
            strategy_class = strategy_config['class']
            params = strategy_config['params']
            strategy = strategy_class(**params)

            # Créer le moteur
            engine = BacktestEngine(**self.engine_config)

            # Exécuter le backtest
            result = engine.run_backtest(data, strategy)

            # Compiler les résultats
            return {
                'asset_type': asset_type,
                'asset_symbol': asset_symbol,
                'asset_name': asset_name,
                'strategy_name': strategy_name,
                'strategy_params': str(params),
                'total_return_pct': result.get('total_return_pct', 0.0),
                'final_capital': result.get('final_capital', 0.0),
                'sharpe_ratio': result.get('sharpe_ratio', 0.0),
                'max_drawdown': result.get('max_drawdown', 0.0),
                'win_rate': result.get('win_rate', 0.0),
                'profit_factor': result.get('profit_factor', 0.0),
                'total_trades': result.get('total_trades', 0),
                'winning_trades': result.get('winning_trades', 0),
                'losing_trades': result.get('losing_trades', 0),
            }

        except Exception as e:
            print(f"❌ Erreur backtest {strategy_name} sur {asset_name}: {e}")
            return None

    def run_comprehensive_backtest(self,
                                   asset_categories: Optional[List[str]] = None,
                                   strategy_names: Optional[List[str]] = None,
                                   use_sample: bool = False) -> pd.DataFrame:
        """
        Lance un backtest complet sur tous les actifs et stratégies

        Args:
            asset_categories: Liste des catégories d'actifs à tester (None = toutes)
            strategy_names: Liste des stratégies à tester (None = toutes)
            use_sample: Utiliser des données synthétiques

        Returns:
            DataFrame avec tous les résultats
        """
        print("\n" + "="*80)
        print("🚀 BACKTEST MULTI-ASSET COMPLET")
        print("="*80)
        print(f"Période: {self.start_date} → {self.end_date}")
        print(f"Capital initial: ${self.engine_config['initial_capital']:,.0f}")
        print(f"Stop-Loss: {self.engine_config.get('stop_loss_pct', 'N/A')}%")
        print(f"Take-Profit: {self.engine_config.get('take_profit_pct', 'N/A')}%")

        # Sélectionner les catégories d'actifs
        if asset_categories is None:
            asset_categories = list(ASSET_UNIVERSE.keys())

        # Sélectionner les stratégies
        if strategy_names is None:
            strategy_names = list(STRATEGY_CONFIGS.keys())

        # Compter le nombre total de tests
        total_tests = sum(len(ASSET_UNIVERSE[cat]) for cat in asset_categories) * len(strategy_names)
        print(f"\n📊 Tests à exécuter: {total_tests}")
        print("-"*80)

        results = []
        test_count = 0

        for category in asset_categories:
            assets = ASSET_UNIVERSE[category]

            print(f"\n📈 Catégorie: {category.upper()}")

            for asset in assets:
                symbol = asset['symbol']
                name = asset['name']

                print(f"\n  💰 {name} ({symbol})")

                for strategy_name in strategy_names:
                    test_count += 1
                    strategy_config = STRATEGY_CONFIGS[strategy_name]

                    print(f"    🔄 {strategy_name}... ", end='', flush=True)

                    result = self.test_strategy_on_asset(
                        category,
                        symbol,
                        name,
                        strategy_name,
                        strategy_config,
                        use_sample
                    )

                    if result:
                        results.append(result)
                        ret = result['total_return_pct']
                        print(f"✓ Rendement: {ret:+.2f}%")
                    else:
                        print("✗ Erreur")

                    # Progress
                    if test_count % 10 == 0:
                        progress = test_count / total_tests * 100
                        print(f"\n  📊 Progrès global: {progress:.0f}%")

        # Créer le DataFrame
        results_df = pd.DataFrame(results)

        print("\n" + "="*80)
        print(f"✅ Backtest terminé: {len(results)} résultats sur {total_tests} tests")
        print("="*80)

        return results_df

    def get_best_setups(self, results_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """
        Extrait les meilleurs setups (Actif + Stratégie)

        Args:
            results_df: DataFrame de résultats
            top_n: Nombre de meilleurs résultats

        Returns:
            DataFrame trié
        """
        return results_df.nlargest(top_n, 'total_return_pct')

    def get_best_by_asset_type(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Trouve la meilleure stratégie par type d'actif

        Args:
            results_df: DataFrame de résultats

        Returns:
            DataFrame avec meilleure stratégie par catégorie
        """
        return results_df.loc[results_df.groupby('asset_type')['total_return_pct'].idxmax()]

    def get_best_by_strategy(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Performance moyenne de chaque stratégie sur tous les actifs

        Args:
            results_df: DataFrame de résultats

        Returns:
            DataFrame agrégé par stratégie
        """
        return results_df.groupby('strategy_name').agg({
            'total_return_pct': ['mean', 'std', 'max', 'min'],
            'sharpe_ratio': 'mean',
            'win_rate': 'mean',
            'total_trades': 'sum'
        }).round(2)


def print_summary_report(results_df: pd.DataFrame):
    """
    Affiche un rapport détaillé des résultats

    Args:
        results_df: DataFrame de résultats
    """
    if len(results_df) == 0:
        print("Aucun résultat à afficher")
        return

    print("\n" + "="*80)
    print("📊 RAPPORT DE RÉSULTATS")
    print("="*80)

    # Top 10 meilleurs setups
    print("\n🏆 TOP 10 MEILLEURS SETUPS (Actif + Stratégie)")
    print("-"*80)
    top_10 = results_df.nlargest(10, 'total_return_pct')
    display_cols = ['asset_name', 'strategy_name', 'total_return_pct',
                   'sharpe_ratio', 'max_drawdown', 'win_rate', 'total_trades']
    print(top_10[display_cols].to_string(index=False))

    # Meilleure stratégie par type d'actif
    print("\n" + "="*80)
    print("🎯 MEILLEURE STRATÉGIE PAR TYPE D'ACTIF")
    print("-"*80)
    best_by_type = results_df.loc[results_df.groupby('asset_type')['total_return_pct'].idxmax()]
    print(best_by_type[display_cols].to_string(index=False))

    # Performance moyenne par stratégie
    print("\n" + "="*80)
    print("📈 PERFORMANCE MOYENNE PAR STRATÉGIE")
    print("-"*80)
    strategy_perf = results_df.groupby('strategy_name').agg({
        'total_return_pct': ['mean', 'std'],
        'sharpe_ratio': 'mean',
        'win_rate': 'mean'
    }).round(2)
    strategy_perf.columns = ['Return_Mean', 'Return_Std', 'Sharpe', 'WinRate']
    strategy_perf = strategy_perf.sort_values('Return_Mean', ascending=False)
    print(strategy_perf.to_string())

    # Statistiques globales
    print("\n" + "="*80)
    print("📊 STATISTIQUES GLOBALES")
    print("-"*80)
    print(f"Nombre total de tests: {len(results_df)}")
    print(f"Tests profitables: {len(results_df[results_df['total_return_pct'] > 0])} ({len(results_df[results_df['total_return_pct'] > 0])/len(results_df)*100:.1f}%)")
    print(f"Rendement moyen: {results_df['total_return_pct'].mean():.2f}%")
    print(f"Rendement médian: {results_df['total_return_pct'].median():.2f}%")
    print(f"Meilleur rendement: {results_df['total_return_pct'].max():.2f}%")
    print(f"Pire rendement: {results_df['total_return_pct'].min():.2f}%")
    print(f"Sharpe ratio moyen: {results_df['sharpe_ratio'].mean():.2f}")
