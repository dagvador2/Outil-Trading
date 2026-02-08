"""
Module d'optimisation des paramètres de stratégies
- Grid Search: teste toutes les combinaisons de paramètres
- Walk-Forward Analysis: évite l'overfitting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from itertools import product
from datetime import datetime, timedelta
from backtesting_engine import BacktestEngine
import warnings
warnings.filterwarnings('ignore')


class GridSearchOptimizer:
    """
    Optimise les paramètres d'une stratégie via Grid Search
    """

    def __init__(self, engine_config: Optional[Dict] = None):
        """
        Args:
            engine_config: Configuration du BacktestEngine
                           (initial_capital, commission, slippage, stop_loss_pct, etc.)
        """
        self.engine_config = engine_config or {
            'initial_capital': 10000,
            'commission': 0.001,
            'slippage': 0.0005
        }

    def optimize(self,
                data: pd.DataFrame,
                strategy_class,
                param_grid: Dict[str, List],
                metric: str = 'total_return_pct',
                top_n: int = 10) -> pd.DataFrame:
        """
        Effectue une recherche exhaustive des meilleurs paramètres

        Args:
            data: DataFrame OHLCV
            strategy_class: Classe de stratégie à optimiser
            param_grid: Dictionnaire des paramètres à tester
                       Ex: {'fast_period': [10, 20, 30], 'slow_period': [50, 100]}
            metric: Métrique à optimiser ('total_return_pct', 'sharpe_ratio', 'profit_factor', etc.)
            top_n: Nombre de meilleurs résultats à retourner

        Returns:
            DataFrame avec les résultats triés par performance
        """
        print(f"🔍 Optimisation de {strategy_class.__name__}...")
        print(f"   Paramètres: {param_grid}")

        # Générer toutes les combinaisons
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))

        print(f"   Total de combinaisons: {len(combinations)}")

        results = []

        for i, combo in enumerate(combinations):
            # Créer le dictionnaire de paramètres
            params = dict(zip(param_names, combo))

            try:
                # Créer la stratégie avec ces paramètres
                strategy = strategy_class(**params)

                # Créer le moteur
                engine = BacktestEngine(**self.engine_config)

                # Exécuter le backtest
                result = engine.run_backtest(data, strategy)

                # Stocker les résultats
                results.append({
                    **params,
                    'total_return_pct': result['total_return_pct'],
                    'sharpe_ratio': result['sharpe_ratio'],
                    'max_drawdown': result['max_drawdown'],
                    'win_rate': result['win_rate'],
                    'profit_factor': result['profit_factor'],
                    'total_trades': result['total_trades'],
                    'final_capital': result['final_capital']
                })

            except Exception as e:
                # Si erreur, continuer
                print(f"   ⚠️  Erreur avec params {params}: {e}")
                continue

            # Progress
            if (i + 1) % max(1, len(combinations) // 10) == 0:
                progress = (i + 1) / len(combinations) * 100
                print(f"   Progrès: {progress:.0f}%")

        # Créer le DataFrame des résultats
        results_df = pd.DataFrame(results)

        if len(results_df) == 0:
            print("❌ Aucun résultat valide")
            return pd.DataFrame()

        # Trier par métrique
        results_df = results_df.sort_values(by=metric, ascending=False)

        print(f"✅ Optimisation terminée!")
        print(f"   Meilleur résultat: {results_df.iloc[0][metric]:.2f} {metric}")

        return results_df.head(top_n)


class WalkForwardOptimizer:
    """
    Walk-Forward Analysis pour éviter l'overfitting

    Principe:
    1. Diviser les données en périodes (in-sample / out-sample)
    2. Optimiser sur in-sample
    3. Tester sur out-sample
    4. Répéter sur chaque période
    """

    def __init__(self, engine_config: Optional[Dict] = None):
        """
        Args:
            engine_config: Configuration du BacktestEngine
        """
        self.engine_config = engine_config or {
            'initial_capital': 10000,
            'commission': 0.001,
            'slippage': 0.0005
        }

    def walk_forward_analysis(self,
                             data: pd.DataFrame,
                             strategy_class,
                             param_grid: Dict[str, List],
                             train_period_days: int = 180,
                             test_period_days: int = 60,
                             metric: str = 'sharpe_ratio') -> Dict:
        """
        Effectue une Walk-Forward Analysis

        Args:
            data: DataFrame OHLCV complet
            strategy_class: Classe de stratégie
            param_grid: Grille de paramètres
            train_period_days: Période d'entraînement (jours)
            test_period_days: Période de test (jours)
            metric: Métrique d'optimisation

        Returns:
            Dictionnaire avec résultats détaillés
        """
        print("\n" + "="*80)
        print("🔬 WALK-FORWARD ANALYSIS")
        print("="*80)
        print(f"Période d'entraînement: {train_period_days} jours")
        print(f"Période de test: {test_period_days} jours")
        print(f"Métrique d'optimisation: {metric}")

        # Diviser en périodes
        total_days = len(data)
        period_length = train_period_days + test_period_days

        if total_days < period_length:
            raise ValueError(f"Pas assez de données. Besoin d'au moins {period_length} jours, obtenu {total_days}")

        # Calcul du nombre de périodes
        n_periods = (total_days - train_period_days) // test_period_days

        if n_periods < 1:
            raise ValueError("Pas assez de données pour créer des périodes de test")

        print(f"Nombre de périodes: {n_periods}")

        all_results = []
        out_sample_results = []

        optimizer = GridSearchOptimizer(self.engine_config)

        for period in range(n_periods):
            print(f"\n📊 Période {period + 1}/{n_periods}")
            print("-" * 80)

            # Définir les indices
            train_start = period * test_period_days
            train_end = train_start + train_period_days
            test_start = train_end
            test_end = test_start + test_period_days

            # S'assurer de ne pas dépasser
            if test_end > total_days:
                test_end = total_days

            # Extraire les données
            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[test_start:test_end]

            if len(test_data) == 0:
                print("⚠️  Pas de données de test, fin de l'analyse")
                break

            print(f"  Train: {train_data.index[0]} → {train_data.index[-1]} ({len(train_data)} jours)")
            print(f"  Test:  {test_data.index[0]} → {test_data.index[-1]} ({len(test_data)} jours)")

            # Optimiser sur train
            print(f"  🔍 Optimisation sur période d'entraînement...")
            train_results = optimizer.optimize(
                train_data,
                strategy_class,
                param_grid,
                metric=metric,
                top_n=1
            )

            if len(train_results) == 0:
                print("  ⚠️  Pas de résultats valides, passage à la période suivante")
                continue

            # Meilleurs paramètres
            best_params = train_results.iloc[0].to_dict()
            strategy_params = {}
            for k, v in best_params.items():
                if k in param_grid.keys():
                    # Convertir en int si nécessaire (pour les périodes)
                    if isinstance(param_grid[k][0], int):
                        strategy_params[k] = int(v)
                    else:
                        strategy_params[k] = v

            print(f"  ✓ Meilleurs params: {strategy_params}")
            print(f"    Performance (train): {best_params[metric]:.2f}")

            # Tester sur out-sample
            print(f"  🧪 Test sur période hors-échantillon...")
            strategy = strategy_class(**strategy_params)
            engine = BacktestEngine(**self.engine_config)
            test_result = engine.run_backtest(test_data, strategy)

            test_metric = test_result.get(metric, 0.0)
            print(f"    Performance (test): {test_metric:.2f}")
            print(f"    Rendement (test): {test_result.get('total_return_pct', 0.0):.2f}%")

            # Stocker les résultats
            period_result = {
                'period': period + 1,
                'train_start': str(train_data.index[0]),
                'train_end': str(train_data.index[-1]),
                'test_start': str(test_data.index[0]),
                'test_end': str(test_data.index[-1]),
                **{f'param_{k}': v for k, v in strategy_params.items()},
                'train_metric': best_params.get(metric, 0.0),
                'test_return': test_result.get('total_return_pct', 0.0),
                'test_sharpe': test_result.get('sharpe_ratio', 0.0),
                'test_max_dd': test_result.get('max_drawdown', 0.0),
                'test_trades': test_result.get('total_trades', 0),
                'test_win_rate': test_result.get('win_rate', 0.0)
            }

            all_results.append(period_result)
            out_sample_results.append(test_result.get('total_return_pct', 0.0))

        # Résumé
        results_df = pd.DataFrame(all_results)

        if len(results_df) == 0:
            print("\n❌ Aucun résultat valide")
            return {'results_df': pd.DataFrame(), 'summary': {}}

        print("\n" + "="*80)
        print("📈 RÉSULTATS WALK-FORWARD")
        print("="*80)

        avg_out_sample_return = np.mean(out_sample_results)
        std_out_sample_return = np.std(out_sample_results)
        total_periods = len(out_sample_results)
        profitable_periods = sum([1 for r in out_sample_results if r > 0])

        print(f"\nNombre de périodes testées: {total_periods}")
        print(f"Périodes profitables: {profitable_periods}/{total_periods} ({profitable_periods/total_periods*100:.1f}%)")
        print(f"Rendement moyen (out-sample): {avg_out_sample_return:.2f}%")
        print(f"Écart-type des rendements: {std_out_sample_return:.2f}%")

        summary = {
            'total_periods': total_periods,
            'profitable_periods': profitable_periods,
            'profitability_rate': profitable_periods / total_periods * 100,
            'avg_out_sample_return': avg_out_sample_return,
            'std_out_sample_return': std_out_sample_return,
            'best_period_return': max(out_sample_results) if out_sample_results else 0,
            'worst_period_return': min(out_sample_results) if out_sample_results else 0
        }

        return {
            'results_df': results_df,
            'summary': summary
        }


def compare_strategies_on_asset(data: pd.DataFrame,
                                strategies_configs: List[Dict],
                                engine_config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Compare plusieurs stratégies (avec paramètres optimisés) sur un actif

    Args:
        data: DataFrame OHLCV
        strategies_configs: Liste de configs
            Ex: [
                {'class': MovingAverageCrossover, 'params': {'fast_period': 20, 'slow_period': 50}},
                {'class': RSIStrategy, 'params': {'period': 14}}
            ]
        engine_config: Config du moteur

    Returns:
        DataFrame comparatif
    """
    engine_config = engine_config or {
        'initial_capital': 10000,
        'commission': 0.001,
        'slippage': 0.0005
    }

    results = []

    for config in strategies_configs:
        strategy_class = config['class']
        params = config.get('params', {})

        strategy = strategy_class(**params)
        engine = BacktestEngine(**engine_config)
        result = engine.run_backtest(data, strategy)

        results.append({
            'strategy': strategy.name,
            'params': str(params),
            'total_return_pct': result['total_return_pct'],
            'sharpe_ratio': result['sharpe_ratio'],
            'max_drawdown': result['max_drawdown'],
            'win_rate': result['win_rate'],
            'profit_factor': result['profit_factor'],
            'total_trades': result['total_trades']
        })

    return pd.DataFrame(results).sort_values('total_return_pct', ascending=False)
