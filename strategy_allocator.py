"""
Strategy Allocator - Allocation intelligente des strategies de trading

A partir des resultats de backtesting, ce module :
1. Filtre les combinaisons actif/strategie non rentables
2. Score chaque paire avec un composite multi-criteres
3. Selectionne la meilleure strategie par actif
4. Alloue le capital de facon optimale (equal, score-weighted, risk-parity)
5. Valide la robustesse via walk-forward (optionnel)
6. Genere des parametres de risque dynamiques (SL/TP)
7. Exporte un TradingPlan compatible avec signal_generator.py

Usage:
    python strategy_allocator.py --capital 50000 --max-positions 8
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any
import json
import argparse

from backtesting_engine import BacktestEngine
from backtest_library import BacktestLibrary, STRATEGY_MAP, instantiate_strategy
from assets_config import MONITORED_ASSETS, get_category_from_symbol
from yahoo_data_feed import convert_to_yahoo_symbol


# ============================================================================
# Dataclasses
# ============================================================================

@dataclass
class AssetStrategyScore:
    """Score d'une paire actif/strategie avec metriques et allocation."""
    asset: str
    asset_type: str
    asset_name: str
    strategy_name: str
    score: float
    sharpe_ratio: float
    total_return_pct: float
    max_drawdown_pct: float
    win_rate: float
    num_trades: int
    avg_trade_return_pct: float
    stability_score: Optional[float] = None
    allocation_pct: float = 0.0


@dataclass
class RiskParams:
    """Parametres de risque dynamiques pour un actif."""
    stop_loss_pct: float
    take_profit_pct: float
    position_size_pct: float


@dataclass
class TradingPlan:
    """Plan de trading complet genere par l'allocator."""
    assignments: List[AssetStrategyScore]
    risk_params: Dict[str, RiskParams]
    config: Dict[str, Any]
    generated_at: datetime
    summary_stats: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Ranges de risque par categorie d'actif
# ============================================================================

RISK_RANGES = {
    'crypto':         {'sl': (5, 12), 'tp': (10, 25)},
    'tech_stocks':    {'sl': (2, 6),  'tp': (4, 12)},
    'semiconductors': {'sl': (2, 6),  'tp': (4, 12)},
    'finance':        {'sl': (2, 5),  'tp': (3, 10)},
    'healthcare':     {'sl': (2, 5),  'tp': (3, 8)},
    'energy':         {'sl': (3, 7),  'tp': (5, 12)},
    'consumer':       {'sl': (2, 5),  'tp': (3, 8)},
    'commodities':    {'sl': (3, 8),  'tp': (6, 15)},
    'indices':        {'sl': (2, 5),  'tp': (3, 8)},
    'forex':          {'sl': (1, 3),  'tp': (2, 6)},
    'etf':            {'sl': (2, 5),  'tp': (3, 8)},
    'defensive':      {'sl': (2, 4),  'tp': (3, 7)},
}


# ============================================================================
# Classe principale
# ============================================================================

class StrategyAllocator:
    """
    Allocation intelligente des strategies de trading basee sur les backtests.

    Parametres configurables pour filtrage, scoring, allocation et validation.
    S'adapte automatiquement aux actifs definis dans assets_config.py.
    """

    def __init__(
        self,
        # Filtres
        min_trades: int = 20,
        min_sharpe: float = 0.0,
        max_drawdown: float = 50.0,
        min_win_rate: float = 0.0,
        # Poids du score composite
        weight_sharpe: float = 0.40,
        weight_return: float = 0.25,
        weight_drawdown: float = 0.20,
        weight_winrate: float = 0.15,
        # Allocation
        allocation_method: str = 'score_weighted',
        max_alloc_per_asset: float = 25.0,
        max_alloc_per_category: float = 40.0,
        cash_reserve_pct: float = 10.0,
        # Walk-forward
        enable_walk_forward: bool = False,
        wf_in_sample_months: int = 12,
        wf_out_sample_months: int = 6,
        min_stability_score: float = 0.3,
    ):
        # Filtres
        self.min_trades = min_trades
        self.min_sharpe = min_sharpe
        self.max_drawdown = max_drawdown
        self.min_win_rate = min_win_rate

        # Poids
        self.weights = {
            'sharpe': weight_sharpe,
            'return': weight_return,
            'drawdown': weight_drawdown,
            'winrate': weight_winrate,
        }

        # Allocation
        self.allocation_method = allocation_method
        self.max_alloc_per_asset = max_alloc_per_asset
        self.max_alloc_per_category = max_alloc_per_category
        self.cash_reserve_pct = cash_reserve_pct

        # Walk-forward
        self.enable_walk_forward = enable_walk_forward
        self.wf_in_sample_months = wf_in_sample_months
        self.wf_out_sample_months = wf_out_sample_months
        self.min_stability_score = min_stability_score

    # ------------------------------------------------------------------ #
    # API publique                                                        #
    # ------------------------------------------------------------------ #

    def allocate(
        self,
        backtest_results: pd.DataFrame,
        total_capital: float = 10000,
        max_positions: int = 5,
    ) -> TradingPlan:
        """
        Pipeline complet d'allocation.

        Args:
            backtest_results: DataFrame au format BacktestLibrary
            total_capital: Capital total a allouer
            max_positions: Nombre max de positions simultanees

        Returns:
            TradingPlan avec les assignations, allocations et risk params
        """
        # 1. Filtrer
        df = self._filter(backtest_results)
        if len(df) == 0:
            return self._empty_plan(total_capital, max_positions)

        # 2. Scorer
        df = self._score(df)

        # 3. Selectionner la meilleure strategie par actif
        assignments = self._select_best_per_asset(df)
        if not assignments:
            return self._empty_plan(total_capital, max_positions)

        # 4. Walk-forward validation (optionnel)
        if self.enable_walk_forward:
            assignments = self._walk_forward_validate(assignments)
            if not assignments:
                return self._empty_plan(total_capital, max_positions)

        # 5. Limiter au max_positions (top par score)
        assignments.sort(key=lambda a: a.score, reverse=True)
        assignments = assignments[:max_positions]

        # 6. Allouer le capital
        assignments = self._allocate_capital(assignments)

        # 7. Parametres de risque dynamiques
        risk_params = self._compute_risk_params(assignments)

        # 8. Construire le plan
        config = {
            'total_capital': total_capital,
            'max_positions': max_positions,
            'allocation_method': self.allocation_method,
            'filters': {
                'min_trades': self.min_trades,
                'min_sharpe': self.min_sharpe,
                'max_drawdown': self.max_drawdown,
                'min_win_rate': self.min_win_rate,
            },
            'weights': self.weights,
            'walk_forward': self.enable_walk_forward,
            'cash_reserve_pct': self.cash_reserve_pct,
        }

        plan = TradingPlan(
            assignments=assignments,
            risk_params=risk_params,
            config=config,
            generated_at=datetime.now(),
        )
        plan.summary_stats = self._compute_summary(plan)
        return plan

    def allocate_from_library(
        self,
        library: BacktestLibrary,
        total_capital: float = 10000,
        max_positions: int = 5,
    ) -> TradingPlan:
        """Allocation a partir d'une BacktestLibrary chargee."""
        return self.allocate(library.get_all_results(), total_capital, max_positions)

    def export_json(self, plan: TradingPlan, filepath: str):
        """Sauvegarde le plan en JSON."""
        data = {
            'generated_at': plan.generated_at.isoformat(),
            'config': plan.config,
            'summary_stats': plan.summary_stats,
            'assignments': [asdict(a) for a in plan.assignments],
            'risk_params': {k: asdict(v) for k, v in plan.risk_params.items()},
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def export_for_signal_generator(self, plan: TradingPlan) -> Dict:
        """
        Convertit le plan au format signal_generator.RECOMMENDED_STRATEGIES.

        Returns:
            Dict: {'SI=F': [('Combined', CombinedStrategy())], ...}
        """
        result = {}
        for assignment in plan.assignments:
            strategy_instance = instantiate_strategy(assignment.strategy_name)
            if strategy_instance is None:
                continue
            display_name = assignment.strategy_name.replace('_', ' ')
            result[assignment.asset] = [(display_name, strategy_instance)]
        return result

    def print_report(self, plan: TradingPlan):
        """Affiche un rapport lisible dans le terminal."""
        sep = '=' * 80
        print(f"\n{sep}")
        print("  STRATEGY ALLOCATION REPORT")
        print(f"  Genere le : {plan.generated_at.strftime('%Y-%m-%d %H:%M')}")
        print(sep)

        cfg = plan.config
        print(f"\n  Capital total       : {cfg['total_capital']:,.0f} EUR")
        print(f"  Max positions       : {cfg['max_positions']}")
        print(f"  Methode allocation  : {cfg['allocation_method']}")
        print(f"  Reserve cash        : {self.cash_reserve_pct}%")
        print(f"  Walk-forward        : {'Oui' if cfg['walk_forward'] else 'Non'}")

        filt = cfg['filters']
        print(f"\n  FILTRES")
        print(f"  Min trades          : {filt['min_trades']}")
        print(f"  Min Sharpe          : {filt['min_sharpe']}")
        print(f"  Max drawdown        : {filt['max_drawdown']}%")
        print(f"  Min win rate        : {filt['min_win_rate']}%")

        if not plan.assignments:
            print(f"\n  Aucune paire actif/strategie ne passe les filtres.")
            print(sep)
            return

        # Tableau des allocations
        print(f"\n  ALLOCATIONS ({len(plan.assignments)} actifs selectionnes)")
        print(f"  {'-' * 76}")
        header = f"  {'#':<3} {'Actif':<14} {'Strategie':<22} {'Score':>6} {'Sharpe':>7} {'Return':>8} {'Alloc':>6} {'SL/TP':>10}"
        print(header)
        print(f"  {'-' * 76}")

        for i, a in enumerate(plan.assignments, 1):
            rp = plan.risk_params.get(a.asset)
            sl_tp = f"{rp.stop_loss_pct:.1f}/{rp.take_profit_pct:.1f}" if rp else "N/A"
            print(
                f"  {i:<3} {a.asset:<14} {a.strategy_name:<22} "
                f"{a.score:>6.3f} {a.sharpe_ratio:>7.2f} "
                f"{a.total_return_pct:>+7.1f}% {a.allocation_pct:>5.1f}% "
                f"{sl_tp:>10}"
            )

        print(f"  {'-' * 76}")

        # Allocation par categorie
        cat_alloc: Dict[str, float] = {}
        cat_count: Dict[str, int] = {}
        for a in plan.assignments:
            cat_alloc[a.asset_type] = cat_alloc.get(a.asset_type, 0) + a.allocation_pct
            cat_count[a.asset_type] = cat_count.get(a.asset_type, 0) + 1

        print(f"\n  ALLOCATION PAR CATEGORIE")
        for cat in sorted(cat_alloc, key=cat_alloc.get, reverse=True):
            print(f"    {cat:<16} : {cat_alloc[cat]:>5.1f}%  ({cat_count[cat]} actif{'s' if cat_count[cat]>1 else ''})")

        total_alloc = sum(a.allocation_pct for a in plan.assignments)
        print(f"    {'Cash reserve':<16} : {100 - total_alloc:>5.1f}%")

        # Stats de walk-forward si activees
        if self.enable_walk_forward:
            stab_scores = [a.stability_score for a in plan.assignments if a.stability_score is not None]
            if stab_scores:
                print(f"\n  WALK-FORWARD")
                print(f"    Stability score moyen : {np.mean(stab_scores):.2f}")
                print(f"    Min                   : {min(stab_scores):.2f}")
                print(f"    Max                   : {max(stab_scores):.2f}")

        print(f"\n{sep}\n")

    # ------------------------------------------------------------------ #
    # Pipeline interne                                                    #
    # ------------------------------------------------------------------ #

    def _filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filtre les combinaisons ne passant pas les seuils minimaux."""
        if df is None or len(df) == 0:
            return pd.DataFrame()

        filtered = df[
            (df['num_trades'] >= self.min_trades) &
            (df['sharpe_ratio'] >= self.min_sharpe) &
            (df['max_drawdown_pct'] <= self.max_drawdown) &
            (df['win_rate'] >= self.min_win_rate)
        ].copy()

        return filtered

    def _score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule le score composite normalise pour chaque paire."""
        df = df.copy()

        # Normalisation min-max pour chaque metrique
        metrics = {
            'sharpe_ratio': 'sharpe',
            'total_return_pct': 'return',
            'max_drawdown_pct': 'drawdown',
            'win_rate': 'winrate',
        }

        for col, key in metrics.items():
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:
                df[f'_norm_{key}'] = (df[col] - min_val) / (max_val - min_val)
            else:
                df[f'_norm_{key}'] = 0.5

        # Score composite (drawdown inverse : plus bas = mieux)
        df['score'] = (
            self.weights['sharpe']   * df['_norm_sharpe'] +
            self.weights['return']   * df['_norm_return'] +
            self.weights['drawdown'] * (1 - df['_norm_drawdown']) +
            self.weights['winrate']  * df['_norm_winrate']
        )

        return df

    def _select_best_per_asset(self, df: pd.DataFrame) -> List[AssetStrategyScore]:
        """Selectionne la meilleure strategie par actif d'apres le score."""
        if len(df) == 0:
            return []

        best_idx = df.groupby('asset')['score'].idxmax()
        best_df = df.loc[best_idx]

        assignments = []
        for _, row in best_df.iterrows():
            assignments.append(AssetStrategyScore(
                asset=row['asset'],
                asset_type=row.get('asset_type', ''),
                asset_name=row.get('asset_name', row['asset']),
                strategy_name=row['strategy'],
                score=row['score'],
                sharpe_ratio=row['sharpe_ratio'],
                total_return_pct=row['total_return_pct'],
                max_drawdown_pct=row['max_drawdown_pct'],
                win_rate=row['win_rate'],
                num_trades=int(row['num_trades']),
                avg_trade_return_pct=row.get('avg_trade_return_pct', 0),
            ))

        return assignments

    def _walk_forward_validate(
        self, assignments: List[AssetStrategyScore]
    ) -> List[AssetStrategyScore]:
        """
        Valide chaque paire via walk-forward sur sous-periodes.
        Ajoute stability_score et filtre celles en dessous du seuil.
        """
        import yfinance as yf

        window_months = self.wf_in_sample_months + self.wf_out_sample_months
        # Generer les fenetres glissantes sur 2024-2025
        windows = self._generate_wf_windows(
            start='2024-01-01',
            end='2025-12-31',
            in_months=self.wf_in_sample_months,
            out_months=self.wf_out_sample_months,
        )

        if not windows:
            return assignments

        validated = []
        for assignment in assignments:
            yahoo_sym = convert_to_yahoo_symbol(assignment.asset)
            window_scores = []

            for in_start, in_end, out_start, out_end in windows:
                try:
                    # Donnees in-sample
                    ticker = yf.Ticker(yahoo_sym)
                    in_data = ticker.history(start=in_start, end=in_end)
                    if len(in_data) < 50:
                        continue
                    in_data.columns = [c.lower() for c in in_data.columns]

                    # Donnees out-of-sample
                    out_data = ticker.history(start=out_start, end=out_end)
                    if len(out_data) < 20:
                        continue
                    out_data.columns = [c.lower() for c in out_data.columns]

                    # Backtests
                    strategy_in = instantiate_strategy(assignment.strategy_name)
                    strategy_out = instantiate_strategy(assignment.strategy_name)
                    if strategy_in is None or strategy_out is None:
                        continue

                    engine_in = BacktestEngine(initial_capital=10000, commission=0.001, slippage=0.0005)
                    metrics_in = engine_in.run_backtest(in_data, strategy_in)
                    ret_in = metrics_in.get('total_return_pct', 0)

                    engine_out = BacktestEngine(initial_capital=10000, commission=0.001, slippage=0.0005)
                    metrics_out = engine_out.run_backtest(out_data, strategy_out)
                    ret_out = metrics_out.get('total_return_pct', 0)

                    # Degradation
                    if abs(ret_in) > 0.01:
                        degradation = (ret_in - ret_out) / abs(ret_in)
                    else:
                        degradation = 1.0 if ret_out < 0 else 0.0

                    window_scores.append(np.clip(1 - degradation, 0, 1))

                except Exception:
                    continue

            # Stability score = moyenne des scores de fenetre
            if window_scores:
                stability = np.mean(window_scores)
            else:
                stability = 0.0

            assignment.stability_score = round(stability, 4)

            if stability >= self.min_stability_score:
                validated.append(assignment)

        return validated

    def _generate_wf_windows(
        self, start: str, end: str, in_months: int, out_months: int
    ) -> List[tuple]:
        """Genere les fenetres glissantes (in_start, in_end, out_start, out_end)."""
        from dateutil.relativedelta import relativedelta

        start_dt = datetime.strptime(start, '%Y-%m-%d')
        end_dt = datetime.strptime(end, '%Y-%m-%d')
        window_size = relativedelta(months=in_months + out_months)

        windows = []
        current = start_dt
        while current + window_size <= end_dt + relativedelta(days=1):
            in_end = current + relativedelta(months=in_months)
            out_end = in_end + relativedelta(months=out_months)
            windows.append((
                current.strftime('%Y-%m-%d'),
                in_end.strftime('%Y-%m-%d'),
                in_end.strftime('%Y-%m-%d'),
                out_end.strftime('%Y-%m-%d'),
            ))
            # Avancer d'un pas = out_months (fenetres non chevauchantes)
            current += relativedelta(months=out_months)

        return windows

    def _allocate_capital(
        self, assignments: List[AssetStrategyScore]
    ) -> List[AssetStrategyScore]:
        """Alloue le capital selon la methode choisie."""
        if not assignments:
            return assignments

        available_pct = 100.0 - self.cash_reserve_pct

        if self.allocation_method == 'equal':
            raw_alloc = available_pct / len(assignments)
            for a in assignments:
                a.allocation_pct = raw_alloc

        elif self.allocation_method == 'risk_parity':
            # Inverse de la volatilite (proxy = max_drawdown)
            inv_dd = []
            for a in assignments:
                dd = max(a.max_drawdown_pct, 0.1)  # eviter division par 0
                inv_dd.append(1.0 / dd)
            total_inv = sum(inv_dd)
            for a, inv in zip(assignments, inv_dd):
                a.allocation_pct = (inv / total_inv) * available_pct

        else:  # score_weighted (default)
            total_score = sum(a.score for a in assignments)
            if total_score > 0:
                for a in assignments:
                    a.allocation_pct = (a.score / total_score) * available_pct
            else:
                raw_alloc = available_pct / len(assignments)
                for a in assignments:
                    a.allocation_pct = raw_alloc

        # Appliquer les caps
        self._apply_caps(assignments)

        # Arrondir
        for a in assignments:
            a.allocation_pct = round(a.allocation_pct, 2)

        return assignments

    def _apply_caps(self, assignments: List[AssetStrategyScore]):
        """Applique les limites par actif et par categorie, redistribue l'excedent."""
        # Cap par actif
        excess = 0.0
        uncapped = []
        for a in assignments:
            if a.allocation_pct > self.max_alloc_per_asset:
                excess += a.allocation_pct - self.max_alloc_per_asset
                a.allocation_pct = self.max_alloc_per_asset
            else:
                uncapped.append(a)

        # Redistribuer l'excedent aux non-cappes
        if excess > 0 and uncapped:
            total_uncapped = sum(a.allocation_pct for a in uncapped)
            if total_uncapped > 0:
                for a in uncapped:
                    a.allocation_pct += excess * (a.allocation_pct / total_uncapped)

        # Cap par categorie
        cat_alloc: Dict[str, List[AssetStrategyScore]] = {}
        for a in assignments:
            cat_alloc.setdefault(a.asset_type, []).append(a)

        for cat, assets in cat_alloc.items():
            cat_total = sum(a.allocation_pct for a in assets)
            if cat_total > self.max_alloc_per_category:
                scale = self.max_alloc_per_category / cat_total
                for a in assets:
                    a.allocation_pct *= scale

    def _compute_risk_params(
        self, assignments: List[AssetStrategyScore]
    ) -> Dict[str, RiskParams]:
        """Genere les SL/TP dynamiques bases sur les metriques de backtest."""
        params = {}
        for a in assignments:
            cat = a.asset_type
            ranges = RISK_RANGES.get(cat, RISK_RANGES.get('defensive', {'sl': (2, 5), 'tp': (3, 8)}))

            sl_min, sl_max = ranges['sl']
            tp_min, tp_max = ranges['tp']

            # SL base sur le drawdown
            sl = np.clip(a.max_drawdown_pct * 0.4, sl_min, sl_max)

            # TP base sur le rendement moyen par trade
            if a.avg_trade_return_pct > 0:
                tp = np.clip(a.avg_trade_return_pct * 3, tp_min, tp_max)
            else:
                tp = (tp_min + tp_max) / 2

            # Position size proportionnelle au score (base 15-25%)
            pos_size = np.clip(15 + a.score * 15, 10, 30)

            params[a.asset] = RiskParams(
                stop_loss_pct=round(float(sl), 2),
                take_profit_pct=round(float(tp), 2),
                position_size_pct=round(float(pos_size), 2),
            )

        return params

    # ------------------------------------------------------------------ #
    # Utilitaires                                                         #
    # ------------------------------------------------------------------ #

    def _empty_plan(self, total_capital: float, max_positions: int) -> TradingPlan:
        """Retourne un plan vide quand aucune paire ne passe les filtres."""
        return TradingPlan(
            assignments=[],
            risk_params={},
            config={
                'total_capital': total_capital,
                'max_positions': max_positions,
                'allocation_method': self.allocation_method,
                'filters': {
                    'min_trades': self.min_trades,
                    'min_sharpe': self.min_sharpe,
                    'max_drawdown': self.max_drawdown,
                    'min_win_rate': self.min_win_rate,
                },
                'weights': self.weights,
                'walk_forward': self.enable_walk_forward,
                'cash_reserve_pct': self.cash_reserve_pct,
            },
            generated_at=datetime.now(),
            summary_stats={'message': 'Aucune paire actif/strategie ne passe les filtres.'},
        )

    def _compute_summary(self, plan: TradingPlan) -> Dict[str, Any]:
        """Calcule des statistiques resumees du plan."""
        if not plan.assignments:
            return {'n_assets': 0}

        returns = [a.total_return_pct for a in plan.assignments]
        sharpes = [a.sharpe_ratio for a in plan.assignments]
        allocs = [a.allocation_pct for a in plan.assignments]

        stats = {
            'n_assets': len(plan.assignments),
            'total_allocation_pct': round(sum(allocs), 2),
            'avg_return': round(np.mean(returns), 2),
            'avg_sharpe': round(np.mean(sharpes), 2),
            'best_return': round(max(returns), 2),
            'worst_return': round(min(returns), 2),
            'categories': list(set(a.asset_type for a in plan.assignments)),
        }

        stab = [a.stability_score for a in plan.assignments if a.stability_score is not None]
        if stab:
            stats['avg_stability'] = round(np.mean(stab), 2)

        return stats


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Strategy Allocator - Allocation intelligente des strategies de trading'
    )
    parser.add_argument('--capital', type=float, default=10000, help='Capital total (EUR)')
    parser.add_argument('--max-positions', type=int, default=5, help='Nombre max de positions')
    parser.add_argument('--min-trades', type=int, default=20, help='Nombre min de trades pour valider')
    parser.add_argument('--min-sharpe', type=float, default=0.0, help='Sharpe ratio minimum')
    parser.add_argument('--max-drawdown', type=float, default=50.0, help='Drawdown max tolere (%%)')
    parser.add_argument(
        '--allocation',
        choices=['equal', 'score_weighted', 'risk_parity'],
        default='score_weighted',
        help='Methode d\'allocation du capital',
    )
    parser.add_argument('--walk-forward', action='store_true', help='Activer la validation walk-forward')
    parser.add_argument('--output', default=None, help='Fichier de sortie JSON')

    args = parser.parse_args()

    # Charger les backtests
    print("Chargement des resultats de backtest...")
    library = BacktestLibrary()
    if not library.loaded:
        print("Pas de resultats en cache. Lancez d'abord: python backtest_library.py")
        exit(1)

    stats = library.get_statistics()
    print(f"  {stats['total_combinations']} combinaisons chargees "
          f"({stats['num_assets']} actifs x {stats['num_strategies']} strategies)")

    # Creer l'allocator
    allocator = StrategyAllocator(
        min_trades=args.min_trades,
        min_sharpe=args.min_sharpe,
        max_drawdown=args.max_drawdown,
        allocation_method=args.allocation,
        enable_walk_forward=args.walk_forward,
    )

    # Generer le plan
    print("\nAllocation en cours...")
    plan = allocator.allocate_from_library(
        library=library,
        total_capital=args.capital,
        max_positions=args.max_positions,
    )

    # Afficher le rapport
    allocator.print_report(plan)

    # Export JSON
    output_file = args.output or f"trading_plan_{datetime.now().strftime('%Y%m%d')}.json"
    allocator.export_json(plan, output_file)
    print(f"Plan sauvegarde dans : {output_file}")

    # Afficher le format signal_generator
    sg_format = allocator.export_for_signal_generator(plan)
    if sg_format:
        print("\nFormat signal_generator.RECOMMENDED_STRATEGIES :")
        for symbol, strategies in sg_format.items():
            names = [name for name, _ in strategies]
            print(f"  {symbol:<14} -> {', '.join(names)}")
