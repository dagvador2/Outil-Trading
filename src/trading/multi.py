"""
Multi Paper Trading - 20 portefeuilles en parallele
Compare differentes configurations d'allocation pour trouver la meilleure approche.
10 portfolios sans filtre macro (baseline) + 10 avec filtre macro (experimental).

Usage:
    python multi_paper_trading.py --capital 100000 --signal-hour 22 --sltp-interval 2
    python multi_paper_trading.py --single
    python multi_paper_trading.py --reset --single
"""

import os
import sys
import json
import time
import signal
import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from src.trading.auto import AutoPaperTrader, DEFAULT_STATE_DIR
from src.backtest.library import BacktestLibrary
from src.db.database import TradingDatabase


# ============================================================================
# 20 Portfolio Configurations (10 sans macro + 10 avec macro)
# ============================================================================

# Base configurations (sans macro filter)
BASE_CONFIGS: List[Dict[str, Any]] = [
    {
        'name': 'Conservative',
        'allocation_method': 'risk_parity',
        'min_sharpe': 0.8,
        'max_drawdown': 20.0,
        'max_positions': 5,
        'min_trades': 25,
        'min_confidence': 0.65,
        'category_filter': None,
        'description': 'Filtres stricts, faible risque',
    },
    {
        'name': 'Balanced',
        'allocation_method': 'score_weighted',
        'min_sharpe': 0.3,
        'max_drawdown': 40.0,
        'max_positions': 8,
        'min_trades': 20,
        'min_confidence': 0.55,
        'category_filter': None,
        'description': 'Configuration equilibree',
    },
    {
        'name': 'Aggressive',
        'allocation_method': 'score_weighted',
        'min_sharpe': 0.0,
        'max_drawdown': 60.0,
        'max_positions': 12,
        'min_trades': 15,
        'min_confidence': 0.50,
        'category_filter': None,
        'description': 'Filtres larges, plus de positions',
    },
    {
        'name': 'High_Sharpe',
        'allocation_method': 'score_weighted',
        'min_sharpe': 1.0,
        'max_drawdown': 50.0,
        'max_positions': 6,
        'min_trades': 20,
        'min_confidence': 0.55,
        'category_filter': None,
        'description': 'Seules les paires a haut Sharpe',
    },
    {
        'name': 'Low_Drawdown',
        'allocation_method': 'risk_parity',
        'min_sharpe': 0.0,
        'max_drawdown': 15.0,
        'max_positions': 8,
        'min_trades': 20,
        'min_confidence': 0.55,
        'category_filter': None,
        'description': 'Minimise le drawdown',
    },
    {
        'name': 'Equal_Weight',
        'allocation_method': 'equal',
        'min_sharpe': 0.3,
        'max_drawdown': 40.0,
        'max_positions': 8,
        'min_trades': 20,
        'min_confidence': 0.55,
        'category_filter': None,
        'description': 'Allocation egalitaire',
    },
    {
        'name': 'Risk_Parity',
        'allocation_method': 'risk_parity',
        'min_sharpe': 0.3,
        'max_drawdown': 40.0,
        'max_positions': 8,
        'min_trades': 20,
        'min_confidence': 0.55,
        'category_filter': None,
        'description': 'Pondere par inverse volatilite',
    },
    {
        'name': 'Concentrated',
        'allocation_method': 'score_weighted',
        'min_sharpe': 0.5,
        'max_drawdown': 40.0,
        'max_positions': 4,
        'min_trades': 25,
        'min_confidence': 0.60,
        'category_filter': None,
        'description': 'Peu de positions, haute conviction',
    },
    {
        'name': 'Diversified',
        'allocation_method': 'equal',
        'min_sharpe': 0.0,
        'max_drawdown': 50.0,
        'max_positions': 15,
        'min_trades': 15,
        'min_confidence': 0.50,
        'category_filter': None,
        'description': 'Maximum de diversification',
    },
    {
        'name': 'Crypto_Commodities',
        'allocation_method': 'score_weighted',
        'min_sharpe': 0.3,
        'max_drawdown': 50.0,
        'max_positions': 8,
        'min_trades': 15,
        'min_confidence': 0.55,
        'category_filter': ['crypto', 'commodities'],
        'description': 'Filtre crypto + commodities uniquement',
    },
]

# Generate 20 portfolios: 10 without macro + 10 with macro filter
PORTFOLIO_CONFIGS: List[Dict[str, Any]] = []

# First 10: Without macro filter (baseline)
for config in BASE_CONFIGS:
    portfolio = config.copy()
    portfolio['enable_macro_filter'] = False
    PORTFOLIO_CONFIGS.append(portfolio)

# Next 10: With macro filter (experimental)
for config in BASE_CONFIGS:
    portfolio = config.copy()
    portfolio['name'] = f"{config['name']}_Macro"
    portfolio['description'] = f"{config['description']} + Filtre macro"
    portfolio['enable_macro_filter'] = True
    portfolio['macro_threshold'] = 60.0  # Seuil conservateur
    PORTFOLIO_CONFIGS.append(portfolio)


# ============================================================================
# Multi Paper Trader
# ============================================================================

class MultiPaperTrader:
    """
    Orchestre 20 AutoPaperTrader en parallele,
    chacun avec sa propre configuration et son propre state_dir.

    - 10 portfolios sans filtre macro (baseline)
    - 10 portfolios avec filtre macro (experimental)
    """

    def __init__(
        self,
        total_capital: float = 100000,
        configs: Optional[List[Dict]] = None,
        signal_hour_utc: int = 22,
        sltp_interval_minutes: int = 2,
        timeframe: str = '1d',
        base_state_dir: str = DEFAULT_STATE_DIR,
    ):
        self.total_capital = total_capital
        self.configs = configs or PORTFOLIO_CONFIGS
        self.signal_hour_utc = signal_hour_utc
        self.sltp_interval = sltp_interval_minutes
        self.timeframe = timeframe
        self.base_state_dir = base_state_dir

        n = len(self.configs)
        self.capital_per_portfolio = total_capital / n

        self.traders: List[AutoPaperTrader] = []
        self._running = True
        self.consolidated_file = os.path.join(base_state_dir, 'consolidated_state.json')
        self.db = TradingDatabase()

        self.log = logging.getLogger('multi_trader')
        self.log.setLevel(logging.INFO)
        if not self.log.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'
            ))
            self.log.addHandler(handler)

        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def setup(self) -> bool:
        """Cree et configure les 20 traders (10 sans macro + 10 avec macro). Charge le BacktestLibrary une seule fois."""
        self.log.info("=" * 70)
        self.log.info(f"MULTI PAPER TRADING - {len(self.configs)} portefeuilles")
        self.log.info(f"Capital total: {self.total_capital:,.0f} EUR "
                      f"({self.capital_per_portfolio:,.0f} EUR/portefeuille)")
        self.log.info("=" * 70)

        # Charger le BacktestLibrary une seule fois
        self.log.info("Chargement des resultats de backtest...")
        library = BacktestLibrary()
        if not library.loaded:
            self.log.error("Pas de cache backtest. Lancez d'abord: python backtest_library.py")
            return False

        stats = library.get_statistics()
        self.log.info(f"  {stats['total_combinations']} combinaisons chargees")

        # Creer chaque trader
        for i, cfg in enumerate(self.configs, 1):
            name = cfg['name']
            state_dir = os.path.join(self.base_state_dir, f"portfolio_{i:02d}_{name.lower()}")

            self.log.info(f"\n--- [{i:02d}] {name}: {cfg['description']} ---")

            trader = AutoPaperTrader(
                total_capital=self.capital_per_portfolio,
                max_positions=cfg['max_positions'],
                min_trades=cfg['min_trades'],
                min_sharpe=cfg['min_sharpe'],
                max_drawdown=cfg['max_drawdown'],
                allocation_method=cfg['allocation_method'],
                min_confidence=cfg['min_confidence'],
                timeframe=self.timeframe,
                enable_macro_filter=cfg.get('enable_macro_filter', False),
                macro_threshold=cfg.get('macro_threshold', 60.0),
                state_dir=state_dir,
                portfolio_name=name,
                category_filter=cfg.get('category_filter'),
                register_signals=False,
                console_log=False,
                db=self.db,
            )

            if trader.setup(library=library):
                self.traders.append(trader)
                n_assets = len(trader.plan.assignments) if trader.plan else 0
                self.log.info(f"  OK: {n_assets} actifs dans le plan")
            else:
                self.log.warning(f"  ECHEC setup pour {name}")

        self.log.info(f"\n{'='*70}")
        self.log.info(f"{len(self.traders)}/{len(self.configs)} portefeuilles prets")
        self.log.info(f"{'='*70}")

        return len(self.traders) > 0

    def run(self):
        """Boucle infinie avec deux frequences:
        - Signaux (run_cycle): 1x par jour a signal_hour_utc (defaut: 22h UTC)
        - SL/TP + pending orders: toutes les sltp_interval minutes (defaut: 2 min)
        """
        self.log.info(f"\nDemarrage boucle:")
        self.log.info(f"  Signaux: 1x/jour a {self.signal_hour_utc}:00 UTC")
        self.log.info(f"  SL/TP + pending: toutes les {self.sltp_interval} min")
        self.log.info("Ctrl+C pour arreter proprement.\n")

        # Au demarrage: pas de cycle signal immediat (evite les entrees non planifiees)
        # On commence directement en mode SL/TP + pending orders
        now_utc = datetime.utcnow()
        if now_utc.hour >= self.signal_hour_utc:
            last_signal_date = now_utc.strftime('%Y-%m-%d')
        else:
            last_signal_date = (now_utc - timedelta(days=1)).strftime('%Y-%m-%d')

        while self._running:
            try:
                now_utc = datetime.utcnow()

                # Cycle signaux: 1x/jour quand on atteint signal_hour_utc
                today_str = now_utc.strftime('%Y-%m-%d')
                if now_utc.hour >= self.signal_hour_utc and today_str != last_signal_date:
                    self.run_cycle()
                    last_signal_date = today_str
                else:
                    # Sinon, uniquement check SL/TP
                    self.run_sltp_cycle()

                if not self._running:
                    break

                # Attente jusqu'au prochain check SL/TP
                for _ in range(self.sltp_interval * 6):
                    if not self._running:
                        break
                    time.sleep(10)

            except Exception as e:
                self.log.error(f"Erreur dans le cycle: {e}", exc_info=True)
                time.sleep(60)

        self.log.info("Arret propre. Sauvegarde finale...")
        self._save_consolidated()
        self.log.info("Bye.")

    def run_cycle(self):
        """Execute un cycle complet (signaux + SL/TP) pour chacun des 10 traders."""
        self.log.info(f"\n{'='*70}")
        self.log.info(f"CYCLE COMPLET (signaux + SL/TP) - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log.info(f"{'='*70}")

        for trader in self.traders:
            try:
                trader.run_cycle()
            except Exception as e:
                self.log.warning(f"  Erreur {trader.portfolio_name}: {e}")

        self._print_comparison()
        self._save_consolidated()

    def run_sltp_cycle(self):
        """Check SL/TP uniquement (rapide, pas de generation de signaux)."""
        total_positions = sum(len(t.positions) for t in self.traders)
        if total_positions == 0:
            return

        self.log.info(f"\n--- SL/TP CHECK - {datetime.now().strftime('%H:%M:%S')} "
                      f"({total_positions} positions ouvertes) ---")

        closed_count = 0
        for trader in self.traders:
            if not trader.positions:
                continue
            try:
                before = len(trader.positions)
                trader.check_sl_tp()
                after = len(trader.positions)
                closed_count += (before - after)
            except Exception as e:
                self.log.warning(f"  Erreur SL/TP {trader.portfolio_name}: {e}")

        if closed_count > 0:
            self.log.info(f"  >> {closed_count} position(s) fermee(s) par SL/TP")
        self._save_consolidated()

    def _print_comparison(self):
        """Affiche un tableau comparatif des 10 portefeuilles."""
        self.log.info(f"\n{'='*70}")
        self.log.info("COMPARAISON DES PORTEFEUILLES")
        self.log.info(f"{'='*70}")

        header = (f"{'#':>2} {'Nom':<18} {'Valeur':>10} {'P&L':>10} {'P&L%':>7} "
                  f"{'Pos':>4} {'Pend':>4} {'Trades':>6} {'WR%':>5}")
        self.log.info(header)
        self.log.info("-" * 75)

        statuses = []
        for i, trader in enumerate(self.traders, 1):
            try:
                s = trader.get_status()
                statuses.append(s)
                pnl_sign = '+' if s['pnl'] >= 0 else ''
                pending = s.get('pending_orders', 0)
                line = (
                    f"{i:>2} {s['portfolio_name']:<18} "
                    f"{s['total_value']:>10,.2f} "
                    f"{pnl_sign}{s['pnl']:>9,.2f} "
                    f"{pnl_sign}{s['pnl_pct']:>6.2f}% "
                    f"{s['positions']:>3}/{s['max_positions']:<1} "
                    f"{pending:>4} "
                    f"{s['trades_closed']:>5} "
                    f"{s['win_rate']:>5.1f}"
                )
                self.log.info(line)
            except Exception as e:
                self.log.warning(f"{i:>2} {trader.portfolio_name:<18} ERREUR: {e}")

        # Total
        if statuses:
            total_val = sum(s['total_value'] for s in statuses)
            total_pnl = sum(s['pnl'] for s in statuses)
            total_pnl_pct = (total_pnl / self.total_capital * 100) if self.total_capital > 0 else 0
            total_pos = sum(s['positions'] for s in statuses)
            total_trades = sum(s['trades_closed'] for s in statuses)
            pnl_sign = '+' if total_pnl >= 0 else ''

            self.log.info("-" * 70)
            self.log.info(
                f"   {'TOTAL':<18} {total_val:>10,.2f} "
                f"{pnl_sign}{total_pnl:>9,.2f} "
                f"{pnl_sign}{total_pnl_pct:>6.2f}% "
                f"{total_pos:>4} {total_trades:>6}"
            )

        self.log.info(f"{'='*70}\n")

    def _save_consolidated(self):
        """Sauvegarde l'etat global dans consolidated_state.json."""
        os.makedirs(self.base_state_dir, exist_ok=True)

        portfolios = {}
        for i, trader in enumerate(self.traders, 1):
            cfg = self.configs[i - 1] if i <= len(self.configs) else {}
            try:
                s = trader.get_status()
                portfolios[s['portfolio_name']] = {
                    'value': s['total_value'],
                    'cash': s['cash'],
                    'pnl': s['pnl'],
                    'pnl_pct': s['pnl_pct'],
                    'realized_pnl': s['realized_pnl'],
                    'positions': s['positions'],
                    'pending_orders': s.get('pending_orders', 0),
                    'max_positions': s['max_positions'],
                    'trades_closed': s['trades_closed'],
                    'win_rate': s['win_rate'],
                    'cycle_count': s['cycle_count'],
                    'n_assets_in_plan': s['n_assets_in_plan'],
                    'config_summary': (
                        f"{cfg.get('allocation_method', '?')} | "
                        f"sharpe>{cfg.get('min_sharpe', '?')} | "
                        f"dd<{cfg.get('max_drawdown', '?')}% | "
                        f"{cfg.get('max_positions', '?')} pos"
                    ),
                    'description': cfg.get('description', ''),
                }
            except Exception:
                pass

        consolidated = {
            'last_update': datetime.now().isoformat(),
            'total_capital': self.total_capital,
            'capital_per_portfolio': self.capital_per_portfolio,
            'n_portfolios': len(self.traders),
            'portfolios': portfolios,
        }

        with open(self.consolidated_file, 'w') as f:
            json.dump(consolidated, f, indent=2, default=str)

    def _handle_shutdown(self, signum, frame):
        self.log.info("\nSignal d'arret recu. Fermeture en cours...")
        self._running = False


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Multi Paper Trading - 10 portefeuilles en parallele'
    )
    parser.add_argument('--capital', type=float, default=100000,
                        help='Capital total reparti sur les 10 portefeuilles (defaut: 100000 = 10k/portfolio)')
    parser.add_argument('--signal-hour', type=int, default=22,
                        help='Heure UTC du cycle quotidien de signaux (defaut: 22 = 22h UTC, apres cloture US)')
    parser.add_argument('--sltp-interval', type=int, default=2,
                        help='Intervalle entre checks SL/TP en minutes (defaut: 2)')
    parser.add_argument('--single', action='store_true',
                        help='Executer un seul cycle puis quitter')
    parser.add_argument('--timeframe', default='1d',
                        choices=['5m', '15m', '1h', '4h', '1d'],
                        help='Timeframe des donnees (defaut: 1d)')
    parser.add_argument('--reset', action='store_true',
                        help='Reinitialiser tous les portefeuilles')

    args = parser.parse_args()

    base_dir = DEFAULT_STATE_DIR

    # Reset si demande
    if args.reset:
        import shutil
        for i, cfg in enumerate(PORTFOLIO_CONFIGS, 1):
            pdir = os.path.join(base_dir, f"portfolio_{i:02d}_{cfg['name'].lower()}")
            if os.path.exists(pdir):
                shutil.rmtree(pdir)
                print(f"  Supprime: {pdir}")
        consolidated = os.path.join(base_dir, 'consolidated_state.json')
        if os.path.exists(consolidated):
            os.remove(consolidated)
        print("Reset complet.")

    multi = MultiPaperTrader(
        total_capital=args.capital,
        signal_hour_utc=args.signal_hour,
        sltp_interval_minutes=args.sltp_interval,
        timeframe=args.timeframe,
        base_state_dir=base_dir,
    )

    if not multi.setup():
        sys.exit(1)

    if args.single:
        multi.run_cycle()
        multi.log.info("Mode single: cycle termine.")
    else:
        multi.run()


if __name__ == '__main__':
    main()
