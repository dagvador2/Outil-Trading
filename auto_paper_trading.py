"""
Auto Paper Trading - Serveur de trading fictif autonome
Integre le Strategy Allocator pour choisir les meilleures paires actif/strategie
et execute des trades papier automatiquement via Yahoo Finance.

Usage standalone:
    python auto_paper_trading.py --capital 10000 --max-positions 8 --interval 15
    python auto_paper_trading.py --single
    python auto_paper_trading.py --capital 10000 --walk-forward

Usage via multi_paper_trading.py (10 portfolios en parallele):
    Voir multi_paper_trading.py
"""

import os
import sys
import json
import time
import signal as signal_module
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from logging.handlers import RotatingFileHandler
import math

import yfinance as yf


def _smart_round(value: float, min_decimals: int = 6) -> float:
    """Round a price preserving enough significant digits.

    For prices like 0.00000637, round(x, 6) = 0.000006 which loses
    precision. This function ensures at least 4 significant digits
    are preserved regardless of how small the price is.
    """
    if value == 0:
        return 0.0
    # Number of decimals needed to keep 4 significant digits
    sig_decimals = -int(math.floor(math.log10(abs(value)))) + 3
    decimals = max(min_decimals, sig_decimals)
    return round(value, decimals)


def _fmt_price(value: float) -> str:
    """Format a price with enough decimals to be readable."""
    if value == 0:
        return "0"
    if abs(value) >= 1:
        return f"{value:,.4f}"
    sig_decimals = -int(math.floor(math.log10(abs(value)))) + 3
    decimals = max(4, sig_decimals)
    return f"{value:.{decimals}f}"

from backtest_library import BacktestLibrary, STRATEGY_MAP, instantiate_strategy
from strategy_allocator import StrategyAllocator, TradingPlan, RiskParams
from assets_config import get_category_from_symbol, get_asset_info
from yahoo_data_feed import convert_to_yahoo_symbol
from indicators import TechnicalIndicators

# Import macro integration (with error handling)
try:
    from macro_integration import MacroFilter
    MACRO_AVAILABLE = True
except ImportError:
    MACRO_AVAILABLE = False


# ============================================================================
# Default paths (used in standalone mode)
# ============================================================================

DEFAULT_STATE_DIR = 'paper_trading_state'


# ============================================================================
# Dataclasses
# ============================================================================

@dataclass
class PaperPosition:
    symbol: str
    side: str            # 'LONG' ou 'SHORT'
    entry_price: float
    quantity: float
    entry_time: str
    stop_loss: float
    take_profit: float
    strategy: str
    allocation_pct: float

    def pnl(self, current_price: float) -> float:
        if self.side == 'LONG':
            return (current_price - self.entry_price) * self.quantity
        return (self.entry_price - current_price) * self.quantity

    def pnl_pct(self, current_price: float) -> float:
        if self.side == 'LONG':
            return ((current_price - self.entry_price) / self.entry_price) * 100
        return ((self.entry_price - current_price) / self.entry_price) * 100

    def should_close(self, current_price: float) -> Optional[str]:
        if self.side == 'LONG':
            if current_price <= self.stop_loss:
                return 'STOP_LOSS'
            if current_price >= self.take_profit:
                return 'TAKE_PROFIT'
        else:
            if current_price >= self.stop_loss:
                return 'STOP_LOSS'
            if current_price <= self.take_profit:
                return 'TAKE_PROFIT'
        return None


@dataclass
class ClosedTrade:
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: str
    exit_time: str
    pnl: float
    pnl_pct: float
    exit_reason: str
    strategy: str


# ============================================================================
# Logger
# ============================================================================

def setup_logger(name: str, state_dir: str, console: bool = True) -> logging.Logger:
    os.makedirs(state_dir, exist_ok=True)
    log_file = os.path.join(state_dir, 'auto_trader.log')

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # Eviter duplication de handlers si deja configure
    if logger.handlers:
        return logger

    file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'
        ))
        logger.addHandler(console_handler)

    return logger


# ============================================================================
# Classe principale
# ============================================================================

class AutoPaperTrader:
    """
    Serveur de paper trading autonome.

    - Genere un TradingPlan via StrategyAllocator
    - Execute des trades papier en boucle
    - Persiste l'etat (positions, trades, plan)
    - Compatible VPS (systemd / screen / nohup)
    - Supporte un state_dir custom pour multi-portfolio
    """

    def __init__(
        self,
        total_capital: float = 10000,
        max_positions: int = 8,
        check_interval_minutes: int = 15,
        # Allocator params
        min_trades: int = 20,
        min_sharpe: float = 0.3,
        max_drawdown: float = 50.0,
        allocation_method: str = 'score_weighted',
        enable_walk_forward: bool = False,
        # Trading params
        min_confidence: float = 0.55,
        timeframe: str = '1d',
        # Macro filter params
        enable_macro_filter: bool = False,
        macro_threshold: float = 60.0,
        # Multi-portfolio params
        state_dir: Optional[str] = None,
        portfolio_name: Optional[str] = None,
        category_filter: Optional[List[str]] = None,
        register_signals: bool = True,
        console_log: bool = True,
    ):
        self.total_capital = total_capital
        self.max_positions = max_positions
        self.check_interval = check_interval_minutes
        self.min_confidence = min_confidence
        self.timeframe = timeframe
        self.category_filter = category_filter
        self.portfolio_name = portfolio_name or 'default'

        # Paths
        self.state_dir = state_dir or DEFAULT_STATE_DIR
        self.state_file = os.path.join(self.state_dir, 'auto_state.json')
        self.trades_file = os.path.join(self.state_dir, 'auto_trades.csv')
        self.plan_file = os.path.join(self.state_dir, 'auto_plan.json')

        # Logger
        self.log = setup_logger(
            f'trader_{self.portfolio_name}', self.state_dir, console=console_log
        )

        # Allocator
        self.allocator = StrategyAllocator(
            min_trades=min_trades,
            min_sharpe=min_sharpe,
            max_drawdown=max_drawdown,
            allocation_method=allocation_method,
            enable_walk_forward=enable_walk_forward,
        )

        # Macro Filter
        self.enable_macro_filter = enable_macro_filter and MACRO_AVAILABLE
        self.macro_filter = None
        if self.enable_macro_filter:
            try:
                self.macro_filter = MacroFilter(
                    enable=True,
                    strong_threshold=macro_threshold,
                    cache_hours=4
                )
                self.log.info(f"✅ Macro filter enabled (threshold={macro_threshold})")
            except Exception as e:
                self.log.error(f"Failed to init macro filter: {e}")
                self.enable_macro_filter = False
        else:
            if enable_macro_filter and not MACRO_AVAILABLE:
                self.log.warning("Macro filter requested but macro_integration not available")

        # State
        self.cash = total_capital
        self.positions: Dict[str, PaperPosition] = {}
        self.closed_trades: List[ClosedTrade] = []
        self.plan: Optional[TradingPlan] = None
        self.strategy_instances: Dict[str, Any] = {}
        self.cycle_count = 0

        # Graceful shutdown (only in standalone mode)
        self._running = True
        if register_signals:
            signal_module.signal(signal_module.SIGINT, self._handle_shutdown)
            signal_module.signal(signal_module.SIGTERM, self._handle_shutdown)

    # ------------------------------------------------------------------ #
    # Setup                                                                #
    # ------------------------------------------------------------------ #

    def setup(self, library: Optional[BacktestLibrary] = None) -> bool:
        """
        Genere le plan d'allocation et charge l'etat precedent.

        Args:
            library: BacktestLibrary pre-chargee (evite de recharger 10x).
                     Si None, charge depuis le cache.
        """
        self.log.info("=" * 60)
        self.log.info(f"SETUP [{self.portfolio_name}]")
        self.log.info("=" * 60)

        # Charger backtest results
        if library is None:
            self.log.info("Chargement des resultats de backtest...")
            library = BacktestLibrary()

        if not library.loaded:
            self.log.error("Pas de cache backtest. Lancez d'abord: python backtest_library.py")
            return False

        stats = library.get_statistics()
        self.log.info(f"  {stats['total_combinations']} combinaisons "
                      f"({stats['num_assets']} actifs x {stats['num_strategies']} strategies)")

        # Filtrer par categorie si demande
        results_df = library.get_all_results()
        if self.category_filter:
            results_df = results_df[results_df['asset_type'].isin(self.category_filter)]
            self.log.info(f"  Filtre categorie: {', '.join(self.category_filter)} "
                          f"-> {len(results_df)} combinaisons")

        # Generer le plan
        self.log.info("Generation du plan d'allocation...")
        self.plan = self.allocator.allocate(
            backtest_results=results_df,
            total_capital=self.total_capital,
            max_positions=self.max_positions,
        )

        if not self.plan.assignments:
            self.log.warning("Aucune paire actif/strategie ne passe les filtres.")
            # Pas un echec fatal: le portfolio reste vide
            os.makedirs(self.state_dir, exist_ok=True)
            self._save_state()
            return True

        self.log.info(f"  {len(self.plan.assignments)} actifs selectionnes")

        # Instancier les strategies
        for assignment in self.plan.assignments:
            instance = instantiate_strategy(assignment.strategy_name)
            if instance:
                self.strategy_instances[assignment.asset] = instance
                self.log.info(
                    f"  {assignment.asset:<14} -> {assignment.strategy_name:<22} "
                    f"(alloc: {assignment.allocation_pct:.1f}%)"
                )

        # Sauvegarder le plan
        os.makedirs(self.state_dir, exist_ok=True)
        self.allocator.export_json(self.plan, self.plan_file)

        # Charger etat precedent
        self._load_state()

        self.log.info(f"  Capital: {self.cash:,.2f} EUR | "
                      f"Positions: {len(self.positions)} | "
                      f"Trades clos: {len(self.closed_trades)}")
        return True

    # ------------------------------------------------------------------ #
    # Boucle principale                                                    #
    # ------------------------------------------------------------------ #

    def run(self):
        """Boucle infinie avec intervalle configurable."""
        self.log.info(f"\nDemarrage boucle (intervalle: {self.check_interval} min)")
        self.log.info("Ctrl+C pour arreter proprement.\n")

        while self._running:
            try:
                self.run_cycle()
                if not self._running:
                    break
                self.log.info(f"Prochain cycle dans {self.check_interval} min...")
                for _ in range(self.check_interval * 6):
                    if not self._running:
                        break
                    time.sleep(10)
            except Exception as e:
                self.log.error(f"Erreur dans le cycle: {e}", exc_info=True)
                time.sleep(60)

        self.log.info("Arret propre. Sauvegarde finale...")
        self._save_state()
        self.log.info("Bye.")

    def run_cycle(self):
        """Un cycle complet: signaux + trades + SL/TP + sauvegarde."""
        self.cycle_count += 1
        self.log.info(f"\n{'='*50}")
        self.log.info(f"[{self.portfolio_name}] CYCLE #{self.cycle_count} - "
                      f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log.info(f"{'='*50}")

        if not self.plan or not self.plan.assignments:
            self.log.info("Pas de plan d'allocation. Portfolio vide.")
            self._save_state()
            return

        # 1. Verifier SL/TP des positions ouvertes
        self._check_positions()

        # 2. Generer signaux et trader
        for assignment in self.plan.assignments:
            symbol = assignment.asset
            strategy = self.strategy_instances.get(symbol)
            if not strategy:
                continue

            try:
                signal_result = self._get_signal(symbol, strategy)
                if signal_result:
                    self._process_signal(symbol, signal_result, assignment)
            except Exception as e:
                self.log.warning(f"  {symbol}: erreur signal - {e}")

        # 3. Sauvegarder
        self._save_state()

    # ------------------------------------------------------------------ #
    # Signaux                                                              #
    # ------------------------------------------------------------------ #

    def _get_signal(self, symbol: str, strategy) -> Optional[Dict]:
        """Genere un signal pour un actif via sa strategie assignee."""
        yahoo_sym = convert_to_yahoo_symbol(symbol)

        try:
            ticker = yf.Ticker(yahoo_sym)
            if self.timeframe == '1d':
                data = ticker.history(period='6mo', interval='1d')
            elif self.timeframe in ('1h', '60m'):
                data = ticker.history(period='1mo', interval='1h')
            elif self.timeframe in ('4h',):
                data = ticker.history(period='3mo', interval='1d')
            else:
                data = ticker.history(period='5d', interval=self.timeframe)
        except Exception as e:
            self.log.warning(f"  {symbol}: pas de donnees Yahoo ({e})")
            return None

        if len(data) < 50:
            self.log.warning(f"  {symbol}: donnees insuffisantes ({len(data)} barres < 50)")
            return None

        data.columns = [c.lower() for c in data.columns]
        required = ['open', 'high', 'low', 'close', 'volume']
        available = [c for c in required if c in data.columns]
        if len(available) < 4:
            self.log.warning(f"  {symbol}: colonnes manquantes (seulement {available})")
            return None
        data = data[available]

        current_price = float(data['close'].iloc[-1])

        try:
            signals_df = strategy.generate_signals(data)
            if len(signals_df) == 0:
                self.log.warning(f"  {symbol}: strategie n'a genere aucun signal")
                return None

            # Apply macro filter if enabled
            macro_info = None
            if self.enable_macro_filter and self.macro_filter:
                try:
                    signals_df, macro_info = self.macro_filter.filter_signals(signals_df, symbol)
                except Exception as e:
                    self.log.warning(f"Macro filter error for {symbol}: {e}")
                    # Continue with unfiltered signals

            last_signal = signals_df.iloc[-1]
            position = last_signal.get('position', last_signal.get('signal', 0))
            confidence = self._estimate_confidence(data, position)

            if position == 1:
                signal_type = 'BUY'
            elif position == -1:
                signal_type = 'SELL'
            else:
                signal_type = 'HOLD'

            result = {
                'symbol': symbol,
                'signal': signal_type,
                'confidence': confidence,
                'current_price': current_price,
            }

            # Add macro info if available
            if macro_info:
                result['macro'] = macro_info

            return result
        except Exception as e:
            self.log.warning(f"  {symbol}: erreur generation signal - {e}")
            return None

    def _estimate_confidence(self, data: pd.DataFrame, position: int) -> float:
        """Estime la confiance basee sur les indicateurs techniques."""
        try:
            confidence = 0.5
            rsi = TechnicalIndicators.rsi(data['close'], 14).iloc[-1]
            bb = TechnicalIndicators.bollinger_bands(data['close'], 20, 2)
            price = data['close'].iloc[-1]
            bb_upper = bb['upper'].iloc[-1]
            bb_lower = bb['lower'].iloc[-1]

            if bb_upper != bb_lower:
                bb_pos = (price - bb_lower) / (bb_upper - bb_lower) * 100
            else:
                bb_pos = 50

            if position == 1:
                if rsi < 35:
                    confidence += 0.15
                elif rsi < 45:
                    confidence += 0.05
                if bb_pos < 20:
                    confidence += 0.15
                elif bb_pos < 40:
                    confidence += 0.05
            elif position == -1:
                if rsi > 70:
                    confidence += 0.15
                elif rsi > 60:
                    confidence += 0.05
                if bb_pos > 80:
                    confidence += 0.15
                elif bb_pos > 60:
                    confidence += 0.05

            return min(1.0, max(0.0, confidence))
        except Exception:
            return 0.5

    # ------------------------------------------------------------------ #
    # Trading                                                              #
    # ------------------------------------------------------------------ #

    def _process_signal(self, symbol: str, sig: Dict, assignment):
        signal_type = sig['signal']
        confidence = sig['confidence']
        price = sig['current_price']

        self.log.info(f"  {symbol}: signal={signal_type}, confidence={confidence:.2f}, price={_fmt_price(price)}")

        if symbol in self.positions:
            pos = self.positions[symbol]
            if (pos.side == 'LONG' and signal_type == 'SELL') or \
               (pos.side == 'SHORT' and signal_type == 'BUY'):
                self._close_position(symbol, price, 'SIGNAL_EXIT')
            else:
                self.log.info(f"  {symbol}: SKIP - position {pos.side} deja ouverte (signal={signal_type})")
            return

        if signal_type == 'HOLD':
            self.log.info(f"  {symbol}: SKIP - signal HOLD")
            return

        if confidence < self.min_confidence:
            self.log.info(f"  {symbol}: SKIP - confiance trop basse ({confidence:.2f} < {self.min_confidence})")
            return

        if len(self.positions) >= self.max_positions:
            self.log.info(f"  {symbol}: SKIP - max positions atteint ({len(self.positions)}/{self.max_positions})")
            return

        self._open_position(symbol, signal_type, price, assignment)

    def _open_position(self, symbol: str, signal_type: str, price: float, assignment):
        alloc_pct = assignment.allocation_pct
        position_value = self.total_capital * (alloc_pct / 100.0)
        position_value = min(position_value, self.cash * 0.95)
        if position_value < 1:
            return

        quantity = position_value / price

        rp = self.plan.risk_params.get(symbol)
        if rp:
            sl_pct = rp.stop_loss_pct
            tp_pct = rp.take_profit_pct
        else:
            sl_pct, tp_pct = 5.0, 10.0

        if signal_type == 'BUY':
            side = 'LONG'
            stop_loss = price * (1 - sl_pct / 100)
            take_profit = price * (1 + tp_pct / 100)
        else:
            side = 'SHORT'
            stop_loss = price * (1 + sl_pct / 100)
            take_profit = price * (1 - tp_pct / 100)

        pos = PaperPosition(
            symbol=symbol, side=side, entry_price=price, quantity=quantity,
            entry_time=datetime.now().isoformat(),
            stop_loss=_smart_round(stop_loss), take_profit=_smart_round(take_profit),
            strategy=assignment.strategy_name, allocation_pct=alloc_pct,
        )

        self.positions[symbol] = pos
        self.cash -= position_value

        self.log.info(
            f"  >> OPEN {side} {symbol} @ {_fmt_price(price)} "
            f"(qty: {quantity:.6f}, val: {position_value:,.2f} EUR, "
            f"SL: {_fmt_price(stop_loss)}, TP: {_fmt_price(take_profit)})"
        )

    def _close_position(self, symbol: str, exit_price: float, reason: str):
        pos = self.positions.pop(symbol, None)
        if not pos:
            return

        pnl = pos.pnl(exit_price)
        pnl_pct = pos.pnl_pct(exit_price)
        self.cash += exit_price * pos.quantity

        trade = ClosedTrade(
            symbol=symbol, side=pos.side, entry_price=pos.entry_price,
            exit_price=exit_price, quantity=pos.quantity,
            entry_time=pos.entry_time, exit_time=datetime.now().isoformat(),
            pnl=round(pnl, 2), pnl_pct=round(pnl_pct, 2),
            exit_reason=reason, strategy=pos.strategy,
        )
        self.closed_trades.append(trade)

        icon = '+' if pnl >= 0 else ''
        self.log.info(
            f"  << CLOSE {pos.side} {symbol} @ {_fmt_price(exit_price)} | "
            f"P&L: {icon}{pnl:,.2f} EUR ({icon}{pnl_pct:.2f}%) | {reason}"
        )

    def check_sl_tp(self):
        """Verifie uniquement les SL/TP des positions ouvertes et sauvegarde.
        Methode publique pour les checks rapides entre les cycles de signaux."""
        if not self.positions:
            return
        self.log.info(f"  [{self.portfolio_name}] Check SL/TP ({len(self.positions)} positions)")
        self._check_positions()
        self._save_state()

    def _check_positions(self):
        symbols_to_close = []
        for symbol, pos in self.positions.items():
            yahoo_sym = convert_to_yahoo_symbol(symbol)
            try:
                data = yf.Ticker(yahoo_sym).history(period='1d', interval='1d')
                if len(data) == 0:
                    continue
                current_price = float(data['Close'].iloc[-1])
                reason = pos.should_close(current_price)
                if reason:
                    symbols_to_close.append((symbol, current_price, reason))
            except Exception:
                continue

        for symbol, price, reason in symbols_to_close:
            self._close_position(symbol, price, reason)

    # ------------------------------------------------------------------ #
    # Status (retourne un dict pour le multi-portfolio)                    #
    # ------------------------------------------------------------------ #

    def get_status(self) -> Dict[str, Any]:
        """Retourne le status du portfolio sous forme de dict."""
        positions_value = 0.0
        for symbol, pos in self.positions.items():
            yahoo_sym = convert_to_yahoo_symbol(symbol)
            try:
                data = yf.Ticker(yahoo_sym).history(period='1d', interval='1d')
                if len(data) > 0:
                    positions_value += float(data['Close'].iloc[-1]) * pos.quantity
                else:
                    positions_value += pos.entry_price * pos.quantity
            except Exception:
                positions_value += pos.entry_price * pos.quantity

        total_value = self.cash + positions_value
        total_pnl = total_value - self.total_capital
        n_trades = len(self.closed_trades)
        realized_pnl = sum(t.pnl for t in self.closed_trades) if n_trades > 0 else 0
        wins = [t for t in self.closed_trades if t.pnl > 0]
        win_rate = (len(wins) / n_trades * 100) if n_trades > 0 else 0

        return {
            'portfolio_name': self.portfolio_name,
            'total_value': round(total_value, 2),
            'cash': round(self.cash, 2),
            'total_capital': self.total_capital,
            'pnl': round(total_pnl, 2),
            'pnl_pct': round(total_pnl / self.total_capital * 100, 2) if self.total_capital > 0 else 0,
            'realized_pnl': round(realized_pnl, 2),
            'positions': len(self.positions),
            'max_positions': self.max_positions,
            'trades_closed': n_trades,
            'win_rate': round(win_rate, 1),
            'cycle_count': self.cycle_count,
            'n_assets_in_plan': len(self.plan.assignments) if self.plan else 0,
        }

    # ------------------------------------------------------------------ #
    # Persistance                                                          #
    # ------------------------------------------------------------------ #

    def _save_state(self):
        os.makedirs(self.state_dir, exist_ok=True)

        state = {
            'portfolio_name': self.portfolio_name,
            'cash': self.cash,
            'total_capital': self.total_capital,
            'cycle_count': self.cycle_count,
            'last_update': datetime.now().isoformat(),
            'positions': {sym: asdict(pos) for sym, pos in self.positions.items()},
            'closed_trades_count': len(self.closed_trades),
        }

        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)

        if self.closed_trades:
            rows = [asdict(t) for t in self.closed_trades]
            pd.DataFrame(rows).to_csv(self.trades_file, index=False)

    def _load_state(self):
        if not os.path.exists(self.state_file):
            return

        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)

            self.cash = state.get('cash', self.total_capital)
            self.cycle_count = state.get('cycle_count', 0)

            for sym, pos_dict in state.get('positions', {}).items():
                self.positions[sym] = PaperPosition(**pos_dict)

            if os.path.exists(self.trades_file):
                df = pd.read_csv(self.trades_file)
                self.closed_trades = [
                    ClosedTrade(**row) for _, row in df.iterrows()
                ]

            # Verification de coherence comptable:
            # cash + cout_positions_ouvertes = total_capital + realized_pnl
            realized_pnl = sum(t.pnl for t in self.closed_trades) if self.closed_trades else 0
            positions_cost = sum(p.entry_price * p.quantity for p in self.positions.values())
            actual_total = self.cash + positions_cost
            expected_total = self.total_capital + realized_pnl

            if actual_total > 0 and abs(actual_total - expected_total) > 1.0:
                ratio = expected_total / actual_total
                self.log.warning(
                    f"  Incoherence: cash+positions ({actual_total:,.2f}) != "
                    f"capital+realized ({expected_total:,.2f}). Ajustement (ratio={ratio:.4f})"
                )
                self.cash *= ratio
                for pos in self.positions.values():
                    pos.quantity *= ratio
                self._save_state()

            self.log.info(f"  Etat charge: {self.cycle_count} cycles, "
                          f"{len(self.positions)} positions, "
                          f"{len(self.closed_trades)} trades clos")
        except Exception as e:
            self.log.warning(f"Impossible de charger l'etat: {e}")

    def _handle_shutdown(self, signum, frame):
        self.log.info("\nSignal d'arret recu. Fermeture en cours...")
        self._running = False


# ============================================================================
# CLI (standalone mode)
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Auto Paper Trading - Serveur de trading fictif autonome'
    )
    parser.add_argument('--capital', type=float, default=10000,
                        help='Capital total en EUR (defaut: 10000)')
    parser.add_argument('--max-positions', type=int, default=8,
                        help='Nombre max de positions simultanees (defaut: 8)')
    parser.add_argument('--interval', type=int, default=15,
                        help='Intervalle entre cycles en minutes (defaut: 15)')
    parser.add_argument('--single', action='store_true',
                        help='Executer un seul cycle puis quitter')
    parser.add_argument('--min-trades', type=int, default=20,
                        help='Min trades pour filtre allocator (defaut: 20)')
    parser.add_argument('--min-sharpe', type=float, default=0.3,
                        help='Min Sharpe ratio pour filtre (defaut: 0.3)')
    parser.add_argument('--max-drawdown', type=float, default=50.0,
                        help='Max drawdown %% pour filtre (defaut: 50)')
    parser.add_argument('--allocation', choices=['equal', 'score_weighted', 'risk_parity'],
                        default='score_weighted',
                        help="Methode d'allocation (defaut: score_weighted)")
    parser.add_argument('--walk-forward', action='store_true',
                        help='Activer la validation walk-forward')
    parser.add_argument('--min-confidence', type=float, default=0.55,
                        help='Confiance minimum pour ouvrir (defaut: 0.55)')
    parser.add_argument('--timeframe', default='1d',
                        choices=['5m', '15m', '1h', '4h', '1d'],
                        help='Timeframe des donnees (defaut: 1d)')
    parser.add_argument('--reset', action='store_true',
                        help='Reinitialiser completement le paper trading')

    args = parser.parse_args()

    state_dir = DEFAULT_STATE_DIR

    # Reset si demande
    if args.reset:
        for fname in ['auto_state.json', 'auto_trades.csv', 'auto_plan.json']:
            fpath = os.path.join(state_dir, fname)
            if os.path.exists(fpath):
                os.remove(fpath)
        print("Etat reinitialise.")

    trader = AutoPaperTrader(
        total_capital=args.capital,
        max_positions=args.max_positions,
        check_interval_minutes=args.interval,
        min_trades=args.min_trades,
        min_sharpe=args.min_sharpe,
        max_drawdown=args.max_drawdown,
        allocation_method=args.allocation,
        enable_walk_forward=args.walk_forward,
        min_confidence=args.min_confidence,
        timeframe=args.timeframe,
        state_dir=state_dir,
        portfolio_name='standalone',
    )

    if not trader.setup():
        sys.exit(1)

    if args.single:
        trader.run_cycle()
        trader._save_state()
        trader.log.info("Mode single: cycle termine.")
    else:
        trader.run()


if __name__ == '__main__':
    main()
