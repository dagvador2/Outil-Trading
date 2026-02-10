"""
Système de Paper Trading (Trading Fictif) Automatisé
Capital virtuel, ordres automatiques, suivi P&L en temps réel
"""

import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import os


@dataclass
class Position:
    """Représente une position ouverte"""
    symbol: str
    entry_price: float
    quantity: float
    side: str  # 'LONG' ou 'SHORT'
    entry_time: datetime
    stop_loss: float
    take_profit: float
    strategy: str
    confidence: float

    def current_pnl(self, current_price: float) -> float:
        """Calcule le P&L actuel"""
        if self.side == 'LONG':
            return (current_price - self.entry_price) * self.quantity
        else:  # SHORT
            return (self.entry_price - current_price) * self.quantity

    def current_pnl_pct(self, current_price: float) -> float:
        """Calcule le P&L en %"""
        if self.side == 'LONG':
            return ((current_price - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - current_price) / self.entry_price) * 100

    def should_close(self, current_price: float) -> Optional[str]:
        """Vérifie si la position doit être fermée (SL/TP)"""
        if self.side == 'LONG':
            if current_price <= self.stop_loss:
                return 'STOP_LOSS'
            elif current_price >= self.take_profit:
                return 'TAKE_PROFIT'
        else:  # SHORT
            if current_price >= self.stop_loss:
                return 'STOP_LOSS'
            elif current_price <= self.take_profit:
                return 'TAKE_PROFIT'
        return None


@dataclass
class Trade:
    """Représente un trade fermé"""
    symbol: str
    entry_price: float
    exit_price: float
    quantity: float
    side: str
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_pct: float
    exit_reason: str  # 'STOP_LOSS', 'TAKE_PROFIT', 'SIGNAL_EXIT'
    strategy: str
    confidence: float


class PaperTradingPortfolio:
    """
    Portefeuille de Paper Trading avec exécution automatique
    """

    def __init__(self, initial_capital: float = 10000.0,
                 min_confidence: float = 0.70,
                 position_size_pct: float = 20.0,
                 max_positions: int = 5):
        """
        Args:
            initial_capital: Capital initial en €
            min_confidence: Confiance minimale pour ouvrir position
            position_size_pct: % du capital par position
            max_positions: Nombre max de positions simultanées
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.min_confidence = min_confidence
        self.position_size_pct = position_size_pct
        self.max_positions = max_positions

        # Positions et historique
        self.positions: Dict[str, Position] = {}  # symbol -> Position
        self.closed_trades: List[Trade] = []

        # Fichiers de sauvegarde
        self.portfolio_file = "paper_trading_portfolio.json"
        self.trades_file = "paper_trading_trades.csv"

        # Charger état précédent
        self._load_state()

        print(f"\n{'='*80}")
        print("💼 PAPER TRADING PORTFOLIO")
        print(f"{'='*80}")
        print(f"Capital initial   : €{self.initial_capital:,.2f}")
        print(f"Confiance min     : {self.min_confidence:.0%}")
        print(f"Taille position   : {self.position_size_pct:.0f}% du capital")
        print(f"Max positions     : {self.max_positions}")
        print(f"{'='*80}\n")

    def _load_state(self):
        """Charge l'état du portfolio depuis les fichiers"""
        # Charger positions ouvertes
        if os.path.exists(self.portfolio_file):
            try:
                with open(self.portfolio_file, 'r') as f:
                    data = json.load(f)
                    self.cash = data.get('cash', self.initial_capital)
                    # TODO: Recharger positions (nécessite conversion dict -> Position)
            except Exception as e:
                print(f"⚠️  Impossible de charger portfolio: {e}")

        # Charger historique trades
        if os.path.exists(self.trades_file):
            try:
                trades_df = pd.read_csv(self.trades_file)
                # TODO: Convertir en objets Trade
                print(f"✅ {len(trades_df)} trades historiques chargés")
            except:
                pass

    def _save_state(self):
        """Sauvegarde l'état actuel"""
        # Sauvegarder portfolio
        portfolio_data = {
            'cash': self.cash,
            'positions': {
                symbol: {
                    'symbol': pos.symbol,
                    'entry_price': pos.entry_price,
                    'quantity': pos.quantity,
                    'side': pos.side,
                    'entry_time': pos.entry_time.isoformat(),
                    'stop_loss': pos.stop_loss,
                    'take_profit': pos.take_profit,
                    'strategy': pos.strategy,
                    'confidence': pos.confidence
                }
                for symbol, pos in self.positions.items()
            },
            'last_update': datetime.now().isoformat()
        }

        with open(self.portfolio_file, 'w') as f:
            json.dump(portfolio_data, f, indent=2)

        # Sauvegarder trades
        if len(self.closed_trades) > 0:
            trades_data = []
            for trade in self.closed_trades:
                trades_data.append({
                    'symbol': trade.symbol,
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'quantity': trade.quantity,
                    'side': trade.side,
                    'entry_time': trade.entry_time.isoformat(),
                    'exit_time': trade.exit_time.isoformat(),
                    'pnl': trade.pnl,
                    'pnl_pct': trade.pnl_pct,
                    'exit_reason': trade.exit_reason,
                    'strategy': trade.strategy,
                    'confidence': trade.confidence
                })

            trades_df = pd.DataFrame(trades_data)
            trades_df.to_csv(self.trades_file, index=False)

    def process_signal(self, signal: Dict) -> Optional[str]:
        """
        Traite un signal et prend une décision de trading

        Args:
            signal: Dictionnaire du signal (depuis LiveSignalGenerator)

        Returns:
            Message d'action ('OPENED', 'CLOSED', 'HOLD', 'IGNORED')
        """
        symbol = signal.get('symbol')
        signal_type = signal.get('signal')
        confidence = signal.get('confidence', 0)
        current_price = signal.get('current_price', 0)

        # Vérifier positions existantes
        if symbol in self.positions:
            return self._check_exit_position(symbol, current_price, signal_type)

        # Vérifier si on peut ouvrir une nouvelle position
        if signal_type in ['BUY', 'SELL'] and confidence >= self.min_confidence:
            if len(self.positions) < self.max_positions:
                return self._open_position(signal)
            else:
                return 'IGNORED: Max positions atteint'

        return 'HOLD'

    def _open_position(self, signal: Dict) -> str:
        """Ouvre une nouvelle position"""
        symbol = signal.get('symbol')
        signal_type = signal.get('signal')
        current_price = signal.get('current_price', 0)
        confidence = signal.get('confidence', 0)

        if current_price == 0:
            return 'IGNORED: Prix invalide'

        # Calculer taille de position
        position_value = self.cash * (self.position_size_pct / 100)
        quantity = position_value / current_price

        if quantity == 0:
            return 'IGNORED: Quantité trop faible'

        # Déterminer SL/TP
        if 'BTC' in symbol or 'ETH' in symbol:
            sl_pct = 8.0
            tp_pct = 15.0
        else:
            sl_pct = 3.0
            tp_pct = 6.0

        if signal_type == 'BUY':
            side = 'LONG'
            stop_loss = current_price * (1 - sl_pct / 100)
            take_profit = current_price * (1 + tp_pct / 100)
        else:  # SELL
            side = 'SHORT'
            stop_loss = current_price * (1 + sl_pct / 100)
            take_profit = current_price * (1 - tp_pct / 100)

        # Créer position
        position = Position(
            symbol=symbol,
            entry_price=current_price,
            quantity=quantity,
            side=side,
            entry_time=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy=signal.get('consensus', 'Unknown'),
            confidence=confidence
        )

        # Enregistrer
        self.positions[symbol] = position
        self.cash -= position_value

        # Sauvegarder
        self._save_state()

        return f"OPENED {side}: {symbol} @ €{current_price:,.2f} (qty: {quantity:.4f})"

    def _check_exit_position(self, symbol: str, current_price: float,
                              signal_type: str) -> str:
        """Vérifie si une position doit être fermée"""
        if symbol not in self.positions:
            return 'HOLD'

        position = self.positions[symbol]

        # Vérifier SL/TP
        exit_reason = position.should_close(current_price)

        # Ou signal inverse
        if not exit_reason:
            if (position.side == 'LONG' and signal_type == 'SELL') or \
               (position.side == 'SHORT' and signal_type == 'BUY'):
                exit_reason = 'SIGNAL_EXIT'

        if exit_reason:
            return self._close_position(symbol, current_price, exit_reason)

        return 'HOLD'

    def _close_position(self, symbol: str, exit_price: float,
                        exit_reason: str) -> str:
        """Ferme une position"""
        if symbol not in self.positions:
            return 'ERROR: Position not found'

        position = self.positions[symbol]

        # Calculer P&L
        pnl = position.current_pnl(exit_price)
        pnl_pct = position.current_pnl_pct(exit_price)

        # Remettre cash
        position_value = exit_price * position.quantity
        self.cash += position_value

        # Créer trade fermé
        trade = Trade(
            symbol=symbol,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            side=position.side,
            entry_time=position.entry_time,
            exit_time=datetime.now(),
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=exit_reason,
            strategy=position.strategy,
            confidence=position.confidence
        )

        self.closed_trades.append(trade)

        # Supprimer position
        del self.positions[symbol]

        # Sauvegarder
        self._save_state()

        emoji = "🟢" if pnl > 0 else "🔴"
        return f"CLOSED {position.side}: {symbol} @ €{exit_price:,.2f} | {emoji} P&L: €{pnl:,.2f} ({pnl_pct:+.2f}%) | {exit_reason}"

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calcule la valeur totale du portfolio"""
        positions_value = sum(
            current_prices.get(symbol, pos.entry_price) * pos.quantity
            for symbol, pos in self.positions.items()
        )
        return self.cash + positions_value

    def get_total_pnl(self, current_prices: Dict[str, float]) -> float:
        """P&L total (réalisé + non réalisé)"""
        # P&L réalisé
        realized_pnl = sum(trade.pnl for trade in self.closed_trades)

        # P&L non réalisé
        unrealized_pnl = sum(
            pos.current_pnl(current_prices.get(symbol, pos.entry_price))
            for symbol, pos in self.positions.items()
        )

        return realized_pnl + unrealized_pnl

    def get_statistics(self, current_prices: Dict[str, float]) -> Dict:
        """Statistiques du portfolio"""
        portfolio_value = self.get_portfolio_value(current_prices)
        total_pnl = self.get_total_pnl(current_prices)

        # Stats trades fermés
        if len(self.closed_trades) > 0:
            winning_trades = [t for t in self.closed_trades if t.pnl > 0]
            losing_trades = [t for t in self.closed_trades if t.pnl < 0]

            win_rate = len(winning_trades) / len(self.closed_trades) * 100

            avg_win = sum(t.pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0
            avg_loss = sum(t.pnl for t in losing_trades) / len(losing_trades) if losing_trades else 0

            profit_factor = abs(sum(t.pnl for t in winning_trades) / sum(t.pnl for t in losing_trades)) if losing_trades else float('inf')
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0

        return {
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'total_pnl': total_pnl,
            'total_pnl_pct': (total_pnl / self.initial_capital) * 100,
            'open_positions': len(self.positions),
            'closed_trades': len(self.closed_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor
        }

    def display_status(self, current_prices: Dict[str, float]):
        """Affiche le statut du portfolio"""
        stats = self.get_statistics(current_prices)

        print(f"\n{'='*80}")
        print("💼 PAPER TRADING PORTFOLIO STATUS")
        print(f"{'='*80}")

        print(f"\n📊 Valeur Portfolio")
        print(f"  Valeur totale    : €{stats['portfolio_value']:,.2f}")
        print(f"  Cash disponible  : €{stats['cash']:,.2f}")
        print(f"  P&L Total        : €{stats['total_pnl']:,.2f} ({stats['total_pnl_pct']:+.2f}%)")

        print(f"\n📈 Positions")
        print(f"  Ouvertes         : {stats['open_positions']}/{self.max_positions}")
        print(f"  Trades fermés    : {stats['closed_trades']}")

        if stats['closed_trades'] > 0:
            print(f"\n📊 Performance")
            print(f"  Win Rate         : {stats['win_rate']:.1f}%")
            print(f"  Gain moyen       : €{stats['avg_win']:,.2f}")
            print(f"  Perte moyenne    : €{stats['avg_loss']:,.2f}")
            print(f"  Profit Factor    : {stats['profit_factor']:.2f}")

        if len(self.positions) > 0:
            print(f"\n🔓 Positions Ouvertes")
            for symbol, pos in self.positions.items():
                current_price = current_prices.get(symbol, pos.entry_price)
                pnl = pos.current_pnl(current_price)
                pnl_pct = pos.current_pnl_pct(current_price)
                emoji = "🟢" if pnl > 0 else "🔴"

                print(f"  {emoji} {symbol:15} {pos.side:5} | "
                      f"Entry: €{pos.entry_price:,.2f} | "
                      f"Current: €{current_price:,.2f} | "
                      f"P&L: €{pnl:,.2f} ({pnl_pct:+.2f}%)")

        print(f"\n{'='*80}\n")


# ============================================================================
# Test
# ============================================================================

if __name__ == '__main__':
    # Créer portfolio
    portfolio = PaperTradingPortfolio(
        initial_capital=10000.0,
        min_confidence=0.70,
        position_size_pct=20.0,
        max_positions=5
    )

    # Simuler signaux
    test_signal = {
        'symbol': 'BTC/USDT',
        'signal': 'BUY',
        'confidence': 0.85,
        'current_price': 45000.0,
        'consensus': 'Combined Strategy'
    }

    action = portfolio.process_signal(test_signal)
    print(action)

    # Afficher statut
    current_prices = {'BTC/USDT': 46000.0}
    portfolio.display_status(current_prices)
