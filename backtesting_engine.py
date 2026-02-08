"""
Moteur de backtesting pour stratégies de trading
Supporte : Crypto, Actions, Forex, Indices, Matières premières
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Trade:
    """Représente une transaction individuelle"""
    entry_date: datetime
    exit_date: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    position_type: str  # 'LONG' ou 'SHORT'
    size: float
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    exit_reason: Optional[str] = None  # 'signal', 'stop_loss', 'take_profit', 'trailing_stop'
    highest_price: Optional[float] = None  # Pour trailing stop
    capital_invested: Optional[float] = None  # Capital investi dans ce trade

    def close_trade(self, exit_date: datetime, exit_price: float, exit_reason: str = 'signal'):
        """Clôture le trade et calcule le P&L"""
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.exit_reason = exit_reason

        if self.position_type == 'LONG':
            self.pnl = (exit_price - self.entry_price) * self.size
            self.pnl_pct = ((exit_price - self.entry_price) / self.entry_price) * 100
        else:  # SHORT
            self.pnl = (self.entry_price - exit_price) * self.size
            self.pnl_pct = ((self.entry_price - exit_price) / self.entry_price) * 100


class BacktestEngine:
    """
    Moteur de backtesting principal
    """
    
    def __init__(self,
                 initial_capital: float = 10000,
                 commission: float = 0.001,  # 0.1% par transaction
                 slippage: float = 0.0005,   # 0.05% de slippage
                 stop_loss_pct: Optional[float] = None,  # Stop-loss en % (ex: 2.0 pour 2%)
                 take_profit_pct: Optional[float] = None,  # Take-profit en % (ex: 5.0 pour 5%)
                 trailing_stop_pct: Optional[float] = None,  # Trailing stop en %
                 position_size_pct: float = 100.0):  # % du capital à risquer par trade
        """
        Args:
            initial_capital: Capital de départ
            commission: Commission par trade (en fraction, ex: 0.001 = 0.1%)
            slippage: Slippage moyen (en fraction)
            stop_loss_pct: Stop-loss en pourcentage (None = désactivé)
            take_profit_pct: Take-profit en pourcentage (None = désactivé)
            trailing_stop_pct: Trailing stop en pourcentage (None = désactivé)
            position_size_pct: Pourcentage du capital à utiliser par trade (100 = tout le capital)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.position_size_pct = position_size_pct

        # État du backtesting
        self.capital = initial_capital
        self.position = None
        self.trades: List[Trade] = []
        self.equity_curve = []
        
    def reset(self):
        """Réinitialise l'état du moteur"""
        self.capital = self.initial_capital
        self.position = None
        self.trades = []
        self.equity_curve = []
        
    def open_position(self, date: datetime, price: float,
                     position_type: str, size: float = None):
        """
        Ouvre une position

        Args:
            date: Date d'entrée
            price: Prix d'entrée
            position_type: 'LONG' ou 'SHORT'
            size: Taille de la position (si None, utilise position_size_pct du capital)
        """
        if self.position is not None:
            return  # Position déjà ouverte

        # Appliquer le slippage
        if position_type == 'LONG':
            entry_price = price * (1 + self.slippage)
        else:
            entry_price = price * (1 - self.slippage)

        # Calculer la taille si non spécifiée
        if size is None:
            # Utiliser position_size_pct du capital disponible
            capital_to_use = self.capital * (self.position_size_pct / 100.0)
            size = (capital_to_use * (1 - self.commission)) / entry_price

        # Calculer le capital investi (taille * prix + commission)
        capital_invested = size * entry_price
        commission_cost = capital_invested * self.commission
        total_invested = capital_invested + commission_cost

        # Déduire du capital disponible
        self.capital -= total_invested

        # Créer le trade
        self.position = Trade(
            entry_date=date,
            exit_date=None,
            entry_price=entry_price,
            exit_price=None,
            position_type=position_type,
            size=size,
            highest_price=entry_price if position_type == 'LONG' else None,
            capital_invested=capital_invested
        )
        
    def close_position(self, date: datetime, price: float, exit_reason: str = 'signal'):
        """Ferme la position actuelle"""
        if self.position is None:
            return

        # Appliquer le slippage
        if self.position.position_type == 'LONG':
            exit_price = price * (1 - self.slippage)
        else:
            exit_price = price * (1 + self.slippage)

        # Clôturer le trade
        self.position.close_trade(date, exit_price, exit_reason)

        # Calculer les proceeds
        if self.position.position_type == 'LONG':
            proceeds = self.position.size * exit_price
        else:
            # Pour un short, on récupère la différence
            proceeds = self.position.size * (2 * self.position.entry_price - exit_price)

        # Déduire la commission de sortie
        commission_cost = proceeds * self.commission
        net_proceeds = proceeds - commission_cost

        # Ajouter les proceeds au capital (et non remplacer)
        self.capital += net_proceeds

        # Enregistrer le trade
        self.trades.append(self.position)
        self.position = None
        
    def update_equity(self, date: datetime, current_price: float):
        """Met à jour la courbe d'équité"""
        if self.position is None:
            equity = self.capital
        else:
            # Calculer la valeur actuelle de la position
            if self.position.position_type == 'LONG':
                position_value = self.position.size * current_price
            else:
                # Pour SHORT: valeur = capital_investi + (entry_price - current_price) * size
                position_value = self.position.capital_invested + (self.position.entry_price - current_price) * self.position.size

            # Equity totale = cash restant + valeur de la position
            equity = self.capital + position_value

        self.equity_curve.append({
            'date': date,
            'equity': equity
        })
        
    def check_stop_loss_take_profit(self, date: datetime, high: float, low: float) -> bool:
        """
        Vérifie si le stop-loss, take-profit ou trailing stop est atteint

        Args:
            date: Date actuelle
            high: Prix haut de la période
            low: Prix bas de la période

        Returns:
            True si la position a été fermée, False sinon
        """
        if self.position is None:
            return False

        entry_price = self.position.entry_price
        position_type = self.position.position_type

        # Pour position LONG
        if position_type == 'LONG':
            # Mettre à jour le highest_price pour trailing stop
            if self.trailing_stop_pct is not None:
                if self.position.highest_price is None or high > self.position.highest_price:
                    self.position.highest_price = high

            # Vérifier Stop Loss
            if self.stop_loss_pct is not None:
                stop_loss_price = entry_price * (1 - self.stop_loss_pct / 100.0)
                if low <= stop_loss_price:
                    self.close_position(date, stop_loss_price, 'stop_loss')
                    return True

            # Vérifier Trailing Stop
            if self.trailing_stop_pct is not None and self.position.highest_price is not None:
                trailing_stop_price = self.position.highest_price * (1 - self.trailing_stop_pct / 100.0)
                if low <= trailing_stop_price:
                    self.close_position(date, trailing_stop_price, 'trailing_stop')
                    return True

            # Vérifier Take Profit
            if self.take_profit_pct is not None:
                take_profit_price = entry_price * (1 + self.take_profit_pct / 100.0)
                if high >= take_profit_price:
                    self.close_position(date, take_profit_price, 'take_profit')
                    return True

        # Pour position SHORT
        else:
            # Vérifier Stop Loss (prix monte)
            if self.stop_loss_pct is not None:
                stop_loss_price = entry_price * (1 + self.stop_loss_pct / 100.0)
                if high >= stop_loss_price:
                    self.close_position(date, stop_loss_price, 'stop_loss')
                    return True

            # Vérifier Take Profit (prix baisse)
            if self.take_profit_pct is not None:
                take_profit_price = entry_price * (1 - self.take_profit_pct / 100.0)
                if low <= take_profit_price:
                    self.close_position(date, take_profit_price, 'take_profit')
                    return True

        return False

    def run_backtest(self, data: pd.DataFrame, strategy):
        """
        Exécute le backtest avec une stratégie donnée

        Args:
            data: DataFrame avec colonnes OHLCV
            strategy: Objet stratégie avec méthode generate_signals()

        Returns:
            DataFrame avec résultats
        """
        self.reset()

        # Générer les signaux de trading
        signals = strategy.generate_signals(data)

        # Parcourir les données
        for i in range(len(data)):
            date = data.index[i]
            row = data.iloc[i]
            signal = signals.iloc[i]

            # Vérifier SL/TP/Trailing Stop en premier (si position ouverte)
            if self.position is not None:
                closed = self.check_stop_loss_take_profit(date, row['high'], row['low'])
                if closed:
                    # Position fermée par SL/TP, continuer
                    pass

            # Gérer les signaux (seulement si pas de position)
            if self.position is None:
                if signal['signal'] == 1:
                    # Signal d'achat
                    self.open_position(date, row['close'], 'LONG')
                elif signal['signal'] == -1:
                    # Signal de vente à découvert
                    self.open_position(date, row['close'], 'SHORT')

            # Signal de sortie
            elif signal['signal'] == 0 and self.position is not None:
                self.close_position(date, row['close'], 'signal')

            # Mettre à jour l'équité
            self.update_equity(date, row['close'])

        # Clôturer la position finale si elle existe
        if self.position is not None:
            last_date = data.index[-1]
            last_price = data.iloc[-1]['close']
            self.close_position(last_date, last_price, 'end_of_data')

        return self.get_results()
    
    def get_results(self) -> Dict:
        """Calcule et retourne les métriques de performance"""
        if not self.trades:
            return {
                'total_trades': 0,
                'final_capital': self.capital,
                'total_return': 0,
                'total_return_pct': 0
            }
        
        # DataFrame des trades
        trades_df = pd.DataFrame([{
            'entry_date': t.entry_date,
            'exit_date': t.exit_date,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'position_type': t.position_type,
            'pnl': t.pnl,
            'pnl_pct': t.pnl_pct,
            'exit_reason': t.exit_reason
        } for t in self.trades])
        
        # Equity curve DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('date', inplace=True)
        
        # Calculs de performance
        total_return = self.capital - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100
        
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] <= 0]
        
        win_rate = (len(winning_trades) / len(trades_df)) * 100 if len(trades_df) > 0 else 0
        
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        
        # Profit factor
        gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.inf
        
        # Drawdown
        equity_series = equity_df['equity']
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax * 100
        max_drawdown = drawdown.min()
        
        # Sharpe ratio (annualisé, approximation)
        returns = equity_series.pct_change().dropna()
        sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252)) if len(returns) > 0 and returns.std() != 0 else 0
        
        return {
            'total_trades': len(trades_df),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'initial_capital': self.initial_capital,
            'final_capital': self.capital,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'trades_df': trades_df,
            'equity_df': equity_df
        }
