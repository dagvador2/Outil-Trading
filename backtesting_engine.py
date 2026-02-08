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
    
    def close_trade(self, exit_date: datetime, exit_price: float):
        """Clôture le trade et calcule le P&L"""
        self.exit_date = exit_date
        self.exit_price = exit_price
        
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
                 slippage: float = 0.0005):   # 0.05% de slippage
        """
        Args:
            initial_capital: Capital de départ
            commission: Commission par trade (en fraction, ex: 0.001 = 0.1%)
            slippage: Slippage moyen (en fraction)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
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
            size: Taille de la position (si None, utilise tout le capital disponible)
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
            size = (self.capital * (1 - self.commission)) / entry_price
        
        # Déduire la commission
        commission_cost = size * entry_price * self.commission
        self.capital -= commission_cost
        
        # Créer le trade
        self.position = Trade(
            entry_date=date,
            exit_date=None,
            entry_price=entry_price,
            exit_price=None,
            position_type=position_type,
            size=size
        )
        
    def close_position(self, date: datetime, price: float):
        """Ferme la position actuelle"""
        if self.position is None:
            return
            
        # Appliquer le slippage
        if self.position.position_type == 'LONG':
            exit_price = price * (1 - self.slippage)
        else:
            exit_price = price * (1 + self.slippage)
        
        # Clôturer le trade
        self.position.close_trade(date, exit_price)
        
        # Calculer le capital après clôture
        if self.position.position_type == 'LONG':
            proceeds = self.position.size * exit_price
        else:
            # Pour un short, on récupère la différence
            proceeds = self.position.size * (2 * self.position.entry_price - exit_price)
        
        commission_cost = proceeds * self.commission
        self.capital = proceeds - commission_cost
        
        # Enregistrer le trade
        self.trades.append(self.position)
        self.position = None
        
    def update_equity(self, date: datetime, current_price: float):
        """Met à jour la courbe d'équité"""
        if self.position is None:
            equity = self.capital
        else:
            # Calculer la valeur non réalisée
            if self.position.position_type == 'LONG':
                unrealized_pnl = (current_price - self.position.entry_price) * self.position.size
            else:
                unrealized_pnl = (self.position.entry_price - current_price) * self.position.size
            
            equity = self.capital + unrealized_pnl
        
        self.equity_curve.append({
            'date': date,
            'equity': equity
        })
        
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
            
            # Gérer les signaux
            if signal['signal'] == 1 and self.position is None:
                # Signal d'achat
                self.open_position(date, row['close'], 'LONG')
                
            elif signal['signal'] == -1 and self.position is None:
                # Signal de vente à découvert
                self.open_position(date, row['close'], 'SHORT')
                
            elif signal['signal'] == 0 and self.position is not None:
                # Signal de sortie
                self.close_position(date, row['close'])
            
            # Mettre à jour l'équité
            self.update_equity(date, row['close'])
        
        # Clôturer la position finale si elle existe
        if self.position is not None:
            last_date = data.index[-1]
            last_price = data.iloc[-1]['close']
            self.close_position(last_date, last_price)
        
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
            'pnl_pct': t.pnl_pct
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
