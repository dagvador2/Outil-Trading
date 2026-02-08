"""
Visualisation des résultats de backtesting
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, Optional


class BacktestVisualizer:
    """
    Classe pour visualiser les résultats de backtesting
    """
    
    @staticmethod
    def plot_equity_curve(equity_df: pd.DataFrame, 
                         title: str = "Courbe d'équité",
                         figsize: tuple = (12, 6)):
        """
        Affiche la courbe d'équité
        
        Args:
            equity_df: DataFrame avec la courbe d'équité
            title: Titre du graphique
            figsize: Taille de la figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(equity_df.index, equity_df['equity'], linewidth=2, color='#2E86AB')
        ax.fill_between(equity_df.index, equity_df['equity'], alpha=0.3, color='#2E86AB')
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Équité ($)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_drawdown(equity_df: pd.DataFrame, 
                     title: str = "Drawdown",
                     figsize: tuple = (12, 4)):
        """
        Affiche la courbe de drawdown
        
        Args:
            equity_df: DataFrame avec la courbe d'équité
            title: Titre du graphique
            figsize: Taille de la figure
        """
        equity_series = equity_df['equity']
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax * 100
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
        ax.plot(drawdown.index, drawdown, linewidth=2, color='darkred')
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_trades(data: pd.DataFrame, 
                   trades_df: pd.DataFrame,
                   title: str = "Trades sur le graphique de prix",
                   figsize: tuple = (14, 7)):
        """
        Affiche les trades sur le graphique de prix
        
        Args:
            data: DataFrame OHLCV original
            trades_df: DataFrame des trades
            title: Titre du graphique
            figsize: Taille de la figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Prix de clôture
        ax.plot(data.index, data['close'], linewidth=1.5, color='black', label='Prix')
        
        # Trades longs
        long_trades = trades_df[trades_df['position_type'] == 'LONG']
        if not long_trades.empty:
            # Entrées
            ax.scatter(long_trades['entry_date'], long_trades['entry_price'],
                      marker='^', color='green', s=100, label='Achat (Long)', zorder=5)
            # Sorties
            ax.scatter(long_trades['exit_date'], long_trades['exit_price'],
                      marker='v', color='lightgreen', s=100, label='Vente (Long)', zorder=5)
        
        # Trades shorts
        short_trades = trades_df[trades_df['position_type'] == 'SHORT']
        if not short_trades.empty:
            # Entrées
            ax.scatter(short_trades['entry_date'], short_trades['entry_price'],
                      marker='v', color='red', s=100, label='Vente (Short)', zorder=5)
            # Sorties
            ax.scatter(short_trades['exit_date'], short_trades['exit_price'],
                      marker='^', color='lightcoral', s=100, label='Rachat (Short)', zorder=5)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Prix', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_returns_distribution(trades_df: pd.DataFrame,
                                  title: str = "Distribution des rendements",
                                  figsize: tuple = (10, 6)):
        """
        Affiche la distribution des rendements par trade
        
        Args:
            trades_df: DataFrame des trades
            title: Titre du graphique
            figsize: Taille de la figure
        """
        if trades_df.empty:
            print("Aucun trade à afficher")
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Histogramme des P&L en %
        ax1.hist(trades_df['pnl_pct'], bins=30, color='#2E86AB', alpha=0.7, edgecolor='black')
        ax1.axvline(0, color='red', linestyle='--', linewidth=2)
        ax1.set_title('Distribution des rendements (%)', fontsize=14)
        ax1.set_xlabel('Rendement (%)', fontsize=12)
        ax1.set_ylabel('Fréquence', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot([trades_df[trades_df['pnl'] > 0]['pnl_pct'],
                     trades_df[trades_df['pnl'] <= 0]['pnl_pct']],
                    labels=['Gains', 'Pertes'],
                    patch_artist=True,
                    boxprops=dict(facecolor='#2E86AB', alpha=0.7))
        ax2.set_title('Gains vs Pertes', fontsize=14)
        ax2.set_ylabel('Rendement (%)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_performance_summary(results: Dict, figsize: tuple = (14, 10)):
        """
        Affiche un dashboard complet des performances
        
        Args:
            results: Dictionnaire de résultats du backtest
            figsize: Taille de la figure
        """
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Courbe d'équité
        ax1 = fig.add_subplot(gs[0, :])
        equity_df = results['equity_df']
        ax1.plot(equity_df.index, equity_df['equity'], linewidth=2, color='#2E86AB')
        ax1.fill_between(equity_df.index, equity_df['equity'], alpha=0.3, color='#2E86AB')
        ax1.set_title("Courbe d'équité", fontsize=14, fontweight='bold')
        ax1.set_ylabel('Équité ($)', fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # 2. Drawdown
        ax2 = fig.add_subplot(gs[1, :])
        equity_series = equity_df['equity']
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax * 100
        ax2.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
        ax2.plot(drawdown.index, drawdown, linewidth=2, color='darkred')
        ax2.set_title("Drawdown", fontsize=14, fontweight='bold')
        ax2.set_ylabel('Drawdown (%)', fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # 3. Distribution des rendements
        ax3 = fig.add_subplot(gs[2, 0])
        if not results['trades_df'].empty:
            ax3.hist(results['trades_df']['pnl_pct'], bins=20, 
                    color='#2E86AB', alpha=0.7, edgecolor='black')
            ax3.axvline(0, color='red', linestyle='--', linewidth=2)
        ax3.set_title('Distribution des rendements', fontsize=14)
        ax3.set_xlabel('Rendement (%)', fontsize=11)
        ax3.set_ylabel('Fréquence', fontsize=11)
        ax3.grid(True, alpha=0.3)
        
        # 4. Métriques textuelles
        ax4 = fig.add_subplot(gs[2, 1])
        ax4.axis('off')
        
        metrics_text = f"""
        RÉSULTATS DU BACKTEST
        {'='*35}
        
        Capital initial:      ${results['initial_capital']:,.2f}
        Capital final:        ${results['final_capital']:,.2f}
        Rendement total:      {results['total_return_pct']:.2f}%
        
        Nombre de trades:     {results['total_trades']}
        Trades gagnants:      {results['winning_trades']}
        Trades perdants:      {results['losing_trades']}
        Win rate:            {results['win_rate']:.2f}%
        
        Gain moyen:          ${results['avg_win']:.2f}
        Perte moyenne:       ${results['avg_loss']:.2f}
        Profit factor:       {results['profit_factor']:.2f}
        
        Max drawdown:        {results['max_drawdown']:.2f}%
        Sharpe ratio:        {results['sharpe_ratio']:.2f}
        """
        
        ax4.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
                verticalalignment='center')
        
        plt.suptitle('Dashboard de Performance', fontsize=16, fontweight='bold', y=0.995)
        
        return fig
    
    @staticmethod
    def save_plots(results: Dict, prefix: str = 'backtest'):
        """
        Sauvegarde tous les graphiques
        
        Args:
            results: Dictionnaire de résultats
            prefix: Préfixe pour les noms de fichiers
        """
        # Performance summary
        fig1 = BacktestVisualizer.plot_performance_summary(results)
        fig1.savefig(f'{prefix}_summary.png', dpi=300, bbox_inches='tight')
        plt.close(fig1)
        
        print(f"Graphiques sauvegardés avec le préfixe: {prefix}")
