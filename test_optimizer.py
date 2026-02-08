"""
Test du module d'optimisation
"""

from optimizer import GridSearchOptimizer, WalkForwardOptimizer
from strategies import MovingAverageCrossover, RSIStrategy, MACDStrategy
from data_fetcher import get_data
import pandas as pd

print("="*80)
print("TEST DU MODULE D'OPTIMISATION")
print("="*80)

# Récupérer des données
print("\n📊 Récupération des données (2 ans)...")
data = get_data('sample', 'DEMO', '2023-01-01', '2025-01-01',
                trend=0.0003, volatility=0.02)  # Tendance légèrement positive
print(f"✓ Données: {len(data)} jours")

# Configuration du moteur (avec SL/TP pour plus de réalisme)
engine_config = {
    'initial_capital': 10000,
    'commission': 0.001,
    'slippage': 0.0005,
    'stop_loss_pct': 3.0,       # SL à 3%
    'take_profit_pct': 6.0,      # TP à 6%
    'position_size_pct': 100.0   # 100% du capital
}

# ============================================================================
# TEST 1: Grid Search - Optimisation MovingAverageCrossover
# ============================================================================
print("\n" + "="*80)
print("TEST 1: GRID SEARCH - MovingAverageCrossover")
print("="*80)

optimizer = GridSearchOptimizer(engine_config)

param_grid = {
    'fast_period': [10, 15, 20, 25, 30],
    'slow_period': [40, 50, 60, 70, 80, 100]
}

results = optimizer.optimize(
    data,
    MovingAverageCrossover,
    param_grid,
    metric='total_return_pct',
    top_n=5
)

print("\n📊 Top 5 configurations:")
print(results[['fast_period', 'slow_period', 'total_return_pct',
               'sharpe_ratio', 'max_drawdown', 'win_rate']].to_string(index=False))

# ============================================================================
# TEST 2: Grid Search - Optimisation RSI
# ============================================================================
print("\n" + "="*80)
print("TEST 2: GRID SEARCH - RSIStrategy")
print("="*80)

param_grid_rsi = {
    'period': [10, 12, 14, 16, 18, 20],
    'oversold': [20, 25, 30, 35],
    'overbought': [65, 70, 75, 80]
}

results_rsi = optimizer.optimize(
    data,
    RSIStrategy,
    param_grid_rsi,
    metric='sharpe_ratio',  # Optimiser pour le Sharpe ratio
    top_n=5
)

print("\n📊 Top 5 configurations:")
print(results_rsi[['period', 'oversold', 'overbought', 'total_return_pct',
                   'sharpe_ratio', 'max_drawdown', 'win_rate']].to_string(index=False))

# ============================================================================
# TEST 3: Walk-Forward Analysis
# ============================================================================
print("\n" + "="*80)
print("TEST 3: WALK-FORWARD ANALYSIS")
print("="*80)

wf_optimizer = WalkForwardOptimizer(engine_config)

# Paramètres à tester (grid plus petit pour walk-forward)
param_grid_wf = {
    'fast_period': [15, 20, 25],
    'slow_period': [50, 60, 70]
}

wf_results = wf_optimizer.walk_forward_analysis(
    data,
    MovingAverageCrossover,
    param_grid_wf,
    train_period_days=250,  # ~1 an d'entraînement
    test_period_days=60,    # ~2 mois de test
    metric='sharpe_ratio'
)

# Afficher le résumé
summary = wf_results['summary']
print("\n" + "="*80)
print("RÉSUMÉ WALK-FORWARD")
print("="*80)
print(f"Taux de profitabilité: {summary.get('profitability_rate', 0):.1f}%")
print(f"Rendement moyen (out-sample): {summary.get('avg_out_sample_return', 0):.2f}%")
print(f"Meilleure période: {summary.get('best_period_return', 0):.2f}%")
print(f"Pire période: {summary.get('worst_period_return', 0):.2f}%")

# Afficher les résultats par période
if len(wf_results['results_df']) > 0:
    print("\n📊 Résultats par période:")
    display_cols = ['period', 'test_return', 'test_sharpe', 'test_max_dd',
                   'test_win_rate', 'test_trades']
    available_cols = [col for col in display_cols if col in wf_results['results_df'].columns]
    print(wf_results['results_df'][available_cols].to_string(index=False))

print("\n" + "="*80)
print("✅ Tests de l'optimiseur terminés avec succès!")
print("="*80)

# Sauvegarder les résultats
results.to_csv('optimization_results_ma.csv', index=False)
results_rsi.to_csv('optimization_results_rsi.csv', index=False)
if len(wf_results['results_df']) > 0:
    wf_results['results_df'].to_csv('walkforward_results.csv', index=False)

print("\n💾 Résultats sauvegardés:")
print("  - optimization_results_ma.csv")
print("  - optimization_results_rsi.csv")
print("  - walkforward_results.csv")
