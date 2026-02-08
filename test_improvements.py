"""
Script de test pour les nouvelles fonctionnalités
- Stop-Loss & Take-Profit
- Position Sizing
"""

from backtesting_engine import BacktestEngine
from strategies import MovingAverageCrossover, RSIStrategy
from data_fetcher import get_data
import pandas as pd

print("=" * 80)
print("TEST DES NOUVELLES FONCTIONNALITÉS")
print("=" * 80)

# Récupérer des données de test
print("\n📊 Récupération des données de test...")
data = get_data('sample', 'DEMO', '2024-01-01', '2025-01-01')
print(f"✓ Données récupérées: {len(data)} jours")

# Stratégie de test
strategy = MovingAverageCrossover(fast_period=20, slow_period=50)

# ============================================================================
# TEST 1: Sans Stop-Loss ni Take-Profit (comportement original)
# ============================================================================
print("\n" + "=" * 80)
print("TEST 1: Sans SL/TP (comportement original)")
print("=" * 80)

engine1 = BacktestEngine(
    initial_capital=10000,
    commission=0.001,
    slippage=0.0005
)

results1 = engine1.run_backtest(data, strategy)

print(f"\n📈 Résultats:")
print(f"  Capital initial:     ${results1['initial_capital']:,.2f}")
print(f"  Capital final:       ${results1['final_capital']:,.2f}")
print(f"  Rendement:           {results1['total_return_pct']:.2f}%")
print(f"  Nombre de trades:    {results1['total_trades']}")
print(f"  Win rate:            {results1['win_rate']:.2f}%")

if results1['total_trades'] > 0:
    print(f"\n  Raisons de sortie:")
    exit_reasons = results1['trades_df']['exit_reason'].value_counts()
    for reason, count in exit_reasons.items():
        print(f"    - {reason}: {count}")

# ============================================================================
# TEST 2: Avec Stop-Loss 2% et Take-Profit 5%
# ============================================================================
print("\n" + "=" * 80)
print("TEST 2: Avec SL 2% et TP 5%")
print("=" * 80)

engine2 = BacktestEngine(
    initial_capital=10000,
    commission=0.001,
    slippage=0.0005,
    stop_loss_pct=2.0,      # Stop-loss à 2%
    take_profit_pct=5.0     # Take-profit à 5%
)

results2 = engine2.run_backtest(data, strategy)

print(f"\n📈 Résultats:")
print(f"  Capital initial:     ${results2['initial_capital']:,.2f}")
print(f"  Capital final:       ${results2['final_capital']:,.2f}")
print(f"  Rendement:           {results2['total_return_pct']:.2f}%")
print(f"  Nombre de trades:    {results2['total_trades']}")
print(f"  Win rate:            {results2['win_rate']:.2f}%")

if results2['total_trades'] > 0:
    print(f"\n  Raisons de sortie:")
    exit_reasons = results2['trades_df']['exit_reason'].value_counts()
    for reason, count in exit_reasons.items():
        print(f"    - {reason}: {count}")

# ============================================================================
# TEST 3: Avec Trailing Stop 3%
# ============================================================================
print("\n" + "=" * 80)
print("TEST 3: Avec Trailing Stop 3%")
print("=" * 80)

engine3 = BacktestEngine(
    initial_capital=10000,
    commission=0.001,
    slippage=0.0005,
    trailing_stop_pct=3.0   # Trailing stop à 3%
)

results3 = engine3.run_backtest(data, strategy)

print(f"\n📈 Résultats:")
print(f"  Capital initial:     ${results3['initial_capital']:,.2f}")
print(f"  Capital final:       ${results3['final_capital']:,.2f}")
print(f"  Rendement:           {results3['total_return_pct']:.2f}%")
print(f"  Nombre de trades:    {results3['total_trades']}")
print(f"  Win rate:            {results3['win_rate']:.2f}%")

if results3['total_trades'] > 0:
    print(f"\n  Raisons de sortie:")
    exit_reasons = results3['trades_df']['exit_reason'].value_counts()
    for reason, count in exit_reasons.items():
        print(f"    - {reason}: {count}")

# ============================================================================
# TEST 4: Avec Position Sizing à 50% du capital
# ============================================================================
print("\n" + "=" * 80)
print("TEST 4: Position Sizing à 50% (gestion du risque)")
print("=" * 80)

engine4 = BacktestEngine(
    initial_capital=10000,
    commission=0.001,
    slippage=0.0005,
    stop_loss_pct=2.0,
    take_profit_pct=5.0,
    position_size_pct=50.0  # Utiliser seulement 50% du capital par trade
)

results4 = engine4.run_backtest(data, strategy)

print(f"\n📈 Résultats:")
print(f"  Capital initial:     ${results4['initial_capital']:,.2f}")
print(f"  Capital final:       ${results4['final_capital']:,.2f}")
print(f"  Rendement:           {results4['total_return_pct']:.2f}%")
print(f"  Nombre de trades:    {results4['total_trades']}")
print(f"  Win rate:            {results4['win_rate']:.2f}%")
print(f"  Max drawdown:        {results4['max_drawdown']:.2f}%")

if results4['total_trades'] > 0:
    print(f"\n  Raisons de sortie:")
    exit_reasons = results4['trades_df']['exit_reason'].value_counts()
    for reason, count in exit_reasons.items():
        print(f"    - {reason}: {count}")

# ============================================================================
# COMPARAISON
# ============================================================================
print("\n" + "=" * 80)
print("📊 COMPARAISON DES CONFIGURATIONS")
print("=" * 80)

comparison = pd.DataFrame({
    'Configuration': [
        'Sans SL/TP',
        'SL 2% + TP 5%',
        'Trailing Stop 3%',
        'SL/TP + 50% capital'
    ],
    'Rendement (%)': [
        results1['total_return_pct'],
        results2['total_return_pct'],
        results3['total_return_pct'],
        results4['total_return_pct']
    ],
    'Trades': [
        results1['total_trades'],
        results2['total_trades'],
        results3['total_trades'],
        results4['total_trades']
    ],
    'Win Rate (%)': [
        results1['win_rate'],
        results2['win_rate'],
        results3['win_rate'],
        results4['win_rate']
    ],
    'Max DD (%)': [
        results1['max_drawdown'],
        results2['max_drawdown'],
        results3['max_drawdown'],
        results4['max_drawdown']
    ]
})

print("\n" + comparison.to_string(index=False))

print("\n" + "=" * 80)
print("✅ Tests terminés avec succès!")
print("=" * 80)
