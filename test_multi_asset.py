"""
Test du système de backtesting multi-asset
"""

from multi_asset_backtester import MultiAssetBacktester, print_summary_report, ASSET_UNIVERSE
import pandas as pd

print("="*80)
print("TEST DU SYSTÈME MULTI-ASSET")
print("="*80)

# Configuration du moteur
engine_config = {
    'initial_capital': 10000,
    'commission': 0.001,
    'slippage': 0.0005,
    'stop_loss_pct': 3.0,      # SL à 3%
    'take_profit_pct': 6.0,     # TP à 6%
    'position_size_pct': 100.0  # 100% du capital
}

# Créer le backtester
backtester = MultiAssetBacktester(
    start_date='2024-01-01',
    end_date='2025-01-01',
    engine_config=engine_config
)

# ============================================================================
# TEST 1: Backtest sur données synthétiques (rapide)
# ============================================================================
print("\n" + "="*80)
print("TEST 1: BACKTEST SUR DONNÉES SYNTHÉTIQUES")
print("="*80)
print("(Utilisé pour tester rapidement sans dépendances externes)")

# Tester seulement quelques actifs et stratégies
results_sample = backtester.run_comprehensive_backtest(
    asset_categories=['stocks', 'crypto'],  # Seulement 2 catégories
    strategy_names=['MA_Crossover_20_50', 'RSI_14_30_70', 'Combined'],  # 3 stratégies
    use_sample=True  # Données synthétiques
)

# Afficher le rapport
print_summary_report(results_sample)

# Sauvegarder les résultats
results_sample.to_csv('multi_asset_results_sample.csv', index=False)
print("\n💾 Résultats sauvegardés dans: multi_asset_results_sample.csv")

# ============================================================================
# Affichage de la matrice Actif × Stratégie
# ============================================================================
print("\n" + "="*80)
print("🗂️  MATRICE RENDEMENT: ACTIF × STRATÉGIE")
print("="*80)

pivot = results_sample.pivot_table(
    values='total_return_pct',
    index='asset_name',
    columns='strategy_name',
    aggfunc='mean'
).round(2)

print(pivot.to_string())

print("\n" + "="*80)
print("✅ Test terminé!")
print("="*80)

print("\n💡 PROCHAINE ÉTAPE:")
print("   Pour tester avec des données RÉELLES (2024-2025), installer les dépendances:")
print("   pip install yfinance ccxt")
print("   Puis utiliser: use_sample=False")
