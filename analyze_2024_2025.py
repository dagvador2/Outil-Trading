"""
ANALYSE COMPLÈTE 2024-2025 - DONNÉES RÉELLES
Objectif: Identifier les meilleures stratégies pour application en 2026
"""

from multi_asset_backtester import MultiAssetBacktester, print_summary_report
import pandas as pd
from datetime import datetime

print("="*80)
print("🎯 ANALYSE COMPLÈTE 2024-2025 - DONNÉES RÉELLES")
print("="*80)
print(f"Date de l'analyse: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()
print("Objectif: Identifier les meilleures stratégies de trading pour 2026")
print("Période analysée: 2024-01-01 → 2025-12-31")
print("="*80)

# Configuration du moteur (paramètres réalistes)
engine_config = {
    'initial_capital': 10000,      # 10k€ de capital
    'commission': 0.001,            # 0.1% de commission
    'slippage': 0.0005,             # 0.05% de slippage
    'stop_loss_pct': 3.0,           # Stop-loss à 3%
    'take_profit_pct': 6.0,         # Take-profit à 6%
    'position_size_pct': 100.0      # 100% du capital par trade
}

print("\n⚙️  CONFIGURATION DU BACKTESTING")
print("-"*80)
print(f"Capital initial:     {engine_config['initial_capital']:,}€")
print(f"Commission:          {engine_config['commission']*100:.2f}%")
print(f"Slippage:            {engine_config['slippage']*100:.3f}%")
print(f"Stop-Loss:           {engine_config['stop_loss_pct']:.1f}%")
print(f"Take-Profit:         {engine_config['take_profit_pct']:.1f}%")
print(f"Position Size:       {engine_config['position_size_pct']:.0f}% du capital")

# Créer le backtester
backtester = MultiAssetBacktester(
    start_date='2024-01-01',
    end_date='2025-12-31',
    engine_config=engine_config
)

# ============================================================================
# PHASE 1: TEST RAPIDE SUR UN SOUS-ENSEMBLE (pour vérifier que tout fonctionne)
# ============================================================================
print("\n" + "="*80)
print("📊 PHASE 1: TEST RAPIDE (Cryptos uniquement)")
print("="*80)
print("(Pour vérifier que les données sont accessibles)")

try:
    results_crypto = backtester.run_comprehensive_backtest(
        asset_categories=['crypto'],  # Seulement crypto
        strategy_names=['MA_Crossover_20_50', 'RSI_14_35_80', 'Combined'],
        use_sample=False  # VRAIES DONNÉES
    )

    if len(results_crypto) > 0:
        print("\n✅ Données crypto récupérées avec succès!")
        print(f"   Nombre de résultats: {len(results_crypto)}")

        # Meilleur résultat crypto
        best_crypto = results_crypto.nlargest(1, 'total_return_pct').iloc[0]
        print(f"\n🏆 Meilleur setup crypto:")
        print(f"   {best_crypto['asset_name']} + {best_crypto['strategy_name']}")
        print(f"   Rendement: {best_crypto['total_return_pct']:+.2f}%")

        # Sauvegarder
        results_crypto.to_csv('results_2024_2025_CRYPTO.csv', index=False)
        print(f"\n💾 Résultats cryptos sauvegardés: results_2024_2025_CRYPTO.csv")
    else:
        print("\n⚠️  Aucune donnée crypto récupérée")

except Exception as e:
    print(f"\n❌ Erreur lors de la récupération des données crypto: {e}")
    print("   → Continuons avec les autres catégories...")

# ============================================================================
# PHASE 2: ANALYSE COMPLÈTE (TOUS LES ACTIFS)
# ============================================================================
print("\n" + "="*80)
print("📊 PHASE 2: ANALYSE COMPLÈTE - TOUS LES ACTIFS")
print("="*80)
print("Cela peut prendre 5-10 minutes...")
print()

input("Appuyez sur ENTRÉE pour lancer l'analyse complète (ou Ctrl+C pour annuler)...")

try:
    # Lancer le backtest complet
    results_full = backtester.run_comprehensive_backtest(
        asset_categories=None,  # Toutes les catégories
        strategy_names=None,    # Toutes les stratégies
        use_sample=False        # VRAIES DONNÉES
    )

    if len(results_full) == 0:
        print("\n❌ Aucun résultat obtenu")
        exit(1)

    # ========================================================================
    # RAPPORT COMPLET
    # ========================================================================
    print_summary_report(results_full)

    # ========================================================================
    # ANALYSES SUPPLÉMENTAIRES
    # ========================================================================

    print("\n" + "="*80)
    print("💎 RECOMMANDATIONS POUR 2026")
    print("="*80)

    # Top 5 absolus
    top_5 = results_full.nlargest(5, 'total_return_pct')
    print("\n🥇 TOP 5 SETUPS ABSOLUS (Rendement maximal)")
    print("-"*80)
    for i, row in top_5.iterrows():
        print(f"{row['asset_name']:20} + {row['strategy_name']:25} → {row['total_return_pct']:+7.2f}%")

    # Top 5 par Sharpe (meilleur ratio rendement/risque)
    top_5_sharpe = results_full.nlargest(5, 'sharpe_ratio')
    print("\n📊 TOP 5 PAR SHARPE RATIO (Meilleur ratio rendement/risque)")
    print("-"*80)
    for i, row in top_5_sharpe.iterrows():
        print(f"{row['asset_name']:20} + {row['strategy_name']:25} → Sharpe: {row['sharpe_ratio']:+6.2f}")

    # Meilleure stratégie par catégorie d'actif
    print("\n🎯 MEILLEURE STRATÉGIE PAR CATÉGORIE D'ACTIF")
    print("-"*80)
    best_by_category = results_full.loc[results_full.groupby('asset_type')['total_return_pct'].idxmax()]
    for _, row in best_by_category.iterrows():
        print(f"{row['asset_type']:15} : {row['asset_name']:20} + {row['strategy_name']:25} → {row['total_return_pct']:+7.2f}%")

    # Stratégies les plus consistantes (meilleur taux de succès)
    print("\n🎲 STRATÉGIES LES PLUS CONSISTANTES")
    print("-"*80)
    strategy_consistency = results_full.groupby('strategy_name').agg({
        'total_return_pct': lambda x: (x > 0).sum() / len(x) * 100,  # Taux de succès
        'asset_name': 'count'  # Nombre de tests
    }).round(2)
    strategy_consistency.columns = ['Success_Rate_%', 'Tests']
    strategy_consistency = strategy_consistency.sort_values('Success_Rate_%', ascending=False)
    print(strategy_consistency.to_string())

    # ========================================================================
    # SAUVEGARDES
    # ========================================================================
    print("\n" + "="*80)
    print("💾 SAUVEGARDE DES RÉSULTATS")
    print("="*80)

    # CSV complet
    results_full.to_csv('RESULTS_2024_2025_COMPLETE.csv', index=False)
    print("✓ RESULTS_2024_2025_COMPLETE.csv - Tous les résultats")

    # Top 20
    top_20 = results_full.nlargest(20, 'total_return_pct')
    top_20.to_csv('RESULTS_2024_2025_TOP20.csv', index=False)
    print("✓ RESULTS_2024_2025_TOP20.csv - Top 20 meilleurs setups")

    # Résumé par stratégie
    strategy_summary = results_full.groupby('strategy_name').agg({
        'total_return_pct': ['mean', 'std', 'max', 'min'],
        'sharpe_ratio': 'mean',
        'win_rate': 'mean',
        'total_trades': 'sum'
    }).round(2)
    strategy_summary.to_csv('RESULTS_2024_2025_BY_STRATEGY.csv')
    print("✓ RESULTS_2024_2025_BY_STRATEGY.csv - Résumé par stratégie")

    # Résumé par actif
    asset_summary = results_full.groupby('asset_name').agg({
        'total_return_pct': ['mean', 'max'],
        'sharpe_ratio': 'mean',
        'strategy_name': lambda x: x[results_full.loc[x.index, 'total_return_pct'].idxmax()]
    }).round(2)
    asset_summary.to_csv('RESULTS_2024_2025_BY_ASSET.csv')
    print("✓ RESULTS_2024_2025_BY_ASSET.csv - Résumé par actif")

    # Matrice pivot
    pivot_matrix = results_full.pivot_table(
        values='total_return_pct',
        index='asset_name',
        columns='strategy_name',
        aggfunc='mean'
    ).round(2)
    pivot_matrix.to_csv('RESULTS_2024_2025_MATRIX.csv')
    print("✓ RESULTS_2024_2025_MATRIX.csv - Matrice Actif × Stratégie")

    print("\n" + "="*80)
    print("✅ ANALYSE COMPLÈTE TERMINÉE!")
    print("="*80)
    print("\n📊 Résultats disponibles:")
    print("   - RESULTS_2024_2025_COMPLETE.csv (tous les résultats)")
    print("   - RESULTS_2024_2025_TOP20.csv (20 meilleurs setups)")
    print("   - RESULTS_2024_2025_BY_STRATEGY.csv (performance par stratégie)")
    print("   - RESULTS_2024_2025_BY_ASSET.csv (performance par actif)")
    print("   - RESULTS_2024_2025_MATRIX.csv (matrice complète)")

    print("\n💡 PROCHAINE ÉTAPE:")
    print("   → Analyser les fichiers CSV pour identifier vos meilleures opportunités 2026")
    print("   → Valider avec Walk-Forward Analysis sur les top setups")
    print("   → Commencer le paper trading avec les meilleures stratégies")

except KeyboardInterrupt:
    print("\n\n⚠️  Analyse interrompue par l'utilisateur")

except Exception as e:
    print(f"\n❌ Erreur lors de l'analyse: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("Fin de l'analyse")
print("="*80)
