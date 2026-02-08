"""
GUIDE DE DÉMARRAGE RAPIDE
Script interactif pour apprendre à utiliser le framework
"""

from backtesting_engine import BacktestEngine
from strategies import *
from data_fetcher import get_data
from visualizer import BacktestVisualizer
from indicators import TechnicalIndicators
import matplotlib.pyplot as plt


def tutorial_step_1():
    """Étape 1: Récupérer des données"""
    print("\n" + "="*70)
    print("ÉTAPE 1: RÉCUPÉRER DES DONNÉES")
    print("="*70)
    
    print("""
Pour commencer, nous devons récupérer des données de marché.
Vous avez plusieurs options :

1. Données synthétiques (idéal pour tester) - Aucune installation requise
2. Actions réelles (nécessite yfinance)
3. Crypto (nécessite ccxt)
4. Forex (nécessite yfinance)

Pour ce tutoriel, nous utilisons des données synthétiques.
    """)
    
    print("Code:")
    print("-" * 70)
    code = """
from data_fetcher import get_data

# Générer des données synthétiques
data = get_data('sample', 'DEMO', '2023-01-01', '2024-01-01',
               initial_price=100, trend=0.0003, volatility=0.02)

print(f"Données chargées: {len(data)} jours")
print(data.head())
    """
    print(code)
    print("-" * 70)
    
    # Exécuter
    print("\nExécution...")
    data = get_data('sample', 'DEMO', '2023-01-01', '2024-01-01',
                   initial_price=100, trend=0.0003, volatility=0.02)
    
    print(f"\n✓ Données chargées: {len(data)} jours")
    print("\nAperçu des données:")
    print(data.head())
    
    return data


def tutorial_step_2(data):
    """Étape 2: Calculer des indicateurs"""
    print("\n\n" + "="*70)
    print("ÉTAPE 2: CALCULER DES INDICATEURS TECHNIQUES")
    print("="*70)
    
    print("""
Les indicateurs techniques nous aident à identifier des opportunités.
Le framework inclut les indicateurs les plus populaires :

- Moyennes mobiles (SMA, EMA)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Stochastique
- Bandes de Bollinger
- ATR (Average True Range)
    """)
    
    print("\nCode:")
    print("-" * 70)
    code = """
from indicators import TechnicalIndicators

# Calculer un RSI
rsi = TechnicalIndicators.rsi(data['close'], period=14)

# Ou ajouter TOUS les indicateurs d'un coup
data_with_indicators = TechnicalIndicators.add_all_indicators(data)

print(data_with_indicators[['close', 'sma_20', 'sma_50', 'rsi']].tail())
    """
    print(code)
    print("-" * 70)
    
    # Exécuter
    print("\nExécution...")
    data_with_indicators = TechnicalIndicators.add_all_indicators(data)
    
    print("\n✓ Indicateurs calculés!")
    print("\nDernières valeurs:")
    print(data_with_indicators[['close', 'sma_20', 'sma_50', 'rsi', 'macd']].tail())
    
    return data_with_indicators


def tutorial_step_3(data):
    """Étape 3: Créer une stratégie"""
    print("\n\n" + "="*70)
    print("ÉTAPE 3: CRÉER UNE STRATÉGIE DE TRADING")
    print("="*70)
    
    print("""
Une stratégie définit QUAND acheter et vendre.

Stratégies prédéfinies disponibles:
1. MovingAverageCrossover - Croisement de moyennes mobiles
2. RSIStrategy - Basée sur le RSI
3. MACDStrategy - Basée sur le MACD
4. BollingerBandsStrategy - Rebonds sur les bandes
5. CombinedStrategy - Combine plusieurs indicateurs

Utilisons la stratégie de croisement de moyennes mobiles:
- Acheter quand MA rapide > MA lente
- Vendre quand MA rapide < MA lente
    """)
    
    print("\nCode:")
    print("-" * 70)
    code = """
from strategies import MovingAverageCrossover

# Créer la stratégie avec moyennes 20 et 50 jours
strategy = MovingAverageCrossover(fast_period=20, slow_period=50)

print(f"Stratégie créée: {strategy.name}")
    """
    print(code)
    print("-" * 70)
    
    # Exécuter
    print("\nExécution...")
    strategy = MovingAverageCrossover(fast_period=20, slow_period=50)
    
    print(f"\n✓ Stratégie créée: {strategy.name}")
    
    return strategy


def tutorial_step_4(data, strategy):
    """Étape 4: Exécuter le backtest"""
    print("\n\n" + "="*70)
    print("ÉTAPE 4: EXÉCUTER LE BACKTEST")
    print("="*70)
    
    print("""
Le backtest simule comment votre stratégie aurait performé.
Le moteur gère automatiquement:
- Les entrées/sorties de positions
- Les commissions et le slippage
- Le calcul des métriques de performance
    """)
    
    print("\nCode:")
    print("-" * 70)
    code = """
from backtesting_engine import BacktestEngine

# Configurer le moteur
engine = BacktestEngine(
    initial_capital=10000,  # $10,000 de départ
    commission=0.001,       # 0.1% de commission
    slippage=0.0005        # 0.05% de slippage
)

# Exécuter le backtest
results = engine.run_backtest(data, strategy)

# Afficher les résultats
print(f"Rendement: {results['total_return_pct']:.2f}%")
print(f"Nombre de trades: {results['total_trades']}")
print(f"Win rate: {results['win_rate']:.2f}%")
    """
    print(code)
    print("-" * 70)
    
    # Exécuter
    print("\nExécution...")
    engine = BacktestEngine(
        initial_capital=10000,
        commission=0.001,
        slippage=0.0005
    )
    
    results = engine.run_backtest(data, strategy)
    
    print("\n" + "="*70)
    print("RÉSULTATS DU BACKTEST")
    print("="*70)
    print(f"\n💰 Performance:")
    print(f"   Capital initial:  ${results['initial_capital']:,.2f}")
    print(f"   Capital final:    ${results['final_capital']:,.2f}")
    print(f"   Rendement:        {results['total_return_pct']:.2f}%")
    
    print(f"\n📊 Statistiques:")
    print(f"   Trades totaux:    {results['total_trades']}")
    print(f"   Trades gagnants:  {results['winning_trades']}")
    print(f"   Trades perdants:  {results['losing_trades']}")
    print(f"   Win rate:         {results['win_rate']:.2f}%")
    
    print(f"\n📈 Métriques de risque:")
    print(f"   Max drawdown:     {results['max_drawdown']:.2f}%")
    print(f"   Sharpe ratio:     {results['sharpe_ratio']:.2f}")
    print(f"   Profit factor:    {results['profit_factor']:.2f}")
    
    return results


def tutorial_step_5(results):
    """Étape 5: Visualiser les résultats"""
    print("\n\n" + "="*70)
    print("ÉTAPE 5: VISUALISER LES RÉSULTATS")
    print("="*70)
    
    print("""
Les graphiques vous aident à comprendre la performance de votre stratégie:
- Courbe d'équité: Évolution de votre capital
- Drawdown: Périodes de pertes
- Distribution des rendements: Analyse des gains/pertes par trade
    """)
    
    print("\nCode:")
    print("-" * 70)
    code = """
from visualizer import BacktestVisualizer
import matplotlib.pyplot as plt

# Créer le dashboard complet
fig = BacktestVisualizer.plot_performance_summary(results)
plt.savefig('/mnt/user-data/outputs/tutorial_results.png', dpi=300, bbox_inches='tight')
plt.show()
    """
    print(code)
    print("-" * 70)
    
    # Exécuter
    print("\nGénération des graphiques...")
    fig = BacktestVisualizer.plot_performance_summary(results)
    plt.savefig('/mnt/user-data/outputs/tutorial_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n✓ Graphique sauvegardé: tutorial_results.png")


def tutorial_step_6():
    """Étape 6: Prochaines étapes"""
    print("\n\n" + "="*70)
    print("ÉTAPE 6: ET MAINTENANT ?")
    print("="*70)
    
    print("""
🎉 Félicitations ! Vous savez maintenant utiliser le framework de backtesting.

Voici ce que vous pouvez faire ensuite:

1️⃣  TESTER DIFFÉRENTES STRATÉGIES
    Essayez RSIStrategy, MACDStrategy, ou CombinedStrategy

2️⃣  OPTIMISER LES PARAMÈTRES
    Testez différentes périodes pour vos moyennes mobiles
    Exemple: (10,30), (20,50), (50,200)

3️⃣  CRÉER VOTRE PROPRE STRATÉGIE
    Héritez de BaseStrategy et implémentez votre logique

4️⃣  UTILISER DES DONNÉES RÉELLES
    Installez yfinance: pip install yfinance
    Puis: data = get_data('stock', 'AAPL', '2023-01-01', '2024-01-01')

5️⃣  COMPARER PLUSIEURS STRATÉGIES
    Exécutez plusieurs backtests et comparez les résultats

6️⃣  ANALYSER EN PROFONDEUR
    Regardez les trades individuels, les patterns de gains/pertes

📚 RESSOURCES:
    - Consultez le README.md pour la documentation complète
    - Regardez example_usage.py pour plus d'exemples
    - Les fichiers sont bien commentés pour faciliter la compréhension

⚠️  RAPPEL IMPORTANT:
    Les performances passées ne garantissent pas les résultats futurs.
    Utilisez cet outil pour apprendre et tester, pas comme seule
    base de décision pour vos investissements réels.

🚀 Bon trading !
    """)


def run_complete_tutorial():
    """Exécute le tutoriel complet"""
    print("\n" + "🎓 "*30)
    print("TUTORIEL COMPLET - FRAMEWORK DE BACKTESTING")
    print("🎓 "*30)
    
    print("""
Bienvenue ! Ce tutoriel vous guidera à travers toutes les étapes
pour utiliser le framework de backtesting.

Durée estimée: 5-10 minutes
    """)
    
    input("Appuyez sur Entrée pour commencer...")
    
    # Étape 1
    data = tutorial_step_1()
    input("\nAppuyez sur Entrée pour continuer...")
    
    # Étape 2
    data_with_ind = tutorial_step_2(data)
    input("\nAppuyez sur Entrée pour continuer...")
    
    # Étape 3
    strategy = tutorial_step_3(data)
    input("\nAppuyez sur Entrée pour continuer...")
    
    # Étape 4
    results = tutorial_step_4(data, strategy)
    input("\nAppuyez sur Entrée pour continuer...")
    
    # Étape 5
    tutorial_step_5(results)
    input("\nAppuyez sur Entrée pour continuer...")
    
    # Étape 6
    tutorial_step_6()
    
    print("\n" + "✅ "*30)
    print("TUTORIEL TERMINÉ !")
    print("✅ "*30 + "\n")


if __name__ == "__main__":
    # Pour un tutoriel interactif, décommentez:
    # run_complete_tutorial()
    
    # Pour une démo rapide non-interactive:
    print("\nExécution de la démo rapide...\n")
    data = tutorial_step_1()
    data_with_ind = tutorial_step_2(data)
    strategy = tutorial_step_3(data)
    results = tutorial_step_4(data, strategy)
    tutorial_step_5(results)
    tutorial_step_6()
