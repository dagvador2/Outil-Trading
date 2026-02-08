"""
Exemple d'utilisation du framework de backtesting

Ce script montre comment:
1. Récupérer des données de marché
2. Tester différentes stratégies
3. Analyser les résultats
4. Visualiser les performances
"""

from backtesting_engine import BacktestEngine
from strategies import (MovingAverageCrossover, RSIStrategy, MACDStrategy, 
                       CombinedStrategy, BollingerBandsStrategy)
from data_fetcher import get_data
from visualizer import BacktestVisualizer
from indicators import TechnicalIndicators
import matplotlib.pyplot as plt


def run_single_backtest_example():
    """
    Exemple de backtest simple avec une stratégie
    """
    print("="*60)
    print("EXEMPLE 1: Backtest simple avec stratégie MA Crossover")
    print("="*60)
    
    # 1. Récupérer les données (on utilise des données synthétiques pour l'exemple)
    print("\n1. Récupération des données...")
    data = get_data('sample', 'DEMO', '2023-01-01', '2024-01-01', 
                   initial_price=100, trend=0.0003, volatility=0.02)
    print(f"   - {len(data)} jours de données chargés")
    print(f"   - Période: {data.index[0].date()} à {data.index[-1].date()}")
    
    # 2. Créer la stratégie
    print("\n2. Création de la stratégie...")
    strategy = MovingAverageCrossover(fast_period=20, slow_period=50)
    print(f"   - Stratégie: {strategy.name}")
    
    # 3. Initialiser le moteur de backtesting
    print("\n3. Configuration du backtest...")
    engine = BacktestEngine(
        initial_capital=10000,
        commission=0.001,  # 0.1%
        slippage=0.0005    # 0.05%
    )
    print(f"   - Capital initial: ${engine.initial_capital:,.2f}")
    print(f"   - Commission: {engine.commission*100}%")
    print(f"   - Slippage: {engine.slippage*100}%")
    
    # 4. Exécuter le backtest
    print("\n4. Exécution du backtest...")
    results = engine.run_backtest(data, strategy)
    
    # 5. Afficher les résultats
    print("\n" + "="*60)
    print("RÉSULTATS")
    print("="*60)
    print(f"\nCapital initial:      ${results['initial_capital']:,.2f}")
    print(f"Capital final:        ${results['final_capital']:,.2f}")
    print(f"Rendement total:      ${results['total_return']:,.2f} ({results['total_return_pct']:.2f}%)")
    print(f"\nNombre de trades:     {results['total_trades']}")
    print(f"Trades gagnants:      {results['winning_trades']}")
    print(f"Trades perdants:      {results['losing_trades']}")
    print(f"Win rate:            {results['win_rate']:.2f}%")
    print(f"\nGain moyen:          ${results['avg_win']:,.2f}")
    print(f"Perte moyenne:       ${results['avg_loss']:,.2f}")
    print(f"Profit factor:       {results['profit_factor']:.2f}")
    print(f"\nMax drawdown:        {results['max_drawdown']:.2f}%")
    print(f"Sharpe ratio:        {results['sharpe_ratio']:.2f}")
    
    # 6. Visualiser
    print("\n5. Génération des visualisations...")
    fig = BacktestVisualizer.plot_performance_summary(results)
    plt.savefig('/mnt/user-data/outputs/backtest_example.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   - Graphique sauvegardé: backtest_example.png")
    
    return results


def compare_strategies():
    """
    Compare plusieurs stratégies sur les mêmes données
    """
    print("\n\n" + "="*60)
    print("EXEMPLE 2: Comparaison de plusieurs stratégies")
    print("="*60)
    
    # Récupérer les données
    print("\n1. Récupération des données...")
    data = get_data('sample', 'DEMO', '2023-01-01', '2024-01-01',
                   initial_price=100, trend=0.0003, volatility=0.02)
    
    # Définir les stratégies à tester
    strategies = [
        MovingAverageCrossover(20, 50),
        RSIStrategy(14, 30, 70),
        MACDStrategy(),
        BollingerBandsStrategy(),
        CombinedStrategy()
    ]
    
    print(f"\n2. Test de {len(strategies)} stratégies...")
    
    # Tester chaque stratégie
    results_comparison = []
    
    for strategy in strategies:
        print(f"\n   Testing: {strategy.name}...")
        engine = BacktestEngine(initial_capital=10000, commission=0.001)
        results = engine.run_backtest(data, strategy)
        
        results_comparison.append({
            'Stratégie': strategy.name,
            'Rendement (%)': results['total_return_pct'],
            'Trades': results['total_trades'],
            'Win Rate (%)': results['win_rate'],
            'Profit Factor': results['profit_factor'],
            'Max DD (%)': results['max_drawdown'],
            'Sharpe': results['sharpe_ratio']
        })
    
    # Afficher le tableau comparatif
    import pandas as pd
    comparison_df = pd.DataFrame(results_comparison)
    comparison_df = comparison_df.sort_values('Rendement (%)', ascending=False)
    
    print("\n" + "="*60)
    print("TABLEAU COMPARATIF DES STRATÉGIES")
    print("="*60)
    print("\n" + comparison_df.to_string(index=False))
    
    # Sauvegarder le tableau
    comparison_df.to_csv('/mnt/user-data/outputs/strategies_comparison.csv', index=False)
    print("\n✓ Tableau sauvegardé: strategies_comparison.csv")
    
    return comparison_df


def backtest_with_real_data():
    """
    Exemple avec des données réelles (nécessite yfinance)
    """
    print("\n\n" + "="*60)
    print("EXEMPLE 3: Backtest avec données réelles")
    print("="*60)
    
    try:
        print("\n1. Tentative de récupération de données réelles (Apple)...")
        data = get_data('stock', 'AAPL', '2023-01-01', '2024-01-01')
        
        if data is not None and not data.empty:
            print(f"   ✓ {len(data)} jours de données AAPL chargés")
            
            # Tester une stratégie
            strategy = MovingAverageCrossover(20, 50)
            engine = BacktestEngine(initial_capital=10000)
            results = engine.run_backtest(data, strategy)
            
            print("\n" + "="*60)
            print(f"RÉSULTATS - {strategy.name} sur AAPL")
            print("="*60)
            print(f"\nRendement: {results['total_return_pct']:.2f}%")
            print(f"Trades: {results['total_trades']}")
            print(f"Win rate: {results['win_rate']:.2f}%")
            
            # Visualiser
            fig = BacktestVisualizer.plot_performance_summary(results)
            plt.savefig('/mnt/user-data/outputs/backtest_aapl.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("\n✓ Graphique sauvegardé: backtest_aapl.png")
            
        else:
            print("   ⚠ Impossible de récupérer les données.")
            print("   Installez yfinance: pip install yfinance")
            
    except Exception as e:
        print(f"   ⚠ Erreur: {e}")
        print("   Pour utiliser des données réelles, installez: pip install yfinance")


def analyze_indicators():
    """
    Analyse des indicateurs techniques sur des données
    """
    print("\n\n" + "="*60)
    print("EXEMPLE 4: Analyse des indicateurs techniques")
    print("="*60)
    
    # Générer des données
    print("\n1. Génération de données...")
    data = get_data('sample', 'DEMO', '2023-01-01', '2024-01-01')
    
    # Ajouter tous les indicateurs
    print("\n2. Calcul des indicateurs...")
    data_with_indicators = TechnicalIndicators.add_all_indicators(data)
    
    print("\n3. Indicateurs disponibles:")
    indicators = [col for col in data_with_indicators.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
    for i, ind in enumerate(indicators, 1):
        print(f"   {i}. {ind}")
    
    # Afficher un échantillon
    print("\n4. Échantillon de données (5 dernières lignes):")
    print(data_with_indicators[['close', 'sma_20', 'sma_50', 'rsi', 'macd']].tail())
    
    # Sauvegarder les données avec indicateurs
    data_with_indicators.to_csv('/mnt/user-data/outputs/data_with_indicators.csv')
    print("\n✓ Données avec indicateurs sauvegardées: data_with_indicators.csv")


def main():
    """
    Fonction principale qui exécute tous les exemples
    """
    print("\n" + "🚀 "*25)
    print("FRAMEWORK DE BACKTESTING - EXEMPLES")
    print("🚀 "*25 + "\n")
    
    # Exemple 1: Backtest simple
    run_single_backtest_example()
    
    # Exemple 2: Comparaison de stratégies
    compare_strategies()
    
    # Exemple 3: Données réelles
    backtest_with_real_data()
    
    # Exemple 4: Analyse d'indicateurs
    analyze_indicators()
    
    print("\n\n" + "✅ "*25)
    print("TOUS LES EXEMPLES TERMINÉS!")
    print("✅ "*25)
    print("\nFichiers générés dans /mnt/user-data/outputs/:")
    print("  - backtest_example.png")
    print("  - strategies_comparison.csv")
    print("  - backtest_aapl.png (si données disponibles)")
    print("  - data_with_indicators.csv")
    print("\n")


if __name__ == "__main__":
    main()
