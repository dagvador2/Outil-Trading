# 📈 Framework de Backtesting pour Trading

Un outil complet de backtesting pour tester vos stratégies de trading sur données historiques.

## 🎯 Fonctionnalités

- ✅ **Multi-marchés** : Actions, Crypto, Forex, Indices, Matières premières
- ✅ **Indicateurs techniques** : SMA, EMA, RSI, MACD, Stochastique, Bollinger Bands, ATR
- ✅ **Stratégies prédéfinies** : 5+ stratégies ready-to-use
- ✅ **Métriques complètes** : Win rate, Profit factor, Sharpe ratio, Max drawdown
- ✅ **Visualisations** : Graphiques professionnels pour analyser les performances
- ✅ **Gestion des coûts** : Commission et slippage intégrés
- ✅ **Extensible** : Facile de créer vos propres stratégies

## 📦 Installation

### Prérequis
```bash
pip install pandas numpy matplotlib
```

### Optionnel (pour données réelles)
```bash
# Pour les actions
pip install yfinance

# Pour les cryptos
pip install ccxt
```

## 🚀 Démarrage rapide

### 1. Backtest simple

```python
from backtesting_engine import BacktestEngine
from strategies import MovingAverageCrossover
from data_fetcher import get_data

# Récupérer des données
data = get_data('sample', 'DEMO', '2023-01-01', '2024-01-01')

# Créer la stratégie
strategy = MovingAverageCrossover(fast_period=20, slow_period=50)

# Exécuter le backtest
engine = BacktestEngine(initial_capital=10000)
results = engine.run_backtest(data, strategy)

# Afficher les résultats
print(f"Rendement: {results['total_return_pct']:.2f}%")
print(f"Win rate: {results['win_rate']:.2f}%")
```

### 2. Avec données réelles (Actions)

```python
from data_fetcher import get_data

# Récupérer données Apple
data = get_data('stock', 'AAPL', '2023-01-01', '2024-01-01')

# Tester votre stratégie...
```

### 3. Comparer plusieurs stratégies

```python
from strategies import *

strategies = [
    MovingAverageCrossover(20, 50),
    RSIStrategy(14, 30, 70),
    MACDStrategy(),
    CombinedStrategy()
]

for strategy in strategies:
    engine = BacktestEngine(initial_capital=10000)
    results = engine.run_backtest(data, strategy)
    print(f"{strategy.name}: {results['total_return_pct']:.2f}%")
```

## 📊 Structure du projet

```
trading-backtest/
├── backtesting_engine.py   # Moteur principal de backtesting
├── strategies.py           # Stratégies de trading
├── indicators.py           # Indicateurs techniques
├── data_fetcher.py         # Récupération de données
├── visualizer.py           # Visualisations
├── example_usage.py        # Exemples d'utilisation
└── README.md              # Ce fichier
```

## 🎨 Stratégies disponibles

### 1. MovingAverageCrossover
Croisement de moyennes mobiles (SMA ou EMA).

```python
strategy = MovingAverageCrossover(fast_period=20, slow_period=50)
```

### 2. RSIStrategy
Basée sur les niveaux de surachat/survente du RSI.

```python
strategy = RSIStrategy(period=14, oversold=30, overbought=70)
```

### 3. MACDStrategy
Croisement MACD / ligne de signal.

```python
strategy = MACDStrategy(fast=12, slow=26, signal=9)
```

### 4. BollingerBandsStrategy
Rebond sur les bandes de Bollinger.

```python
strategy = BollingerBandsStrategy(period=20, num_std=2)
```

### 5. CombinedStrategy
Combinaison de plusieurs indicateurs (MA + RSI + MACD).

```python
strategy = CombinedStrategy()
```

## 🔧 Créer votre propre stratégie

```python
from strategies import BaseStrategy
import pandas as pd

class MaStrategiePersonnalisee(BaseStrategy):
    def __init__(self, param1, param2):
        super().__init__("Ma Stratégie")
        self.param1 = param1
        self.param2 = param2
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        # Votre logique ici
        # signals['signal'] = 1  -> Acheter (LONG)
        # signals['signal'] = -1 -> Vendre (SHORT)
        # signals['signal'] = 0  -> Sortir de position
        
        return signals
```

## 📈 Métriques calculées

Le backtesting calcule automatiquement :

- **Rendement total** : % et $ de profit/perte
- **Nombre de trades** : Total, gagnants, perdants
- **Win rate** : % de trades gagnants
- **Profit factor** : Ratio gains/pertes
- **Gain/Perte moyens** : Par trade
- **Max drawdown** : Plus grande baisse depuis un sommet
- **Sharpe ratio** : Ratio rendement/risque annualisé

## 📊 Visualisations

```python
from visualizer import BacktestVisualizer

# Dashboard complet
fig = BacktestVisualizer.plot_performance_summary(results)
plt.show()

# Courbe d'équité seule
fig = BacktestVisualizer.plot_equity_curve(results['equity_df'])
plt.show()

# Trades sur le graphique
fig = BacktestVisualizer.plot_trades(data, results['trades_df'])
plt.show()
```

## 💾 Sources de données

### Données synthétiques (pour tester)
```python
data = get_data('sample', 'DEMO', '2023-01-01', '2024-01-01')
```

### Actions (via yfinance)
```python
data = get_data('stock', 'AAPL', '2023-01-01', '2024-01-01')
data = get_data('stock', 'TSLA', '2023-01-01', '2024-01-01', interval='1h')
```

### Cryptomonnaies (via ccxt)
```python
data = get_data('crypto', 'BTC/USDT', '2023-01-01', '2024-01-01',
                exchange='binance', timeframe='1d')
```

### Forex
```python
data = get_data('forex', 'EURUSD', '2023-01-01', '2024-01-01')
```

## ⚙️ Configuration du moteur

```python
engine = BacktestEngine(
    initial_capital=10000,    # Capital de départ
    commission=0.001,         # 0.1% par transaction
    slippage=0.0005          # 0.05% de slippage
)
```

## 📝 Indicateurs disponibles

### Moyennes mobiles
- `sma()` - Simple Moving Average
- `ema()` - Exponential Moving Average

### Oscillateurs
- `rsi()` - Relative Strength Index
- `stochastic()` - Stochastic Oscillator

### Tendance
- `macd()` - Moving Average Convergence Divergence
- `bollinger_bands()` - Bandes de Bollinger

### Volatilité
- `atr()` - Average True Range

### Tout en un
```python
from indicators import TechnicalIndicators

# Ajouter tous les indicateurs au DataFrame
data_with_indicators = TechnicalIndicators.add_all_indicators(data)
```

## 🎯 Cas d'usage

### 1. Tester une idée de stratégie
Avant de risquer de l'argent réel, testez votre idée sur des données historiques.

### 2. Optimiser les paramètres
Trouvez les meilleurs paramètres pour vos indicateurs (périodes, seuils).

### 3. Comparer différentes approches
Quel indicateur fonctionne le mieux sur votre marché ?

### 4. Analyser la performance
Comprendre les forces et faiblesses de votre stratégie.

## ⚠️ Avertissements

- **Les performances passées ne garantissent pas les résultats futurs**
- Ce framework est à but éducatif et de recherche
- Utilisez-le pour apprendre et tester, pas comme seule base de décision
- Le trading réel comporte des risques de perte en capital
- Toujours tester en paper trading avant le live

## 🔮 Améliorations futures

Fonctionnalités prévues :
- [ ] Stop-loss et take-profit
- [ ] Position sizing dynamique
- [ ] Walk-forward analysis
- [ ] Monte Carlo simulation
- [ ] Optimisation des paramètres
- [ ] Backtesting multi-actifs
- [ ] Export des rapports en PDF
- [ ] API pour live trading (paper trading)

## 📚 Ressources

- [Documentation pandas](https://pandas.pydata.org/)
- [Guide yfinance](https://github.com/ranaroussi/yfinance)
- [CCXT Documentation](https://docs.ccxt.com/)
- [Analyse technique](https://www.investopedia.com/technical-analysis-4689657)

## 🤝 Contribution

N'hésitez pas à améliorer ce framework ! Quelques idées :
- Ajouter de nouvelles stratégies
- Implémenter de nouveaux indicateurs
- Améliorer les visualisations
- Optimiser les performances

## 📄 License

Ce projet est libre d'utilisation pour l'apprentissage et la recherche personnelle.

---

**Bon trading ! 📈**

*Remember: La discipline et la gestion du risque sont plus importantes que la stratégie elle-même.*
