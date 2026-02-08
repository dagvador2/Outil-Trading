# Guide Utilisateur - Framework de Backtesting

## Table des matières

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Architecture du système](#architecture)
4. [Guide d'utilisation](#guide-utilisation)
5. [Référence des API](#api-reference)
6. [Exemples pratiques](#exemples)
7. [FAQ](#faq)
8. [Bonnes pratiques](#bonnes-pratiques)

---

## 1. Introduction

### Qu'est-ce que le backtesting ?

Le backtesting est le processus de test d'une stratégie de trading sur des données historiques pour évaluer sa performance potentielle. C'est une étape cruciale avant de déployer une stratégie en trading réel.

### Pourquoi utiliser ce framework ?

- ✅ **Complet** : Tous les outils nécessaires en un seul endroit
- ✅ **Modulaire** : Facile d'ajouter vos propres stratégies et indicateurs
- ✅ **Réaliste** : Inclut commissions, slippage, et métriques professionnelles
- ✅ **Éducatif** : Code bien documenté et exemples nombreux
- ✅ **Multi-marchés** : Actions, crypto, forex, indices

### Avertissement

⚠️ **IMPORTANT** : Les performances passées ne garantissent pas les résultats futurs. Ce framework est destiné à l'apprentissage et à la recherche, pas comme seul outil de décision d'investissement.

---

## 2. Installation

### Étape 1 : Prérequis

Python 3.8 ou supérieur est requis.

### Étape 2 : Installation des dépendances

```bash
# Dépendances principales (REQUISES)
pip install pandas numpy matplotlib

# Optionnel : Pour données réelles
pip install yfinance    # Actions et Forex
pip install ccxt        # Cryptomonnaies
```

Ou utilisez le fichier requirements.txt :

```bash
pip install -r requirements.txt
```

### Étape 3 : Vérification

```python
python example_usage.py
```

Si le script s'exécute sans erreur, l'installation est réussie !

---

## 3. Architecture du système

### Structure des fichiers

```
trading-backtest/
│
├── backtesting_engine.py    # Moteur principal
│   └── Classes : BacktestEngine, Trade
│
├── strategies.py            # Stratégies de trading
│   └── Classes : BaseStrategy, MovingAverageCrossover, RSIStrategy, etc.
│
├── indicators.py            # Indicateurs techniques
│   └── Classe : TechnicalIndicators (méthodes statiques)
│
├── data_fetcher.py         # Récupération de données
│   └── Classe : DataFetcher, fonction get_data()
│
├── visualizer.py           # Visualisations
│   └── Classe : BacktestVisualizer
│
├── example_usage.py        # Exemples d'utilisation
├── quick_start_guide.py    # Tutoriel interactif
└── README.md              # Documentation
```

### Flux de données

```
Données de marché
       ↓
Calcul d'indicateurs
       ↓
Génération de signaux (Stratégie)
       ↓
Exécution des trades (BacktestEngine)
       ↓
Calcul des métriques
       ↓
Visualisation des résultats
```

---

## 4. Guide d'utilisation

### 4.1 Récupération de données

#### Option 1 : Données synthétiques (recommandé pour débuter)

```python
from data_fetcher import get_data

data = get_data('sample', 'DEMO', '2023-01-01', '2024-01-01')
```

#### Option 2 : Actions réelles

```python
# Nécessite: pip install yfinance
data = get_data('stock', 'AAPL', '2023-01-01', '2024-01-01')
```

#### Option 3 : Cryptomonnaies

```python
# Nécessite: pip install ccxt
data = get_data('crypto', 'BTC/USDT', '2023-01-01', '2024-01-01',
                exchange='binance', timeframe='1d')
```

#### Option 4 : Forex

```python
data = get_data('forex', 'EURUSD', '2023-01-01', '2024-01-01')
```

### 4.2 Calcul des indicateurs

```python
from indicators import TechnicalIndicators

# Un seul indicateur
rsi = TechnicalIndicators.rsi(data['close'], period=14)

# Tous les indicateurs
data_enriched = TechnicalIndicators.add_all_indicators(data)
```

### 4.3 Utilisation d'une stratégie

```python
from strategies import MovingAverageCrossover

# Créer la stratégie
strategy = MovingAverageCrossover(fast_period=20, slow_period=50)

# Générer les signaux
signals = strategy.generate_signals(data)
```

### 4.4 Exécution du backtest

```python
from backtesting_engine import BacktestEngine

# Configurer le moteur
engine = BacktestEngine(
    initial_capital=10000,
    commission=0.001,    # 0.1%
    slippage=0.0005     # 0.05%
)

# Exécuter
results = engine.run_backtest(data, strategy)

# Afficher les résultats
print(f"Rendement: {results['total_return_pct']:.2f}%")
```

### 4.5 Visualisation

```python
from visualizer import BacktestVisualizer
import matplotlib.pyplot as plt

# Dashboard complet
fig = BacktestVisualizer.plot_performance_summary(results)
plt.show()

# Courbe d'équité seule
fig = BacktestVisualizer.plot_equity_curve(results['equity_df'])
plt.show()
```

---

## 5. Référence des API

### 5.1 BacktestEngine

```python
engine = BacktestEngine(
    initial_capital: float = 10000,
    commission: float = 0.001,
    slippage: float = 0.0005
)
```

**Méthodes principales :**

- `run_backtest(data, strategy)` : Exécute le backtest
- `get_results()` : Retourne les métriques de performance

**Résultats retournés :**

```python
{
    'total_trades': int,
    'winning_trades': int,
    'losing_trades': int,
    'win_rate': float,          # %
    'initial_capital': float,
    'final_capital': float,
    'total_return': float,      # $
    'total_return_pct': float,  # %
    'avg_win': float,
    'avg_loss': float,
    'profit_factor': float,
    'max_drawdown': float,      # %
    'sharpe_ratio': float,
    'trades_df': DataFrame,
    'equity_df': DataFrame
}
```

### 5.2 Stratégies

Toutes les stratégies héritent de `BaseStrategy` et implémentent :

```python
def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
    # Retourne DataFrame avec colonne 'signal':
    # 1  = Acheter (LONG)
    # -1 = Vendre (SHORT)
    # 0  = Sortir de position
```

**Stratégies disponibles :**

1. `MovingAverageCrossover(fast_period, slow_period)`
2. `RSIStrategy(period, oversold, overbought)`
3. `MACDStrategy(fast, slow, signal)`
4. `BollingerBandsStrategy(period, num_std)`
5. `CombinedStrategy()`

### 5.3 Indicateurs

Tous les indicateurs sont des méthodes statiques de `TechnicalIndicators` :

```python
# Moyennes mobiles
sma = TechnicalIndicators.sma(data['close'], period)
ema = TechnicalIndicators.ema(data['close'], period)

# Oscillateurs
rsi = TechnicalIndicators.rsi(data['close'], period)
macd_df = TechnicalIndicators.macd(data['close'])
stoch_df = TechnicalIndicators.stochastic(high, low, close)

# Bandes et volatilité
bb_df = TechnicalIndicators.bollinger_bands(data['close'])
atr = TechnicalIndicators.atr(high, low, close)

# Tout en un
data_full = TechnicalIndicators.add_all_indicators(data)
```

---

## 6. Exemples pratiques

### Exemple 1 : Backtest simple

```python
from backtesting_engine import BacktestEngine
from strategies import MovingAverageCrossover
from data_fetcher import get_data

# 1. Données
data = get_data('sample', 'DEMO', '2023-01-01', '2024-01-01')

# 2. Stratégie
strategy = MovingAverageCrossover(20, 50)

# 3. Backtest
engine = BacktestEngine(initial_capital=10000)
results = engine.run_backtest(data, strategy)

# 4. Résultats
print(f"Rendement: {results['total_return_pct']:.2f}%")
```

### Exemple 2 : Comparer plusieurs stratégies

```python
strategies = [
    MovingAverageCrossover(20, 50),
    RSIStrategy(14, 30, 70),
    MACDStrategy()
]

for strat in strategies:
    engine = BacktestEngine(initial_capital=10000)
    results = engine.run_backtest(data, strat)
    print(f"{strat.name}: {results['total_return_pct']:.2f}%")
```

### Exemple 3 : Créer sa propre stratégie

```python
from strategies import BaseStrategy
import pandas as pd

class MyCustomStrategy(BaseStrategy):
    def __init__(self, threshold):
        super().__init__("Ma Stratégie Custom")
        self.threshold = threshold
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        # Votre logique ici
        # Par exemple : acheter quand le prix monte de X%
        returns = data['close'].pct_change()
        signals.loc[returns > self.threshold, 'signal'] = 1
        signals.loc[returns < -self.threshold, 'signal'] = -1
        
        return signals

# Utilisation
strategy = MyCustomStrategy(threshold=0.02)  # 2%
results = engine.run_backtest(data, strategy)
```

### Exemple 4 : Optimisation de paramètres

```python
# Tester différentes périodes de moyennes mobiles
best_params = None
best_return = -float('inf')

for fast in [10, 20, 30]:
    for slow in [50, 100, 200]:
        if fast >= slow:
            continue
        
        strategy = MovingAverageCrossover(fast, slow)
        engine = BacktestEngine(initial_capital=10000)
        results = engine.run_backtest(data, strategy)
        
        if results['total_return_pct'] > best_return:
            best_return = results['total_return_pct']
            best_params = (fast, slow)

print(f"Meilleurs paramètres: MA{best_params[0]}/MA{best_params[1]}")
print(f"Rendement: {best_return:.2f}%")
```

---

## 7. FAQ

### Q1 : Comment gérer les coûts de trading ?

Les commissions et le slippage sont configurables dans BacktestEngine :

```python
engine = BacktestEngine(
    commission=0.001,  # 0.1% par transaction
    slippage=0.0005    # 0.05% de slippage
)
```

### Q2 : Puis-je faire du short selling ?

Oui ! Les stratégies peuvent générer des signaux -1 pour vendre à découvert.

### Q3 : Comment ajouter un stop-loss ?

Actuellement non supporté nativement, mais vous pouvez le coder dans votre stratégie personnalisée en surveillant les prix et générant un signal de sortie.

### Q4 : Les données sont-elles ajustées pour les dividendes ?

Si vous utilisez yfinance, oui (colonne 'adj close'). Pour les données synthétiques, non.

### Q5 : Puis-je backtester sur plusieurs actifs simultanément ?

Pas actuellement. Le framework teste un actif à la fois. Pour un portefeuille, exécutez plusieurs backtests.

### Q6 : Comment interpréter le Sharpe ratio ?

- < 1 : Médiocre
- 1-2 : Acceptable
- 2-3 : Bon
- > 3 : Excellent

### Q7 : Que faire si mes résultats sont trop beaux ?

Attention à l'overfitting ! Vérifiez :
- Vos paramètres ne sont pas sur-optimisés
- Vous n'utilisez pas de données futures (look-ahead bias)
- Vos coûts de transaction sont réalistes

---

## 8. Bonnes pratiques

### 8.1 Avant de commencer

1. **Définissez votre hypothèse** : Pourquoi cette stratégie devrait-elle fonctionner ?
2. **Choisissez la bonne période** : Au moins 2-3 ans de données
3. **Soyez réaliste** : Incluez commissions et slippage

### 8.2 Pendant le backtesting

1. **Évitez l'overfitting** : Ne sur-optimisez pas les paramètres
2. **Test out-of-sample** : Gardez une période de données pour validation finale
3. **Walk-forward** : Testez sur plusieurs périodes consécutives

### 8.3 Interprétation des résultats

1. **Ne vous fiez pas qu'au rendement total** : Regardez aussi le drawdown et le Sharpe
2. **Analysez les trades individuels** : Y a-t-il des patterns ?
3. **Contexte de marché** : La stratégie fonctionne-t-elle dans tous les contextes ?

### 8.4 Métriques importantes

**Pour évaluer une stratégie, regardez :**

1. **Rendement total** : Combien avez-vous gagné/perdu ?
2. **Win rate** : % de trades gagnants (minimum 40-50%)
3. **Profit factor** : Ratio gains/pertes (minimum 1.5)
4. **Max drawdown** : Plus grande perte depuis un sommet (maximum acceptable 20-30%)
5. **Sharpe ratio** : Rendement ajusté au risque (minimum 1.0)

### 8.5 Pièges à éviter

❌ **Look-ahead bias** : N'utilisez jamais de données futures
❌ **Survivorship bias** : Testez sur des actifs qui existaient à l'époque
❌ **Curve fitting** : Sur-optimiser les paramètres sur les données historiques
❌ **Ignorer les coûts** : Toujours inclure commissions et slippage
❌ **Cherry picking** : Ne montrez pas que les bons résultats

### 8.6 Checklist avant le trading réel

Avant de déployer une stratégie en trading réel :

- [ ] Backtesté sur au moins 2-3 ans de données
- [ ] Testé sur période out-of-sample
- [ ] Sharpe ratio > 1.0
- [ ] Max drawdown acceptable pour votre tolérance au risque
- [ ] Profit factor > 1.5
- [ ] Compris pourquoi la stratégie fonctionne (théoriquement)
- [ ] Testé en paper trading pendant au moins 1-3 mois
- [ ] Défini un plan de gestion du risque
- [ ] Prêt à arrêter si les résultats divergent du backtest

---

## Conclusion

Ce framework vous donne tous les outils pour tester rigoureusement vos idées de trading. Utilisez-le pour apprendre, expérimenter, et développer votre compréhension des marchés.

**Rappel final** : Le backtesting est un outil puissant, mais ce n'est qu'un outil. La discipline, la gestion du risque, et la compréhension des marchés sont tout aussi importantes que la stratégie elle-même.

Bonne chance dans votre parcours de trading ! 📈

---

*Dernière mise à jour : Février 2026*
