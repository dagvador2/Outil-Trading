# Plan d'Intégration des Signaux Macro au Paper Trading

## 🎯 Objectif

Intégrer les signaux macroéconomiques dans le système de paper trading pour améliorer les décisions de trading en combinant :
- **Analyse technique** (existant) : Indicateurs, patterns, momentum
- **Analyse macro** (nouveau) : News, sentiment, événements, indicateurs économiques

---

## 📋 État Actuel

### Système Existant

```
┌─────────────────────────────────────────────────────────────┐
│ PAPER TRADING ACTUEL                                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  [Stratégies Techniques]                                     │
│    ↓                                                         │
│  [signal_generator.py]                                       │
│    ↓                                                         │
│  [multi_paper_trading.py] → 10 portfolios                   │
│    ↓                                                         │
│  [app_dashboard.py] → Visualisation                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘

Signaux basés UNIQUEMENT sur :
- MA, RSI, MACD, Bollinger, ADX, VWAP, Ichimoku
- Prix, volume, volatilité
```

### Nouveau Système Macro

```
┌─────────────────────────────────────────────────────────────┐
│ SYSTÈME MACRO (Nouveau)                                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  [news_fetcher.py] → Récupération signaux                   │
│    ↓                                                         │
│  [macro_signal_scorer.py] → Scoring & recommandations       │
│    ↓                                                         │
│  MacroScore (par actif + global)                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔄 Stratégies d'Intégration

### Option 1 : Filtre Macro (CONSERVATEUR) ⭐ RECOMMANDÉ POUR DÉMARRER

**Concept :** Les signaux techniques sont **filtrés** par le contexte macro

```python
Signal Final = Signal Technique × Filtre Macro

Filtre Macro:
- Si macro score < -60 (très bearish) : Annuler signaux LONG
- Si macro score > +60 (très bullish) : Annuler signaux SHORT
- Si macro score modéré : Laisser passer signaux techniques
```

**Avantages :**
- ✅ Changement minimal du code existant
- ✅ Évite les trades contre-tendance macro
- ✅ Facile à tester et désactiver
- ✅ Préserve la logique technique éprouvée

**Implémentation :**
```python
# Dans signal_generator.py

from macro_signal_scorer import MacroSignalScorer

def generate_signals_with_macro_filter(data, strategy, asset_symbol, scorer):
    # 1. Signaux techniques classiques
    tech_signals = strategy.generate_signals(data)

    # 2. Score macro pour cet actif
    macro_score = scorer.compute_asset_score(asset_symbol, lookback_hours=48)

    # 3. Filtrage
    filtered_signals = tech_signals.copy()

    for idx in filtered_signals.index:
        original_signal = filtered_signals.loc[idx, 'position']

        # Environnement très bearish
        if macro_score.score < -60:
            if original_signal == 1:  # LONG
                filtered_signals.loc[idx, 'position'] = 0  # Annuler

        # Environnement très bullish
        elif macro_score.score > 60:
            if original_signal == -1:  # SHORT
                filtered_signals.loc[idx, 'position'] = 0  # Annuler

    return filtered_signals, macro_score
```

---

### Option 2 : Pondération Macro (MODÉRÉ)

**Concept :** Ajuster la **taille des positions** selon le contexte macro

```python
Position Size = Base Size × Multiplicateur Macro

Multiplicateur:
- Macro très favorable + signal technique : 1.5× (position plus grande)
- Macro neutre : 1.0× (position normale)
- Macro défavorable : 0.5× (position réduite)
- Macro très défavorable : 0× (pas de position)
```

**Avantages :**
- ✅ Exploite les environnements favorables
- ✅ Réduit l'exposition dans les environnements risqués
- ✅ Plus sophistiqué que simple filtre

**Inconvénients :**
- ⚠️ Plus complexe à tester
- ⚠️ Peut sur-exposer dans euphorie

---

### Option 3 : Stratégies Event-Aware (AVANCÉ)

**Concept :** Utiliser les stratégies `strategies_event_aware.py` (déjà dans le code !)

```python
# Wrapper de stratégie technique avec conscience macro
EventFilteredMAStrategy
EventFilteredRSIStrategy
EventFilteredCombinedStrategy
```

Ces stratégies combinent **nativement** technique + macro dans `generate_signals()`

**Avantages :**
- ✅ Intégration la plus profonde
- ✅ Utilise le code déjà écrit
- ✅ Chaque stratégie adapte sa logique

**Inconvénients :**
- ⚠️ Nécessite de modifier STRATEGY_MAP
- ⚠️ Plus difficile à débugger
- ⚠️ Dépendance forte aux signaux macro

---

## 🎯 Plan d'Implémentation Recommandé

### Phase 1 : Setup & Test (Semaine 1)

**1.1 Configuration des APIs**
```bash
cd ~/Desktop/Outil\ trading/Outil-Trading/

# Copier le template
cp .env.example .env

# Configurer AU MINIMUM
# - FRED_KEY (gratuit, essentiel)
nano .env

# Installer dépendances
pip install feedparser python-dotenv requests
```

**1.2 Test du module de récupération**
```bash
# Test des fetchers
python news_fetcher.py

# Devrait afficher :
# - Signaux RSS récupérés
# - Fear & Greed Index
# - Test du cache
```

**1.3 Test du scoring**
```bash
# Test du scorer
python macro_signal_scorer.py

# Ou pour des actifs spécifiques
python macro_signal_scorer.py BTC/USDT AAPL NVDA

# Devrait calculer et afficher les scores
```

---

### Phase 2 : Intégration Simple - Filtre Macro (Semaine 2)

**2.1 Créer le module d'intégration**

Créer `macro_integration.py` :

```python
"""
Intégration légère des signaux macro au paper trading
Mode : Filtre conservateur
"""

from macro_signal_scorer import MacroSignalScorer, MacroNewsAggregator
import pandas as pd
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MacroFilter:
    """Filtre les signaux techniques selon contexte macro"""

    def __init__(self, enable: bool = True,
                 strong_threshold: float = 60.0,
                 moderate_threshold: float = 30.0):
        """
        Args:
            enable: Activer/désactiver le filtre (pour A/B test)
            strong_threshold: Seuil pour blocage fort
            moderate_threshold: Seuil pour réduction position
        """
        self.enable = enable
        self.strong_threshold = strong_threshold
        self.moderate_threshold = moderate_threshold

        if self.enable:
            self.scorer = MacroSignalScorer()
            self.cache = {}  # Cache des scores par actif
            self.last_update = None

    def filter_signals(self, signals: pd.DataFrame,
                      asset_symbol: str) -> Tuple[pd.DataFrame, Optional[dict]]:
        """
        Filtre les signaux techniques selon macro

        Returns:
            (signals_filtrés, macro_info)
        """
        if not self.enable:
            return signals, None

        # Récupérer score macro (avec cache)
        try:
            macro_score = self._get_macro_score(asset_symbol)

            # Appliquer filtre
            filtered = signals.copy()

            for idx in filtered.index:
                pos = filtered.loc[idx, 'position']

                # Filtre fort : annuler positions contre-tendance macro
                if abs(macro_score.score) > self.strong_threshold:
                    if macro_score.score < -self.strong_threshold and pos == 1:
                        # Très bearish, annuler LONG
                        filtered.loc[idx, 'position'] = 0
                        logger.info(f"{asset_symbol}: LONG annulé (macro: {macro_score.score:.1f})")

                    elif macro_score.score > self.strong_threshold and pos == -1:
                        # Très bullish, annuler SHORT
                        filtered.loc[idx, 'position'] = 0
                        logger.info(f"{asset_symbol}: SHORT annulé (macro: {macro_score.score:.1f})")

            # Recalculer les signaux
            filtered['signal'] = filtered['position'].diff()

            macro_info = {
                'score': macro_score.score,
                'confidence': macro_score.confidence,
                'recommendation': macro_score.recommendation,
                'num_signals': macro_score.num_signals
            }

            return filtered, macro_info

        except Exception as e:
            logger.error(f"Macro filter error for {asset_symbol}: {e}")
            return signals, None

    def _get_macro_score(self, asset_symbol: str):
        """Récupère score avec cache"""
        # Cache de 4 heures
        from datetime import datetime, timedelta

        now = datetime.now()
        cache_key = asset_symbol

        if cache_key in self.cache:
            score, timestamp = self.cache[cache_key]
            if now - timestamp < timedelta(hours=4):
                return score

        # Calculer nouveau score
        score = self.scorer.compute_asset_score(asset_symbol, lookback_hours=48)
        self.cache[cache_key] = (score, now)

        return score
```

**2.2 Intégrer au signal generator**

Modifier `signal_generator.py` :

```python
# Ajout en haut du fichier
from macro_integration import MacroFilter

# Dans SignalGenerator.__init__
self.macro_filter = MacroFilter(enable=True)  # Paramétrable

# Dans generate_current_signals
def generate_current_signals(self):
    signals = {}
    macro_info = {}

    for symbol, strategies in self.strategies.items():
        # ... code existant pour générer signaux techniques ...

        # NOUVEAU : Appliquer filtre macro
        if self.macro_filter.enable:
            final_signals, macro_data = self.macro_filter.filter_signals(
                tech_signals,
                symbol
            )
            signals[symbol] = final_signals
            macro_info[symbol] = macro_data
        else:
            signals[symbol] = tech_signals

    return signals, macro_info
```

**2.3 Modifier AutoPaperTrader**

Dans `auto_paper_trading.py`, stocker les infos macro :

```python
# Sauvegarder macro_info dans state
state['macro_context'] = macro_info
```

---

### Phase 3 : Test & Validation (Semaine 3)

**3.1 Backtest comparatif**

Créer `backtest_macro_comparison.py` :

```python
"""
Compare les performances avec/sans filtre macro
"""

from backtest_library import BacktestLibrary
from backtesting_engine import BacktestEngine
from macro_integration import MacroFilter

# Test sur période historique (2024-2025)
# Comparer:
# - Stratégie pure (sans macro)
# - Stratégie + filtre macro

# Métriques:
# - Total return
# - Sharpe ratio
# - Max drawdown
# - Win rate
# - Trades évités (grâce au filtre)
```

**3.2 Paper trading parallèle**

Lancer 2 portfolios en parallèle :
- Portfolio 1 : Sans macro (baseline)
- Portfolio 2 : Avec macro filter

Comparer sur 2-4 semaines.

---

### Phase 4 : Déploiement Production (Semaine 4)

**4.1 Configuration serveur**

Sur le serveur Hetzner :

```bash
ssh root@188.245.184.69
cd /opt/trading/

# Pull les nouveaux fichiers
git pull origin main

# Configurer .env
nano .env
# Ajouter au minimum FRED_KEY

# Installer dépendances
source venv/bin/activate
pip install feedparser python-dotenv

# Tester
python news_fetcher.py
python macro_signal_scorer.py BTC/USDT
```

**4.2 Update du service**

Modifier si besoin `paper-trading.service` pour passer des flags :

```bash
# Option pour activer/désactiver macro
--enable-macro
--macro-threshold 60
```

**4.3 Monitoring**

Ajouter au dashboard (Tab 8) :
- Score macro par actif
- Trades filtrés par macro
- Performance avec/sans macro

---

## 📊 Métriques de Succès

### KPIs à suivre

1. **Impact sur performance**
   - Sharpe ratio amélioré ?
   - Drawdown réduit ?
   - Return total augmenté ?

2. **Efficacité du filtre**
   - % trades filtrés
   - % trades filtrés qui auraient été perdants (bonne décision)
   - % trades filtrés qui auraient été gagnants (mauvaise décision)

3. **Qualité des signaux macro**
   - Corrélation score macro / mouvement prix
   - Lead time (le signal précède-t-il le mouvement ?)

---

## ⚙️ Configuration Avancée

### Paramètres à tuner

```python
# Seuils du filtre
STRONG_THRESHOLD = 60  # Blocage total
MODERATE_THRESHOLD = 30  # Réduction position

# Fenêtre temporelle
LOOKBACK_HOURS = 48  # Signaux des 48h

# Poids des sources
SOURCE_WEIGHTS = {
    'alphavantage': 1.0,
    'finnhub': 0.9,
    'newsapi': 0.7,
    'rss': 0.5,
    'fred': 1.2,
}

# Poids des catégories
CATEGORY_WEIGHTS = {
    'fed': 1.5,  # Impact majeur
    'geopolitical': 1.2,
    'earnings': 0.8,
    'technology': 0.7,
}
```

### A/B Testing

Créer plusieurs configurations dans `multi_paper_trading.py` :

```python
PORTFOLIO_CONFIGS = [
    # ... configs existants ...

    # Nouveau : avec macro
    {
        'name': 'Balanced_Macro',
        'allocation_method': 'score_weighted',
        'enable_macro_filter': True,
        'macro_threshold': 60,
        ...
    },

    {
        'name': 'Aggressive_Macro',
        'enable_macro_filter': True,
        'macro_threshold': 40,  # Plus agressif
        ...
    }
]
```

---

## 🚨 Risques & Mitigations

### Risques identifiés

1. **Sur-filtrage**
   - Risque : Filtrer trop de trades, réduire opportunités
   - Mitigation : Seuils élevés (60+), A/B testing

2. **Faux signaux**
   - Risque : Signaux macro bruités ou retardés
   - Mitigation : Pondération par confiance, multiples sources

3. **Latence**
   - Risque : Fetch des signaux ralentit le système
   - Mitigation : Cache 4h, async si nécessaire

4. **API limits**
   - Risque : Dépasser quotas gratuits
   - Mitigation : Priorité aux RSS (illimité), cache agressif

### Kill Switch

Toujours pouvoir désactiver rapidement :

```python
# Désactiver via flag
macro_filter = MacroFilter(enable=False)

# Ou via variable d'environnement
ENABLE_MACRO_FILTER=false
```

---

## 📚 Prochaines Évolutions

### Phase 5+ : Optimisations

- **Sentiment NLP avancé** : Utiliser transformers (FinBERT) pour meilleur scoring
- **Événements calendrier** : Intégrer calendrier earnings, Fed meetings
- **ML pour pondération** : Apprendre les meilleurs poids via ML
- **Signaux prédictifs** : Détecter patterns pré-mouvement
- **Multi-timeframe** : Macro LT (semaines) + court terme (heures)

---

## 🎓 Ressources

- [news_fetcher.py](news_fetcher.py) - Module de récupération
- [macro_signal_scorer.py](macro_signal_scorer.py) - Système de scoring
- [MACRO_SIGNALS_GUIDE.md](MACRO_SIGNALS_GUIDE.md) - Guide des APIs
- [strategies_event_aware.py](strategies_event_aware.py) - Stratégies macro-aware (Option 3)

---

## ✅ Checklist de Déploiement

- [ ] Configurer au moins FRED_KEY dans .env
- [ ] Tester news_fetcher.py localement
- [ ] Tester macro_signal_scorer.py localement
- [ ] Créer macro_integration.py
- [ ] Modifier signal_generator.py
- [ ] Lancer backtest comparatif
- [ ] Tester en paper trading local (1-2 semaines)
- [ ] Déployer sur serveur
- [ ] Monitorer performance
- [ ] Ajuster seuils si nécessaire
- [ ] Documenter dans MEMORY.md

---

**Prêt à commencer ? On démarre par la Phase 1 quand tu veux !** 🚀
