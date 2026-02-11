# 📊 Module de Signaux Macro - Guide de Démarrage Rapide

## 🎯 Qu'est-ce que c'est ?

Un système complet pour intégrer des **signaux macroéconomiques** dans ton framework de trading :

- 📰 **Récupération de news** depuis multiples sources (APIs + RSS)
- 🎭 **Analyse de sentiment** du marché (Fear & Greed, VIX, etc.)
- 📈 **Indicateurs macro** (inflation, chômage, taux Fed, etc.)
- 🤖 **Scoring automatique** : Score composite -100 (très bearish) à +100 (très bullish)
- 🔄 **Intégration au paper trading** : Filtre ou pondère les signaux techniques

---

## 📦 Fichiers Créés

### Modules Python

1. **[news_fetcher.py](news_fetcher.py)** - Récupération des signaux
   - 6 sources : NewsAPI, AlphaVantage, Finnhub, RSS, Fear&Greed, FRED
   - Support gratuit + payant
   - Cache intégré

2. **[macro_signal_scorer.py](macro_signal_scorer.py)** - Système de scoring
   - Agrège les signaux multi-sources
   - Calcule scores pondérés (général + par actif)
   - Génère recommandations (strong_sell → strong_buy)

3. **[macro_events.py](macro_events.py)** - Base historique 2024-2025 (déjà présent)
   - Événements Fed, Trump, géopolitique, etc.
   - Utilisé par les stratégies event-aware

4. **[strategies_event_aware.py](strategies_event_aware.py)** - Stratégies macro-aware (déjà présent)
   - Combine technique + macro nativement

### Documentation

5. **[MACRO_SIGNALS_GUIDE.md](MACRO_SIGNALS_GUIDE.md)** - Guide complet des APIs
   - Détail de toutes les sources disponibles
   - Setup pour chaque API
   - Recommandations de stack (gratuit → premium)
   - Cas d'usage par type d'actif

6. **[MACRO_INTEGRATION_PLAN.md](MACRO_INTEGRATION_PLAN.md)** - Plan d'intégration
   - 4 phases de déploiement
   - 3 options d'intégration (filtre, pondération, stratégies)
   - Backtest comparatif
   - Métriques de succès

### Configuration

7. **[.env.example](.env.example)** - Template de configuration
   - APIs à configurer
   - Instructions de setup

8. **[requirements_macro.txt](requirements_macro.txt)** - Dépendances
   - feedparser, requests, python-dotenv

---

## 🚀 Démarrage Rapide (5 minutes)

### 1. Installer les dépendances

```bash
cd ~/Desktop/Outil\ trading/Outil-Trading/
pip install -r requirements_macro.txt
```

### 2. Configuration (optionnel pour test)

```bash
# Copier le template
cp .env.example .env

# Pour test basique, AUCUNE CONFIG nécessaire
# Le module fonctionne avec RSS + Fear&Greed (gratuit)

# Pour aller plus loin (recommandé) :
# Créer compte FRED : https://fred.stlouisfed.org/
# Obtenir clé : https://fredaccount.stlouisfed.org/apikeys
nano .env
# Ajouter : FRED_KEY=your_key_here
```

### 3. Test du module de récupération

```bash
python3 news_fetcher.py
```

**Sortie attendue :**
```
================================================================================
MODULE DE RÉCUPÉRATION DE NEWS & SIGNAUX MACRO
================================================================================

📋 Configuration des APIs:
...

🧪 TEST DU MODULE (utilise RSS feeds)
================================================================================

1. Signaux généraux...
   ✅ 15 signaux récupérés
   Exemple de signaux:
   - [2026-02-10] Fed holds rates at 5.25-5.50%
     Impact: +3.0 | Sentiment: bullish

2. Sentiment du marché...
   ✅ Sentiment: greed
   ✅ Macro score: +4.0
   ✅ Fear & Greed: 68/100

3. Test du cache...
   ✅ 5 signaux en cache

✅ Module prêt à l'emploi!
```

### 4. Test du scoring

```bash
# Score général du marché
python3 macro_signal_scorer.py

# Score pour actifs spécifiques
python3 macro_signal_scorer.py BTC/USDT AAPL NVDA
```

**Sortie attendue :**
```
======================================================================
  MACRO SCORE REPORT - all
  2026-02-10 15:30
======================================================================

  📈 SCORE COMPOSITE: +42.5/100
  📊 Confiance: 67.3%
  💡 Recommandation: BUY

  SIGNAUX ANALYSÉS: 15
    🟢 Positifs: 8
    🔴 Négatifs: 3
    ⚪ Neutres: 4

  DÉCOMPOSITION:
    📰 News:          +38.2
    🎭 Sentiment:     +40.0
    📈 Économie:      +55.0

  CONTEXTE:
    Sentiment marché: greed
    Fear & Greed:     68/100
    VIX:              16.3

======================================================================
```

### 5. Export des scores

Les scores sont automatiquement exportés en JSON :

```bash
cat macro_scores.json
```

---

## 📖 Guide d'Utilisation

### Récupération de signaux généraux

```python
from news_fetcher import MacroNewsAggregator

# Initialiser (avec ou sans API keys)
aggregator = MacroNewsAggregator()

# Récupérer signaux généraux (affectent tous les actifs)
signals = aggregator.fetch_general_signals(days_back=7)

# Récupérer signaux pour un actif spécifique
btc_signals = aggregator.fetch_asset_signals('BTC/USDT', days_back=7)

# Sentiment du marché
sentiment = aggregator.get_market_sentiment()
print(f"Fear & Greed: {sentiment.fear_greed_index}/100")
print(f"Sentiment: {sentiment.general_sentiment}")
print(f"Macro score: {sentiment.macro_score}")

# Indicateurs économiques (si FRED configuré)
indicators = aggregator.get_economic_indicators()
print(f"Inflation (CPI): {indicators.get('cpi')}%")
print(f"Chômage: {indicators.get('unemployment')}%")
print(f"VIX: {indicators.get('vix')}")
```

### Scoring des signaux

```python
from macro_signal_scorer import MacroSignalScorer

scorer = MacroSignalScorer()

# Score général du marché
market_score = scorer.compute_market_score(lookback_hours=48)
print(f"Score marché: {market_score.score}/100")
print(f"Recommandation: {market_score.recommendation}")
print(f"Confiance: {market_score.confidence:.1%}")

# Score pour un actif
btc_score = scorer.compute_asset_score('BTC/USDT', lookback_hours=48)
scorer.print_score_report(btc_score)

# Scores pour multiple actifs
assets = ['BTC/USDT', 'AAPL', 'NVDA', 'GOOGL']
all_scores = scorer.compute_all_assets_scores(assets)

# Export
scorer.export_scores(all_scores, 'scores.json')
```

### Cache des signaux

```python
from news_fetcher import SignalCache

cache = SignalCache('my_cache.json')

# Ajouter des signaux
cache.add_signals(signals)

# Récupérer signaux récents
recent = cache.get_recent_signals(hours=24)
print(f"{len(recent)} signaux des dernières 24h")
```

---

## 🎯 Prochaines Étapes

### Option A : Test en local d'abord

1. ✅ Modules créés et testés
2. ⏳ Lancer en background pour collecter des signaux
3. ⏳ Analyser les scores vs mouvements de marché
4. ⏳ Valider la pertinence avant intégration

```bash
# Lancer collection en background
nohup python3 -c "
from news_fetcher import MacroNewsAggregator, SignalCache
import time
aggregator = MacroNewsAggregator()
cache = SignalCache()
while True:
    signals = aggregator.fetch_general_signals(days_back=1)
    cache.add_signals(signals)
    print(f'✅ {len(signals)} signaux collectés')
    time.sleep(7200)  # Toutes les 2h
" > macro_collection.log 2>&1 &
```

### Option B : Intégrer directement au paper trading

Suivre le plan détaillé dans [MACRO_INTEGRATION_PLAN.md](MACRO_INTEGRATION_PLAN.md)

**Phase 1 : Filtre Macro Simple**
- Modifier `signal_generator.py`
- Ajouter `MacroFilter` qui annule les trades contre-tendance macro
- Tester en paper trading parallèle

---

## 📚 Documentation Complète

- **[MACRO_SIGNALS_GUIDE.md](MACRO_SIGNALS_GUIDE.md)** : Détail de toutes les APIs et sources
- **[MACRO_INTEGRATION_PLAN.md](MACRO_INTEGRATION_PLAN.md)** : Plan complet d'intégration au paper trading
- **[news_fetcher.py](news_fetcher.py)** : Code source avec documentation
- **[macro_signal_scorer.py](macro_signal_scorer.py)** : Code source avec documentation

---

## 💡 Tips

### Stack Recommandé pour Démarrer (0€)

```bash
# AUCUNE API key nécessaire
✅ RSS Feeds (Bloomberg, Reuters, CNBC, Fed)
✅ Fear & Greed Index
✅ Cache local

# C'est déjà suffisant pour avoir des signaux pertinents !
```

### Pour aller plus loin (gratuit)

```bash
# Configurer FRED (gratuit, 5 min)
✅ Indicateurs macro officiels (CPI, chômage, taux, VIX)
✅ Données fiables et à jour
✅ API illimitée

# Configuration :
# 1. Créer compte : https://fred.stlouisfed.org/
# 2. Obtenir clé : https://fredaccount.stlouisfed.org/apikeys
# 3. Ajouter dans .env : FRED_KEY=your_key
```

### Stack Pro (~200€/mois)

Si tu veux passer à l'échelle et avoir un edge compétitif :
- NewsAPI payant (449€/mois) OU Benzinga (99$/mois)
- AlphaVantage payant (50$/mois) pour sentiment analysis
- Twitter API (100$/mois) pour sentiment social

---

## 🔍 Exemples de Cas d'Usage

### 1. Éviter les pièges

**Cas :** Fed annonce taux hawkish (très négatif pour crypto)
- **Score macro :** -75 (très bearish)
- **Signal technique :** LONG sur BTC
- **Décision filtre :** ❌ Annuler le LONG
- **Résultat :** Évite une perte pendant le dump

### 2. Confirmer les opportunités

**Cas :** Bitcoin ETF approval + score technique LONG
- **Score macro :** +85 (très bullish)
- **Signal technique :** LONG sur BTC
- **Décision filtre :** ✅ Augmenter position size
- **Résultat :** Maximise le gain sur le rally

### 3. Détecter les retournements

**Cas :** Accumulation de news négatives (guerre, inflation, etc.)
- **Score macro passe de +30 à -40**
- **Signaux techniques encore positifs**
- **Décision :** Réduire exposition, préparer protection
- **Résultat :** Sortir avant le krach

---

## ❓ FAQ

**Q : Ça marche sans aucune API key ?**
A : Oui ! RSS + Fear&Greed sont gratuits et suffisent pour démarrer.

**Q : Combien ça coûte pour un setup complet ?**
A : 0€ pour test, ~50€/mois pour usage sérieux (FRED gratuit + AlphaVantage payant)

**Q : C'est compatible avec le paper trading actuel ?**
A : Oui, l'intégration est conçue pour être non-intrusive. Tu peux activer/désactiver facilement.

**Q : Ça va ralentir le système ?**
A : Non, avec le cache (4h) et les limites de requêtes, l'impact est minimal.

**Q : Comment tester l'efficacité ?**
A : Backtest comparatif (avec/sans macro) + paper trading parallèle. Voir le plan d'intégration.

---

## 🎓 Ressources

- **APIs gratuites :**
  - FRED : https://fred.stlouisfed.org/
  - Fear & Greed : https://alternative.me/crypto/fear-and-greed-index/
  - NewsAPI : https://newsapi.org/ (100 req/jour gratuit)
  - Finnhub : https://finnhub.io/ (60 req/min gratuit)

- **Lectures recommandées :**
  - [How Macro Events Move Markets](https://www.investopedia.com/articles/investing/072913/how-interest-rates-affect-stock-market.asp)
  - [Sentiment Analysis in Trading](https://www.sciencedirect.com/science/article/abs/pii/S0378426619301797)

---

## ✅ Status

- ✅ Module de récupération créé
- ✅ Système de scoring créé
- ✅ Documentation complète
- ✅ Plan d'intégration défini
- ⏳ Configuration des APIs
- ⏳ Test en local
- ⏳ Intégration au paper trading
- ⏳ Déploiement serveur

---

**Prêt à intégrer les signaux macro ? Dis-moi par quelle phase tu veux commencer !** 🚀
