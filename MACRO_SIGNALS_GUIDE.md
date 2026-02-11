# Guide Complet : APIs et Sources de Signaux Macro

## 📊 Vue d'ensemble

Ce guide détaille toutes les sources de données macro disponibles, gratuites et payantes, pour alimenter le système de trading.

---

## 🆓 Sources GRATUITES (Recommandées pour démarrer)

### 1. RSS Feeds - **AUCUNE CONFIG NÉCESSAIRE**

✅ **Avantages**
- Totalement gratuit, illimité
- Pas besoin d'API key
- Sources fiables (Bloomberg, Reuters, CNBC, Fed)

❌ **Limites**
- Pas d'analyse de sentiment automatique
- Format texte brut à analyser

**Sources disponibles dans le module :**
```python
'bloomberg_markets': 'https://feeds.bloomberg.com/markets/news.rss'
'reuters_business': 'http://feeds.reuters.com/reuters/businessNews'
'cnbc_markets': 'https://www.cnbc.com/id/100003114/device/rss/rss.html'
'marketwatch': 'http://feeds.marketwatch.com/marketwatch/topstories/'
'coindesk': 'https://www.coindesk.com/arc/outboundfeeds/rss/'
'fed_news': 'https://www.federalreserve.gov/feeds/press_all.xml'
```

**Utilisation :**
```python
from news_fetcher import RSSFeedFetcher

rss = RSSFeedFetcher()
entries = rss.fetch_all_feeds(max_per_feed=20)
```

---

### 2. Fear & Greed Index (Crypto) - **GRATUIT**

🔗 **URL:** https://alternative.me/crypto/fear-and-greed-index/

✅ **Avantages**
- Indicateur de sentiment crypto en temps réel
- Score 0-100 (fear → greed)
- API publique sans limite
- Historique disponible

**Setup :**
```bash
# Aucune config nécessaire !
```

**Utilisation :**
```python
from news_fetcher import FearGreedIndexFetcher

fg = FearGreedIndexFetcher()
current = fg.get_current_sentiment()
# {'value': 42, 'value_classification': 'Fear'}

history = fg.fetch_crypto_fear_greed(limit=30)
```

**Interprétation :**
- 0-20 : Extreme Fear (très bearish) → Opportunité d'achat
- 20-40 : Fear (bearish)
- 40-60 : Neutral
- 60-80 : Greed (bullish)
- 80-100 : Extreme Greed (très bullish) → Signal de prudence

---

### 3. FRED - Federal Reserve Economic Data - **GRATUIT**

🔗 **URL:** https://fred.stlouisfed.org/

✅ **Avantages**
- Données macro officielles de la Fed
- Inflation, chômage, GDP, taux, VIX
- Historique complet
- API gratuite illimitée

**Setup :**
```bash
# 1. Créer compte : https://fred.stlouisfed.org/
# 2. Demander API key : https://fredaccount.stlouisfed.org/apikeys
# 3. Configurer
export FRED_KEY=your_api_key_here
```

**Séries importantes :**
```python
'CPI': 'CPIAUCSL'          # Inflation (Consumer Price Index)
'Unemployment': 'UNRATE'    # Taux de chômage
'Fed Funds Rate': 'DFF'     # Taux directeur Fed
'GDP': 'GDP'                # Produit Intérieur Brut
'VIX': 'VIXCLS'            # Volatilité (indice de peur)
'10Y Yields': 'DGS10'       # Rendement obligations 10 ans
```

**Utilisation :**
```python
from news_fetcher import FredAPIFetcher

fred = FredAPIFetcher()
indicators = fred.get_latest_indicators()
# {'cpi': 3.2, 'unemployment': 4.1, 'fed_funds': 5.25, 'vix': 18.5}

# Série spécifique
df = fred.fetch_series('CPIAUCSL', limit=100)
```

**Interprétation :**
- **CPI en hausse** → Inflation forte → Fed hawkish → Bearish pour risk assets
- **Unemployment en baisse** → Économie forte → Bullish
- **VIX > 25** → Peur élevée → Volatilité, opportunités contrarian
- **10Y Yields en hausse** → Taux up → Bearish pour growth stocks/crypto

---

## 💰 Sources GRATUITES avec Limites (Puis payantes)

### 4. NewsAPI - **100 req/jour gratuit**

🔗 **URL:** https://newsapi.org/

✅ **Avantages**
- Agrégateur de 80,000+ sources
- Recherche par mots-clés
- Filtres puissants
- Facile à utiliser

❌ **Limites gratuites**
- 100 requêtes/jour
- News jusqu'à 1 mois seulement
- Pas de données historiques

💵 **Version payante : 449€/mois**
- 250,000 requêtes/mois
- Historique complet
- Support

**Setup :**
```bash
# 1. Créer compte : https://newsapi.org/register
# 2. Copier API key du dashboard
export NEWSAPI_KEY=your_api_key_here
```

**Utilisation :**
```python
from news_fetcher import NewsAPIFetcher

newsapi = NewsAPIFetcher()

# News générales macro
articles = newsapi.fetch_general_news(
    query='economy OR inflation OR fed OR interest rates',
    days_back=7
)

# News spécifiques à un actif
apple_news = newsapi.fetch_asset_news('AAPL', days_back=7)
```

---

### 5. Alpha Vantage - **25 req/jour gratuit**

🔗 **URL:** https://www.alphavantage.co/

✅ **Avantages**
- **Analyse de sentiment incluse** (score -1 à +1)
- News financières de qualité
- Données de marché aussi disponibles

❌ **Limites gratuites**
- 25 requêtes/jour (très limité)
- 5 requêtes/minute

💵 **Version payante : 49.99$/mois**
- 75 requêtes/minute
- Support prioritaire

**Setup :**
```bash
# 1. Créer compte : https://www.alphavantage.co/support/#api-key
export ALPHAVANTAGE_KEY=your_api_key_here
```

**Utilisation :**
```python
from news_fetcher import AlphaVantageFetcher

av = AlphaVantageFetcher()

# News avec sentiment analysis
news = av.fetch_news_sentiment(
    topics='economy',  # ou 'earnings', 'technology', etc.
    tickers='AAPL,MSFT'  # optionnel
)

# Chaque news a un 'overall_sentiment_score' de -1 (bearish) à +1 (bullish)
```

---

### 6. Finnhub - **60 req/min gratuit**

🔗 **URL:** https://finnhub.io/

✅ **Avantages**
- News en temps réel
- Données sur earnings, IPOs, etc.
- Bon rate limit gratuit (60/min)
- Market data aussi disponible

❌ **Limites gratuites**
- Features avancées limitées

💵 **Version payante : 59.99$/mois**
- Données historiques
- WebSocket pour temps réel
- Plus de features

**Setup :**
```bash
# 1. Créer compte : https://finnhub.io/register
export FINNHUB_KEY=your_api_key_here
```

**Utilisation :**
```python
from news_fetcher import FinnhubFetcher

finnhub = FinnhubFetcher()

# News du marché
market_news = finnhub.fetch_market_news(category='general')
# Categories: 'general', 'forex', 'crypto', 'merger'

# News d'une entreprise
aapl_news = finnhub.fetch_company_news('AAPL', days_back=7)
```

---

## 💎 Sources PAYANTES Avancées

### 7. Twitter/X API - **Sentiment Social**

🔗 **URL:** https://developer.twitter.com/

💵 **Prix : 100$/mois (Basic)**

✅ **Avantages**
- Sentiment social en temps réel
- Trending topics
- Influence des personnalités (Elon, etc.)

**Cas d'usage :**
- Détecter le buzz sur Bitcoin avant les mouvements
- Sentiment retail sur stocks (WSB, FinTwit)

---

### 8. Bloomberg Terminal - **PREMIUM**

💵 **Prix : ~24,000$/an**

✅ **Avantages**
- News institutionnelles en temps réel
- Données ultra-complètes
- Analyses professionnelles

❌ **Limites**
- Très cher
- Overkill pour trading algorithmique

---

### 9. Benzinga News API

🔗 **URL:** https://www.benzinga.com/apis/

💵 **Prix : 99$/mois - 999$/mois**

✅ **Avantages**
- News financières rapides
- Earnings calendars
- Ratings changes
- FDA approvals (pharma)

---

### 10. Polygon.io - **Market Data + News**

🔗 **URL:** https://polygon.io/

💵 **Prix : 29$/mois - 249$/mois**

✅ **Avantages**
- News + market data combinés
- Données crypto incluses
- WebSockets temps réel

---

## 🎯 Recommandation de Stack

### 🥉 **Stack Gratuit (0€/mois)**
```
✅ RSS Feeds (Bloomberg, Reuters, CNBC, Fed)
✅ Fear & Greed Index (crypto sentiment)
✅ FRED (indicateurs macro officiels)

→ Suffisant pour démarrer et avoir de bons signaux
```

### 🥈 **Stack Starter (0€ + limites)**
```
✅ Stack Gratuit
✅ NewsAPI (100 req/jour)
✅ Finnhub (60 req/min)

→ Bon équilibre pour paper trading avec diversité de sources
```

### 🥇 **Stack Pro (~200€/mois)**
```
✅ Stack Starter
✅ Alpha Vantage payant (50$/mois) → Sentiment analysis
✅ NewsAPI payant (449€/mois) OU Benzinga (99$/mois)
✅ Twitter API Basic (100$/mois) → Sentiment social

→ Pour live trading avec edge compétitif
```

---

## 🛠️ Configuration Rapide

### 1. Créer le fichier `.env`

```bash
cd ~/Desktop/Outil\ trading/Outil-Trading/
nano .env
```

Ajouter :
```bash
# APIs gratuites (recommandé de configurer)
FRED_KEY=your_fred_key_here

# APIs avec limite gratuite (optionnel)
NEWSAPI_KEY=your_newsapi_key_here
FINNHUB_KEY=your_finnhub_key_here
ALPHAVANTAGE_KEY=your_alphavantage_key_here

# APIs payantes (si vous passez à l'échelle)
# TWITTER_BEARER_TOKEN=your_twitter_token_here
```

### 2. Installer les dépendances

```bash
pip install feedparser python-dotenv
```

### 3. Tester le module

```bash
python news_fetcher.py
```

---

## 📈 Cas d'Usage par Type d'Actif

### **Cryptos (BTC, ETH)**
- **Primary:** Fear & Greed Index
- **Secondary:** Twitter sentiment, CoinDesk RSS, Finnhub crypto news
- **Macro:** Fed rate decisions, USD strength, risk-on/risk-off

### **Tech Stocks (AAPL, NVDA, MSFT)**
- **Primary:** Company-specific news (Finnhub, NewsAPI)
- **Secondary:** Earnings calendars, sector sentiment
- **Macro:** Fed policy, tech regulation, innovation trends

### **Gold/Commodities**
- **Primary:** FRED (inflation, USD), geopolitical news
- **Secondary:** Reuters commodities RSS
- **Macro:** Fed dovish/hawkish, conflicts, currency moves

### **Indices (S&P, Nasdaq)**
- **Primary:** VIX (FRED), broad market sentiment
- **Secondary:** Fed announcements, economic data
- **Macro:** GDP, unemployment, corporate earnings season

---

## 🔄 Fréquence de Mise à Jour Recommandée

```python
# Signaux généraux
- RSS feeds : Toutes les 2 heures
- Fear & Greed : 1x par jour (mis à jour quotidiennement)
- FRED : 1x par semaine (données mensuelles)

# Signaux spécifiques
- Company news : Toutes les 4 heures
- Earnings : Daily check pendant earnings season

# Sentiment
- Market sentiment : 1x par jour avant génération signaux
```

---

## 📊 Prochaines Étapes

1. ✅ Module de récupération créé ([news_fetcher.py](news_fetcher.py))
2. 🔄 Configurer au moins FRED (gratuit, très utile)
3. 🔄 Créer le système de scoring composite
4. 🔄 Intégrer au paper trading actuel

---

## 💡 Tips

- **Commencer par RSS + Fear & Greed** : 0€, déjà très efficace
- **FRED est essentiel** : Indicateurs macro officiels, gratuit
- **Ne pas surcharger** : Qualité > Quantité. 3-4 sources bien utilisées > 10 mal exploitées
- **Tester d'abord en paper trading** : Valider l'impact avant de payer des APIs
- **Combiner signaux** : Un signal isolé = bruit. Convergence de plusieurs signaux = signal fort
