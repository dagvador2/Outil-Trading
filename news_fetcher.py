"""
Module de récupération et agrégation de news/signaux macro
Supporte multiples sources : APIs, RSS feeds, scraping

Architecture:
- Signaux généraux : Fed, inflation, géopolitique, sentiment de marché
- Signaux spécifiques : Earnings, annonces d'entreprises, news sectorielles

Sources disponibles:
1. NewsAPI (gratuit limité) - news générales
2. Alpha Vantage (gratuit limité) - news financières
3. Finnhub (gratuit limité) - news et sentiment
4. RSS feeds (gratuit) - Bloomberg, Reuters, CNBC
5. Fear & Greed Index (gratuit) - sentiment de marché
6. Fred API (gratuit) - données macro économiques
7. Twitter/X API - sentiment social (payant)
"""

import os
import requests
import feedparser
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import json
import time
import logging

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class MacroSignal:
    """Signal macroéconomique"""
    timestamp: datetime
    source: str
    category: str  # 'fed', 'geopolitical', 'earnings', 'sector', 'sentiment'
    title: str
    description: str
    impact_score: float  # -10 (très bearish) à +10 (très bullish)
    confidence: float  # 0-1
    affected_assets: List[str]  # ['all', 'crypto', 'BTC/USDT', 'AAPL', etc.]
    url: Optional[str] = None
    sentiment: Optional[str] = None  # 'bullish', 'bearish', 'neutral'

    def to_dict(self):
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class MarketSentiment:
    """Sentiment général du marché"""
    timestamp: datetime
    fear_greed_index: Optional[float]  # 0-100
    volatility_index: Optional[float]  # VIX
    general_sentiment: str  # 'extreme_fear', 'fear', 'neutral', 'greed', 'extreme_greed'
    macro_score: float  # Score composite -10 à +10


# ============================================================================
# News Fetchers
# ============================================================================

class NewsAPIFetcher:
    """
    NewsAPI.org - Agrégateur de news généralistes

    Gratuit: 100 requêtes/jour, news jusqu'à 1 mois
    Payant: 250$/mois pour données historiques et plus de requêtes

    Setup:
    1. Créer un compte sur https://newsapi.org/
    2. Copier votre API key
    3. Export NEWSAPI_KEY=your_key_here
    """

    BASE_URL = "https://newsapi.org/v2"

    def __init__(self, api_key: Optional[str] = None):
        import os
        self.api_key = api_key or os.getenv('NEWSAPI_KEY')

    def fetch_general_news(self, query: str = 'economy OR inflation OR fed OR interest rates',
                          days_back: int = 7) -> List[Dict]:
        """Récupère les news générales macro"""
        if not self.api_key:
            logger.warning("NewsAPI key not configured")
            return []

        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

        url = f"{self.BASE_URL}/everything"
        params = {
            'q': query,
            'from': from_date,
            'sortBy': 'relevancy',
            'language': 'en',
            'apiKey': self.api_key
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get('articles', [])
        except Exception as e:
            logger.error(f"NewsAPI fetch error: {e}")
            return []

    def fetch_asset_news(self, asset_symbol: str, days_back: int = 7) -> List[Dict]:
        """Récupère les news spécifiques à un actif"""
        if not self.api_key:
            return []

        # Convertir symbole en recherche (ex: AAPL -> Apple)
        query = self._symbol_to_company(asset_symbol)

        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

        url = f"{self.BASE_URL}/everything"
        params = {
            'q': query,
            'from': from_date,
            'sortBy': 'relevancy',
            'language': 'en',
            'apiKey': self.api_key
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get('articles', [])
        except Exception as e:
            logger.error(f"NewsAPI fetch error for {asset_symbol}: {e}")
            return []

    def _symbol_to_company(self, symbol: str) -> str:
        """Map symboles vers noms de compagnies pour recherche"""
        mapping = {
            'AAPL': 'Apple',
            'MSFT': 'Microsoft',
            'GOOGL': 'Google OR Alphabet',
            'AMZN': 'Amazon',
            'NVDA': 'Nvidia',
            'TSLA': 'Tesla',
            'META': 'Meta OR Facebook',
            'BTC/USDT': 'Bitcoin',
            'ETH/USDT': 'Ethereum',
            'GC=F': 'Gold',
            'SI=F': 'Silver',
        }
        return mapping.get(symbol, symbol)


class AlphaVantageFetcher:
    """
    Alpha Vantage - Données financières et news

    Gratuit: 25 requêtes/jour
    Payant: 49.99$/mois pour 75 req/min

    Setup:
    1. Créer compte sur https://www.alphavantage.co/
    2. Copier API key
    3. Export ALPHAVANTAGE_KEY=your_key_here
    """

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: Optional[str] = None):
        import os
        self.api_key = api_key or os.getenv('ALPHAVANTAGE_KEY')

    def fetch_news_sentiment(self, topics: str = 'economy',
                            tickers: Optional[str] = None) -> List[Dict]:
        """
        Récupère news avec analyse de sentiment

        Args:
            topics: 'economy', 'earnings', 'technology', etc.
            tickers: 'AAPL,MSFT' pour filtrer par ticker
        """
        if not self.api_key:
            logger.warning("AlphaVantage key not configured")
            return []

        params = {
            'function': 'NEWS_SENTIMENT',
            'topics': topics,
            'apikey': self.api_key
        }

        if tickers:
            params['tickers'] = tickers

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get('feed', [])
        except Exception as e:
            logger.error(f"AlphaVantage fetch error: {e}")
            return []


class FinnhubFetcher:
    """
    Finnhub - News financières en temps réel

    Gratuit: 60 requêtes/min
    Payant: 59.99$/mois pour plus de features

    Setup:
    1. Créer compte sur https://finnhub.io/
    2. Copier API key
    3. Export FINNHUB_KEY=your_key_here
    """

    BASE_URL = "https://finnhub.io/api/v1"

    def __init__(self, api_key: Optional[str] = None):
        import os
        self.api_key = api_key or os.getenv('FINNHUB_KEY')

    def fetch_market_news(self, category: str = 'general') -> List[Dict]:
        """
        Récupère les news du marché

        Categories: 'general', 'forex', 'crypto', 'merger'
        """
        if not self.api_key:
            logger.warning("Finnhub key not configured")
            return []

        url = f"{self.BASE_URL}/news"
        params = {
            'category': category,
            'token': self.api_key
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Finnhub fetch error: {e}")
            return []

    def fetch_company_news(self, symbol: str, days_back: int = 7) -> List[Dict]:
        """Récupère news spécifiques à une entreprise"""
        if not self.api_key:
            return []

        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        to_date = datetime.now().strftime('%Y-%m-%d')

        url = f"{self.BASE_URL}/company-news"
        params = {
            'symbol': symbol,
            'from': from_date,
            'to': to_date,
            'token': self.api_key
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Finnhub company news error for {symbol}: {e}")
            return []


class RSSFeedFetcher:
    """
    Récupération de flux RSS - GRATUIT

    Sources principales:
    - Bloomberg: https://www.bloomberg.com/feeds/
    - Reuters: http://feeds.reuters.com/reuters/
    - CNBC: https://www.cnbc.com/id/100003114/device/rss/rss.html
    - MarketWatch: http://feeds.marketwatch.com/
    - CoinDesk (crypto): https://www.coindesk.com/arc/outboundfeeds/rss/
    """

    FEEDS = {
        'bloomberg_markets': 'https://feeds.bloomberg.com/markets/news.rss',
        'reuters_business': 'http://feeds.reuters.com/reuters/businessNews',
        'cnbc_markets': 'https://www.cnbc.com/id/100003114/device/rss/rss.html',
        'marketwatch': 'http://feeds.marketwatch.com/marketwatch/topstories/',
        'coindesk': 'https://www.coindesk.com/arc/outboundfeeds/rss/',
        'fed_news': 'https://www.federalreserve.gov/feeds/press_all.xml',
    }

    def fetch_feed(self, feed_name: str, max_entries: int = 50) -> List[Dict]:
        """Récupère un flux RSS spécifique"""
        url = self.FEEDS.get(feed_name)
        if not url:
            logger.warning(f"Unknown feed: {feed_name}")
            return []

        try:
            feed = feedparser.parse(url)
            entries = []

            for entry in feed.entries[:max_entries]:
                entries.append({
                    'title': entry.get('title', ''),
                    'description': entry.get('summary', ''),
                    'link': entry.get('link', ''),
                    'published': entry.get('published', ''),
                    'source': feed_name
                })

            return entries
        except Exception as e:
            logger.error(f"RSS fetch error for {feed_name}: {e}")
            return []

    def fetch_all_feeds(self, max_per_feed: int = 20) -> List[Dict]:
        """Récupère tous les flux RSS configurés"""
        all_entries = []
        for feed_name in self.FEEDS.keys():
            entries = self.fetch_feed(feed_name, max_per_feed)
            all_entries.extend(entries)
            time.sleep(0.5)  # Rate limiting courtois

        return all_entries


class FearGreedIndexFetcher:
    """
    Fear & Greed Index - CNN Money (GRATUIT)
    Alternative.me pour crypto Fear & Greed

    Mesure le sentiment du marché 0-100
    """

    CRYPTO_URL = "https://api.alternative.me/fng/"

    def fetch_crypto_fear_greed(self, limit: int = 30) -> List[Dict]:
        """Récupère l'index Fear & Greed crypto"""
        try:
            params = {'limit': limit}
            response = requests.get(self.CRYPTO_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get('data', [])
        except Exception as e:
            logger.error(f"Fear & Greed fetch error: {e}")
            return []

    def get_current_sentiment(self) -> Optional[Dict]:
        """Récupère le sentiment actuel"""
        data = self.fetch_crypto_fear_greed(limit=1)
        if data:
            return data[0]
        return None


class FredAPIFetcher:
    """
    Federal Reserve Economic Data (FRED) - GRATUIT

    Données macro officielles : inflation, taux, emploi, GDP, etc.

    Setup:
    1. Créer compte sur https://fred.stlouisfed.org/
    2. Demander API key: https://fredaccount.stlouisfed.org/apikeys
    3. Export FRED_KEY=your_key_here
    """

    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    # Séries importantes
    SERIES = {
        'cpi': 'CPIAUCSL',  # Consumer Price Index (inflation)
        'unemployment': 'UNRATE',  # Unemployment rate
        'fed_funds': 'DFF',  # Federal Funds Rate
        'gdp': 'GDP',  # Gross Domestic Product
        'vix': 'VIXCLS',  # VIX volatility
        'yields_10y': 'DGS10',  # 10-Year Treasury
    }

    def __init__(self, api_key: Optional[str] = None):
        import os
        self.api_key = api_key or os.getenv('FRED_KEY')

    def fetch_series(self, series_id: str, limit: int = 100) -> pd.DataFrame:
        """Récupère une série temporelle"""
        if not self.api_key:
            logger.warning("FRED API key not configured")
            return pd.DataFrame()

        params = {
            'series_id': series_id,
            'api_key': self.api_key,
            'file_type': 'json',
            'limit': limit,
            'sort_order': 'desc'
        }

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            observations = data.get('observations', [])
            df = pd.DataFrame(observations)

            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                df = df.dropna(subset=['value'])

            return df
        except Exception as e:
            logger.error(f"FRED fetch error for {series_id}: {e}")
            return pd.DataFrame()

    def get_latest_indicators(self) -> Dict[str, float]:
        """Récupère les dernières valeurs des indicateurs clés"""
        indicators = {}

        for name, series_id in self.SERIES.items():
            df = self.fetch_series(series_id, limit=1)
            if not df.empty:
                indicators[name] = df.iloc[0]['value']

        return indicators


# ============================================================================
# Main News Aggregator
# ============================================================================

class MacroNewsAggregator:
    """
    Agrégateur principal qui combine toutes les sources
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: Dict avec les clés API optionnelles
                {
                    'newsapi_key': '...',
                    'alphavantage_key': '...',
                    'finnhub_key': '...',
                    'fred_key': '...'
                }
        """
        config = config or {}

        self.newsapi = NewsAPIFetcher(config.get('newsapi_key'))
        self.alphavantage = AlphaVantageFetcher(config.get('alphavantage_key'))
        self.finnhub = FinnhubFetcher(config.get('finnhub_key'))
        self.rss = RSSFeedFetcher()
        self.fear_greed = FearGreedIndexFetcher()
        self.fred = FredAPIFetcher(config.get('fred_key'))

    def fetch_general_signals(self, days_back: int = 7) -> List[MacroSignal]:
        """
        Récupère les signaux généraux affectant tous les actifs

        Returns:
            Liste de MacroSignal avec affected_assets=['all']
        """
        signals = []

        # 1. RSS Feeds (gratuit, toujours disponible)
        logger.info("Fetching RSS feeds...")
        rss_entries = self.rss.fetch_all_feeds(max_per_feed=10)
        for entry in rss_entries:
            signal = self._rss_to_signal(entry, affected_assets=['all'])
            if signal:
                signals.append(signal)

        # 2. NewsAPI si configuré
        if self.newsapi.api_key:
            logger.info("Fetching NewsAPI general news...")
            articles = self.newsapi.fetch_general_news(days_back=days_back)
            for article in articles[:20]:  # Limiter
                signal = self._newsapi_to_signal(article, affected_assets=['all'])
                if signal:
                    signals.append(signal)

        # 3. AlphaVantage sentiment si configuré
        if self.alphavantage.api_key:
            logger.info("Fetching AlphaVantage sentiment...")
            news = self.alphavantage.fetch_news_sentiment(topics='economy')
            for item in news[:15]:
                signal = self._alphavantage_to_signal(item, affected_assets=['all'])
                if signal:
                    signals.append(signal)

        # 4. Finnhub market news si configuré
        if self.finnhub.api_key:
            logger.info("Fetching Finnhub market news...")
            news = self.finnhub.fetch_market_news(category='general')
            for item in news[:15]:
                signal = self._finnhub_to_signal(item, affected_assets=['all'])
                if signal:
                    signals.append(signal)

        logger.info(f"Collected {len(signals)} general signals")
        return signals

    def fetch_asset_signals(self, asset_symbol: str,
                           days_back: int = 7) -> List[MacroSignal]:
        """
        Récupère les signaux spécifiques à un actif

        Args:
            asset_symbol: 'AAPL', 'BTC/USDT', etc.

        Returns:
            Liste de MacroSignal avec affected_assets=[asset_symbol]
        """
        signals = []

        # Déterminer la catégorie d'actif
        category = self._categorize_asset(asset_symbol)

        # 1. NewsAPI si configuré
        if self.newsapi.api_key:
            logger.info(f"Fetching NewsAPI news for {asset_symbol}...")
            articles = self.newsapi.fetch_asset_news(asset_symbol, days_back)
            for article in articles[:10]:
                signal = self._newsapi_to_signal(article,
                                                affected_assets=[asset_symbol, category])
                if signal:
                    signals.append(signal)

        # 2. Finnhub company news si c'est un ticker US
        if self.finnhub.api_key and self._is_us_ticker(asset_symbol):
            logger.info(f"Fetching Finnhub company news for {asset_symbol}...")
            clean_symbol = asset_symbol.replace('/USDT', '')
            news = self.finnhub.fetch_company_news(clean_symbol, days_back)
            for item in news[:10]:
                signal = self._finnhub_to_signal(item,
                                                affected_assets=[asset_symbol, category])
                if signal:
                    signals.append(signal)

        logger.info(f"Collected {len(signals)} signals for {asset_symbol}")
        return signals

    def get_market_sentiment(self) -> MarketSentiment:
        """Récupère le sentiment général du marché"""
        # Fear & Greed Index crypto
        fg_data = self.fear_greed.get_current_sentiment()
        fg_value = None
        if fg_data:
            fg_value = float(fg_data.get('value', 50))

        # VIX depuis FRED
        vix_value = None
        if self.fred.api_key:
            df = self.fred.fetch_series(self.fred.SERIES['vix'], limit=1)
            if not df.empty:
                vix_value = df.iloc[0]['value']

        # Déterminer sentiment général
        if fg_value:
            if fg_value <= 20:
                general = 'extreme_fear'
                macro_score = -8
            elif fg_value <= 40:
                general = 'fear'
                macro_score = -4
            elif fg_value <= 60:
                general = 'neutral'
                macro_score = 0
            elif fg_value <= 80:
                general = 'greed'
                macro_score = 4
            else:
                general = 'extreme_greed'
                macro_score = 8
        else:
            general = 'neutral'
            macro_score = 0

        return MarketSentiment(
            timestamp=datetime.now(),
            fear_greed_index=fg_value,
            volatility_index=vix_value,
            general_sentiment=general,
            macro_score=macro_score
        )

    def get_economic_indicators(self) -> Dict[str, Any]:
        """Récupère les derniers indicateurs économiques"""
        if not self.fred.api_key:
            return {}

        return self.fred.get_latest_indicators()

    # ------------------------------------------------------------------ #
    # Helpers de conversion
    # ------------------------------------------------------------------ #

    def _rss_to_signal(self, entry: Dict, affected_assets: List[str]) -> Optional[MacroSignal]:
        """Convertit entrée RSS en MacroSignal"""
        try:
            # Parse date
            published = entry.get('published', '')
            try:
                timestamp = pd.to_datetime(published)
            except:
                timestamp = datetime.now()

            # Sentiment basique par mots-clés
            text = f"{entry.get('title', '')} {entry.get('description', '')}".lower()
            impact_score, sentiment = self._simple_sentiment(text)

            return MacroSignal(
                timestamp=timestamp,
                source=entry.get('source', 'rss'),
                category=self._categorize_news(text),
                title=entry.get('title', '')[:200],
                description=entry.get('description', '')[:500],
                impact_score=impact_score,
                confidence=0.4,  # RSS = faible confiance
                affected_assets=affected_assets,
                url=entry.get('link'),
                sentiment=sentiment
            )
        except Exception as e:
            logger.debug(f"RSS conversion error: {e}")
            return None

    def _newsapi_to_signal(self, article: Dict,
                          affected_assets: List[str]) -> Optional[MacroSignal]:
        """Convertit article NewsAPI en MacroSignal"""
        try:
            timestamp = pd.to_datetime(article.get('publishedAt', datetime.now()))

            text = f"{article.get('title', '')} {article.get('description', '')}".lower()
            impact_score, sentiment = self._simple_sentiment(text)

            return MacroSignal(
                timestamp=timestamp,
                source='newsapi',
                category=self._categorize_news(text),
                title=article.get('title', '')[:200],
                description=article.get('description', '')[:500],
                impact_score=impact_score,
                confidence=0.5,
                affected_assets=affected_assets,
                url=article.get('url'),
                sentiment=sentiment
            )
        except Exception as e:
            logger.debug(f"NewsAPI conversion error: {e}")
            return None

    def _alphavantage_to_signal(self, item: Dict,
                               affected_assets: List[str]) -> Optional[MacroSignal]:
        """Convertit item AlphaVantage en MacroSignal"""
        try:
            timestamp = pd.to_datetime(item.get('time_published', datetime.now()))

            # AlphaVantage fournit un score de sentiment
            sentiment_score = float(item.get('overall_sentiment_score', 0))
            # Convertir de [-1, 1] à [-10, 10]
            impact_score = sentiment_score * 10

            if sentiment_score > 0.15:
                sentiment = 'bullish'
            elif sentiment_score < -0.15:
                sentiment = 'bearish'
            else:
                sentiment = 'neutral'

            return MacroSignal(
                timestamp=timestamp,
                source='alphavantage',
                category=self._categorize_news(item.get('title', '').lower()),
                title=item.get('title', '')[:200],
                description=item.get('summary', '')[:500],
                impact_score=impact_score,
                confidence=0.7,  # AlphaVantage a analyse de sentiment
                affected_assets=affected_assets,
                url=item.get('url'),
                sentiment=sentiment
            )
        except Exception as e:
            logger.debug(f"AlphaVantage conversion error: {e}")
            return None

    def _finnhub_to_signal(self, item: Dict,
                          affected_assets: List[str]) -> Optional[MacroSignal]:
        """Convertit item Finnhub en MacroSignal"""
        try:
            timestamp = pd.to_datetime(item.get('datetime', time.time()), unit='s')

            text = f"{item.get('headline', '')} {item.get('summary', '')}".lower()
            impact_score, sentiment = self._simple_sentiment(text)

            return MacroSignal(
                timestamp=timestamp,
                source='finnhub',
                category=item.get('category', 'general'),
                title=item.get('headline', '')[:200],
                description=item.get('summary', '')[:500],
                impact_score=impact_score,
                confidence=0.6,
                affected_assets=affected_assets,
                url=item.get('url'),
                sentiment=sentiment
            )
        except Exception as e:
            logger.debug(f"Finnhub conversion error: {e}")
            return None

    def _simple_sentiment(self, text: str) -> Tuple[float, str]:
        """
        Analyse de sentiment simple par mots-clés

        Returns:
            (impact_score, sentiment)
        """
        # Mots-clés bullish
        bullish_keywords = [
            'rally', 'surge', 'soar', 'breakthrough', 'record high', 'bullish',
            'positive', 'growth', 'expansion', 'beat expectations', 'upgrade',
            'strong earnings', 'dovish', 'rate cut', 'stimulus'
        ]

        # Mots-clés bearish
        bearish_keywords = [
            'crash', 'plunge', 'drop', 'decline', 'bearish', 'recession',
            'inflation', 'hawkish', 'rate hike', 'downgrade', 'miss',
            'weak', 'crisis', 'risk', 'uncertainty', 'war', 'conflict'
        ]

        bullish_count = sum(1 for kw in bullish_keywords if kw in text)
        bearish_count = sum(1 for kw in bearish_keywords if kw in text)

        net_score = bullish_count - bearish_count

        # Normaliser à [-10, 10]
        impact_score = np.clip(net_score * 2, -10, 10)

        if impact_score > 2:
            sentiment = 'bullish'
        elif impact_score < -2:
            sentiment = 'bearish'
        else:
            sentiment = 'neutral'

        return float(impact_score), sentiment

    def _categorize_news(self, text: str) -> str:
        """Catégorise une news"""
        if any(kw in text for kw in ['fed', 'federal reserve', 'powell', 'interest rate']):
            return 'fed'
        elif any(kw in text for kw in ['earnings', 'revenue', 'profit', 'eps']):
            return 'earnings'
        elif any(kw in text for kw in ['war', 'conflict', 'geopolitic', 'sanction']):
            return 'geopolitical'
        elif any(kw in text for kw in ['bitcoin', 'crypto', 'ethereum', 'blockchain']):
            return 'crypto'
        elif any(kw in text for kw in ['tech', 'ai', 'artificial intelligence', 'chip']):
            return 'technology'
        else:
            return 'general'

    def _categorize_asset(self, symbol: str) -> str:
        """Détermine la catégorie d'un actif"""
        if 'BTC' in symbol or 'ETH' in symbol or '/USDT' in symbol:
            return 'crypto'
        elif any(x in symbol for x in ['GC=F', 'SI=F', 'CL=F']):
            return 'commodities'
        elif '^' in symbol:
            return 'indices'
        else:
            return 'stocks'

    def _is_us_ticker(self, symbol: str) -> bool:
        """Vérifie si c'est un ticker US standard"""
        # Tickers US = lettres majuscules sans caractères spéciaux
        return symbol.replace('.', '').isalpha() and symbol.isupper()


# ============================================================================
# Storage & Cache
# ============================================================================

class SignalCache:
    """Cache local des signaux pour éviter de re-fetcher"""

    def __init__(self, cache_file: str = 'macro_signals_cache.json'):
        self.cache_file = cache_file
        self.signals: List[MacroSignal] = []
        self._load()

    def _load(self):
        """Charge le cache depuis le fichier"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    # Reconstituer les objets MacroSignal
                    for item in data:
                        item['timestamp'] = pd.to_datetime(item['timestamp'])
                        self.signals.append(MacroSignal(**item))
            except Exception as e:
                logger.error(f"Cache load error: {e}")

    def save(self):
        """Sauvegarde le cache"""
        try:
            data = [s.to_dict() for s in self.signals]
            with open(self.cache_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Cache save error: {e}")

    def add_signals(self, signals: List[MacroSignal]):
        """Ajoute des signaux au cache"""
        self.signals.extend(signals)
        # Garder seulement les 30 derniers jours
        cutoff = datetime.now() - timedelta(days=30)
        self.signals = [s for s in self.signals if s.timestamp > cutoff]
        self.save()

    def get_recent_signals(self, hours: int = 24) -> List[MacroSignal]:
        """Récupère les signaux récents"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [s for s in self.signals if s.timestamp > cutoff]


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    print("="*80)
    print("MODULE DE RÉCUPÉRATION DE NEWS & SIGNAUX MACRO")
    print("="*80)

    print("\n📋 Configuration des APIs:")
    print("\n1. NewsAPI (https://newsapi.org/)")
    print("   - Gratuit: 100 req/jour, news jusqu'à 1 mois")
    print("   - export NEWSAPI_KEY=your_key")

    print("\n2. Alpha Vantage (https://www.alphavantage.co/)")
    print("   - Gratuit: 25 req/jour avec sentiment analysis")
    print("   - export ALPHAVANTAGE_KEY=your_key")

    print("\n3. Finnhub (https://finnhub.io/)")
    print("   - Gratuit: 60 req/min, news en temps réel")
    print("   - export FINNHUB_KEY=your_key")

    print("\n4. FRED (https://fred.stlouisfed.org/)")
    print("   - Gratuit: données macro officielles")
    print("   - export FRED_KEY=your_key")

    print("\n5. RSS Feeds - TOUJOURS GRATUIT (pas de config)")
    print("\n6. Fear & Greed Index - GRATUIT (pas de config)")

    print("\n" + "="*80)
    print("🧪 TEST DU MODULE (utilise RSS feeds)")
    print("="*80)

    # Test avec RSS seulement (gratuit)
    aggregator = MacroNewsAggregator()

    print("\n1. Signaux généraux...")
    general_signals = aggregator.fetch_general_signals(days_back=3)
    print(f"   ✅ {len(general_signals)} signaux récupérés")

    if general_signals:
        print("\n   Exemple de signaux:")
        for sig in general_signals[:3]:
            print(f"   - [{sig.timestamp.strftime('%Y-%m-%d')}] {sig.title[:60]}")
            print(f"     Impact: {sig.impact_score:+.1f} | Sentiment: {sig.sentiment}")

    print("\n2. Sentiment du marché...")
    sentiment = aggregator.get_market_sentiment()
    print(f"   ✅ Sentiment: {sentiment.general_sentiment}")
    print(f"   ✅ Macro score: {sentiment.macro_score:+.1f}")
    if sentiment.fear_greed_index:
        print(f"   ✅ Fear & Greed: {sentiment.fear_greed_index:.0f}/100")

    print("\n3. Test du cache...")
    cache = SignalCache('test_cache.json')
    cache.add_signals(general_signals[:5])
    print(f"   ✅ {len(cache.signals)} signaux en cache")

    print("\n" + "="*80)
    print("✅ Module prêt à l'emploi!")
    print("="*80)
