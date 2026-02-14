"""
Système de récupération de données en temps réel
Sources : Yahoo Finance (principal), Binance (crypto), FRED (macro)
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
from typing import Dict, Optional
import ccxt


class LiveDataFeed:
    """
    Récupère les données en temps réel pour le trading live
    Utilise Yahoo Finance (gratuit, illimité) + Binance (crypto)
    """

    def __init__(self, alpha_vantage_key: str, fred_key: str, use_yahoo: bool = True, db=None):
        self.av_key = alpha_vantage_key
        self.fred_key = fred_key
        self.use_yahoo = use_yahoo
        self.db = db

        # Exchange crypto (Binance)
        self.binance = ccxt.binance({
            'enableRateLimit': True,
        })

        # Yahoo Finance (si activé)
        if self.use_yahoo:
            try:
                from src.data.yahoo import YahooDataFeed, convert_to_yahoo_symbol
                self.yahoo = YahooDataFeed()
                self.convert_symbol = convert_to_yahoo_symbol
                print("✅ Yahoo Finance activé (gratuit, illimité)")
            except ImportError:
                print("⚠️  yahoo_data_feed non trouvé, fallback sur Alpha Vantage")
                self.use_yahoo = False
                self.yahoo = None

        # Cache pour éviter trop de requêtes
        self.cache = {}
        self.cache_duration = 60  # 60 secondes

    def get_live_crypto_price(self, symbol: str) -> Dict:
        """
        Récupère le prix crypto en temps réel depuis Binance

        Args:
            symbol: Paire crypto (ex: 'BTC/USDT')

        Returns:
            Dict avec 'price', 'timestamp', 'volume', 'change_24h'
        """
        try:
            ticker = self.binance.fetch_ticker(symbol)

            return {
                'symbol': symbol,
                'price': ticker['last'],
                'timestamp': datetime.now(),
                'volume_24h': ticker['quoteVolume'],
                'change_24h_pct': ticker['percentage'],
                'high_24h': ticker['high'],
                'low_24h': ticker['low'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'spread_pct': ((ticker['ask'] - ticker['bid']) / ticker['bid']) * 100
            }

        except Exception as e:
            print(f"❌ Erreur Binance pour {symbol}: {e}")
            return None

    def get_live_stock_price(self, ticker: str) -> Dict:
        """
        Récupère le prix stock en temps réel
        Essaie Yahoo Finance d'abord (gratuit), puis Alpha Vantage si échec

        Args:
            ticker: Symbole action (ex: 'NVDA')

        Returns:
            Dict avec 'price', 'timestamp', 'volume', 'change'
        """
        # Check cache
        cache_key = f"stock_{ticker}"
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if (datetime.now() - cached_time).seconds < self.cache_duration:
                return cached_data

        # PRIORITÉ 1: Yahoo Finance (si activé)
        if self.use_yahoo and self.yahoo:
            try:
                # Convertir symbole si nécessaire
                yahoo_symbol = self.convert_symbol(ticker) if hasattr(self, 'convert_symbol') else ticker
                result = self.yahoo.get_live_price(yahoo_symbol)

                if result and result.get('price', 0) > 0:
                    # Cache
                    self.cache[cache_key] = (datetime.now(), result)
                    return result
            except Exception as e:
                print(f"⚠️  Yahoo Finance fallback pour {ticker}")

        # FALLBACK: Alpha Vantage (rate limited)
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': ticker,
            'apikey': self.av_key
        }

        try:
            response = requests.get(url, params=params)
            data = response.json()

            if 'Global Quote' in data and data['Global Quote']:
                quote = data['Global Quote']

                result = {
                    'symbol': ticker,
                    'price': float(quote['05. price']),
                    'timestamp': datetime.now(),
                    'volume': int(quote['06. volume']),
                    'change': float(quote['09. change']),
                    'change_pct': float(quote['10. change percent'].rstrip('%')),
                    'open': float(quote['02. open']),
                    'high': float(quote['03. high']),
                    'low': float(quote['04. low']),
                    'previous_close': float(quote['08. previous close'])
                }

                # Cache
                self.cache[cache_key] = (datetime.now(), result)

                return result
            else:
                print(f"⚠️  Pas de données Alpha Vantage pour {ticker}")
                if 'Note' in data:
                    print(f"    Note API: {data['Note']}")
                return None

        except Exception as e:
            print(f"❌ Erreur Alpha Vantage pour {ticker}: {e}")
            return None

    def get_live_forex_rate(self, from_currency: str, to_currency: str) -> Dict:
        """
        Récupère le taux de change forex en temps réel

        Args:
            from_currency: Devise source (ex: 'EUR')
            to_currency: Devise cible (ex: 'USD')

        Returns:
            Dict avec 'rate', 'timestamp'
        """
        # Check cache
        cache_key = f"forex_{from_currency}{to_currency}"
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if (datetime.now() - cached_time).seconds < self.cache_duration:
                return cached_data

        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'CURRENCY_EXCHANGE_RATE',
            'from_currency': from_currency,
            'to_currency': to_currency,
            'apikey': self.av_key
        }

        try:
            response = requests.get(url, params=params)
            data = response.json()

            if 'Realtime Currency Exchange Rate' in data:
                rate_data = data['Realtime Currency Exchange Rate']

                result = {
                    'pair': f"{from_currency}/{to_currency}",
                    'rate': float(rate_data['5. Exchange Rate']),
                    'timestamp': datetime.strptime(
                        rate_data['6. Last Refreshed'],
                        '%Y-%m-%d %H:%M:%S'
                    ),
                    'bid': float(rate_data['8. Bid Price']),
                    'ask': float(rate_data['9. Ask Price']),
                }

                # Cache
                self.cache[cache_key] = (datetime.now(), result)

                return result
            else:
                print(f"⚠️  Pas de données forex pour {from_currency}/{to_currency}")
                return None

        except Exception as e:
            print(f"❌ Erreur forex pour {from_currency}/{to_currency}: {e}")
            return None

    def get_live_commodity_price(self, commodity: str) -> Dict:
        """
        Récupère le prix commodité en temps réel depuis FRED ou Alpha Vantage

        Args:
            commodity: Code commodité (ex: 'CL=F' pour crude oil)

        Returns:
            Dict avec 'price', 'timestamp'
        """
        # Pour commodités, utiliser Alpha Vantage stocks (même API)
        return self.get_live_stock_price(commodity)

    def get_intraday_data(self, symbol: str, interval: str = '5min',
                          asset_type: str = 'stock') -> pd.DataFrame:
        """
        Récupère les données intraday (pour calcul indicateurs)
        Essaie Yahoo Finance d'abord, puis Alpha Vantage

        Args:
            symbol: Symbole actif
            interval: Intervalle ('1min', '5min', '15min', '30min', '60min')
            asset_type: 'stock', 'crypto', 'forex'

        Returns:
            DataFrame avec OHLCV
        """
        if asset_type == 'crypto':
            return self._get_crypto_intraday(symbol, interval)
        else:
            # PRIORITÉ 1: Yahoo Finance (si activé)
            if self.use_yahoo and self.yahoo:
                try:
                    yahoo_symbol = self.convert_symbol(symbol) if hasattr(self, 'convert_symbol') else symbol

                    # Convertir format interval pour Yahoo Finance
                    interval_yahoo_map = {
                        '1min': '1m',
                        '5min': '5m',
                        '15min': '15m',
                        '30min': '30m',
                        '60min': '1h',
                        '1h': '1h',
                        '4h': '1h',  # Yahoo n'a pas 4h, utiliser 1h (plus de données)
                        '1d': '1d'
                    }
                    interval_yahoo = interval_yahoo_map.get(interval, interval)

                    df = self.yahoo.get_intraday_data(yahoo_symbol, interval=interval_yahoo)

                    if len(df) > 0:
                        if self.db:
                            try:
                                self.db.upsert_candles(symbol, interval_yahoo, df, source='yahoo')
                            except Exception:
                                pass
                        return df
                except Exception as e:
                    print(f"⚠️  Yahoo intraday fallback pour {symbol}")

            # FALLBACK: Alpha Vantage
            return self._get_av_intraday(symbol, interval)

    def _get_crypto_intraday(self, symbol: str, interval: str) -> pd.DataFrame:
        """Récupère données intraday crypto depuis Binance"""
        try:
            # Convertir interval format
            interval_map = {
                '1min': '1m',
                '5min': '5m',
                '15min': '15m',
                '30min': '30m',
                '60min': '1h',
                '1h': '1h',
                '4h': '4h',
                '1d': '1d'
            }
            binance_interval = interval_map.get(interval, '5m')

            # Récupérer 100 dernières bougies
            ohlcv = self.binance.fetch_ohlcv(symbol, binance_interval, limit=100)

            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            if self.db:
                try:
                    self.db.upsert_candles(symbol, binance_interval, df, source='binance')
                except Exception:
                    pass

            return df

        except Exception as e:
            print(f"❌ Erreur intraday crypto {symbol}: {e}")
            return pd.DataFrame()

    def _get_av_intraday(self, symbol: str, interval: str) -> pd.DataFrame:
        """Récupère données intraday stocks/forex depuis Alpha Vantage"""
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': symbol,
            'interval': interval,
            'apikey': self.av_key,
            'outputsize': 'compact'  # 100 dernières données
        }

        try:
            response = requests.get(url, params=params)
            data = response.json()

            time_series_key = f'Time Series ({interval})'

            if time_series_key in data:
                df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
                df.index = pd.to_datetime(df.index)
                df.columns = ['open', 'high', 'low', 'close', 'volume']
                df = df.astype(float)
                df = df.sort_index()

                return df
            else:
                print(f"⚠️  Pas de données intraday pour {symbol}")
                return pd.DataFrame()

        except Exception as e:
            print(f"❌ Erreur intraday {symbol}: {e}")
            return pd.DataFrame()

    def get_latest_macro_indicators(self) -> Dict:
        """
        Récupère les derniers indicateurs macroéconomiques depuis FRED

        Returns:
            Dict avec indicateurs clés (Fed Funds, CPI, etc.)
        """
        indicators = {
            'FEDFUNDS': 'Federal Funds Rate',
            'CPIAUCSL': 'Consumer Price Index',
            'UNRATE': 'Unemployment Rate',
            'DGS10': '10-Year Treasury Yield',
            'DTWEXBGS': 'Dollar Index',
        }

        results = {}

        for code, name in indicators.items():
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                'series_id': code,
                'api_key': self.fred_key,
                'file_type': 'json',
                'sort_order': 'desc',
                'limit': 1
            }

            try:
                response = requests.get(url, params=params)
                data = response.json()

                if 'observations' in data and len(data['observations']) > 0:
                    latest = data['observations'][0]
                    results[name] = {
                        'value': float(latest['value']) if latest['value'] != '.' else None,
                        'date': latest['date'],
                        'code': code
                    }

            except Exception as e:
                print(f"⚠️  Erreur FRED {code}: {e}")
                results[name] = None

        return results

    def get_market_status(self) -> Dict:
        """
        Vérifie le statut des marchés (ouvert/fermé)

        Returns:
            Dict avec statut par marché
        """
        now = datetime.now()

        # NYSE/NASDAQ (9:30 - 16:00 ET, lun-ven)
        # Simplifié : vérifier jour de semaine + heure approximative
        is_weekday = now.weekday() < 5

        # TODO: Améliorer avec vraie timezone et holidays
        us_market_open = is_weekday and (9 <= now.hour < 16)

        return {
            'us_stocks': 'OPEN' if us_market_open else 'CLOSED',
            'crypto': 'OPEN',  # 24/7
            'forex': 'OPEN' if now.weekday() < 5 else 'CLOSED',  # 24/5
            'timestamp': now
        }


# ============================================================================
# Script de test
# ============================================================================

if __name__ == '__main__':
    # Charger les keys depuis macro_data_fetcher
    import sys
    sys.path.append('.')
    from src.data.macro_fetcher import MacroDataFetcher

    fetcher_macro = MacroDataFetcher()
    av_key = fetcher_macro.alpha_vantage_key
    fred_key = fetcher_macro.fred_api_key

    if not av_key or not fred_key:
        print("❌ API keys non configurées")
        exit(1)

    feed = LiveDataFeed(av_key, fred_key)

    print("="*80)
    print("🔴 DONNÉES EN TEMPS RÉEL")
    print("="*80)

    # Test crypto
    print("\n📊 CRYPTO (Binance)")
    print("-"*80)
    btc = feed.get_live_crypto_price('BTC/USDT')
    if btc:
        print(f"Bitcoin     : ${btc['price']:,.2f}")
        print(f"Change 24h  : {btc['change_24h_pct']:+.2f}%")
        print(f"Volume 24h  : ${btc['volume_24h']:,.0f}")
        print(f"Spread      : {btc['spread_pct']:.3f}%")

    eth = feed.get_live_crypto_price('ETH/USDT')
    if eth:
        print(f"\nEthereum    : ${eth['price']:,.2f}")
        print(f"Change 24h  : {eth['change_24h_pct']:+.2f}%")

    # Test stocks
    print("\n📈 STOCKS (Alpha Vantage)")
    print("-"*80)
    print("⚠️  Limite: 25 calls/jour gratuit (attendre entre chaque...)")

    nvda = feed.get_live_stock_price('NVDA')
    if nvda:
        print(f"Nvidia      : ${nvda['price']:,.2f}")
        print(f"Change      : {nvda['change_pct']:+.2f}%")
        print(f"Volume      : {nvda['volume']:,}")

    time.sleep(12)  # Respecter rate limit Alpha Vantage (5 calls/min)

    # Test forex
    print("\n💱 FOREX (Alpha Vantage)")
    print("-"*80)
    eur_usd = feed.get_live_forex_rate('EUR', 'USD')
    if eur_usd:
        print(f"EUR/USD     : {eur_usd['rate']:.4f}")
        print(f"Spread      : {eur_usd['ask'] - eur_usd['bid']:.5f}")

    # Indicateurs macro
    print("\n🌍 INDICATEURS MACRO (FRED)")
    print("-"*80)
    macro = feed.get_latest_macro_indicators()
    for name, data in macro.items():
        if data and data['value'] is not None:
            print(f"{name:30} : {data['value']:8.2f} (au {data['date']})")
        else:
            print(f"{name:30} : N/A")

    # Statut marchés
    print("\n🕐 STATUT DES MARCHÉS")
    print("-"*80)
    status = feed.get_market_status()
    print(f"US Stocks  : {status['us_stocks']}")
    print(f"Crypto     : {status['crypto']}")
    print(f"Forex      : {status['forex']}")

    # Test données intraday
    print("\n📊 DONNÉES INTRADAY (100 dernières bougies)")
    print("-"*80)
    btc_intraday = feed.get_intraday_data('BTC/USDT', interval='5min', asset_type='crypto')
    if len(btc_intraday) > 0:
        print(f"✅ Bitcoin 5min: {len(btc_intraday)} bougies récupérées")
        print(f"   Dernière close: ${btc_intraday.iloc[-1]['close']:,.2f}")
        print(f"   Timestamp: {btc_intraday.index[-1]}")

    print("\n" + "="*80)
    print("✅ Test terminé!")
    print("="*80)
