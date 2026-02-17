"""
Yahoo Finance Data Feed (Alternative gratuite à Alpha Vantage)
Couvre : Stocks, Indices, Forex, Commodities
API non-officielle mais gratuite et illimitée
"""

import yfinance as yf
from datetime import datetime
from typing import Dict, Optional
import pandas as pd


class YahooDataFeed:
    """
    Récupère données via Yahoo Finance (gratuit, illimité)
    Compatible avec stocks, indices, forex, commodities
    """

    def __init__(self):
        self.cache = {}
        self.cache_duration = 60  # 60 secondes

    def get_live_price(self, symbol: str) -> Optional[Dict]:
        """
        Prix actuel d'un actif (stock/commodity/forex/indice)

        Args:
            symbol: Symbole Yahoo Finance

        Returns:
            Dict avec prix et métadonnées
        """
        # Check cache
        cache_key = f"price_{symbol}"
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if (datetime.now() - cached_time).seconds < self.cache_duration:
                return cached_data

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Prix actuel (plusieurs fallbacks possibles)
            price = (
                info.get('regularMarketPrice') or
                info.get('currentPrice') or
                info.get('previousClose') or
                0
            )

            if price == 0:
                return None

            result = {
                'symbol': symbol,
                'price': price,
                'timestamp': datetime.now(),
                'volume': info.get('regularMarketVolume', info.get('volume', 0)),
                'change': info.get('regularMarketChange', 0),
                'change_pct': info.get('regularMarketChangePercent', 0),
                'open': info.get('regularMarketOpen', info.get('open', 0)),
                'high': info.get('regularMarketDayHigh', info.get('dayHigh', 0)),
                'low': info.get('regularMarketDayLow', info.get('dayLow', 0)),
                'previous_close': info.get('regularMarketPreviousClose', info.get('previousClose', 0))
            }

            # Cache
            self.cache[cache_key] = (datetime.now(), result)

            return result

        except Exception as e:
            print(f"⚠️  Yahoo Finance {symbol}: {str(e)[:50]}")
            return None

    def get_intraday_data(self, symbol: str, interval: str = '5m',
                          period: str = '1d') -> pd.DataFrame:
        """
        Données intraday pour calcul d'indicateurs

        Args:
            symbol: Symbole Yahoo Finance
            interval: '1m', '2m', '5m', '15m', '30m', '60m', '1h'
            period: '1d', '5d', '1mo', '3mo', '6mo', '1y'

        Returns:
            DataFrame OHLCV
        """
        try:
            ticker = yf.Ticker(symbol)

            # Mapping interval → period optimal
            period_map = {
                '1m': '1d',
                '2m': '1d',
                '5m': '1d',
                '15m': '5d',
                '30m': '5d',
                '60m': '1mo',
                '1h': '1mo',
                '4h': '3mo',  # 4h nécessite plus de période pour 100 points
                '1d': '2y'    # 1 jour nécessite 2 ans pour 100+ points
            }

            if period == '1d':  # Utiliser mapping si period par défaut
                period = period_map.get(interval, '1d')

            # Récupérer données
            data = ticker.history(period=period, interval=interval)

            if len(data) == 0:
                return pd.DataFrame()

            # Nettoyer et formater
            data.columns = [col.lower() for col in data.columns]

            # Garder seulement OHLCV
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            available_cols = [col for col in required_cols if col in data.columns]

            if len(available_cols) < 4:  # Minimum O, H, L, C
                return pd.DataFrame()

            return data[available_cols]

        except Exception as e:
            print(f"⚠️  Yahoo intraday {symbol}: {str(e)[:50]}")
            return pd.DataFrame()

    def get_historical_data(self, symbol: str, start_date: str,
                            end_date: str) -> pd.DataFrame:
        """
        Données historiques daily

        Args:
            symbol: Symbole Yahoo Finance
            start_date: Date début 'YYYY-MM-DD'
            end_date: Date fin 'YYYY-MM-DD'

        Returns:
            DataFrame OHLCV
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)

            if len(data) == 0:
                return pd.DataFrame()

            data.columns = [col.lower() for col in data.columns]

            return data

        except Exception as e:
            print(f"⚠️  Yahoo historical {symbol}: {str(e)[:50]}")
            return pd.DataFrame()

    def test_symbol(self, symbol: str) -> bool:
        """
        Teste si un symbole est valide

        Args:
            symbol: Symbole à tester

        Returns:
            True si données disponibles
        """
        data = self.get_live_price(symbol)
        return data is not None and data.get('price', 0) > 0


# ============================================================================
# Mapping des symboles (votre config → Yahoo Finance)
# ============================================================================

SYMBOL_MAPPING = {
    # Vos symboles → Symboles Yahoo Finance

    # Commodities (=F reste identique)
    'GC=F': 'GC=F',      # Gold
    'SI=F': 'SI=F',      # Silver
    'CL=F': 'CL=F',      # Crude Oil
    'NG=F': 'NG=F',      # Natural Gas
    'HG=F': 'HG=F',      # Copper
    'PL=F': 'PL=F',      # Platinum
    'PA=F': 'PA=F',      # Palladium
    'ZC=F': 'ZC=F',      # Corn
    'ZW=F': 'ZW=F',      # Wheat
    'ZS=F': 'ZS=F',      # Soybeans
    'KC=F': 'KC=F',      # Coffee
    'CT=F': 'CT=F',      # Cotton

    # Indices (^ reste identique)
    '^GSPC': '^GSPC',    # S&P 500
    '^IXIC': '^IXIC',    # Nasdaq
    '^DJI': '^DJI',      # Dow Jones
    '^FTSE': '^FTSE',    # FTSE 100
    '^GDAXI': '^GDAXI',  # DAX
    '^FCHI': '^FCHI',    # CAC 40
    '^N225': '^N225',    # Nikkei 225
    '^RUT': '^RUT',      # Russell 2000
    '^HSI': '^HSI',      # Hang Seng
    '^STOXX50E': '^STOXX50E',  # Euro Stoxx 50

    # Forex (=X reste identique)
    'EURUSD=X': 'EURUSD=X',
    'GBPUSD=X': 'GBPUSD=X',
    'USDJPY=X': 'JPY=X',     # Yahoo utilise JPY=X
    'USDCHF=X': 'CHF=X',     # Yahoo utilise CHF=X
    'AUDUSD=X': 'AUDUSD=X',
    'NZDUSD=X': 'NZDUSD=X',
    'USDCAD=X': 'CAD=X',     # Yahoo utilise CAD=X
    'EURGBP=X': 'EURGBP=X',
    'EURJPY=X': 'EURJPY=X',
    'GBPJPY=X': 'GBPJPY=X',

    # Tech Stocks
    'NVDA': 'NVDA',
    'AAPL': 'AAPL',
    'MSFT': 'MSFT',
    'GOOGL': 'GOOGL',
    'AMZN': 'AMZN',
    'META': 'META',
    'TSLA': 'TSLA',
    'AMD': 'AMD',
    'NFLX': 'NFLX',
    'COIN': 'COIN',

    # Semiconductors
    'AVGO': 'AVGO',
    'INTC': 'INTC',
    'QCOM': 'QCOM',

    # Finance
    'JPM': 'JPM',
    'GS': 'GS',
    'V': 'V',
    'MA': 'MA',
    'BAC': 'BAC',

    # Healthcare
    'UNH': 'UNH',
    'ABBV': 'ABBV',
    'LLY': 'LLY',
    'MRK': 'MRK',
    'PFE': 'PFE',

    # Energy
    'XOM': 'XOM',
    'CVX': 'CVX',
    'COP': 'COP',
    'SLB': 'SLB',
    'EOG': 'EOG',

    # Consumer
    'NKE': 'NKE',
    'SBUX': 'SBUX',
    'MCD': 'MCD',
    'HD': 'HD',
    'COST': 'COST',

    # Defensive Stocks
    'JNJ': 'JNJ',
    'PG': 'PG',
    'KO': 'KO',
    'PEP': 'PEP',
    'WMT': 'WMT',

    # ETF
    'SPY': 'SPY',
    'QQQ': 'QQQ',
    'IWM': 'IWM',
    'EEM': 'EEM',
    'GLD': 'GLD',
    'SLV': 'SLV',
    'TLT': 'TLT',
    'XLF': 'XLF',
    'XLE': 'XLE',
    'VNQ': 'VNQ',

    # Cryptos (Yahoo utilise -USD au lieu de /USDT)
    'BTC/USDT': 'BTC-USD',
    'ETH/USDT': 'ETH-USD',
    'BNB/USDT': 'BNB-USD',
    'SOL/USDT': 'SOL-USD',
    'XRP/USDT': 'XRP-USD',
    'ADA/USDT': 'ADA-USD',
    'AVAX/USDT': 'AVAX-USD',
    'DOT/USDT': 'DOT-USD',
    'LINK/USDT': 'LINK-USD',
    'TRX/USDT': 'TRX-USD',
    'UNI/USDT': 'UNI7083-USD',  # Yahoo a un ID spécial pour UNI
    'LTC/USDT': 'LTC-USD',
    'ATOM/USDT': 'ATOM-USD',
    'ARB/USDT': 'ARB11841-USD',  # Yahoo a un ID spécial
    'OP/USDT': 'OP-USD',
    'DOGE/USDT': 'DOGE-USD',
    'SHIB/USDT': 'SHIB-USD',
    'NEAR/USDT': 'NEAR-USD',
    'SUI/USDT': 'SUI20947-USD',
    'ALGO/USDT': 'ALGO-USD',

    # ============================================================
    # NOUVEAUX ACTIFS (100 supplementaires)
    # ============================================================

    # Crypto supplementaires
    'MATIC/USDT': 'MATIC-USD',
    'FTM/USDT': 'FTM-USD',
    'AAVE/USDT': 'AAVE-USD',
    'MKR/USDT': 'MKR-USD',
    'CRV/USDT': 'CRV-USD',
    'APE/USDT': 'APE18876-USD',
    'FIL/USDT': 'FIL-USD',
    'SAND/USDT': 'SAND-USD',
    'MANA/USDT': 'MANA-USD',
    'AXS/USDT': 'AXS-USD',

    # Tech & Growth US extra
    'CRM': 'CRM',
    'ORCL': 'ORCL',
    'IBM': 'IBM',
    'DIS': 'DIS',
    'UBER': 'UBER',
    'SQ': 'SQ',
    'PYPL': 'PYPL',
    'PLTR': 'PLTR',
    'RIVN': 'RIVN',
    'BABA': 'BABA',
    'SHOP': 'SHOP',
    'SNAP': 'SNAP',
    'ROKU': 'ROKU',
    'ZM': 'ZM',
    'ABNB': 'ABNB',

    # Commodities extra
    'SB=F': 'SB=F',
    'CC=F': 'CC=F',
    'OJ=F': 'OJ=F',
    'LBS=F': 'LBS=F',
    'LE=F': 'LE=F',

    # Indices extra
    '^IBEX': '^IBEX',
    '^AEX': '^AEX',
    '^SSMI': '^SSMI',
    '^KS11': '^KS11',
    '^BSESN': '^BSESN',

    # Forex extra
    'EURCHF=X': 'EURCHF=X',
    'EURAUD=X': 'EURAUD=X',
    'GBPCHF=X': 'GBPCHF=X',
    'USDNOK=X': 'USDNOK=X',
    'USDSEK=X': 'USDSEK=X',

    # ETF Europe
    'EWG': 'EWG',
    'EWU': 'EWU',
    'EWQ': 'EWQ',
    'EWP': 'EWP',
    'EWI': 'EWI',
    'EWL': 'EWL',
    'EWN': 'EWN',
    'IEUR': 'IEUR',
    'VGK': 'VGK',
    'EZU': 'EZU',

    # Actions Francaises (Euronext Paris)
    'MC.PA': 'MC.PA',
    'TTE.PA': 'TTE.PA',
    'SAN.PA': 'SAN.PA',
    'AI.PA': 'AI.PA',
    'SU.PA': 'SU.PA',
    'OR.PA': 'OR.PA',
    'RMS.PA': 'RMS.PA',
    'DSY.PA': 'DSY.PA',
    'BNP.PA': 'BNP.PA',
    'CS.PA': 'CS.PA',

    # Actions Allemandes (Xetra)
    'SAP.DE': 'SAP.DE',
    'SIE.DE': 'SIE.DE',
    'ALV.DE': 'ALV.DE',
    'BAS.DE': 'BAS.DE',
    'DTE.DE': 'DTE.DE',
    'BMW.DE': 'BMW.DE',
    'ADS.DE': 'ADS.DE',
    'MBG.DE': 'MBG.DE',
    'IFX.DE': 'IFX.DE',
    'BAYN.DE': 'BAYN.DE',

    # Actions UK (London Stock Exchange)
    'SHEL.L': 'SHEL.L',
    'AZN.L': 'AZN.L',
    'HSBA.L': 'HSBA.L',
    'ULVR.L': 'ULVR.L',
    'RIO.L': 'RIO.L',
    'BP.L': 'BP.L',
    'GSK.L': 'GSK.L',
    'DGE.L': 'DGE.L',
    'BARC.L': 'BARC.L',
    'RR.L': 'RR.L',

    # Europe Nord & Benelux
    'ASML.AS': 'ASML.AS',
    'PHIA.AS': 'PHIA.AS',
    'INGA.AS': 'INGA.AS',
    'NESN.SW': 'NESN.SW',
    'NOVN.SW': 'NOVN.SW',
    'ROG.SW': 'ROG.SW',
    'UBSG.SW': 'UBSG.SW',
    'ITX.MC': 'ITX.MC',
    'SAN.MC': 'SAN.MC',
    'IBE.MC': 'IBE.MC',

    # Actions Asiatiques
    '7203.T': '7203.T',
    '6758.T': '6758.T',
    '9984.T': '9984.T',
    '6861.T': '6861.T',
    '8306.T': '8306.T',
    '005930.KS': '005930.KS',
    '9988.HK': '9988.HK',
    '0700.HK': '0700.HK',
    '2330.TW': '2330.TW',
    'RELIANCE.NS': 'RELIANCE.NS',
}


def convert_to_yahoo_symbol(symbol: str) -> str:
    """
    Convertit votre symbole en symbole Yahoo Finance

    Args:
        symbol: Votre symbole

    Returns:
        Symbole Yahoo Finance
    """
    return SYMBOL_MAPPING.get(symbol, symbol)


# ============================================================================
# Script de test
# ============================================================================

if __name__ == '__main__':
    feed = YahooDataFeed()

    print("="*80)
    print("🧪 TEST YAHOO FINANCE DATA FEED")
    print("="*80)

    # Test quelques symboles
    test_symbols = [
        ('AAPL', 'Apple'),
        ('GC=F', 'Gold'),
        ('^GSPC', 'S&P 500'),
        ('EURUSD=X', 'EUR/USD'),
        ('BTC-USD', 'Bitcoin')
    ]

    print("\n📊 Test Prix Actuel")
    print("-"*80)

    success = 0
    for symbol, name in test_symbols:
        data = feed.get_live_price(symbol)
        if data and data['price'] > 0:
            print(f"✅ {name:15} ({symbol:12}) : ${data['price']:,.2f}")
            success += 1
        else:
            print(f"❌ {name:15} ({symbol:12}) : NO DATA")

    print(f"\n📈 Résultat: {success}/{len(test_symbols)} succès")

    # Test données intraday
    print("\n📊 Test Données Intraday (Apple)")
    print("-"*80)

    intraday = feed.get_intraday_data('AAPL', interval='5m')
    if len(intraday) > 0:
        print(f"✅ {len(intraday)} bougies récupérées")
        print(f"   Dernière close: ${intraday.iloc[-1]['close']:,.2f}")
        print(f"   Timestamp: {intraday.index[-1]}")
    else:
        print("❌ Pas de données intraday")

    print("\n" + "="*80)
    print("✅ Test terminé!")
    print("="*80)
