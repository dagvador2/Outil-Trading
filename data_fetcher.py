"""
Récupération de données de marché
Supporte: Actions, Crypto, Forex, Indices
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional


class DataFetcher:
    """
    Classe pour récupérer les données de différents marchés
    """
    
    @staticmethod
    def fetch_stock_data(symbol: str, 
                        start_date: str, 
                        end_date: str,
                        interval: str = '1d') -> pd.DataFrame:
        """
        Récupère les données d'actions via yfinance
        
        Args:
            symbol: Ticker de l'action (ex: 'AAPL', 'MSFT')
            start_date: Date de début (format 'YYYY-MM-DD')
            end_date: Date de fin (format 'YYYY-MM-DD')
            interval: Intervalle ('1d', '1h', '5m', etc.)
            
        Returns:
            DataFrame avec colonnes OHLCV
        """
        try:
            import yfinance as yf
            
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)
            
            # Normaliser les noms de colonnes
            df.columns = [col.lower() for col in df.columns]
            df = df.rename(columns={'adj close': 'adj_close'})
            
            # Garder seulement les colonnes OHLCV
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            return df
            
        except ImportError:
            print("yfinance n'est pas installé. Installez-le avec: pip install yfinance")
            return None
        except Exception as e:
            print(f"Erreur lors de la récupération des données: {e}")
            return None
    
    @staticmethod
    def fetch_crypto_data(symbol: str,
                         start_date: str,
                         end_date: str,
                         exchange: str = 'binance',
                         timeframe: str = '1d') -> pd.DataFrame:
        """
        Récupère les données de crypto via ccxt
        
        Args:
            symbol: Paire de trading (ex: 'BTC/USDT', 'ETH/USDT')
            start_date: Date de début
            end_date: Date de fin
            exchange: Nom de l'exchange ('binance', 'coinbase', etc.)
            timeframe: Intervalle ('1m', '5m', '1h', '1d', etc.)
            
        Returns:
            DataFrame avec colonnes OHLCV
        """
        try:
            import ccxt
            
            # Initialiser l'exchange
            exchange_class = getattr(ccxt, exchange)
            exchange_obj = exchange_class()
            
            # Convertir les dates en timestamps
            start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
            end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)
            
            # Récupérer les données
            all_ohlcv = []
            current_ts = start_ts
            
            while current_ts < end_ts:
                ohlcv = exchange_obj.fetch_ohlcv(symbol, timeframe, current_ts, limit=1000)
                if not ohlcv:
                    break
                all_ohlcv.extend(ohlcv)
                current_ts = ohlcv[-1][0] + 1
            
            # Créer le DataFrame
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Filtrer par date
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            return df
            
        except ImportError:
            print("ccxt n'est pas installé. Installez-le avec: pip install ccxt")
            return None
        except Exception as e:
            print(f"Erreur lors de la récupération des données crypto: {e}")
            return None
    
    @staticmethod
    def generate_sample_data(symbol: str = 'SAMPLE',
                           start_date: str = '2023-01-01',
                           end_date: str = '2024-01-01',
                           initial_price: float = 100,
                           trend: float = 0.0002,
                           volatility: float = 0.02) -> pd.DataFrame:
        """
        Génère des données synthétiques pour tester (mouvement brownien géométrique)
        
        Args:
            symbol: Nom du symbole
            start_date: Date de début
            end_date: Date de fin
            initial_price: Prix initial
            trend: Tendance moyenne (drift)
            volatility: Volatilité
            
        Returns:
            DataFrame avec colonnes OHLCV
        """
        # Créer la plage de dates
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n = len(dates)
        
        # Générer les rendements
        np.random.seed(42)
        returns = np.random.normal(trend, volatility, n)
        
        # Calculer les prix
        prices = initial_price * np.exp(np.cumsum(returns))
        
        # Générer OHLCV
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            # Simuler open, high, low
            daily_volatility = volatility * close
            open_price = close + np.random.normal(0, daily_volatility * 0.5)
            high = max(open_price, close) + abs(np.random.normal(0, daily_volatility * 0.3))
            low = min(open_price, close) - abs(np.random.normal(0, daily_volatility * 0.3))
            volume = np.random.uniform(1000000, 5000000)
            
            data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        df = pd.DataFrame(data, index=dates)
        df.index.name = 'date'
        
        return df
    
    @staticmethod
    def fetch_forex_data(pair: str,
                        start_date: str,
                        end_date: str,
                        timeframe: str = 'D') -> pd.DataFrame:
        """
        Récupère les données Forex via yfinance
        
        Args:
            pair: Paire de devises (ex: 'EURUSD=X', 'GBPUSD=X')
            start_date: Date de début
            end_date: Date de fin
            timeframe: Intervalle
            
        Returns:
            DataFrame avec colonnes OHLCV
        """
        # Ajouter =X si pas déjà présent
        if not pair.endswith('=X'):
            pair = f"{pair}=X"
        
        return DataFetcher.fetch_stock_data(pair, start_date, end_date, timeframe)
    
    @staticmethod
    def save_data(df: pd.DataFrame, filename: str):
        """Sauvegarde les données dans un fichier CSV"""
        df.to_csv(filename)
        print(f"Données sauvegardées dans {filename}")
    
    @staticmethod
    def load_data(filename: str) -> pd.DataFrame:
        """Charge les données depuis un fichier CSV"""
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
        return df


# Fonction helper pour obtenir des données facilement
def get_data(asset_type: str,
            symbol: str,
            start_date: str = '2023-01-01',
            end_date: str = '2024-01-01',
            **kwargs) -> pd.DataFrame:
    """
    Fonction wrapper pour récupérer des données selon le type d'actif
    
    Args:
        asset_type: Type d'actif ('stock', 'crypto', 'forex', 'sample')
        symbol: Symbole de l'actif
        start_date: Date de début
        end_date: Date de fin
        **kwargs: Arguments supplémentaires spécifiques au type
        
    Returns:
        DataFrame avec données OHLCV
    """
    fetcher = DataFetcher()
    
    if asset_type.lower() == 'stock':
        return fetcher.fetch_stock_data(symbol, start_date, end_date, 
                                       kwargs.get('interval', '1d'))
    
    elif asset_type.lower() == 'crypto':
        return fetcher.fetch_crypto_data(symbol, start_date, end_date,
                                        kwargs.get('exchange', 'binance'),
                                        kwargs.get('timeframe', '1d'))
    
    elif asset_type.lower() == 'forex':
        return fetcher.fetch_forex_data(symbol, start_date, end_date,
                                       kwargs.get('timeframe', 'D'))
    
    elif asset_type.lower() == 'sample':
        return fetcher.generate_sample_data(symbol, start_date, end_date,
                                           kwargs.get('initial_price', 100),
                                           kwargs.get('trend', 0.0002),
                                           kwargs.get('volatility', 0.02))
    
    else:
        raise ValueError(f"Type d'actif non supporté: {asset_type}")
