"""
Indicateurs techniques pour le trading
"""

import pandas as pd
import numpy as np


class TechnicalIndicators:
    """
    Collection d'indicateurs techniques
    """
    
    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """
        Simple Moving Average (Moyenne Mobile Simple)
        
        Args:
            data: Série de prix
            period: Période de la moyenne
            
        Returns:
            Série avec la SMA
        """
        return data.rolling(window=period).mean()
    
    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """
        Exponential Moving Average (Moyenne Mobile Exponentielle)
        
        Args:
            data: Série de prix
            period: Période de la moyenne
            
        Returns:
            Série avec l'EMA
        """
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index
        
        Args:
            data: Série de prix
            period: Période du RSI (généralement 14)
            
        Returns:
            Série avec le RSI (0-100)
        """
        # Calculer les variations
        delta = data.diff()
        
        # Séparer gains et pertes
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculer les moyennes
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculer RS et RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def macd(data: pd.Series, 
             fast_period: int = 12, 
             slow_period: int = 26, 
             signal_period: int = 9) -> pd.DataFrame:
        """
        Moving Average Convergence Divergence
        
        Args:
            data: Série de prix
            fast_period: Période de l'EMA rapide
            slow_period: Période de l'EMA lente
            signal_period: Période de la ligne de signal
            
        Returns:
            DataFrame avec colonnes: macd, signal, histogram
        """
        # Calculer les EMAs
        ema_fast = TechnicalIndicators.ema(data, fast_period)
        ema_slow = TechnicalIndicators.ema(data, slow_period)
        
        # MACD line
        macd_line = ema_fast - ema_slow
        
        # Signal line
        signal_line = TechnicalIndicators.ema(macd_line, signal_period)
        
        # Histogram
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        })
    
    @staticmethod
    def stochastic(high: pd.Series, 
                   low: pd.Series, 
                   close: pd.Series,
                   k_period: int = 14,
                   d_period: int = 3) -> pd.DataFrame:
        """
        Stochastic Oscillator
        
        Args:
            high: Série des prix hauts
            low: Série des prix bas
            close: Série des prix de clôture
            k_period: Période pour %K
            d_period: Période pour %D (moyenne de %K)
            
        Returns:
            DataFrame avec colonnes: k, d
        """
        # Plus bas sur la période
        lowest_low = low.rolling(window=k_period).min()
        
        # Plus haut sur la période
        highest_high = high.rolling(window=k_period).max()
        
        # %K
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        
        # %D (moyenne mobile de %K)
        d = k.rolling(window=d_period).mean()
        
        return pd.DataFrame({
            'k': k,
            'd': d
        })
    
    @staticmethod
    def bollinger_bands(data: pd.Series, 
                        period: int = 20, 
                        num_std: float = 2) -> pd.DataFrame:
        """
        Bollinger Bands
        
        Args:
            data: Série de prix
            period: Période de la moyenne mobile
            num_std: Nombre d'écarts-types
            
        Returns:
            DataFrame avec colonnes: middle, upper, lower
        """
        middle = TechnicalIndicators.sma(data, period)
        std = data.rolling(window=period).std()
        
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        
        return pd.DataFrame({
            'middle': middle,
            'upper': upper,
            'lower': lower
        })
    
    @staticmethod
    def atr(high: pd.Series, 
            low: pd.Series, 
            close: pd.Series,
            period: int = 14) -> pd.Series:
        """
        Average True Range
        
        Args:
            high: Série des prix hauts
            low: Série des prix bas
            close: Série des prix de clôture
            period: Période de l'ATR
            
        Returns:
            Série avec l'ATR
        """
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR (moyenne du TR)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute tous les indicateurs principaux à un DataFrame OHLCV
        
        Args:
            df: DataFrame avec colonnes: open, high, low, close, volume
            
        Returns:
            DataFrame enrichi avec tous les indicateurs
        """
        df = df.copy()
        
        # Moyennes mobiles
        df['sma_20'] = TechnicalIndicators.sma(df['close'], 20)
        df['sma_50'] = TechnicalIndicators.sma(df['close'], 50)
        df['sma_200'] = TechnicalIndicators.sma(df['close'], 200)
        df['ema_12'] = TechnicalIndicators.ema(df['close'], 12)
        df['ema_26'] = TechnicalIndicators.ema(df['close'], 26)
        
        # RSI
        df['rsi'] = TechnicalIndicators.rsi(df['close'], 14)
        
        # MACD
        macd_data = TechnicalIndicators.macd(df['close'])
        df['macd'] = macd_data['macd']
        df['macd_signal'] = macd_data['signal']
        df['macd_histogram'] = macd_data['histogram']
        
        # Stochastic
        stoch_data = TechnicalIndicators.stochastic(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch_data['k']
        df['stoch_d'] = stoch_data['d']
        
        # Bollinger Bands
        bb_data = TechnicalIndicators.bollinger_bands(df['close'])
        df['bb_middle'] = bb_data['middle']
        df['bb_upper'] = bb_data['upper']
        df['bb_lower'] = bb_data['lower']
        
        # ATR
        df['atr'] = TechnicalIndicators.atr(df['high'], df['low'], df['close'])
        
        return df
