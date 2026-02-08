"""
Exemples de stratégies de trading
"""

import pandas as pd
import numpy as np
from indicators import TechnicalIndicators


class BaseStrategy:
    """Classe de base pour toutes les stratégies"""
    
    def __init__(self, name: str):
        self.name = name
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Génère les signaux de trading
        
        Returns:
            DataFrame avec colonne 'signal':
                1 = Acheter (LONG)
                -1 = Vendre à découvert (SHORT)
                0 = Sortir de position
        """
        raise NotImplementedError("Cette méthode doit être implémentée par les sous-classes")


class MovingAverageCrossover(BaseStrategy):
    """
    Stratégie de croisement de moyennes mobiles
    
    Signal d'achat: quand la moyenne rapide croise au-dessus de la lente
    Signal de vente: quand la moyenne rapide croise en-dessous de la lente
    """
    
    def __init__(self, fast_period: int = 20, slow_period: int = 50):
        super().__init__(f"MA Crossover ({fast_period}/{slow_period})")
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        # Calculer les moyennes mobiles
        signals['ma_fast'] = TechnicalIndicators.sma(data['close'], self.fast_period)
        signals['ma_slow'] = TechnicalIndicators.sma(data['close'], self.slow_period)
        
        # Générer les signaux de croisement
        signals['position'] = 0
        signals.loc[signals['ma_fast'] > signals['ma_slow'], 'position'] = 1
        signals.loc[signals['ma_fast'] < signals['ma_slow'], 'position'] = -1
        
        # Détecter les changements de position
        signals['signal'] = signals['position'].diff()
        
        return signals


class RSIStrategy(BaseStrategy):
    """
    Stratégie basée sur le RSI
    
    Signal d'achat: RSI < seuil de survente (ex: 30)
    Signal de vente: RSI > seuil de surachat (ex: 70)
    """
    
    def __init__(self, period: int = 14, oversold: int = 30, overbought: int = 70):
        super().__init__(f"RSI Strategy ({period}, {oversold}/{overbought})")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        # Calculer le RSI
        signals['rsi'] = TechnicalIndicators.rsi(data['close'], self.period)
        
        # Positions basées sur RSI
        signals['position'] = 0
        signals.loc[signals['rsi'] < self.oversold, 'position'] = 1  # Survente -> Acheter
        signals.loc[signals['rsi'] > self.overbought, 'position'] = -1  # Surachat -> Vendre
        
        # Détecter les changements
        signals['signal'] = signals['position'].diff()
        
        return signals


class MACDStrategy(BaseStrategy):
    """
    Stratégie basée sur le MACD
    
    Signal d'achat: MACD croise au-dessus de la ligne de signal
    Signal de vente: MACD croise en-dessous de la ligne de signal
    """
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__(f"MACD Strategy ({fast}/{slow}/{signal})")
        self.fast = fast
        self.slow = slow
        self.signal_period = signal
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        # Calculer le MACD
        macd_data = TechnicalIndicators.macd(data['close'], self.fast, self.slow, self.signal_period)
        signals['macd'] = macd_data['macd']
        signals['macd_signal'] = macd_data['signal']
        
        # Position basée sur le croisement
        signals['position'] = 0
        signals.loc[signals['macd'] > signals['macd_signal'], 'position'] = 1
        signals.loc[signals['macd'] < signals['macd_signal'], 'position'] = -1
        
        # Détecter les changements
        signals['signal'] = signals['position'].diff()
        
        return signals


class CombinedStrategy(BaseStrategy):
    """
    Stratégie combinée: MA Crossover + RSI + MACD
    
    Tous les signaux doivent être alignés pour entrer en position
    """
    
    def __init__(self, 
                 ma_fast: int = 20, 
                 ma_slow: int = 50,
                 rsi_period: int = 14,
                 rsi_oversold: int = 30,
                 rsi_overbought: int = 70):
        super().__init__("Combined Strategy")
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        # Moyennes mobiles
        signals['ma_fast'] = TechnicalIndicators.sma(data['close'], self.ma_fast)
        signals['ma_slow'] = TechnicalIndicators.sma(data['close'], self.ma_slow)
        
        # RSI
        signals['rsi'] = TechnicalIndicators.rsi(data['close'], self.rsi_period)
        
        # MACD
        macd_data = TechnicalIndicators.macd(data['close'])
        signals['macd'] = macd_data['macd']
        signals['macd_signal'] = macd_data['signal']
        
        # Conditions pour LONG
        ma_bullish = signals['ma_fast'] > signals['ma_slow']
        rsi_oversold = signals['rsi'] < self.rsi_oversold
        macd_bullish = signals['macd'] > signals['macd_signal']
        
        # Conditions pour SHORT
        ma_bearish = signals['ma_fast'] < signals['ma_slow']
        rsi_overbought = signals['rsi'] > self.rsi_overbought
        macd_bearish = signals['macd'] < signals['macd_signal']
        
        # Signaux combinés (au moins 2 sur 3 doivent être alignés)
        signals['position'] = 0
        bullish_score = ma_bullish.astype(int) + rsi_oversold.astype(int) + macd_bullish.astype(int)
        bearish_score = ma_bearish.astype(int) + rsi_overbought.astype(int) + macd_bearish.astype(int)
        
        signals.loc[bullish_score >= 2, 'position'] = 1
        signals.loc[bearish_score >= 2, 'position'] = -1
        
        # Détecter les changements
        signals['signal'] = signals['position'].diff()
        
        return signals


class BollingerBandsStrategy(BaseStrategy):
    """
    Stratégie basée sur les Bandes de Bollinger
    
    Signal d'achat: prix touche la bande basse
    Signal de vente: prix touche la bande haute
    """
    
    def __init__(self, period: int = 20, num_std: float = 2):
        super().__init__(f"Bollinger Bands ({period}, {num_std}σ)")
        self.period = period
        self.num_std = num_std
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        # Calculer les Bollinger Bands
        bb = TechnicalIndicators.bollinger_bands(data['close'], self.period, self.num_std)
        signals['bb_upper'] = bb['upper']
        signals['bb_lower'] = bb['lower']
        signals['bb_middle'] = bb['middle']
        
        # Signaux
        signals['position'] = 0
        
        # Acheter quand le prix touche la bande basse
        signals.loc[data['close'] <= signals['bb_lower'], 'position'] = 1
        
        # Vendre quand le prix touche la bande haute
        signals.loc[data['close'] >= signals['bb_upper'], 'position'] = -1
        
        # Sortir quand on retourne à la moyenne
        signals.loc[(data['close'] > signals['bb_lower']) & 
                   (data['close'] < signals['bb_upper']), 'position'] = 0
        
        # Détecter les changements
        signals['signal'] = signals['position'].diff()
        
        return signals
