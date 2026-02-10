"""
3 Stratégies supplémentaires pour atteindre 10 stratégies totales
Total : 7 existantes + 3 nouvelles = 10 stratégies
"""

import pandas as pd
import numpy as np
from strategies import BaseStrategy
from indicators import TechnicalIndicators


class ADXTrendStrategy(BaseStrategy):
    """
    Stratégie basée sur l'ADX (Average Directional Index)
    Entre en position uniquement sur tendances fortes
    """

    def __init__(self, adx_period: int = 14, adx_threshold: float = 25):
        super().__init__(f"ADX Trend ({adx_period}, {adx_threshold})")
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0

        # Calculer ADX
        signals['adx'] = TechnicalIndicators.adx(
            data['high'],
            data['low'],
            data['close'],
            self.adx_period
        )

        # Calculer +DI et -DI pour direction
        signals['plus_di'] = TechnicalIndicators.plus_di(
            data['high'],
            data['low'],
            data['close'],
            self.adx_period
        )

        signals['minus_di'] = TechnicalIndicators.minus_di(
            data['high'],
            data['low'],
            data['close'],
            self.adx_period
        )

        # Position
        signals['position'] = 0

        # LONG: ADX > threshold ET +DI > -DI (tendance haussière forte)
        long_condition = (
            (signals['adx'] > self.adx_threshold) &
            (signals['plus_di'] > signals['minus_di'])
        )

        # SHORT: ADX > threshold ET -DI > +DI (tendance baissière forte)
        short_condition = (
            (signals['adx'] > self.adx_threshold) &
            (signals['minus_di'] > signals['plus_di'])
        )

        signals.loc[long_condition, 'position'] = 1
        signals.loc[short_condition, 'position'] = -1

        # Signaux d'entrée/sortie
        signals['signal'] = signals['position'].diff()

        return signals


class VWAPStrategy(BaseStrategy):
    """
    Stratégie basée sur le VWAP (Volume Weighted Average Price)
    Utilisée par les institutionnels pour exécution optimale
    """

    def __init__(self, lookback_period: int = 20):
        super().__init__(f"VWAP ({lookback_period})")
        self.lookback_period = lookback_period

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0

        # Calculer VWAP rolling
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        signals['vwap'] = (typical_price * data['volume']).rolling(
            window=self.lookback_period
        ).sum() / data['volume'].rolling(window=self.lookback_period).sum()

        # Bandes VWAP (écart-type)
        signals['vwap_std'] = typical_price.rolling(
            window=self.lookback_period
        ).std()

        signals['vwap_upper'] = signals['vwap'] + 2 * signals['vwap_std']
        signals['vwap_lower'] = signals['vwap'] - 2 * signals['vwap_std']

        # Position
        signals['position'] = 0

        # LONG: Prix < VWAP lower band (sous-évalué)
        long_condition = data['close'] < signals['vwap_lower']

        # SHORT: Prix > VWAP upper band (sur-évalué)
        short_condition = data['close'] > signals['vwap_upper']

        # Sortir quand prix revient au VWAP
        exit_condition = (
            (data['close'] > signals['vwap'] * 0.99) &
            (data['close'] < signals['vwap'] * 1.01)
        )

        for i in range(len(signals)):
            if long_condition.iloc[i]:
                signals.iloc[i, signals.columns.get_loc('position')] = 1
            elif short_condition.iloc[i]:
                signals.iloc[i, signals.columns.get_loc('position')] = -1
            elif exit_condition.iloc[i]:
                signals.iloc[i, signals.columns.get_loc('position')] = 0
            elif i > 0:
                # Conserver position précédente
                signals.iloc[i, signals.columns.get_loc('position')] = \
                    signals.iloc[i-1, signals.columns.get_loc('position')]

        # Signaux d'entrée/sortie
        signals['signal'] = signals['position'].diff()

        return signals


class IchimokuStrategy(BaseStrategy):
    """
    Stratégie Ichimoku Cloud (Nuage de Kumo)
    Système complet japonais avec Tenkan, Kijun, Senkou Span, Chikou
    """

    def __init__(self, tenkan_period: int = 9, kijun_period: int = 26,
                 senkou_span_b_period: int = 52):
        super().__init__(f"Ichimoku ({tenkan_period}/{kijun_period}/{senkou_span_b_period})")
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_span_b_period = senkou_span_b_period

    def _calculate_ichimoku(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcule les composantes Ichimoku"""

        high = data['high']
        low = data['low']
        close = data['close']

        # Tenkan-sen (ligne de conversion)
        tenkan_high = high.rolling(window=self.tenkan_period).max()
        tenkan_low = low.rolling(window=self.tenkan_period).min()
        tenkan = (tenkan_high + tenkan_low) / 2

        # Kijun-sen (ligne de base)
        kijun_high = high.rolling(window=self.kijun_period).max()
        kijun_low = low.rolling(window=self.kijun_period).min()
        kijun = (kijun_high + kijun_low) / 2

        # Senkou Span A (Leading Span A)
        senkou_a = ((tenkan + kijun) / 2).shift(self.kijun_period)

        # Senkou Span B (Leading Span B)
        senkou_b_high = high.rolling(window=self.senkou_span_b_period).max()
        senkou_b_low = low.rolling(window=self.senkou_span_b_period).min()
        senkou_b = ((senkou_b_high + senkou_b_low) / 2).shift(self.kijun_period)

        # Chikou Span (Lagging Span)
        chikou = close.shift(-self.kijun_period)

        return pd.DataFrame({
            'tenkan': tenkan,
            'kijun': kijun,
            'senkou_a': senkou_a,
            'senkou_b': senkou_b,
            'chikou': chikou
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0

        # Calculer Ichimoku
        ichimoku = self._calculate_ichimoku(data)
        signals = pd.concat([signals, ichimoku], axis=1)

        # Position
        signals['position'] = 0

        # Conditions LONG:
        # 1. Tenkan > Kijun (momentum haussier)
        # 2. Prix > Nuage (Senkou A et B)
        # 3. Chikou > Prix (confirmation)
        long_condition = (
            (signals['tenkan'] > signals['kijun']) &
            (data['close'] > signals['senkou_a']) &
            (data['close'] > signals['senkou_b']) &
            (signals['chikou'] > data['close'])
        )

        # Conditions SHORT:
        # 1. Tenkan < Kijun (momentum baissier)
        # 2. Prix < Nuage
        # 3. Chikou < Prix
        short_condition = (
            (signals['tenkan'] < signals['kijun']) &
            (data['close'] < signals['senkou_a']) &
            (data['close'] < signals['senkou_b']) &
            (signals['chikou'] < data['close'])
        )

        signals.loc[long_condition, 'position'] = 1
        signals.loc[short_condition, 'position'] = -1

        # Signaux d'entrée/sortie
        signals['signal'] = signals['position'].diff()

        return signals


# ============================================================================
# Liste complète des 10 stratégies
# ============================================================================

STRATEGY_LIBRARY = {
    # 7 stratégies existantes (strategies.py)
    'MA_Crossover_20_50': {
        'class': 'MovingAverageCrossover',
        'params': {'fast_period': 20, 'slow_period': 50},
        'description': 'Croisement moyennes mobiles 20/50'
    },
    'MA_Crossover_10_30': {
        'class': 'MovingAverageCrossover',
        'params': {'fast_period': 10, 'slow_period': 30},
        'description': 'Croisement moyennes mobiles 10/30'
    },
    'RSI_14_30_70': {
        'class': 'RSIStrategy',
        'params': {'period': 14, 'oversold': 30, 'overbought': 70},
        'description': 'RSI 14 avec seuils 30/70'
    },
    'RSI_14_35_80': {
        'class': 'RSIStrategy',
        'params': {'period': 14, 'oversold': 35, 'overbought': 80},
        'description': 'RSI 14 avec seuils 35/80'
    },
    'MACD_Standard': {
        'class': 'MACDStrategy',
        'params': {'fast': 12, 'slow': 26, 'signal': 9},
        'description': 'MACD standard 12/26/9'
    },
    'Bollinger_20_2': {
        'class': 'BollingerBandsStrategy',
        'params': {'period': 20, 'num_std': 2},
        'description': 'Bollinger Bands 20/2'
    },
    'Combined': {
        'class': 'CombinedStrategy',
        'params': {},
        'description': 'Stratégie combinée (MA + RSI + MACD)'
    },

    # 3 nouvelles stratégies (strategies_extended.py)
    'ADX_Trend_14_25': {
        'class': 'ADXTrendStrategy',
        'params': {'adx_period': 14, 'adx_threshold': 25},
        'description': 'ADX Trend 14/25 - Tendances fortes'
    },
    'VWAP_20': {
        'class': 'VWAPStrategy',
        'params': {'lookback_period': 20},
        'description': 'VWAP 20 - Volume Weighted Average Price'
    },
    'Ichimoku_9_26_52': {
        'class': 'IchimokuStrategy',
        'params': {'tenkan_period': 9, 'kijun_period': 26, 'senkou_span_b_period': 52},
        'description': 'Ichimoku Cloud - Système complet japonais'
    },
}


def get_strategy_count():
    """Retourne le nombre total de stratégies disponibles"""
    return len(STRATEGY_LIBRARY)


def get_strategy_names():
    """Retourne la liste des noms de stratégies"""
    return list(STRATEGY_LIBRARY.keys())


def get_strategy_info(strategy_name):
    """Retourne les informations d'une stratégie"""
    return STRATEGY_LIBRARY.get(strategy_name, None)


if __name__ == '__main__':
    print("="*80)
    print("📊 LIBRAIRIE DE STRATÉGIES ÉTENDUE")
    print("="*80)
    print(f"\nTotal stratégies: {get_strategy_count()}")
    print("\nListe des stratégies:")
    for i, (name, info) in enumerate(STRATEGY_LIBRARY.items(), 1):
        print(f"  {i:2d}. {name:25} - {info['description']}")

    print("\n" + "="*80)
    print("✅ 10 stratégies disponibles!")
    print("="*80)
