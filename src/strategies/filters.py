"""
Filtres et wrappers applicables a n'importe quelle strategie existante
- VolumeConfirmedStrategy : supprime les signaux sur faible volume
- RegimeAwareStrategy : utilise ADX pour switcher entre strategie tendance et mean-reversion
- MultiTimeframeStrategy : confirme la direction avec une tendance de long terme (SMA200)
"""

import pandas as pd
import numpy as np
from src.indicators.technical import TechnicalIndicators
from src.strategies.base import BaseStrategy


class VolumeConfirmedStrategy(BaseStrategy):
    """
    Wrapper : supprime les signaux d'entree quand le volume est faible.
    Un breakout sans volume est souvent un faux signal.
    """

    def __init__(self, base_strategy, volume_period: int = 20,
                 volume_multiplier: float = 1.0):
        super().__init__(f"{base_strategy.name} +Vol")
        self.base_strategy = base_strategy
        self.volume_period = volume_period
        self.volume_multiplier = volume_multiplier

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = self.base_strategy.generate_signals(data)

        if 'volume' not in data.columns:
            return signals

        avg_vol = data['volume'].rolling(self.volume_period).mean()
        low_volume = data['volume'] < avg_vol * self.volume_multiplier

        # Sur les barres a faible volume, maintenir la position precedente (pas de nouvelle entree)
        signals = signals.copy()
        for i in range(1, len(signals)):
            if low_volume.iloc[i]:
                current_pos = signals.iloc[i].get('position', 0)
                prev_pos = signals.iloc[i - 1].get('position', 0)
                # Si c'est un changement de position, l'annuler
                if current_pos != prev_pos:
                    signals.iloc[i, signals.columns.get_loc('position')] = prev_pos

        signals['signal'] = signals['position'].diff()
        return signals


class RegimeAwareStrategy(BaseStrategy):
    """
    Switche entre une strategie de tendance (ADX > threshold) et une strategie
    de mean-reversion (ADX < threshold) selon le regime de marche.
    """

    def __init__(self, trend_strategy, range_strategy,
                 adx_period: int = 14, trend_threshold: int = 25,
                 range_threshold: int = 20):
        super().__init__(f"Regime({trend_strategy.name}/{range_strategy.name})")
        self.trend_strategy = trend_strategy
        self.range_strategy = range_strategy
        self.adx_period = adx_period
        self.trend_threshold = trend_threshold
        self.range_threshold = range_threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        adx = TechnicalIndicators.adx(
            data['high'], data['low'], data['close'], self.adx_period
        )

        trend_signals = self.trend_strategy.generate_signals(data)
        range_signals = self.range_strategy.generate_signals(data)

        signals = pd.DataFrame(index=data.index)
        signals['position'] = 0
        signals['signal'] = 0

        for i in range(len(data)):
            adx_val = adx.iloc[i] if not pd.isna(adx.iloc[i]) else 0

            if adx_val > self.trend_threshold:
                signals.iloc[i, signals.columns.get_loc('position')] = (
                    trend_signals.iloc[i].get('position', 0)
                )
            elif adx_val < self.range_threshold:
                signals.iloc[i, signals.columns.get_loc('position')] = (
                    range_signals.iloc[i].get('position', 0)
                )
            # Zone de transition (range_threshold <= ADX <= trend_threshold) : rester flat

        signals['signal'] = signals['position'].diff()
        return signals


class MultiTimeframeStrategy(BaseStrategy):
    """
    Wrapper : confirme les signaux avec la tendance long terme.
    Annule les LONG sous SMA(trend_period), annule les SHORT au-dessus.
    Filtre les entrees contre-tendance.
    """

    def __init__(self, base_strategy, trend_period: int = 200):
        super().__init__(f"{base_strategy.name} +MTF{trend_period}")
        self.base_strategy = base_strategy
        self.trend_period = trend_period

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = self.base_strategy.generate_signals(data)
        signals = signals.copy()

        trend_ma = TechnicalIndicators.sma(data['close'], self.trend_period)

        for i in range(len(signals)):
            pos = signals.iloc[i].get('position', 0)
            ma_val = trend_ma.iloc[i]

            if pd.isna(ma_val):
                continue

            # Annuler LONG sous la tendance
            if pos == 1 and data['close'].iloc[i] < ma_val:
                signals.iloc[i, signals.columns.get_loc('position')] = 0

            # Annuler SHORT au-dessus de la tendance
            elif pos == -1 and data['close'].iloc[i] > ma_val:
                signals.iloc[i, signals.columns.get_loc('position')] = 0

        signals['signal'] = signals['position'].diff()
        return signals
