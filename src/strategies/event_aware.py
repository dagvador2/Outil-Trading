"""
Stratégies de trading "Event-Aware"
Ajustent les signaux en fonction des événements macroéconomiques
"""

import pandas as pd
import numpy as np
from src.strategies.base import BaseStrategy
from src.indicators.technical import TechnicalIndicators
from src.signals.macro_events import MacroEventsDatabase


class EventAwareStrategy(BaseStrategy):
    """
    Classe de base pour stratégies conscientes des événements macro
    """

    def __init__(self, name: str, base_strategy: BaseStrategy,
                 event_sensitivity: float = 1.0,
                 asset_type: str = 'all'):
        """
        Args:
            name: Nom de la stratégie
            base_strategy: Stratégie technique de base
            event_sensitivity: Sensibilité aux événements (0.0 à 2.0)
                             0.0 = ignore complètement les événements
                             1.0 = sensibilité normale
                             2.0 = très sensible aux événements
            asset_type: Type d'actif ('crypto', 'stocks', 'all', etc.)
        """
        super().__init__(name)
        self.base_strategy = base_strategy
        self.event_sensitivity = event_sensitivity
        self.asset_type = asset_type
        self.events_db = MacroEventsDatabase()

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Génère des signaux en combinant analyse technique et événements macro

        Logique:
        1. Obtenir les signaux techniques de base
        2. Calculer le score d'impact macro pour chaque date
        3. Ajuster les signaux selon le contexte macro
        """

        # 1. Signaux techniques de base
        signals = self.base_strategy.generate_signals(data)

        # 2. Calculer l'impact macro pour chaque date
        signals['macro_impact'] = 0.0

        for idx in data.index:
            date_str = idx.strftime('%Y-%m-%d')
            # Impact sur fenêtre de 7 jours
            impact = self.events_db.get_impact_for_date(
                date_str,
                asset=self.asset_type,
                window_days=7
            )
            signals.loc[idx, 'macro_impact'] = impact * self.event_sensitivity

        # 3. Ajuster les positions selon le contexte macro
        signals['adjusted_position'] = signals['position'].copy()

        # Règles d'ajustement:
        # - Impact très positif (>+5) : Amplifier les signaux LONG, réduire SHORT
        # - Impact très négatif (<-5) : Amplifier les signaux SHORT, réduire LONG
        # - Impact modéré : Conserver les signaux

        for idx in signals.index:
            original_pos = signals.loc[idx, 'position']
            macro_impact = signals.loc[idx, 'macro_impact']

            # Environnement très bullish (impact > +5)
            if macro_impact > 5:
                if original_pos == -1:  # SHORT signal
                    # Annuler le SHORT dans un environnement très bullish
                    signals.loc[idx, 'adjusted_position'] = 0
                # Conserver LONG tel quel

            # Environnement très bearish (impact < -5)
            elif macro_impact < -5:
                if original_pos == 1:  # LONG signal
                    # Annuler le LONG dans un environnement très bearish
                    signals.loc[idx, 'adjusted_position'] = 0
                # Conserver SHORT tel quel

            # Environnement modérément bullish (2 < impact < 5)
            elif 2 < macro_impact <= 5:
                if original_pos == -1:  # SHORT signal
                    # Réduire conviction SHORT
                    signals.loc[idx, 'adjusted_position'] = 0

            # Environnement modérément bearish (-5 < impact < -2)
            elif -5 <= macro_impact < -2:
                if original_pos == 1:  # LONG signal
                    # Réduire conviction LONG
                    signals.loc[idx, 'adjusted_position'] = 0

        # 4. Recalculer les signaux d'entrée/sortie
        signals['signal'] = signals['adjusted_position'].diff()

        return signals


class EventFilteredMAStrategy(EventAwareStrategy):
    """
    MA Crossover avec filtre événementiel
    """

    def __init__(self, fast_period: int = 20, slow_period: int = 50,
                 event_sensitivity: float = 1.0, asset_type: str = 'all'):
        from src.strategies.base import MovingAverageCrossover
        base = MovingAverageCrossover(fast_period, slow_period)
        name = f"Event-Aware MA ({fast_period}/{slow_period})"
        super().__init__(name, base, event_sensitivity, asset_type)


class EventFilteredRSIStrategy(EventAwareStrategy):
    """
    RSI avec filtre événementiel
    """

    def __init__(self, period: int = 14, oversold: int = 30, overbought: int = 70,
                 event_sensitivity: float = 1.0, asset_type: str = 'all'):
        from src.strategies.base import RSIStrategy
        base = RSIStrategy(period, oversold, overbought)
        name = f"Event-Aware RSI ({period}, {oversold}/{overbought})"
        super().__init__(name, base, event_sensitivity, asset_type)


class EventFilteredCombinedStrategy(EventAwareStrategy):
    """
    Combined strategy avec filtre événementiel - La plus sophistiquée
    """

    def __init__(self, event_sensitivity: float = 1.0, asset_type: str = 'all'):
        from src.strategies.base import CombinedStrategy
        base = CombinedStrategy()
        name = f"Event-Aware Combined"
        super().__init__(name, base, event_sensitivity, asset_type)


class EventOpportunisticStrategy(BaseStrategy):
    """
    Stratégie OPPORTUNISTE qui trade UNIQUEMENT lors d'événements majeurs

    Logique:
    - Attend des événements avec impact > +/-7
    - Entre en LONG sur événements très bullish
    - Entre en SHORT sur événements très bearish
    - Sort sur signal technique ou après N jours
    """

    def __init__(self, event_threshold: float = 7.0,
                 hold_days: int = 5,
                 asset_type: str = 'all'):
        super().__init__(f"Event Opportunist (threshold={event_threshold})")
        self.event_threshold = event_threshold
        self.hold_days = hold_days
        self.asset_type = asset_type
        self.events_db = MacroEventsDatabase()

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        signals['position'] = 0
        signals['macro_impact'] = 0.0

        # Calculer l'impact macro pour chaque jour
        for idx in data.index:
            date_str = idx.strftime('%Y-%m-%d')
            impact = self.events_db.get_impact_for_date(
                date_str,
                asset=self.asset_type,
                window_days=3  # Fenêtre étroite pour événements ponctuels
            )
            signals.loc[idx, 'macro_impact'] = impact

        # Générer les positions
        in_position = False
        position_entry_idx = None
        days_in_position = 0

        for i, idx in enumerate(data.index):
            impact = signals.loc[idx, 'macro_impact']

            if not in_position:
                # Chercher des opportunités d'entrée
                if impact > self.event_threshold:
                    # Événement très bullish -> LONG
                    signals.loc[idx, 'position'] = 1
                    in_position = True
                    position_entry_idx = i
                    days_in_position = 0

                elif impact < -self.event_threshold:
                    # Événement très bearish -> SHORT
                    signals.loc[idx, 'position'] = -1
                    in_position = True
                    position_entry_idx = i
                    days_in_position = 0
            else:
                # Déjà en position
                days_in_position += 1

                # Sortir si:
                # 1. Holding period atteint
                # 2. Impact change de signe (bullish -> bearish ou inverse)
                current_pos = signals.iloc[position_entry_idx]['position']
                should_exit = False

                if days_in_position >= self.hold_days:
                    should_exit = True

                # Changement de sentiment
                elif current_pos == 1 and impact < -3:  # Long mais devient bearish
                    should_exit = True
                elif current_pos == -1 and impact > 3:  # Short mais devient bullish
                    should_exit = True

                if should_exit:
                    signals.loc[idx, 'position'] = 0
                    in_position = False
                    position_entry_idx = None
                    days_in_position = 0
                else:
                    # Conserver la position
                    signals.loc[idx, 'position'] = signals.iloc[i-1]['position']

        # Calculer les signaux d'entrée/sortie
        signals['signal'] = signals['position'].diff()

        return signals


class SentimentMomentumStrategy(BaseStrategy):
    """
    Stratégie hybride : Momentum technique + Sentiment macroéconomique

    Combine:
    - RSI pour momentum
    - MA 50/200 pour tendance
    - Sentiment macro pour timing
    """

    def __init__(self, asset_type: str = 'all'):
        super().__init__("Sentiment-Momentum Hybrid")
        self.asset_type = asset_type
        self.events_db = MacroEventsDatabase()

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0

        # Indicateurs techniques
        signals['rsi'] = TechnicalIndicators.rsi(data['close'], 14)
        signals['ma_fast'] = TechnicalIndicators.sma(data['close'], 50)
        signals['ma_slow'] = TechnicalIndicators.sma(data['close'], 200)

        # Impact macro
        signals['macro_impact'] = 0.0
        for idx in data.index:
            date_str = idx.strftime('%Y-%m-%d')
            impact = self.events_db.get_impact_for_date(
                date_str,
                asset=self.asset_type,
                window_days=14  # Fenêtre plus large pour contexte général
            )
            signals.loc[idx, 'macro_impact'] = impact

        # Logique de trading
        signals['position'] = 0

        for idx in signals.index:
            rsi = signals.loc[idx, 'rsi']
            ma_fast = signals.loc[idx, 'ma_fast']
            ma_slow = signals.loc[idx, 'ma_slow']
            macro = signals.loc[idx, 'macro_impact']

            # Conditions LONG
            long_conditions = [
                ma_fast > ma_slow,  # Tendance haussière
                rsi < 65,  # Pas suracheté
                macro > -3,  # Pas d'environnement très bearish
            ]

            # Conditions SHORT
            short_conditions = [
                ma_fast < ma_slow,  # Tendance baissière
                rsi > 35,  # Pas survendu
                macro < 3,  # Pas d'environnement très bullish
            ]

            # Boost si sentiment très favorable
            if macro > 5:  # Très bullish
                if all(long_conditions[:2]):  # Ignore condition macro
                    signals.loc[idx, 'position'] = 1

            elif macro < -5:  # Très bearish
                if all(short_conditions[:2]):
                    signals.loc[idx, 'position'] = -1

            # Conditions normales
            elif all(long_conditions):
                signals.loc[idx, 'position'] = 1

            elif all(short_conditions):
                signals.loc[idx, 'position'] = -1

        # Signaux d'entrée/sortie
        signals['signal'] = signals['position'].diff()

        return signals


if __name__ == '__main__':
    # Test rapide
    print("="*80)
    print("🧪 TEST DES STRATÉGIES EVENT-AWARE")
    print("="*80)

    # Test avec données synthétiques
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    data = pd.DataFrame({
        'close': np.cumsum(np.random.randn(len(dates))) + 100,
        'open': np.cumsum(np.random.randn(len(dates))) + 100,
        'high': np.cumsum(np.random.randn(len(dates))) + 102,
        'low': np.cumsum(np.random.randn(len(dates))) + 98,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)

    strategies = [
        EventFilteredMAStrategy(20, 50, event_sensitivity=1.0, asset_type='crypto'),
        EventFilteredRSIStrategy(14, 30, 70, event_sensitivity=1.0, asset_type='stocks'),
        EventOpportunisticStrategy(event_threshold=7.0, asset_type='all'),
        SentimentMomentumStrategy(asset_type='all')
    ]

    for strategy in strategies:
        print(f"\n✅ {strategy.name}")
        signals = strategy.generate_signals(data)
        num_signals = (signals['signal'] != 0).sum()
        print(f"   Signaux générés: {num_signals}")
        print(f"   Impact macro moyen: {signals.get('macro_impact', pd.Series([0])).mean():.2f}")

    print("\n" + "="*80)
    print("✅ Tests terminés!")
    print("="*80)
