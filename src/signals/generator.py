"""
Générateur de signaux de trading en temps réel
Basé sur les stratégies backtestées + événements macro
"""

import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from src.data.feed import LiveDataFeed
from src.strategies.base import *
from src.strategies.event_aware import *
from src.indicators.technical import TechnicalIndicators
from src.signals.macro_events import MacroEventsDatabase


class LiveSignalGenerator:
    """
    Génère des signaux de trading en temps réel
    """

    def __init__(self, live_feed: LiveDataFeed):
        self.feed = live_feed
        self.events_db = MacroEventsDatabase()

        # Stratégies recommandées par actif (basé sur backtests)
        self.recommended_strategies = {
            'SI=F': [  # Silver - Champion absolu
                ('Event-Aware Combined', EventFilteredCombinedStrategy(event_sensitivity=1.5, asset_type='commodities')),
                ('Combined', CombinedStrategy()),
            ],
            'NG=F': [  # Natural Gas
                ('Bollinger 20/2', BollingerBandsStrategy(20, 2)),
                ('RSI 14/35/80', RSIStrategy(14, 35, 80)),
            ],
            'BTC/USDT': [  # Bitcoin
                ('Bollinger 20/2', BollingerBandsStrategy(20, 2)),
                ('Event-Aware Combined', EventFilteredCombinedStrategy(event_sensitivity=1.0, asset_type='crypto')),
            ],
            'ETH/USDT': [  # Ethereum
                ('MA Crossover 10/30', MovingAverageCrossover(10, 30)),
            ],
            'NVDA': [  # Nvidia
                ('RSI 14/35/80', RSIStrategy(14, 35, 80)),
                ('Event-Aware RSI', EventFilteredRSIStrategy(14, 35, 80, event_sensitivity=1.0, asset_type='stocks')),
            ],
            'AAPL': [  # Apple
                ('Combined', CombinedStrategy()),
            ],
        }

    def get_current_signal(self, symbol: str, asset_type: str = 'stock', timeframe: str = '1h') -> Dict:
        """
        Génère le signal actuel pour un actif

        Args:
            symbol: Symbole de l'actif
            asset_type: 'stock', 'crypto', 'forex', 'commodity'
            timeframe: Intervalle de temps ('5min', '15min', '1h', '4h', '1d')

        Returns:
            Dict avec signal, stratégie, confiance, prix, indicateurs
        """

        # 1. Récupérer données intraday (100 bougies) avec le timeframe spécifié
        interval = timeframe  # Utiliser le timeframe fourni
        data = self.feed.get_intraday_data(symbol, interval, asset_type)

        if len(data) < 50:
            return {
                'symbol': symbol,
                'signal': 'NO_DATA',
                'confidence': 0,
                'message': 'Pas assez de données historiques'
            }

        # 2. Récupérer prix actuel
        if asset_type == 'crypto':
            current_price_data = self.feed.get_live_crypto_price(symbol)
        else:
            ticker = symbol.replace('=F', '')  # Remove suffix for commodities
            current_price_data = self.feed.get_live_stock_price(ticker)

        if not current_price_data:
            return {
                'symbol': symbol,
                'signal': 'NO_DATA',
                'confidence': 0,
                'message': 'Impossible de récupérer le prix actuel'
            }

        current_price = current_price_data['price']

        # 3. Obtenir stratégies recommandées pour cet actif
        strategies = self.recommended_strategies.get(symbol, [])

        if not strategies:
            # Utiliser stratégie par défaut
            strategies = [('Combined', CombinedStrategy())]

        # 4. Générer signaux avec chaque stratégie
        signals = []

        for strategy_name, strategy in strategies:
            try:
                # Générer signaux sur données historiques
                signals_df = strategy.generate_signals(data)

                if len(signals_df) == 0:
                    continue

                # Dernier signal
                last_signal = signals_df.iloc[-1]
                last_position = last_signal['position']

                # Calculer indicateurs actuels
                indicators = self._calculate_current_indicators(data)

                # Impact macro
                macro_impact = self.events_db.get_impact_for_date(
                    datetime.now().strftime('%Y-%m-%d'),
                    asset=asset_type,
                    window_days=7
                )

                # Déterminer signal
                if last_position == 1:
                    signal_type = 'BUY'
                    confidence = self._calculate_confidence(indicators, macro_impact, 'BUY')
                elif last_position == -1:
                    signal_type = 'SELL'
                    confidence = self._calculate_confidence(indicators, macro_impact, 'SELL')
                else:
                    signal_type = 'HOLD'
                    confidence = 0.5

                signals.append({
                    'strategy': strategy_name,
                    'signal': signal_type,
                    'confidence': confidence,
                    'indicators': indicators,
                    'macro_impact': macro_impact
                })

            except Exception as e:
                print(f"⚠️  Erreur stratégie {strategy_name}: {e}")
                continue

        # 5. Consensus des signaux
        if len(signals) == 0:
            return {
                'symbol': symbol,
                'signal': 'HOLD',
                'confidence': 0,
                'message': 'Aucune stratégie n\'a pu générer de signal'
            }

        # Voter sur le signal (majorité)
        buy_votes = sum(1 for s in signals if s['signal'] == 'BUY')
        sell_votes = sum(1 for s in signals if s['signal'] == 'SELL')
        hold_votes = sum(1 for s in signals if s['signal'] == 'HOLD')

        if buy_votes > sell_votes and buy_votes > hold_votes:
            final_signal = 'BUY'
        elif sell_votes > buy_votes and sell_votes > hold_votes:
            final_signal = 'SELL'
        else:
            final_signal = 'HOLD'

        # Confiance moyenne
        avg_confidence = sum(s['confidence'] for s in signals) / len(signals)

        # Impact macro moyen
        avg_macro = sum(s['macro_impact'] for s in signals) / len(signals)

        return {
            'symbol': symbol,
            'signal': final_signal,
            'confidence': avg_confidence,
            'current_price': current_price,
            'change_24h': current_price_data.get('change_24h_pct', 0) if asset_type == 'crypto' else current_price_data.get('change_pct', 0),
            'strategies_signals': signals,
            'consensus': f"{buy_votes} BUY / {sell_votes} SELL / {hold_votes} HOLD",
            'macro_impact': avg_macro,
            'timestamp': datetime.now(),
            'indicators': signals[0]['indicators'] if len(signals) > 0 else {}
        }

    def _calculate_current_indicators(self, data: pd.DataFrame) -> Dict:
        """Calcule les indicateurs techniques actuels"""

        try:
            # RSI
            rsi = TechnicalIndicators.rsi(data['close'], 14)
            current_rsi = rsi.iloc[-1]

            # MAs
            ma_20 = TechnicalIndicators.sma(data['close'], 20).iloc[-1]
            ma_50 = TechnicalIndicators.sma(data['close'], 50).iloc[-1]

            # MACD
            macd_data = TechnicalIndicators.macd(data['close'])
            current_macd = macd_data['macd'].iloc[-1]
            current_signal = macd_data['signal'].iloc[-1]
            current_histogram = macd_data['histogram'].iloc[-1]

            # Bollinger
            bb = TechnicalIndicators.bollinger_bands(data['close'], 20, 2)
            current_price = data['close'].iloc[-1]
            bb_upper = bb['upper'].iloc[-1]
            bb_lower = bb['lower'].iloc[-1]
            bb_middle = bb['middle'].iloc[-1]

            # Position dans Bollinger
            bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) * 100

            return {
                'rsi': current_rsi,
                'ma_20': ma_20,
                'ma_50': ma_50,
                'macd': current_macd,
                'macd_signal': current_signal,
                'macd_histogram': current_histogram,
                'bb_upper': bb_upper,
                'bb_lower': bb_lower,
                'bb_middle': bb_middle,
                'bb_position': bb_position,
                'current_price': current_price
            }

        except Exception as e:
            print(f"⚠️  Erreur calcul indicateurs: {e}")
            return {}

    def _calculate_confidence(self, indicators: Dict, macro_impact: float, signal_type: str) -> float:
        """
        Calcule la confiance dans un signal

        Returns:
            Float entre 0 et 1
        """

        if not indicators:
            return 0.5

        confidence = 0.5  # Base

        try:
            rsi = indicators.get('rsi', 50)
            bb_position = indicators.get('bb_position', 50)
            macd_histogram = indicators.get('macd_histogram', 0)

            if signal_type == 'BUY':
                # RSI oversold
                if rsi < 35:
                    confidence += 0.15
                elif rsi < 45:
                    confidence += 0.05

                # Bollinger lower band
                if bb_position < 20:
                    confidence += 0.15

                # MACD positif
                if macd_histogram > 0:
                    confidence += 0.10

                # Macro bullish
                if macro_impact > 3:
                    confidence += 0.10
                elif macro_impact > 0:
                    confidence += 0.05

            elif signal_type == 'SELL':
                # RSI overbought
                if rsi > 70:
                    confidence += 0.15
                elif rsi > 60:
                    confidence += 0.05

                # Bollinger upper band
                if bb_position > 80:
                    confidence += 0.15

                # MACD négatif
                if macd_histogram < 0:
                    confidence += 0.10

                # Macro bearish
                if macro_impact < -3:
                    confidence += 0.10
                elif macro_impact < 0:
                    confidence += 0.05

            # Clamp entre 0 et 1
            confidence = max(0, min(1, confidence))

            return confidence

        except:
            return 0.5

    def get_all_signals(self) -> List[Dict]:
        """
        Génère les signaux pour tous les actifs recommandés

        Returns:
            Liste de dictionnaires avec signaux
        """

        all_signals = []

        for symbol in self.recommended_strategies.keys():
            # Déterminer asset_type
            if 'USDT' in symbol:
                asset_type = 'crypto'
            elif '=F' in symbol:
                asset_type = 'stock'  # Commodities use same API
            else:
                asset_type = 'stock'

            signal = self.get_current_signal(symbol, asset_type)
            all_signals.append(signal)

        return all_signals

    def get_trade_recommendations(self, capital: float = 10000,
                                  max_positions: int = 3) -> List[Dict]:
        """
        Génère des recommandations de trades basées sur les signaux

        Args:
            capital: Capital disponible
            max_positions: Nombre max de positions ouvertes

        Returns:
            Liste de recommandations triées par confiance
        """

        signals = self.get_all_signals()

        # Filtrer signaux BUY avec confiance >= 0.6
        buy_signals = [
            s for s in signals
            if s['signal'] == 'BUY' and s['confidence'] >= 0.6
        ]

        # Trier par confiance décroissante
        buy_signals.sort(key=lambda x: x['confidence'], reverse=True)

        # Limiter au max_positions
        buy_signals = buy_signals[:max_positions]

        # Calculer taille de position
        if len(buy_signals) > 0:
            position_size = capital / len(buy_signals)
        else:
            position_size = 0

        recommendations = []

        for signal in buy_signals:
            price = signal.get('current_price', 0)
            if price == 0:
                continue

            quantity = position_size / price

            # Stop-Loss et Take-Profit recommandés
            if 'BTC' in signal['symbol'] or 'ETH' in signal['symbol']:
                sl_pct = 8.0
                tp_pct = 15.0
            else:
                sl_pct = 3.0
                tp_pct = 6.0

            sl_price = price * (1 - sl_pct / 100)
            tp_price = price * (1 + tp_pct / 100)

            recommendations.append({
                'symbol': signal['symbol'],
                'action': 'BUY',
                'price': price,
                'quantity': quantity,
                'position_value': position_size,
                'stop_loss': sl_price,
                'take_profit': tp_price,
                'confidence': signal['confidence'],
                'reasoning': self._generate_reasoning(signal),
                'risk_reward': tp_pct / sl_pct,
                'timestamp': datetime.now()
            })

        return recommendations

    def _generate_reasoning(self, signal: Dict) -> str:
        """Génère une explication textuelle du signal"""

        reasons = []

        # Confiance
        conf = signal['confidence']
        if conf > 0.8:
            reasons.append("Confiance très élevée")
        elif conf > 0.6:
            reasons.append("Confiance élevée")

        # Consensus
        if 'consensus' in signal:
            reasons.append(f"Consensus: {signal['consensus']}")

        # Macro
        macro = signal.get('macro_impact', 0)
        if macro > 3:
            reasons.append("Contexte macro très favorable")
        elif macro > 0:
            reasons.append("Contexte macro favorable")
        elif macro < -3:
            reasons.append("⚠️ Contexte macro défavorable")

        # Indicateurs
        indicators = signal.get('indicators', {})
        if indicators:
            rsi = indicators.get('rsi', 50)
            if rsi < 35:
                reasons.append(f"RSI en survente ({rsi:.0f})")
            elif rsi > 70:
                reasons.append(f"RSI en surachat ({rsi:.0f})")

        return " | ".join(reasons) if reasons else "Signal technique standard"


# ============================================================================
# Script de test
# ============================================================================

if __name__ == '__main__':
    from src.data.macro_fetcher import MacroDataFetcher

    # Charger API keys
    fetcher = MacroDataFetcher()
    av_key = fetcher.alpha_vantage_key
    fred_key = fetcher.fred_api_key

    # Créer live feed
    feed = LiveDataFeed(av_key, fred_key)

    # Créer générateur de signaux
    generator = LiveSignalGenerator(feed)

    print("="*80)
    print("🎯 SIGNAUX DE TRADING EN TEMPS RÉEL")
    print("="*80)

    # Test sur Bitcoin
    print("\n📊 Signal Bitcoin")
    print("-"*80)
    btc_signal = generator.get_current_signal('BTC/USDT', 'crypto')

    print(f"Symbole       : {btc_signal['symbol']}")
    print(f"Signal        : {btc_signal['signal']}")
    print(f"Confiance     : {btc_signal['confidence']:.1%}")
    print(f"Prix actuel   : ${btc_signal.get('current_price', 0):,.2f}")
    print(f"Change 24h    : {btc_signal.get('change_24h', 0):+.2f}%")
    print(f"Consensus     : {btc_signal.get('consensus', 'N/A')}")
    print(f"Impact macro  : {btc_signal.get('macro_impact', 0):+.2f}")

    if 'indicators' in btc_signal and btc_signal['indicators']:
        ind = btc_signal['indicators']
        print(f"\nIndicateurs:")
        print(f"  RSI         : {ind.get('rsi', 0):.1f}")
        print(f"  BB Position : {ind.get('bb_position', 0):.1f}%")
        print(f"  MACD        : {ind.get('macd', 0):.2f}")

    # Recommandations de trades
    print("\n" + "="*80)
    print("💡 RECOMMANDATIONS DE TRADES")
    print("="*80)

    recommendations = generator.get_trade_recommendations(capital=10000, max_positions=3)

    if len(recommendations) > 0:
        for i, rec in enumerate(recommendations, 1):
            print(f"\n#{i} - {rec['symbol']}")
            print(f"  Action        : {rec['action']}")
            print(f"  Prix entrée   : ${rec['price']:,.2f}")
            print(f"  Quantité      : {rec['quantity']:.4f}")
            print(f"  Valeur        : ${rec['position_value']:,.2f}")
            print(f"  Stop-Loss     : ${rec['stop_loss']:,.2f}")
            print(f"  Take-Profit   : ${rec['take_profit']:,.2f}")
            print(f"  Risk/Reward   : 1:{rec['risk_reward']:.1f}")
            print(f"  Confiance     : {rec['confidence']:.1%}")
            print(f"  Raison        : {rec['reasoning']}")
    else:
        print("\n⚠️  Aucune recommandation actuellement")
        print("   (Confiance < 60% ou aucun signal BUY)")

    print("\n" + "="*80)
    print("✅ Test terminé!")
    print("="*80)
