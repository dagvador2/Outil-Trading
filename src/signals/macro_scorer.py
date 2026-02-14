"""
Système de Scoring des Signaux Macro
Agrège les signaux de multiples sources et calcule un score composite

Architecture:
1. Récupère les signaux de news_fetcher
2. Calcule des scores pondérés par source et confiance
3. Agrège en scores globaux (général) et spécifiques (par actif)
4. Fournit des recommandations de trading

Output:
- Score général du marché : -100 (très bearish) à +100 (très bullish)
- Score par actif : -100 à +100
- Confiance du signal : 0-1
- Recommandations : 'strong_sell', 'sell', 'neutral', 'buy', 'strong_buy'
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import json
import logging

from src.signals.news import (
    MacroNewsAggregator, MacroSignal, MarketSentiment,
    SignalCache
)

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class MacroScore:
    """Score macro composite pour un actif ou le marché global"""
    timestamp: datetime
    asset: str  # 'all' pour score général
    score: float  # -100 (très bearish) à +100 (très bullish)
    confidence: float  # 0-1
    recommendation: str  # 'strong_sell', 'sell', 'neutral', 'buy', 'strong_buy'

    # Détails
    num_signals: int
    positive_signals: int
    negative_signals: int
    neutral_signals: int

    # Scores par source
    news_score: float
    sentiment_score: float
    economic_score: float

    # Context
    market_sentiment: Optional[str] = None
    fear_greed_index: Optional[float] = None
    vix: Optional[float] = None

    def to_dict(self):
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat()
        }


# ============================================================================
# Scorer
# ============================================================================

class MacroSignalScorer:
    """
    Score les signaux macro et génère des recommandations
    """

    # Poids par source
    SOURCE_WEIGHTS = {
        'alphavantage': 1.0,  # Sentiment analysis inclus
        'finnhub': 0.9,
        'newsapi': 0.7,
        'rss': 0.5,  # Plus de bruit
        'fred': 1.2,  # Données officielles
        'fear_greed': 1.0,
    }

    # Poids par catégorie de news
    CATEGORY_WEIGHTS = {
        'fed': 1.5,  # Impact majeur
        'geopolitical': 1.2,
        'earnings': 0.8,  # Impact plus localisé
        'technology': 0.7,
        'crypto': 0.9,
        'general': 0.5,
    }

    def __init__(self, aggregator: Optional[MacroNewsAggregator] = None,
                 cache: Optional[SignalCache] = None):
        """
        Args:
            aggregator: Instance de MacroNewsAggregator
            cache: Cache pour stocker les signaux
        """
        self.aggregator = aggregator or MacroNewsAggregator()
        self.cache = cache or SignalCache()

    def compute_market_score(self, lookback_hours: int = 48) -> MacroScore:
        """
        Calcule le score macro général du marché

        Args:
            lookback_hours: Fenêtre temporelle pour les signaux

        Returns:
            MacroScore avec asset='all'
        """
        logger.info("Computing general market score...")

        # 1. Récupérer les signaux généraux récents
        recent_signals = self.cache.get_recent_signals(hours=lookback_hours)
        general_signals = [s for s in recent_signals if 'all' in s.affected_assets]

        # Si pas assez de signaux en cache, en récupérer de nouveaux
        if len(general_signals) < 10:
            logger.info("Not enough cached signals, fetching new ones...")
            new_signals = self.aggregator.fetch_general_signals(days_back=2)
            self.cache.add_signals(new_signals)
            general_signals = new_signals

        # 2. Récupérer le sentiment du marché
        market_sentiment = self.aggregator.get_market_sentiment()

        # 3. Récupérer les indicateurs économiques
        economic_indicators = self.aggregator.get_economic_indicators()

        # 4. Calculer les scores
        news_score = self._score_signals(general_signals)
        sentiment_score = self._score_market_sentiment(market_sentiment)
        economic_score = self._score_economic_indicators(economic_indicators)

        # 5. Score composite pondéré
        composite_score = (
            news_score * 0.50 +
            sentiment_score * 0.30 +
            economic_score * 0.20
        )

        # 6. Confiance (basée sur nombre de signaux et cohérence)
        confidence = self._calculate_confidence(
            general_signals,
            news_score,
            sentiment_score,
            economic_score
        )

        # 7. Recommandation
        recommendation = self._score_to_recommendation(composite_score, confidence)

        # 8. Compter les signaux
        positive = sum(1 for s in general_signals if s.impact_score > 2)
        negative = sum(1 for s in general_signals if s.impact_score < -2)
        neutral = len(general_signals) - positive - negative

        return MacroScore(
            timestamp=datetime.now(),
            asset='all',
            score=round(composite_score, 2),
            confidence=round(confidence, 3),
            recommendation=recommendation,
            num_signals=len(general_signals),
            positive_signals=positive,
            negative_signals=negative,
            neutral_signals=neutral,
            news_score=round(news_score, 2),
            sentiment_score=round(sentiment_score, 2),
            economic_score=round(economic_score, 2),
            market_sentiment=market_sentiment.general_sentiment,
            fear_greed_index=market_sentiment.fear_greed_index,
            vix=market_sentiment.volatility_index,
        )

    def compute_asset_score(self, asset_symbol: str,
                          lookback_hours: int = 48) -> MacroScore:
        """
        Calcule le score macro pour un actif spécifique

        Combine:
        - Signaux généraux du marché (pondération 0.4)
        - Signaux spécifiques à l'actif (pondération 0.6)

        Args:
            asset_symbol: 'AAPL', 'BTC/USDT', etc.
            lookback_hours: Fenêtre temporelle

        Returns:
            MacroScore pour cet actif
        """
        logger.info(f"Computing score for {asset_symbol}...")

        # 1. Score général du marché (impact de base)
        market_score = self.compute_market_score(lookback_hours)

        # 2. Récupérer signaux spécifiques à l'actif
        asset_signals = self.aggregator.fetch_asset_signals(
            asset_symbol,
            days_back=2
        )

        # Filtrer signaux récents
        cutoff = datetime.now() - timedelta(hours=lookback_hours)
        recent_asset_signals = [
            s for s in asset_signals
            if s.timestamp > cutoff
        ]

        if not recent_asset_signals:
            logger.info(f"No specific signals for {asset_symbol}, using market score")
            # Pas de signaux spécifiques → utiliser score du marché
            return MacroScore(
                timestamp=datetime.now(),
                asset=asset_symbol,
                score=market_score.score * 0.5,  # Atténuer
                confidence=market_score.confidence * 0.7,
                recommendation=self._score_to_recommendation(
                    market_score.score * 0.5,
                    market_score.confidence * 0.7
                ),
                num_signals=0,
                positive_signals=0,
                negative_signals=0,
                neutral_signals=0,
                news_score=0,
                sentiment_score=market_score.sentiment_score,
                economic_score=market_score.economic_score,
                market_sentiment=market_score.market_sentiment,
                fear_greed_index=market_score.fear_greed_index,
                vix=market_score.vix,
            )

        # 3. Scorer les signaux spécifiques
        asset_news_score = self._score_signals(recent_asset_signals)

        # 4. Score composite : 40% marché + 60% actif
        composite_score = (
            market_score.score * 0.4 +
            asset_news_score * 0.6
        )

        # 5. Confiance
        confidence = self._calculate_confidence(
            recent_asset_signals,
            asset_news_score,
            market_score.sentiment_score,
            market_score.economic_score
        )

        # 6. Recommandation
        recommendation = self._score_to_recommendation(composite_score, confidence)

        # 7. Compter les signaux
        positive = sum(1 for s in recent_asset_signals if s.impact_score > 2)
        negative = sum(1 for s in recent_asset_signals if s.impact_score < -2)
        neutral = len(recent_asset_signals) - positive - negative

        return MacroScore(
            timestamp=datetime.now(),
            asset=asset_symbol,
            score=round(composite_score, 2),
            confidence=round(confidence, 3),
            recommendation=recommendation,
            num_signals=len(recent_asset_signals),
            positive_signals=positive,
            negative_signals=negative,
            neutral_signals=neutral,
            news_score=round(asset_news_score, 2),
            sentiment_score=market_score.sentiment_score,
            economic_score=market_score.economic_score,
            market_sentiment=market_score.market_sentiment,
            fear_greed_index=market_score.fear_greed_index,
            vix=market_score.vix,
        )

    def compute_all_assets_scores(self, assets: List[str],
                                  lookback_hours: int = 48) -> Dict[str, MacroScore]:
        """
        Calcule les scores pour une liste d'actifs

        Args:
            assets: Liste de symboles ['AAPL', 'BTC/USDT', ...]
            lookback_hours: Fenêtre temporelle

        Returns:
            Dict {symbol: MacroScore}
        """
        scores = {}

        # Score général d'abord (réutilisé pour tous)
        market_score = self.compute_market_score(lookback_hours)
        scores['market'] = market_score

        # Scores par actif
        for asset in assets:
            try:
                scores[asset] = self.compute_asset_score(asset, lookback_hours)
            except Exception as e:
                logger.error(f"Error scoring {asset}: {e}")
                # Fallback sur score marché atténué
                scores[asset] = MacroScore(
                    timestamp=datetime.now(),
                    asset=asset,
                    score=market_score.score * 0.5,
                    confidence=0.3,
                    recommendation='neutral',
                    num_signals=0,
                    positive_signals=0,
                    negative_signals=0,
                    neutral_signals=0,
                    news_score=0,
                    sentiment_score=0,
                    economic_score=0,
                )

        return scores

    # ------------------------------------------------------------------ #
    # Scoring helpers
    # ------------------------------------------------------------------ #

    def _score_signals(self, signals: List[MacroSignal]) -> float:
        """
        Score un ensemble de signaux en tenant compte des poids

        Returns:
            Score de -100 à +100
        """
        if not signals:
            return 0.0

        weighted_sum = 0.0
        total_weight = 0.0

        for signal in signals:
            # Poids de la source
            source_weight = self.SOURCE_WEIGHTS.get(signal.source, 0.5)

            # Poids de la catégorie
            category_weight = self.CATEGORY_WEIGHTS.get(signal.category, 0.5)

            # Poids de confiance du signal
            confidence_weight = signal.confidence

            # Poids composite
            weight = source_weight * category_weight * confidence_weight

            # Décroissance temporelle (signaux récents = plus importants)
            hours_old = (datetime.now() - signal.timestamp).total_seconds() / 3600
            time_decay = max(0.3, 1.0 - (hours_old / 48))  # Decay sur 48h

            final_weight = weight * time_decay

            # Impact pondéré
            weighted_sum += signal.impact_score * final_weight * 10  # Scale to -100/+100
            total_weight += final_weight

        if total_weight == 0:
            return 0.0

        # Normaliser
        score = weighted_sum / total_weight

        # Clamp à [-100, 100]
        return np.clip(score, -100, 100)

    def _score_market_sentiment(self, sentiment: MarketSentiment) -> float:
        """
        Score le sentiment général du marché

        Returns:
            Score de -100 à +100
        """
        # Sentiment principal (score macro)
        base_score = sentiment.macro_score * 10  # Scale to -100/+100

        # Ajustements basés sur VIX si disponible
        if sentiment.volatility_index:
            vix = sentiment.volatility_index
            if vix > 30:  # VIX élevé = peur = bearish
                base_score -= 20
            elif vix > 20:
                base_score -= 10
            elif vix < 12:  # VIX bas = complaisance = prudence
                base_score -= 5

        return np.clip(base_score, -100, 100)

    def _score_economic_indicators(self, indicators: Dict[str, float]) -> float:
        """
        Score les indicateurs économiques

        Returns:
            Score de -100 à +100
        """
        if not indicators:
            return 0.0

        score = 0.0

        # Inflation (CPI)
        if 'cpi' in indicators:
            cpi = indicators['cpi']
            if cpi > 4.0:  # Inflation élevée = bearish
                score -= 30
            elif cpi > 3.0:
                score -= 15
            elif cpi < 2.0:  # Inflation basse = bullish
                score += 15

        # Chômage
        if 'unemployment' in indicators:
            unemp = indicators['unemployment']
            if unemp > 5.0:  # Chômage élevé = bearish
                score -= 20
            elif unemp < 4.0:  # Chômage bas = économie forte = bullish
                score += 20

        # Taux Fed
        if 'fed_funds' in indicators:
            fed_rate = indicators['fed_funds']
            if fed_rate > 5.0:  # Taux élevés = bearish pour risk assets
                score -= 25
            elif fed_rate > 4.0:
                score -= 10
            elif fed_rate < 2.0:  # Taux bas = bullish
                score += 20

        # VIX
        if 'vix' in indicators:
            vix = indicators['vix']
            if vix > 30:
                score -= 20
            elif vix < 15:
                score += 10

        return np.clip(score, -100, 100)

    def _calculate_confidence(self, signals: List[MacroSignal],
                             news_score: float,
                             sentiment_score: float,
                             economic_score: float) -> float:
        """
        Calcule la confiance dans le score composite

        Basé sur:
        - Nombre de signaux
        - Cohérence entre les scores
        - Qualité des sources

        Returns:
            Confiance de 0 à 1
        """
        # 1. Confiance basée sur le nombre de signaux
        num_signals = len(signals)
        if num_signals == 0:
            signal_confidence = 0.0
        elif num_signals < 5:
            signal_confidence = 0.4
        elif num_signals < 10:
            signal_confidence = 0.6
        elif num_signals < 20:
            signal_confidence = 0.8
        else:
            signal_confidence = 1.0

        # 2. Cohérence entre les sources
        scores = [news_score, sentiment_score, economic_score]
        scores = [s for s in scores if s != 0]  # Ignorer les scores nuls

        if len(scores) < 2:
            consistency = 0.5
        else:
            # Vérifier si les signes sont cohérents
            signs = [np.sign(s) for s in scores]
            if all(s == signs[0] for s in signs):
                # Tous du même signe = haute cohérence
                consistency = 1.0
            elif all(abs(s) < 20 for s in scores):
                # Tous neutres = cohérence moyenne
                consistency = 0.7
            else:
                # Signaux contradictoires = faible cohérence
                consistency = 0.4

        # 3. Qualité des sources
        if not signals:
            source_quality = 0.5
        else:
            avg_confidence = np.mean([s.confidence for s in signals])
            source_quality = avg_confidence

        # Confiance composite
        confidence = (
            signal_confidence * 0.4 +
            consistency * 0.4 +
            source_quality * 0.2
        )

        return np.clip(confidence, 0, 1)

    def _score_to_recommendation(self, score: float, confidence: float) -> str:
        """
        Convertit un score en recommandation

        Args:
            score: -100 à +100
            confidence: 0 à 1

        Returns:
            'strong_sell', 'sell', 'neutral', 'buy', 'strong_buy'
        """
        # Ajuster les seuils selon la confiance
        if confidence < 0.5:
            # Faible confiance → recommandations plus conservatrices
            if score < -50:
                return 'sell'
            elif score < -20:
                return 'neutral'
            elif score < 20:
                return 'neutral'
            elif score < 50:
                return 'neutral'
            else:
                return 'buy'
        else:
            # Confiance normale/haute
            if score < -60:
                return 'strong_sell'
            elif score < -30:
                return 'sell'
            elif score < -10:
                return 'neutral'
            elif score < 30:
                return 'neutral'
            elif score < 60:
                return 'buy'
            else:
                return 'strong_buy'

    # ------------------------------------------------------------------ #
    # Export & Display
    # ------------------------------------------------------------------ #

    def export_scores(self, scores: Dict[str, MacroScore],
                     filepath: str = 'macro_scores.json'):
        """Sauvegarde les scores en JSON"""
        data = {symbol: score.to_dict() for symbol, score in scores.items()}
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Scores exported to {filepath}")

    def print_score_report(self, score: MacroScore):
        """Affiche un rapport lisible pour un score"""
        sep = "=" * 70
        print(f"\n{sep}")
        print(f"  MACRO SCORE REPORT - {score.asset}")
        print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(sep)

        # Score principal
        emoji = self._score_emoji(score.score)
        print(f"\n  {emoji} SCORE COMPOSITE: {score.score:+.1f}/100")
        print(f"  📊 Confiance: {score.confidence:.1%}")
        print(f"  💡 Recommandation: {score.recommendation.upper()}")

        # Détails
        print(f"\n  SIGNAUX ANALYSÉS: {score.num_signals}")
        print(f"    🟢 Positifs: {score.positive_signals}")
        print(f"    🔴 Négatifs: {score.negative_signals}")
        print(f"    ⚪ Neutres: {score.neutral_signals}")

        # Scores par composante
        print(f"\n  DÉCOMPOSITION:")
        print(f"    📰 News:          {score.news_score:+6.1f}")
        print(f"    🎭 Sentiment:     {score.sentiment_score:+6.1f}")
        print(f"    📈 Économie:      {score.economic_score:+6.1f}")

        # Context
        if score.market_sentiment:
            print(f"\n  CONTEXTE:")
            print(f"    Sentiment marché: {score.market_sentiment}")
            if score.fear_greed_index:
                print(f"    Fear & Greed:     {score.fear_greed_index:.0f}/100")
            if score.vix:
                print(f"    VIX:              {score.vix:.1f}")

        print(f"\n{sep}\n")

    def _score_emoji(self, score: float) -> str:
        """Retourne un emoji selon le score"""
        if score > 60:
            return "🚀"
        elif score > 30:
            return "📈"
        elif score > -10:
            return "➡️"
        elif score > -30:
            return "📉"
        else:
            return "⚠️"


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    import sys

    print("="*70)
    print("SYSTÈME DE SCORING DES SIGNAUX MACRO")
    print("="*70)

    # Initialiser
    aggregator = MacroNewsAggregator()
    cache = SignalCache()
    scorer = MacroSignalScorer(aggregator, cache)

    # Récupérer de nouveaux signaux
    print("\n📡 Récupération des signaux...")
    signals = aggregator.fetch_general_signals(days_back=2)
    cache.add_signals(signals)
    print(f"   ✅ {len(signals)} signaux récupérés")

    # Score du marché général
    print("\n🌍 Calcul du score général du marché...")
    market_score = scorer.compute_market_score(lookback_hours=48)
    scorer.print_score_report(market_score)

    # Test sur quelques actifs
    test_assets = ['BTC/USDT', 'AAPL', 'NVDA']

    if len(sys.argv) > 1:
        # Assets passés en argument
        test_assets = sys.argv[1:]

    print(f"\n📊 Calcul des scores pour actifs spécifiques...")
    for asset in test_assets:
        try:
            score = scorer.compute_asset_score(asset, lookback_hours=48)
            scorer.print_score_report(score)
        except Exception as e:
            print(f"   ❌ Erreur pour {asset}: {e}")

    # Export
    print("\n💾 Export des scores...")
    all_scores = {'market': market_score}
    for asset in test_assets:
        try:
            all_scores[asset] = scorer.compute_asset_score(asset, lookback_hours=48)
        except:
            pass

    scorer.export_scores(all_scores, 'macro_scores.json')
    print("   ✅ Scores exportés vers macro_scores.json")

    print("\n" + "="*70)
    print("✅ Système de scoring opérationnel!")
    print("="*70)
