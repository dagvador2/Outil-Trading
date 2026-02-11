"""
Intégration des signaux macro au paper trading
Mode : Filtre conservateur qui annule les trades contre-tendance macro forte

Usage:
    from macro_integration import MacroFilter

    filter = MacroFilter(enable=True)
    filtered_signals, macro_info = filter.filter_signals(signals, 'BTC/USDT')
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict
import logging
import os

# Import avec gestion d'erreur pour dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from macro_signal_scorer import MacroSignalScorer, MacroNewsAggregator

logger = logging.getLogger(__name__)


class MacroFilter:
    """
    Filtre les signaux techniques selon le contexte macro

    Stratégie conservative :
    - Si macro très bearish (< -60) : Annuler signaux LONG
    - Si macro très bullish (> +60) : Annuler signaux SHORT
    - Sinon : Laisser passer les signaux techniques
    """

    def __init__(self,
                 enable: bool = True,
                 strong_threshold: float = 60.0,
                 moderate_threshold: float = 30.0,
                 cache_hours: int = 4):
        """
        Args:
            enable: Activer/désactiver le filtre (pour A/B test)
            strong_threshold: Seuil pour blocage fort (annuler position)
            moderate_threshold: Seuil pour signal modéré (réduire conviction)
            cache_hours: Durée du cache des scores (éviter trop de requêtes)
        """
        self.enable = enable
        self.strong_threshold = strong_threshold
        self.moderate_threshold = moderate_threshold
        self.cache_hours = cache_hours

        if self.enable:
            try:
                self.scorer = MacroSignalScorer()
                self.cache: Dict[str, Tuple] = {}  # {asset: (score, timestamp)}
                self.last_market_score = None
                self.last_market_update = None
                logger.info("MacroFilter initialized and enabled")
            except Exception as e:
                logger.error(f"Error initializing MacroFilter: {e}")
                self.enable = False
                logger.warning("MacroFilter disabled due to initialization error")
        else:
            logger.info("MacroFilter disabled")

    def filter_signals(self,
                      signals: pd.DataFrame,
                      asset_symbol: str) -> Tuple[pd.DataFrame, Optional[Dict]]:
        """
        Filtre les signaux techniques selon le contexte macro

        Args:
            signals: DataFrame avec colonnes 'position', 'signal'
            asset_symbol: Symbole de l'actif ('BTC/USDT', 'AAPL', etc.)

        Returns:
            (signals_filtrés, macro_info)

        macro_info format:
            {
                'score': float,
                'confidence': float,
                'recommendation': str,
                'num_signals': int,
                'market_score': float,
                'filtered_longs': int,
                'filtered_shorts': int
            }
        """
        if not self.enable:
            return signals, None

        try:
            # Récupérer score macro pour cet actif
            macro_score = self._get_macro_score(asset_symbol)

            # Récupérer score du marché général
            market_score = self._get_market_score()

            # Appliquer filtre
            filtered = signals.copy()
            filtered_longs = 0
            filtered_shorts = 0

            for idx in filtered.index:
                original_pos = filtered.loc[idx, 'position']

                # Filtre fort : annuler positions contre-tendance macro
                if abs(macro_score.score) > self.strong_threshold:
                    if macro_score.score < -self.strong_threshold and original_pos == 1:
                        # Très bearish, annuler LONG
                        filtered.loc[idx, 'position'] = 0
                        filtered_longs += 1
                        logger.info(f"{asset_symbol}: LONG annulé (macro: {macro_score.score:.1f})")

                    elif macro_score.score > self.strong_threshold and original_pos == -1:
                        # Très bullish, annuler SHORT
                        filtered.loc[idx, 'position'] = 0
                        filtered_shorts += 1
                        logger.info(f"{asset_symbol}: SHORT annulé (macro: {macro_score.score:.1f})")

                # Filtre modéré : on pourrait réduire la taille des positions
                # Pour l'instant on laisse passer

            # Recalculer les signaux après filtrage
            filtered['signal'] = filtered['position'].diff()

            # Construire macro_info
            macro_info = {
                'timestamp': datetime.now().isoformat(),
                'asset': asset_symbol,
                'score': round(macro_score.score, 2),
                'confidence': round(macro_score.confidence, 3),
                'recommendation': macro_score.recommendation,
                'num_signals': macro_score.num_signals,
                'market_score': round(market_score.score, 2),
                'market_sentiment': market_score.market_sentiment,
                'fear_greed_index': market_score.fear_greed_index,
                'filtered_longs': filtered_longs,
                'filtered_shorts': filtered_shorts,
                'filter_active': abs(macro_score.score) > self.strong_threshold
            }

            return filtered, macro_info

        except Exception as e:
            logger.error(f"Macro filter error for {asset_symbol}: {e}")
            # En cas d'erreur, retourner signaux originaux
            return signals, {
                'timestamp': datetime.now().isoformat(),
                'asset': asset_symbol,
                'error': str(e),
                'filter_active': False
            }

    def _get_macro_score(self, asset_symbol: str):
        """
        Récupère le score macro avec cache

        Cache de X heures pour éviter trop de requêtes
        """
        now = datetime.now()

        # Vérifier cache
        if asset_symbol in self.cache:
            score, timestamp = self.cache[asset_symbol]
            if now - timestamp < timedelta(hours=self.cache_hours):
                logger.debug(f"Using cached macro score for {asset_symbol}")
                return score

        # Calculer nouveau score
        logger.info(f"Computing fresh macro score for {asset_symbol}")
        score = self.scorer.compute_asset_score(asset_symbol, lookback_hours=48)

        # Mettre en cache
        self.cache[asset_symbol] = (score, now)

        return score

    def _get_market_score(self):
        """Récupère le score du marché général avec cache"""
        now = datetime.now()

        # Vérifier cache
        if self.last_market_score and self.last_market_update:
            if now - self.last_market_update < timedelta(hours=self.cache_hours):
                logger.debug("Using cached market score")
                return self.last_market_score

        # Calculer nouveau score
        logger.info("Computing fresh market score")
        score = self.scorer.compute_market_score(lookback_hours=48)

        # Mettre en cache
        self.last_market_score = score
        self.last_market_update = now

        return score

    def get_all_cached_scores(self) -> Dict[str, Dict]:
        """
        Retourne tous les scores en cache (pour dashboard)

        Returns:
            {
                'market': {...},
                'BTC/USDT': {...},
                'AAPL': {...},
                ...
            }
        """
        result = {}

        # Score marché
        if self.last_market_score:
            result['market'] = {
                'score': round(self.last_market_score.score, 2),
                'confidence': round(self.last_market_score.confidence, 3),
                'recommendation': self.last_market_score.recommendation,
                'sentiment': self.last_market_score.market_sentiment,
                'fear_greed': self.last_market_score.fear_greed_index,
                'timestamp': self.last_market_update.isoformat() if self.last_market_update else None
            }

        # Scores par actif
        for asset, (score, timestamp) in self.cache.items():
            result[asset] = {
                'score': round(score.score, 2),
                'confidence': round(score.confidence, 3),
                'recommendation': score.recommendation,
                'num_signals': score.num_signals,
                'timestamp': timestamp.isoformat()
            }

        return result

    def clear_cache(self):
        """Vide le cache (forcer mise à jour)"""
        self.cache.clear()
        self.last_market_score = None
        self.last_market_update = None
        logger.info("Macro scores cache cleared")

    def set_thresholds(self, strong: float = None, moderate: float = None):
        """Ajuster les seuils à chaud"""
        if strong is not None:
            self.strong_threshold = strong
            logger.info(f"Strong threshold set to {strong}")
        if moderate is not None:
            self.moderate_threshold = moderate
            logger.info(f"Moderate threshold set to {moderate}")


class MacroSignalCollector:
    """
    Collecteur de signaux macro en background
    Optionnel : pour pré-remplir le cache
    """

    def __init__(self):
        self.aggregator = MacroNewsAggregator()
        self.last_collection = None

    def collect_signals(self, days_back: int = 2):
        """
        Collecte les signaux généraux et les met en cache

        À appeler périodiquement (ex: toutes les 2h)
        """
        try:
            logger.info("Collecting macro signals...")

            # Récupérer signaux généraux
            signals = self.aggregator.fetch_general_signals(days_back=days_back)

            # Sauvegarder dans cache
            from news_fetcher import SignalCache
            cache = SignalCache()
            cache.add_signals(signals)

            self.last_collection = datetime.now()

            logger.info(f"✅ Collected {len(signals)} macro signals")

            return len(signals)

        except Exception as e:
            logger.error(f"Error collecting macro signals: {e}")
            return 0

    def should_collect(self, interval_hours: int = 2) -> bool:
        """Vérifie s'il est temps de collecter de nouveaux signaux"""
        if not self.last_collection:
            return True

        elapsed = datetime.now() - self.last_collection
        return elapsed > timedelta(hours=interval_hours)


# ============================================================================
# Utilitaires
# ============================================================================

def test_macro_filter():
    """Test rapide du filtre"""
    print("="*70)
    print("TEST DU MACRO FILTER")
    print("="*70)

    # Créer signaux de test
    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    test_signals = pd.DataFrame({
        'position': [1, 1, 0, -1, -1, 0, 1, 0, -1, 0],
        'signal': [1, 0, -1, -1, 0, 1, 1, -1, -1, 1]
    }, index=dates)

    print("\nSignaux originaux:")
    print(test_signals)

    # Créer filtre
    macro_filter = MacroFilter(enable=True, strong_threshold=60)

    # Tester sur BTC
    print("\n📊 Test filtre sur BTC/USDT...")
    filtered, macro_info = macro_filter.filter_signals(test_signals, 'BTC/USDT')

    print("\nMacro Info:")
    for key, value in macro_info.items():
        print(f"  {key}: {value}")

    print("\nSignaux filtrés:")
    print(filtered)

    # Afficher scores en cache
    print("\n📈 Scores en cache:")
    all_scores = macro_filter.get_all_cached_scores()
    for asset, score_info in all_scores.items():
        print(f"\n  {asset}:")
        for key, value in score_info.items():
            print(f"    {key}: {value}")

    print("\n" + "="*70)
    print("✅ Test terminé!")
    print("="*70)


if __name__ == '__main__':
    # Test du module
    test_macro_filter()
