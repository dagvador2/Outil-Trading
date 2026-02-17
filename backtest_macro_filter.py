"""
Filtre Macro pour Backtesting Historique
Reproduit la logique du MacroFilter live avec des donnees historiques.

Le filtre live utilise : news RSS (50%) + sentiment/VIX (30%) + eco (20%)
Le proxy backtest utilise : tendance marche (40%) + tendance actif (60%)

Meme logique d'application :
  - Score < -threshold : annule les LONG (contexte tres bearish)
  - Score > +threshold : annule les SHORT (contexte tres bullish)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Optional, Dict, Tuple
from datetime import datetime, timedelta


class BacktestMacroFilter:
    """
    Proxy historique du MacroFilter live.
    Utilise VIX + tendance S&P 500 + tendance de l'actif
    pour calculer un score macro [-100, +100] a chaque date.
    """

    def __init__(self,
                 strong_threshold: float = 60.0,
                 market_weight: float = 0.4,
                 asset_weight: float = 0.6):
        self.strong_threshold = strong_threshold
        self.market_weight = market_weight
        self.asset_weight = asset_weight

        # Cache des donnees de marche (VIX + S&P 500)
        self._market_data_cache: Optional[pd.DataFrame] = None
        self._cache_start: Optional[str] = None
        self._cache_end: Optional[str] = None

    def load_market_data(self, start_date: str, end_date: str):
        """
        Pre-charge les donnees de marche (VIX + S&P 500) pour la periode.
        A appeler une seule fois avant de backtester tous les actifs.
        """
        if (self._market_data_cache is not None and
                self._cache_start == start_date and
                self._cache_end == end_date):
            return  # Deja charge

        # Charger S&P 500
        sp500 = yf.Ticker('^GSPC').history(start=start_date, end=end_date)
        sp500.columns = [c.lower() for c in sp500.columns]

        # Charger VIX
        vix = yf.Ticker('^VIX').history(start=start_date, end=end_date)
        vix.columns = [c.lower() for c in vix.columns]

        # Construire DataFrame de marche
        market = pd.DataFrame(index=sp500.index)
        market['sp500_close'] = sp500['close']
        market['sp500_sma50'] = sp500['close'].rolling(50).mean()
        market['sp500_sma200'] = sp500['close'].rolling(200).mean()
        market['vix'] = vix['close'].reindex(sp500.index, method='ffill')

        # Calculer score marche pour chaque jour
        market['market_score'] = 0.0

        # VIX component (-30 a +10)
        market.loc[market['vix'] > 30, 'market_score'] -= 30
        market.loc[(market['vix'] > 20) & (market['vix'] <= 30), 'market_score'] -= 10
        market.loc[market['vix'] < 15, 'market_score'] += 10

        # S&P 500 vs SMA 200 (-25 a +25)
        market.loc[market['sp500_close'] > market['sp500_sma200'], 'market_score'] += 25
        market.loc[market['sp500_close'] < market['sp500_sma200'], 'market_score'] -= 25

        # S&P 500 vs SMA 50 (-15 a +15)
        market.loc[market['sp500_close'] > market['sp500_sma50'], 'market_score'] += 15
        market.loc[market['sp500_close'] < market['sp500_sma50'], 'market_score'] -= 15

        # S&P 500 momentum (rendement 20 jours) : -20 a +20
        sp_returns_20d = sp500['close'].pct_change(20) * 100
        market['sp_momentum'] = sp_returns_20d.clip(-10, 10) * 2
        market['market_score'] += market['sp_momentum'].fillna(0)

        # Clamp
        market['market_score'] = market['market_score'].clip(-100, 100)

        self._market_data_cache = market
        self._cache_start = start_date
        self._cache_end = end_date

    def compute_asset_score(self, asset_data: pd.DataFrame) -> pd.Series:
        """
        Calcule le score macro pour un actif a chaque date.

        Combine:
        - Score marche global (40%)
        - Score tendance de l'actif (60%)

        Args:
            asset_data: DataFrame OHLCV de l'actif

        Returns:
            Series avec score macro [-100, +100] pour chaque date
        """
        asset_score = pd.Series(0.0, index=asset_data.index)

        # --- Composante actif (60%) ---

        close = asset_data['close']

        # Actif vs SMA 200 (-25 a +25)
        sma200 = close.rolling(200).mean()
        asset_score[close > sma200] += 25
        asset_score[close < sma200] -= 25

        # Actif vs SMA 50 (-15 a +15)
        sma50 = close.rolling(50).mean()
        asset_score[close > sma50] += 15
        asset_score[close < sma50] -= 15

        # Momentum 20 jours (-20 a +20)
        momentum = close.pct_change(20) * 100
        asset_score += momentum.clip(-10, 10).fillna(0) * 2

        # RSI 14 : extremes
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # RSI extreme : -15 a +15
        asset_score[rsi > 75] -= 15
        asset_score[rsi < 25] += 15

        asset_score = asset_score.clip(-100, 100)

        # --- Composante marche (40%) ---
        if self._market_data_cache is not None:
            market_scores = self._market_data_cache['market_score'].reindex(
                asset_data.index, method='ffill'
            ).fillna(0)
        else:
            market_scores = pd.Series(0.0, index=asset_data.index)

        # --- Score composite ---
        composite = (
            market_scores * self.market_weight +
            asset_score * self.asset_weight
        )

        return composite.clip(-100, 100)

    def filter_signals(self,
                       signals: pd.DataFrame,
                       asset_data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Filtre les signaux techniques selon le contexte macro historique.
        Meme logique que le MacroFilter live.

        Args:
            signals: DataFrame avec colonnes 'position', 'signal'
            asset_data: DataFrame OHLCV de l'actif

        Returns:
            (signals_filtres, macro_info)
        """
        macro_scores = self.compute_asset_score(asset_data)

        filtered = signals.copy()
        filtered_longs = 0
        filtered_shorts = 0

        for idx in filtered.index:
            score = macro_scores.get(idx, 0)
            original_pos = filtered.loc[idx, 'position']

            if abs(score) > self.strong_threshold:
                if score < -self.strong_threshold and original_pos == 1:
                    # Tres bearish → annuler LONG
                    filtered.loc[idx, 'position'] = 0
                    filtered_longs += 1
                elif score > self.strong_threshold and original_pos == -1:
                    # Tres bullish → annuler SHORT
                    filtered.loc[idx, 'position'] = 0
                    filtered_shorts += 1

        # Recalculer les signaux
        filtered['signal'] = filtered['position'].diff()

        macro_info = {
            'avg_score': round(macro_scores.mean(), 2),
            'min_score': round(macro_scores.min(), 2),
            'max_score': round(macro_scores.max(), 2),
            'filtered_longs': filtered_longs,
            'filtered_shorts': filtered_shorts,
            'total_filtered': filtered_longs + filtered_shorts,
            'filter_active_pct': round(
                (macro_scores.abs() > self.strong_threshold).mean() * 100, 1
            ),
        }

        return filtered, macro_info
