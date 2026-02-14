"""
Dashboard Web Interactif - Streamlit
Interface de monitoring en temps réel pour 50 actifs
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import os
import json
from typing import Dict, List

from live_data_feed import LiveDataFeed
from signal_generator import LiveSignalGenerator
from assets_config import MONITORED_ASSETS, get_all_symbols, get_asset_type, get_asset_info
from macro_data_fetcher import MacroDataFetcher
from backtesting_engine import BacktestEngine
from strategies import *
from yahoo_data_feed import convert_to_yahoo_symbol
from backtest_library import instantiate_strategy, BacktestLibrary
from strategy_allocator import StrategyAllocator, TradingPlan
import yfinance as yf
import pytz
import math


def _fmt_price(value: float) -> str:
    """Format a price with enough decimals to be readable (small crypto prices)."""
    if value == 0:
        return "0"
    if abs(value) >= 1:
        return f"{value:,.4f}"
    sig_decimals = -int(math.floor(math.log10(abs(value)))) + 3
    decimals = max(4, sig_decimals)
    return f"{value:.{decimals}f}"


# ============================================================================
# Configuration de la page
# ============================================================================

st.set_page_config(
    page_title="Trading Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1f77b4 0%, #2ca02c 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .buy-signal {
        background-color: #d4edda;
        color: #155724;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .sell-signal {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .hold-signal {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    /* Multi Paper Trading tab styles */
    .mpt-status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .mpt-status-active {
        background-color: #d4edda;
        color: #155724;
    }
    .mpt-status-stale {
        background-color: #fff3cd;
        color: #856404;
    }
    .mpt-kpi-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    .mpt-kpi-card {
        flex: 1;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border-left: 4px solid #1f77b4;
    }
    .mpt-kpi-card.positive { border-left-color: #2ca02c; }
    .mpt-kpi-card.negative { border-left-color: #d62728; }
    .mpt-kpi-card.neutral { border-left-color: #6c757d; }
    .mpt-kpi-card .kpi-value {
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0.3rem 0;
    }
    .mpt-kpi-card .kpi-label {
        font-size: 0.8rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .mpt-kpi-card .kpi-sub {
        font-size: 0.85rem;
        color: #495057;
    }
    .mpt-glossary {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    .mpt-glossary dt {
        font-weight: 600;
        color: #1f77b4;
        margin-top: 0.5rem;
    }
    .mpt-glossary dd {
        margin-left: 1rem;
        margin-bottom: 0.5rem;
        color: #495057;
        font-size: 0.9rem;
    }
    .mpt-section-help {
        font-size: 0.85rem;
        color: #6c757d;
        font-style: italic;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Initialisation (avec cache)
# ============================================================================

@st.cache_resource
def init_feed():
    """Initialise le feed de données (cached)"""
    fetcher = MacroDataFetcher()
    feed = LiveDataFeed(fetcher.alpha_vantage_key, fetcher.fred_api_key)
    return feed


@st.cache_resource
def init_generator(_feed):
    """Initialise le générateur de signaux (cached)"""
    return LiveSignalGenerator(_feed)


# ============================================================================
# Fonctions de récupération de données
# ============================================================================

# Cache TTL par timeframe (en secondes)
SIGNAL_CACHE_TTL = {
    '1d': 4 * 3600,     # 4h pour daily (donnees changent 1x/jour)
    '4h': 2 * 3600,     # 2h
    '1h': 30 * 60,      # 30min
    '15min': 10 * 60,   # 10min
    '5min': 3 * 60,     # 3min
}

SIGNAL_CACHE_DIR = 'signals_cache'


def _get_cache_path(asset_filter: str, timeframe: str) -> str:
    """Retourne le chemin du fichier cache pour ces parametres"""
    os.makedirs(SIGNAL_CACHE_DIR, exist_ok=True)
    safe_name = asset_filter.replace(' ', '_').lower()
    return os.path.join(SIGNAL_CACHE_DIR, f'signals_{safe_name}_{timeframe}.json')


def _load_signals_cache(asset_filter: str, timeframe: str) -> List[Dict]:
    """Charge les signaux depuis le cache disque si frais"""
    cache_path = _get_cache_path(asset_filter, timeframe)
    if not os.path.exists(cache_path):
        return []

    try:
        mtime = os.path.getmtime(cache_path)
        age = time.time() - mtime
        ttl = SIGNAL_CACHE_TTL.get(timeframe, 1800)

        if age > ttl:
            return []  # Cache expire

        with open(cache_path, 'r') as f:
            cached = json.load(f)

        # Reconvertir les timestamps
        for sig in cached:
            if 'timestamp' in sig and isinstance(sig['timestamp'], str):
                try:
                    sig['timestamp'] = datetime.fromisoformat(sig['timestamp'])
                except Exception:
                    sig['timestamp'] = datetime.now()

        return cached
    except Exception:
        return []


def _save_signals_cache(signals: List[Dict], asset_filter: str, timeframe: str):
    """Sauvegarde les signaux en cache disque"""
    cache_path = _get_cache_path(asset_filter, timeframe)
    try:
        serializable = []
        for sig in signals:
            s = dict(sig)
            if 'timestamp' in s and isinstance(s['timestamp'], datetime):
                s['timestamp'] = s['timestamp'].isoformat()
            serializable.append(s)

        with open(cache_path, 'w') as f:
            json.dump(serializable, f, default=str)
    except Exception:
        pass  # Cache write failure is not critical


def get_all_current_signals(generator: LiveSignalGenerator,
                             asset_filter: str = "Tous",
                             timeframe: str = "1h",
                             force_refresh: bool = False) -> List[Dict]:
    """
    Recupere tous les signaux actuels, avec cache disque.

    Args:
        generator: Instance LiveSignalGenerator
        asset_filter: Filtre par categorie ('Tous', 'Crypto', 'Stocks', etc.)
        timeframe: Intervalle de temps (5min, 15min, 1h, 4h, 1d)
        force_refresh: Forcer le recalcul (ignorer le cache)

    Returns:
        Liste de signaux
    """
    # Tenter le cache disque si pas de force refresh
    if not force_refresh:
        cached = _load_signals_cache(asset_filter, timeframe)
        if cached:
            ttl = SIGNAL_CACHE_TTL.get(timeframe, 1800)
            cache_path = _get_cache_path(asset_filter, timeframe)
            age_min = (time.time() - os.path.getmtime(cache_path)) / 60
            st.sidebar.info(f"Signaux depuis cache ({age_min:.0f}min, TTL {ttl//60}min)")
            return cached

    signals = []

    # Filtrer symboles selon categorie
    if asset_filter == "Tous":
        symbols = get_all_symbols()
    else:
        category_map = {
            'Crypto': 'crypto',
            'Tech Stocks': 'tech_stocks',
            'Commodities': 'commodities',
            'Indices': 'indices',
            'Forex': 'forex',
            'Defensive': 'defensive'
        }
        category = category_map.get(asset_filter)
        if category and category in MONITORED_ASSETS:
            symbols = [asset['symbol'] for asset in MONITORED_ASSETS[category]]
        else:
            symbols = []

    # Recuperer signaux
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, symbol in enumerate(symbols):
        try:
            status_text.text(f"Chargement {symbol}... ({i+1}/{len(symbols)})")

            asset_type = get_asset_type(symbol)
            signal = generator.get_current_signal(symbol, asset_type, timeframe=timeframe)

            if signal:
                signals.append(signal)

            progress_bar.progress((i + 1) / len(symbols))

        except Exception as e:
            st.warning(f"Erreur pour {symbol}: {str(e)}")
            continue

    progress_bar.empty()
    status_text.empty()

    # Sauvegarder en cache
    if signals:
        _save_signals_cache(signals, asset_filter, timeframe)

    return signals


def create_price_chart(symbol: str, data: pd.DataFrame,
                       signal: Dict) -> go.Figure:
    """
    Crée un graphique de prix avec indicateurs

    Args:
        symbol: Symbole de l'actif
        data: DataFrame OHLCV
        signal: Dictionnaire du signal

    Returns:
        Figure Plotly
    """
    fig = go.Figure()

    # Chandelier
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close'],
        name='Prix'
    ))

    # Bollinger Bands (si disponibles)
    if 'indicators' in signal and signal['indicators']:
        ind = signal['indicators']
        if 'bb_upper' in ind and 'bb_lower' in ind and 'bb_middle' in ind:
            # Ajouter bandes (simplifiées, sur toute la période)
            latest_close = data['close'].iloc[-1]
            bb_upper = ind['bb_upper']
            bb_lower = ind['bb_lower']
            bb_middle = ind['bb_middle']

            fig.add_hline(y=bb_upper, line_dash="dash", line_color="red",
                          annotation_text="BB Upper")
            fig.add_hline(y=bb_middle, line_dash="dash", line_color="blue",
                          annotation_text="BB Middle")
            fig.add_hline(y=bb_lower, line_dash="dash", line_color="green",
                          annotation_text="BB Lower")

    # Mise en forme
    fig.update_layout(
        title=f"{symbol} - Prix",
        xaxis_title="Date",
        yaxis_title="Prix ($)",
        height=400,
        template="plotly_white",
        xaxis_rangeslider_visible=False
    )

    return fig


def is_market_open(symbol: str) -> tuple[bool, str]:
    """
    Détermine si le marché est ouvert pour un symbole donné

    Args:
        symbol: Symbole de l'actif

    Returns:
        Tuple (is_open, status_message)
    """
    now_utc = datetime.now(pytz.UTC)
    weekday = now_utc.weekday()  # 0=Lundi, 6=Dimanche

    # Crypto : 24/7
    if '/USDT' in symbol or 'BTC' in symbol or 'ETH' in symbol:
        return True, "24/7"

    # Forex : 24/5 (fermé weekend)
    if '=X' in symbol or 'USD' in symbol or 'EUR' in symbol or 'GBP' in symbol or 'JPY' in symbol:
        if weekday >= 5:  # Samedi ou Dimanche
            return False, "Fermé (weekend)"
        return True, "24/5"

    # Stocks US : 14:30-21:00 UTC (9:30-16:00 EST)
    if symbol in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'AMD', 'NFLX', 'COIN', 'JNJ', 'PG', 'KO', 'PEP', 'WMT']:
        if weekday >= 5:
            return False, "Fermé (weekend)"
        hour_utc = now_utc.hour
        # Marché ouvert 14:30-21:00 UTC
        if 14 <= hour_utc < 21 or (hour_utc == 14 and now_utc.minute >= 30):
            return True, "Ouvert (US)"
        return False, "Fermé (hors heures)"

    # Indices US
    if symbol in ['^GSPC', '^IXIC', '^DJI']:
        if weekday >= 5:
            return False, "Fermé (weekend)"
        hour_utc = now_utc.hour
        if 14 <= hour_utc < 21 or (hour_utc == 14 and now_utc.minute >= 30):
            return True, "Ouvert (US)"
        return False, "Fermé (hors heures)"

    # Indices Européens : 8:00-16:30 CET (7:00-15:30 UTC)
    if symbol in ['^FTSE', '^GDAXI', '^FCHI']:
        if weekday >= 5:
            return False, "Fermé (weekend)"
        hour_utc = now_utc.hour
        if 7 <= hour_utc < 16 or (hour_utc == 15 and now_utc.minute < 30):
            return True, "Ouvert (EU)"
        return False, "Fermé (hors heures)"

    # Indices Asie
    if symbol == '^N225':  # Nikkei
        if weekday >= 5:
            return False, "Fermé (weekend)"
        hour_utc = now_utc.hour
        # Tokyo: 0:00-6:00 UTC
        if 0 <= hour_utc < 6:
            return True, "Ouvert (Asia)"
        return False, "Fermé (hors heures)"

    # Commodities : Généralement 23h/24, 5j/7
    if '=F' in symbol:
        if weekday >= 5:
            return False, "Fermé (weekend)"
        return True, "Ouvert (Commodities)"

    # Par défaut : On considère fermé si incertain
    return False, "État inconnu"


def get_trade_recommendations(signal: Dict) -> Dict:
    """
    Génère des recommandations de trading détaillées (LONG ou SHORT)

    Args:
        signal: Dictionnaire du signal

    Returns:
        Dict avec SL, TP, simulation, trailing, etc.
    """
    price = signal.get('current_price', 0)
    symbol = signal.get('symbol', '')
    confidence = signal.get('confidence', 0)
    signal_type = signal.get('signal', 'HOLD')

    if price <= 0 or signal_type not in ['BUY', 'SELL']:
        return {}

    # Déterminer volatilité selon le type d'actif
    is_crypto = '/USDT' in symbol or 'BTC' in symbol or 'ETH' in symbol
    is_volatile_stock = symbol in ['TSLA', 'NVDA', 'AMD', 'COIN']

    # Paramètres selon volatilité
    if is_crypto:
        sl_pct = 8.0
        tp_pct = 15.0
        trailing_recommended = True
        trailing_distance = 5.0
        trade_type = "Swing (2-7 jours)"
    elif is_volatile_stock:
        sl_pct = 5.0
        tp_pct = 10.0
        trailing_recommended = True
        trailing_distance = 3.0
        trade_type = "Day/Swing (1-5 jours)"
    else:
        sl_pct = 3.0
        tp_pct = 6.0
        trailing_recommended = False
        trailing_distance = 2.0
        trade_type = "Position (5-30 jours)"

    # Ajuster selon confiance
    if confidence >= 0.80:
        tp_pct *= 1.3  # Target plus ambitieux si haute confiance
        trade_type = f"{trade_type} - Haute confiance"
    elif confidence < 0.65:
        sl_pct *= 0.8  # SL plus serré si faible confiance

    # Calculs selon direction (LONG ou SHORT)
    if signal_type == 'BUY':
        # LONG : SL en dessous, TP au dessus
        sl_price = price * (1 - sl_pct / 100)
        tp_price = price * (1 + tp_pct / 100)
        direction = "LONG"

        # Simulation sur base €1000
        position_size_eur = 1000
        quantity = position_size_eur / price

        # Scénario gain (TP atteint - prix monte)
        gain_eur = quantity * (tp_price - price)
        gain_pct = tp_pct

        # Scénario perte (SL touché - prix baisse)
        loss_eur = quantity * (price - sl_price)
        loss_pct = sl_pct

    else:  # SELL
        # SHORT : SL au dessus, TP en dessous
        sl_price = price * (1 + sl_pct / 100)
        tp_price = price * (1 - tp_pct / 100)
        direction = "SHORT"

        # Simulation sur base €1000
        position_size_eur = 1000
        quantity = position_size_eur / price

        # Scénario gain (TP atteint - prix baisse)
        gain_eur = quantity * (price - tp_price)
        gain_pct = tp_pct

        # Scénario perte (SL touché - prix monte)
        loss_eur = quantity * (sl_price - price)
        loss_pct = sl_pct

    # Risk/Reward ratio
    risk_reward = gain_eur / loss_eur if loss_eur > 0 else 0

    return {
        'direction': direction,
        'sl_price': sl_price,
        'sl_pct': sl_pct,
        'tp_price': tp_price,
        'tp_pct': tp_pct,
        'trailing_recommended': trailing_recommended,
        'trailing_distance': trailing_distance,
        'trade_type': trade_type,
        'simulation_investment': position_size_eur,
        'simulation_quantity': quantity,
        'simulation_gain_eur': gain_eur,
        'simulation_gain_pct': gain_pct,
        'simulation_loss_eur': loss_eur,
        'simulation_loss_pct': loss_pct,
        'risk_reward_ratio': risk_reward
    }


def create_indicators_gauge(indicators: Dict) -> go.Figure:
    """
    Crée des jauges pour indicateurs techniques

    Args:
        indicators: Dict avec RSI, BB position, etc.

    Returns:
        Figure Plotly avec jauges
    """
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]],
        subplot_titles=("RSI", "Bollinger Position", "MACD Histogram")
    )

    # RSI
    rsi = indicators.get('rsi', 50)
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=rsi,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "lightgray"},
                {'range': [70, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ), row=1, col=1)

    # BB Position
    bb_position = indicators.get('bb_position', 50)
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=bb_position,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "purple"},
            'steps': [
                {'range': [0, 20], 'color': "lightgreen"},
                {'range': [20, 80], 'color': "lightgray"},
                {'range': [80, 100], 'color': "lightcoral"}
            ]
        }
    ), row=1, col=2)

    # MACD
    macd = indicators.get('macd_histogram', 0)
    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=macd,
        domain={'x': [0, 1], 'y': [0, 1]},
        delta={'reference': 0, 'increasing': {'color': "green"},
               'decreasing': {'color': "red"}}
    ), row=1, col=3)

    fig.update_layout(height=250)

    return fig


# ============================================================================
# Fonctions pour l'explorateur de backtests
# ============================================================================

def get_strategy_description(strategy_name: str) -> Dict:
    """
    Retourne la description détaillée d'une stratégie

    Args:
        strategy_name: Nom de la stratégie

    Returns:
        Dict avec description, paramètres, indicateurs
    """
    descriptions = {
        'MA_Crossover_20_50': {
            'name': 'Moving Average Crossover 20/50',
            'description': 'Croisement de moyennes mobiles. Achat quand la MA rapide (20) croise au-dessus de la MA lente (50), vente quand elle croise en-dessous.',
            'parameters': {'MA Rapide': 20, 'MA Lente': 50},
            'indicators': ['SMA 20', 'SMA 50'],
            'best_for': 'Tendances claires',
            'risk': 'Moyen'
        },
        'MA_Crossover_10_30': {
            'name': 'Moving Average Crossover 10/30',
            'description': 'Croisement de moyennes mobiles rapides. Plus reactif que 20/50 mais plus de faux signaux.',
            'parameters': {'MA Rapide': 10, 'MA Lente': 30},
            'indicators': ['SMA 10', 'SMA 30'],
            'best_for': 'Marches volatils',
            'risk': 'Moyen-Eleve'
        },
        'RSI_14_30_70': {
            'name': 'RSI (14, 30/70)',
            'description': 'Achat quand RSI < 30 (survente), vente quand RSI > 70 (surachat).',
            'parameters': {'Periode RSI': 14, 'Survente': 30, 'Surachat': 70},
            'indicators': ['RSI 14'],
            'best_for': 'Marches oscillants',
            'risk': 'Moyen-Eleve'
        },
        'RSI_14_35_80': {
            'name': 'RSI (14, 35/80)',
            'description': 'RSI avec seuils asymetriques. Achat RSI < 35, vente RSI > 80. Plus selectif en surachat.',
            'parameters': {'Periode RSI': 14, 'Survente': 35, 'Surachat': 80},
            'indicators': ['RSI 14'],
            'best_for': 'Marches haussiers',
            'risk': 'Moyen'
        },
        'MACD_Standard': {
            'name': 'MACD (12/26/9)',
            'description': 'Achat quand MACD croise au-dessus de sa ligne de signal, vente quand il croise en-dessous.',
            'parameters': {'Fast': 12, 'Slow': 26, 'Signal': 9},
            'indicators': ['MACD', 'Signal', 'Histogram'],
            'best_for': 'Changements de tendance',
            'risk': 'Moyen'
        },
        'Bollinger_20_2': {
            'name': 'Bollinger Bands (20, 2)',
            'description': 'Achat quand le prix touche la bande inferieure, vente quand il touche la bande superieure.',
            'parameters': {'Periode': 20, 'Ecart-type': 2},
            'indicators': ['BB Upper', 'BB Middle', 'BB Lower'],
            'best_for': 'Marches en range',
            'risk': 'Moyen'
        },
        'Combined': {
            'name': 'Strategie Combinee',
            'description': 'Combine MA + RSI + MACD. Necessite au moins 2 indicateurs sur 3 alignes pour un signal.',
            'parameters': {'MA': '20/50', 'RSI': '14/30/70', 'MACD': '12/26/9'},
            'indicators': ['SMA 20', 'SMA 50', 'RSI 14', 'MACD'],
            'best_for': 'Tous types de marches',
            'risk': 'Faible-Moyen'
        },
        'ADX_Trend_14_25': {
            'name': 'ADX Trend (14, 25)',
            'description': 'Entre en position uniquement sur tendances fortes (ADX > 25). Direction determinee par +DI/-DI.',
            'parameters': {'Periode ADX': 14, 'Seuil': 25},
            'indicators': ['ADX', '+DI', '-DI'],
            'best_for': 'Tendances fortes',
            'risk': 'Moyen'
        },
        'VWAP_20': {
            'name': 'VWAP (20)',
            'description': 'Achat quand prix sous la bande basse VWAP (sous-evalue), vente quand au-dessus de la bande haute.',
            'parameters': {'Periode': 20},
            'indicators': ['VWAP', 'Bandes VWAP'],
            'best_for': 'Retour a la moyenne',
            'risk': 'Moyen'
        },
        'Ichimoku_9_26_52': {
            'name': 'Ichimoku Cloud (9/26/52)',
            'description': 'Systeme complet japonais. Achat quand prix au-dessus du nuage avec Tenkan > Kijun.',
            'parameters': {'Tenkan': 9, 'Kijun': 26, 'Senkou B': 52},
            'indicators': ['Tenkan', 'Kijun', 'Senkou A/B', 'Chikou'],
            'best_for': 'Analyse complete',
            'risk': 'Moyen'
        }
    }

    # Normaliser le nom de la stratégie pour la recherche
    normalized_name = strategy_name.replace('_', ' ').replace('  ', ' ').lower()

    # Recherche partielle si nom exact pas trouvé
    for key, value in descriptions.items():
        normalized_key = key.replace('_', ' ').lower()
        if normalized_name in normalized_key or normalized_key in normalized_name:
            return value

    # Description par défaut avec le nom original
    return {
        'name': strategy_name.replace('_', ' '),
        'description': 'Stratégie de trading technique',
        'parameters': {},
        'indicators': ['Inconnu'],
        'best_for': 'À déterminer',
        'risk': 'Non évalué'
    }


def run_detailed_backtest(symbol: str, strategy_name: str,
                          start_date: str = '2024-01-01',
                          end_date: str = '2025-12-31') -> Dict:
    """
    Exécute un backtest détaillé et retourne tous les détails

    Args:
        symbol: Symbole de l'actif
        strategy_name: Nom de la stratégie
        start_date: Date de début
        end_date: Date de fin

    Returns:
        Dict avec data, trades, metrics, strategy
    """
    try:
        # 1. Convertir le symbole pour Yahoo Finance
        yahoo_symbol = convert_to_yahoo_symbol(symbol)

        # 2. Charger données historiques
        ticker = yf.Ticker(yahoo_symbol)
        data = ticker.history(start=start_date, end=end_date)

        if len(data) == 0:
            st.error(f"Aucune donnée historique trouvée pour {symbol} (Yahoo: {yahoo_symbol})")
            return None

        # Renommer colonnes en minuscules
        data.columns = [col.lower() for col in data.columns]

        # 3. Instancier la stratégie (même fonction que la bibliothèque)
        strategy = instantiate_strategy(strategy_name)
        if strategy is None:
            st.error(f"Stratégie inconnue: {strategy_name}")
            return None

        # 4. Exécuter backtest
        engine = BacktestEngine(
            initial_capital=10000,
            commission=0.001,
            slippage=0.0005
        )

        metrics = engine.run_backtest(data, strategy)

        # 5. Récupérer trades
        trades_list = []
        for trade in engine.trades:
            trades_list.append({
                'entry_date': trade.entry_date,
                'exit_date': trade.exit_date,
                'type': trade.position_type,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'size': trade.size,
                'profit': trade.pnl,  # Utiliser pnl au lieu de profit
                'profit_pct': trade.pnl_pct,  # Utiliser pnl_pct au lieu de profit_pct
                'duration_days': (trade.exit_date - trade.entry_date).days
            })

        return {
            'data': data,
            'trades': trades_list,
            'metrics': metrics,
            'strategy': strategy,
            'equity_curve': engine.equity_curve
        }

    except Exception as e:
        st.error(f"❌ Erreur lors du backtest de {symbol} (Yahoo: {yahoo_symbol if 'yahoo_symbol' in locals() else 'N/A'})")
        st.error(f"Détails: {str(e)}")
        st.info(f"💡 Astuce: Vérifiez que le symbole est correct. Les cryptos doivent utiliser le format Yahoo (ex: BTC-USD au lieu de BTC/USDT)")
        return None


def create_backtest_chart(data: pd.DataFrame, trades: List[Dict],
                          symbol: str) -> go.Figure:
    """
    Crée un graphique avec prix et trades marqués

    Args:
        data: DataFrame OHLCV
        trades: Liste des trades
        symbol: Symbole de l'actif

    Returns:
        Figure Plotly
    """
    fig = go.Figure()

    # Prix (chandelier)
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close'],
        name='Prix',
        increasing_line_color='green',
        decreasing_line_color='red'
    ))

    # Marquer les trades
    for trade in trades:
        # Point d'entrée
        color = 'green' if trade['type'] == 'LONG' else 'red'
        symbol_marker = 'triangle-up' if trade['type'] == 'LONG' else 'triangle-down'

        fig.add_trace(go.Scatter(
            x=[trade['entry_date']],
            y=[trade['entry_price']],
            mode='markers',
            marker=dict(size=12, color=color, symbol=symbol_marker),
            name=f"Entrée {trade['type']}",
            showlegend=False,
            hovertemplate=f"<b>Entrée {trade['type']}</b><br>" +
                         f"Prix: ${_fmt_price(trade['entry_price'])}<br>" +
                         f"Date: {trade['entry_date']}<extra></extra>"
        ))

        # Point de sortie
        exit_color = 'lightgreen' if trade['profit'] > 0 else 'lightcoral'

        fig.add_trace(go.Scatter(
            x=[trade['exit_date']],
            y=[trade['exit_price']],
            mode='markers',
            marker=dict(size=10, color=exit_color, symbol='x'),
            name='Sortie',
            showlegend=False,
            hovertemplate=f"<b>Sortie</b><br>" +
                         f"Prix: ${_fmt_price(trade['exit_price'])}<br>" +
                         f"Profit: ${trade['profit']:.2f} ({trade['profit_pct']:.2f}%)<br>" +
                         f"Durée: {trade['duration_days']} jours<extra></extra>"
        ))

        # Ligne reliant entrée et sortie
        fig.add_trace(go.Scatter(
            x=[trade['entry_date'], trade['exit_date']],
            y=[trade['entry_price'], trade['exit_price']],
            mode='lines',
            line=dict(color=exit_color, width=1, dash='dot'),
            showlegend=False,
            hoverinfo='skip'
        ))

    fig.update_layout(
        title=f"{symbol} - Backtest avec Trades",
        xaxis_title="Date",
        yaxis_title="Prix ($)",
        height=600,
        template="plotly_white",
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )

    return fig


# ============================================================================
# Interface principale
# ============================================================================

def main():
    # Header
    st.markdown('<div class="main-header">📊 Trading Dashboard Live</div>',
                unsafe_allow_html=True)

    # Initialisation
    with st.spinner("Initialisation du système..."):
        feed = init_feed()
        generator = init_generator(feed)

    # Initialiser session state pour les signaux
    if 'all_signals' not in st.session_state:
        st.session_state.all_signals = []
        st.session_state.signals_loaded = False
        st.session_state.last_category = None
        st.session_state.previous_signals = {}  # Track previous signals for change detection
        st.session_state.last_timeframe = None

    # Initialiser session state pour la bibliothèque backtests (évite rechargement)
    if 'backtest_library' not in st.session_state:
        st.session_state.backtest_library = None

    # Initialiser session state pour le backtest détaillé
    if 'detailed_backtest_results' not in st.session_state:
        st.session_state.detailed_backtest_results = None
    if 'detailed_backtest_asset' not in st.session_state:
        st.session_state.detailed_backtest_asset = None
    if 'detailed_backtest_strategy' not in st.session_state:
        st.session_state.detailed_backtest_strategy = None

    # Sidebar
    st.sidebar.title("⚙️ Paramètres")

    # Sélecteur de timeframe
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Timeframe des Signaux")

    timeframe_options = {
        "5min": "5 minutes (Scalping - Très volatil)",
        "15min": "15 minutes (Day trading)",
        "1h": "1 heure (Swing trading)",
        "4h": "4 heures (Position trading)",
        "1d": "1 jour (Paper trading - Recommandé)"
    }

    selected_timeframe = st.sidebar.selectbox(
        "Intervalle de temps",
        options=list(timeframe_options.keys()),
        index=4,  # Default: 1d (cohérent avec le paper trading)
        format_func=lambda x: timeframe_options[x],
        help="Le paper trading utilise des signaux journaliers (1d). Les autres timeframes sont disponibles pour exploration."
    )

    # Info sur le timeframe
    if selected_timeframe == "5min":
        st.sidebar.warning("⚠️ Timeframe très court - Signaux volatils")
    elif selected_timeframe == "1d":
        st.sidebar.success("✅ Timeframe utilisé par le paper trading")
    else:
        st.sidebar.info("ℹ️ Timeframe différent du paper trading (1d)")

    st.sidebar.markdown("---")

    # Filtre de catégorie
    asset_categories = ["Tous", "Crypto", "Tech Stocks", "Commodities",
                        "Indices", "Forex", "Defensive"]
    selected_category = st.sidebar.selectbox(
        "Catégorie d'actifs",
        asset_categories
    )

    # Filtre de confiance
    min_confidence = st.sidebar.slider(
        "Confiance minimale",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05
    )

    # Filtre de signal
    signal_filter = st.sidebar.multiselect(
        "Type de signal",
        ["BUY", "SELL", "HOLD"],
        default=["BUY", "SELL"]
    )

    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
    if auto_refresh:
        st.sidebar.info("Dashboard se rafraîchit automatiquement")

    # Bouton refresh manuel
    refresh_signals = st.sidebar.button("🔄 Rafraichir Signaux (forcer)", type="primary")

    # Detecter si la categorie ou le timeframe a change
    category_changed = (st.session_state.last_category != selected_category or
                        st.session_state.last_timeframe != selected_timeframe)
    if category_changed:
        st.session_state.signals_loaded = False
        st.session_state.last_category = selected_category
        st.session_state.last_timeframe = selected_timeframe

    # Charger les signaux: cache disque en priorite, recalcul si necessaire
    if refresh_signals or not st.session_state.signals_loaded or category_changed:
        # Sauvegarder les signaux precedents pour detecter les changements
        previous_signals = {}
        for sig in st.session_state.all_signals:
            previous_signals[sig.get('symbol')] = sig.get('signal')
        st.session_state.previous_signals = previous_signals

        # force_refresh = True seulement si bouton clique
        force = bool(refresh_signals)

        with st.spinner(f"Chargement des signaux ({selected_timeframe})..."):
            st.session_state.all_signals = get_all_current_signals(
                generator, selected_category, selected_timeframe,
                force_refresh=force
            )
            st.session_state.signals_loaded = True
            if force:
                st.sidebar.success(f"Signaux recalcules ({selected_timeframe})!")

    # Utiliser les signaux en cache
    all_signals = st.session_state.all_signals

    # ========================================================================
    # Onglets
    # ========================================================================

    tab1, tab2, tab3, tab5, tab7, tab8, tab9 = st.tabs([
        "📊 Vue d'ensemble",
        "🎯 Signaux actifs",
        "📈 Analyse détaillée",
        "📚 Bibliothèque Backtests",
        "🧠 Strategy Allocator",
        "🖥️ Multi Paper Trading",
        "📡 Signaux Macro"
    ])

    # ========================================================================
    # TAB 1: Vue d'ensemble
    # ========================================================================

    with tab1:
        st.header("Vue d'ensemble du marché")

        # Utiliser les signaux déjà en cache (pas de rechargement)
        if len(all_signals) == 0:
            st.warning("Aucun signal disponible")
            return

        # Métriques globales
        col1, col2, col3, col4, col5 = st.columns(5)

        total_signals = len(all_signals)
        buy_signals = sum(1 for s in all_signals if s.get('signal') == 'BUY')
        sell_signals = sum(1 for s in all_signals if s.get('signal') == 'SELL')
        avg_confidence = sum(s.get('confidence', 0) for s in all_signals) / len(all_signals)

        # Compter marchés ouverts
        markets_open = sum(1 for s in all_signals if is_market_open(s.get('symbol', ''))[0])
        markets_closed = total_signals - markets_open

        col1.metric("Total Actifs", total_signals)
        col2.metric("🟢 Signaux BUY", buy_signals)
        col3.metric("🔴 Signaux SELL", sell_signals)
        col4.metric("Confiance Moy.", f"{avg_confidence:.1%}")
        col5.metric("📈 Marchés Ouverts", f"{markets_open}/{total_signals}")

        # Distribution des signaux
        st.subheader("Distribution des signaux")

        signal_counts = pd.DataFrame([
            {'Signal': s.get('signal', 'HOLD'), 'Count': 1}
            for s in all_signals
        ]).groupby('Signal').sum().reset_index()

        fig_pie = px.pie(
            signal_counts,
            values='Count',
            names='Signal',
            color='Signal',
            color_discrete_map={'BUY': '#2ca02c', 'SELL': '#d62728', 'HOLD': '#7f7f7f'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # Top signaux par confiance
        st.subheader("Top 10 Signaux par Confiance")

        top_signals = sorted(all_signals, key=lambda s: s.get('confidence', 0),
                             reverse=True)[:10]

        for signal in top_signals:
            col_a, col_b, col_c, col_d, col_e = st.columns([2, 2, 2, 3, 2])

            with col_a:
                st.write(f"**{signal.get('symbol', 'N/A')}**")

            with col_b:
                signal_type = signal.get('signal', 'HOLD')
                if signal_type == 'BUY':
                    st.markdown('<div class="buy-signal">🟢 BUY</div>',
                                unsafe_allow_html=True)
                elif signal_type == 'SELL':
                    st.markdown('<div class="sell-signal">🔴 SELL</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown('<div class="hold-signal">⚪ HOLD</div>',
                                unsafe_allow_html=True)

            with col_c:
                st.write(f"**{signal.get('confidence', 0):.1%}**")

            with col_d:
                st.write(f"${signal.get('current_price', 0):,.2f} "
                         f"({signal.get('change_24h', 0):+.2f}%)")

            with col_e:
                # Statut marché
                is_open, status = is_market_open(signal.get('symbol', ''))
                if is_open:
                    st.markdown(f"🟢 **{status}**")
                else:
                    st.markdown(f"🔴 **{status}**")

            st.divider()

    # ========================================================================
    # TAB 2: Signaux actifs
    # ========================================================================

    with tab2:
        st.header("🎯 Signaux actifs")

        # Info sur le système de détection
        st.info(f"""
        **Timeframe actuel:** {timeframe_options[selected_timeframe]}

        **Légende des statuts:**
        - 🆕 **NOUVEAU** : Signal détecté pour la première fois
        - 🔄 **CHANGEMENT** : Le signal a changé depuis le dernier refresh (ex: BUY → SELL)
        - ♻️ **Maintenu** : La position recommandée est inchangée (pas de nouveau trade à placer)

        💡 **Conseil:** Si un signal est "Maintenu" et que vous avez déjà une position ouverte,
        pas besoin de trader à nouveau. Suivez votre plan SL/TP initial.
        """)

        # Filtrer selon paramètres
        filtered_signals = [
            s for s in all_signals
            if s.get('confidence', 0) >= min_confidence
            and s.get('signal') in signal_filter
        ]

        st.write(f"**{len(filtered_signals)} signaux correspondent aux critères de filtrage**")

        if len(filtered_signals) == 0:
            st.warning("Aucun signal ne correspond aux filtres")
        else:
            # Grouper par catégorie
            signals_by_category = {}
            for signal in filtered_signals:
                symbol = signal.get('symbol', '')
                asset_info = get_asset_info(symbol)
                if asset_info:
                    category = asset_info.get('category', 'autre')
                    if category not in signals_by_category:
                        signals_by_category[category] = []
                    signals_by_category[category].append((signal, asset_info))

            # Mapping catégories → émojis et noms affichables
            category_display = {
                'crypto': ('🪙', 'Cryptomonnaies'),
                'tech_stocks': ('💻', 'Actions Tech'),
                'commodities': ('🛢️', 'Matières Premières'),
                'indices': ('📊', 'Indices'),
                'forex': ('💱', 'Forex'),
                'defensive': ('🛡️', 'Actions Défensives')
            }

            # Afficher par catégorie
            for category, signals_list in signals_by_category.items():
                emoji, category_name = category_display.get(category, ('📦', category.title()))

                st.markdown(f"### {emoji} {category_name} ({len(signals_list)})")

                for signal, asset_info in signals_list:
                    # Statut marché
                    symbol = signal.get('symbol', '')
                    is_open, market_status = is_market_open(symbol)
                    status_emoji = "🟢" if is_open else "🔴"

                    # Détecter changement de signal
                    current_signal = signal.get('signal', 'HOLD')
                    previous_signal = st.session_state.previous_signals.get(symbol, None)

                    if previous_signal is None:
                        signal_status = "🆕 NOUVEAU"
                    elif previous_signal != current_signal:
                        signal_status = "🔄 CHANGEMENT"
                    else:
                        signal_status = "♻️ Maintenu"

                    # Informations détaillées
                    asset_name = asset_info.get('name', symbol)

                    # Informations supplémentaires selon type d'actif
                    extra_info = ""
                    if 'sector' in asset_info:
                        extra_info = f" | {asset_info['sector']}"
                    elif 'region' in asset_info:
                        extra_info = f" | {asset_info['region']}"
                    elif 'pair' in asset_info:
                        extra_info = f" | Paire {asset_info['pair']}"

                    with st.expander(f"{status_emoji} **{symbol}** - {asset_name}{extra_info} - "
                                     f"{signal.get('signal', 'N/A')} - "
                                     f"{signal.get('confidence', 0):.1%} - "
                                     f"{signal_status} - "
                                     f"{market_status}"):

                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Prix actuel",
                                      f"${signal.get('current_price', 0):,.2f}",
                                      f"{signal.get('change_24h', 0):+.2f}%")

                        with col2:
                            st.metric("Confiance",
                                      f"{signal.get('confidence', 0):.1%}")

                        with col3:
                            st.metric("Impact Macro",
                                      f"{signal.get('macro_impact', 0):+.2f}")

                        with col4:
                            if is_open:
                                st.success(f"✅ {market_status}")
                            else:
                                st.warning(f"⏸️ {market_status}")

                        # Consensus
                        st.write(f"**Consensus:** {signal.get('consensus', 'N/A')}")

                        # Indicateurs
                        if 'indicators' in signal and signal['indicators']:
                            st.subheader("Indicateurs Techniques")
                            fig_gauges = create_indicators_gauge(signal['indicators'])
                            st.plotly_chart(fig_gauges, use_container_width=True)

                        # Recommandations détaillées pour signaux BUY et SELL (CFD)
                        if signal.get('signal') in ['BUY', 'SELL']:
                            signal_type = signal.get('signal')
                            direction_emoji = "🟢" if signal_type == 'BUY' else "🔴"
                            direction_text = "LONG" if signal_type == 'BUY' else "SHORT (CFD)"

                            st.subheader(f"{direction_emoji} Plan de Trading Recommandé - {direction_text}")

                            # Avertissement si marché fermé
                            if not is_open:
                                st.warning("⏸️ **Marché actuellement fermé** - Plan à titre indicatif, à exécuter à l'ouverture")

                            reco = get_trade_recommendations(signal)

                            if reco:
                                # Type de trade + Direction
                                if signal_type == 'BUY':
                                    st.info(f"**Type:** {reco['trade_type']} | **Direction:** LONG (achat)")
                                else:
                                    st.info(f"**Type:** {reco['trade_type']} | **Direction:** SHORT (vente à découvert CFD)")

                                # SL/TP avec explications selon direction
                                col_sl, col_tp = st.columns(2)

                                if signal_type == 'BUY':
                                    col_sl.metric("🛡️ Stop-Loss (en dessous)",
                                                  f"${reco['sl_price']:,.2f}",
                                                  f"-{reco['sl_pct']:.1f}%")
                                    col_tp.metric("💰 Take-Profit (au dessus)",
                                                  f"${reco['tp_price']:,.2f}",
                                                  f"+{reco['tp_pct']:.1f}%")
                                else:  # SELL
                                    col_sl.metric("🛡️ Stop-Loss (au dessus)",
                                                  f"${reco['sl_price']:,.2f}",
                                                  f"+{reco['sl_pct']:.1f}%")
                                    col_tp.metric("💰 Take-Profit (en dessous)",
                                                  f"${reco['tp_price']:,.2f}",
                                                  f"-{reco['tp_pct']:.1f}%")

                                # Risk/Reward
                                st.metric("⚖️ Risk/Reward Ratio",
                                          f"1:{reco['risk_reward_ratio']:.2f}")

                                # Trailing Stop
                                if reco['trailing_recommended']:
                                    st.success(f"✅ **Trailing Stop recommandé** : Distance {reco['trailing_distance']:.1f}%")
                                else:
                                    st.info("ℹ️ **Trailing Stop non nécessaire** (actif peu volatile)")

                                # Simulation sur €1000
                                st.subheader("💶 Simulation sur €1,000")

                                col_sim1, col_sim2 = st.columns(2)

                                with col_sim1:
                                    if signal_type == 'BUY':
                                        st.markdown("**📈 Scénario Gain (prix monte à TP)**")
                                    else:
                                        st.markdown("**📉 Scénario Gain (prix baisse à TP)**")
                                    st.write(f"Quantité: {reco['simulation_quantity']:.4f}")
                                    st.success(f"**Gain: +€{reco['simulation_gain_eur']:.2f}** (+{reco['simulation_gain_pct']:.1f}%)")

                                with col_sim2:
                                    if signal_type == 'BUY':
                                        st.markdown("**📉 Scénario Perte (prix baisse à SL)**")
                                    else:
                                        st.markdown("**📈 Scénario Perte (prix monte à SL)**")
                                    st.write(f"Quantité: {reco['simulation_quantity']:.4f}")
                                    st.error(f"**Perte: -€{reco['simulation_loss_eur']:.2f}** (-{reco['simulation_loss_pct']:.1f}%)")

    # ========================================================================
    # TAB 3: Analyse détaillée
    # ========================================================================

    with tab3:
        st.header("📈 Analyse détaillée")

        # Sélection d'un actif
        all_symbols = [s.get('symbol') for s in all_signals]
        selected_symbol = st.selectbox("Choisir un actif", all_symbols)

        if selected_symbol:
            # Trouver signal correspondant
            signal = next((s for s in all_signals
                          if s.get('symbol') == selected_symbol), None)

            if signal:
                # Informations générales
                col1, col2, col3, col4 = st.columns(4)

                col1.metric("Prix", f"${signal.get('current_price', 0):,.2f}")
                col2.metric("Signal", signal.get('signal', 'N/A'))
                col3.metric("Confiance", f"{signal.get('confidence', 0):.1%}")
                col4.metric("Change 24h", f"{signal.get('change_24h', 0):+.2f}%")

                # Récupérer données historiques
                st.subheader("Graphique de prix")

                asset_type = get_asset_type(selected_symbol)
                interval = '5min' if asset_type == 'crypto' else '15min'

                with st.spinner("Chargement des données..."):
                    data = feed.get_intraday_data(selected_symbol, interval, asset_type)

                if len(data) > 0:
                    fig_chart = create_price_chart(selected_symbol, data, signal)
                    st.plotly_chart(fig_chart, use_container_width=True)

                    # Indicateurs détaillés
                    if 'indicators' in signal and signal['indicators']:
                        st.subheader("Indicateurs Techniques Détaillés")

                        ind = signal['indicators']

                        col_ind1, col_ind2, col_ind3 = st.columns(3)

                        with col_ind1:
                            st.metric("RSI", f"{ind.get('rsi', 0):.1f}")
                            st.metric("MA 20", f"${ind.get('ma_20', 0):,.2f}")

                        with col_ind2:
                            st.metric("BB Position", f"{ind.get('bb_position', 0):.0f}%")
                            st.metric("MA 50", f"${ind.get('ma_50', 0):,.2f}")

                        with col_ind3:
                            st.metric("MACD", f"{ind.get('macd', 0):.2f}")
                            st.metric("MACD Signal", f"{ind.get('macd_signal', 0):.2f}")

                else:
                    st.warning("Pas de données historiques disponibles")

    # ========================================================================
    # TAB 5: Bibliothèque Backtests
    # ========================================================================

    with tab5:
        st.header("📚 Bibliothèque des Backtests 2024-2025")

        try:
            from backtest_library import BacktestLibrary

            # Utiliser la bibliothèque en cache (charger une seule fois)
            if st.session_state.backtest_library is None:
                with st.spinner("Chargement de la bibliothèque backtests..."):
                    st.session_state.backtest_library = BacktestLibrary()

            library = st.session_state.backtest_library

            # Bouton pour calculer / recalculer les backtests réels
            col_calc1, col_calc2 = st.columns([3, 1])
            with col_calc1:
                if not library.loaded:
                    st.warning("Aucun backtest calcule. Cliquez sur le bouton pour lancer le calcul avec Yahoo Finance.")
                else:
                    st.success(f"Backtests charges: {len(library.results_df)} combinaisons (source: Yahoo Finance + BacktestEngine)")
            with col_calc2:
                calc_label = "Calculer les Backtests" if not library.loaded else "Recalculer"
                if st.button(f"🔄 {calc_label}", type="primary", key="compute_backtests_btn"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def on_progress(current, total, message):
                        progress_bar.progress(current / total)
                        status_text.text(f"[{current}/{total}] {message}")

                    library.compute_all_backtests(progress_callback=on_progress)
                    st.session_state.backtest_library = library

                    progress_bar.empty()
                    status_text.empty()
                    st.success(f"Backtests calcules: {len(library.results_df)} combinaisons")
                    st.rerun()

            if library.loaded:
                stats = library.get_statistics()

                # Statistiques globales
                col1, col2, col3, col4 = st.columns(4)

                col1.metric("Total Combinaisons", stats['total_combinations'])
                col2.metric("Rendement Moyen", f"{stats['avg_return']:.2f}%")
                col3.metric("Meilleur Rendement", f"{stats['best_return']:.2f}%")
                col4.metric("Taux Positif", f"{stats['positive_rate']:.1f}%")

                # Top 10
                st.subheader("🏆 Top 10 Stratégies")
                top10 = library.get_top_strategies(10)

                if len(top10) > 0:
                    # Créer graphique
                    fig_top10 = go.Figure(data=[
                        go.Bar(
                            x=top10['total_return_pct'],
                            y=[f"{row['asset']} - {row['strategy'][:20]}" for _, row in top10.iterrows()],
                            orientation='h',
                            marker=dict(color=top10['total_return_pct'], colorscale='RdYlGn')
                        )
                    ])
                    fig_top10.update_layout(
                        xaxis_title="Rendement (%)",
                        yaxis_title="",
                        height=400
                    )
                    st.plotly_chart(fig_top10, use_container_width=True)

                # Table recherche
                st.subheader("🔍 Recherche Stratégies")

                col_f1, col_f2, col_f3 = st.columns(3)

                with col_f1:
                    min_return_filter = st.slider("Rendement min (%)", -50, 100, 0)

                with col_f2:
                    min_sharpe_filter = st.slider("Sharpe min", 0.0, 3.0, 0.0, 0.1)

                with col_f3:
                    min_winrate_filter = st.slider("Win Rate min (%)", 0, 100, 0)

                # Rechercher
                filtered_results = library.search_strategies(
                    min_return=min_return_filter,
                    min_sharpe=min_sharpe_filter,
                    min_win_rate=min_winrate_filter
                )

                st.info(f"{len(filtered_results)} stratégies trouvées")

                if len(filtered_results) > 0:
                    # Afficher tableau
                    display_cols = ['asset', 'strategy', 'total_return_pct', 'sharpe_ratio',
                                    'max_drawdown_pct', 'win_rate', 'num_trades']
                    st.dataframe(
                        filtered_results[display_cols].head(50),
                        use_container_width=True
                    )

                # Heatmap
                st.subheader("🗺️ Heatmap Actifs × Stratégies")

                heatmap_data = library.get_heatmap_data()

                if not heatmap_data.empty:
                    # Limiter pour performance
                    heatmap_limited = heatmap_data.head(20)

                    fig_heatmap = go.Figure(data=go.Heatmap(
                        z=heatmap_limited.values,
                        x=heatmap_limited.columns,
                        y=heatmap_limited.index,
                        colorscale='RdYlGn',
                        zmid=0
                    ))
                    fig_heatmap.update_layout(height=600)
                    st.plotly_chart(fig_heatmap, use_container_width=True)

                # ========== EXPLORATEUR DÉTAILLÉ ==========
                st.markdown("---")
                st.subheader("🔬 Explorateur Détaillé de Stratégie")
                st.info("Sélectionnez une combinaison actif + stratégie pour voir le backtest complet avec tous les trades")

                # Sélection actif + stratégie
                col_select1, col_select2 = st.columns(2)

                with col_select1:
                    # Liste des actifs disponibles
                    available_assets = sorted(library.results_df['asset'].unique().tolist())
                    selected_asset = st.selectbox(
                        "Choisir un actif",
                        available_assets,
                        key='backtest_asset_selector'
                    )

                with col_select2:
                    # Liste des stratégies pour cet actif
                    asset_strategies = library.get_results_by_asset(selected_asset)
                    available_strategies = sorted(asset_strategies['strategy'].unique().tolist())
                    selected_strategy = st.selectbox(
                        "Choisir une stratégie",
                        available_strategies,
                        key='backtest_strategy_selector'
                    )

                # Bouton pour lancer l'analyse détaillée
                if st.button("🚀 Analyser en Détail", type="primary", key="run_detailed_backtest_btn"):
                    with st.spinner(f"Exécution du backtest détaillé pour {selected_asset} avec {selected_strategy}..."):
                        # Exécuter backtest détaillé
                        backtest_results = run_detailed_backtest(
                            symbol=selected_asset,
                            strategy_name=selected_strategy,
                            start_date='2024-01-01',
                            end_date='2025-12-31'
                        )

                        # Stocker dans session_state pour persistance
                        st.session_state.detailed_backtest_results = backtest_results
                        st.session_state.detailed_backtest_asset = selected_asset
                        st.session_state.detailed_backtest_strategy = selected_strategy

                # Afficher les résultats s'ils existent (persistant après rerun)
                if st.session_state.detailed_backtest_results is not None:
                    backtest_results = st.session_state.detailed_backtest_results
                    selected_asset = st.session_state.detailed_backtest_asset
                    selected_strategy = st.session_state.detailed_backtest_strategy

                    if backtest_results:
                        # Afficher le symbole Yahoo utilisé
                        yahoo_symbol = convert_to_yahoo_symbol(selected_asset)

                        col_info1, col_info2 = st.columns([3, 1])
                        with col_info1:
                            st.success(f"✅ Backtest chargé: {selected_asset} - {selected_strategy}")
                            if yahoo_symbol != selected_asset:
                                st.info(f"📊 Symbole Yahoo Finance: `{yahoo_symbol}` (converti depuis `{selected_asset}`)")

                        with col_info2:
                            # Bouton pour effacer les résultats
                            if st.button("🗑️ Effacer", key="clear_backtest_btn", use_container_width=True):
                                st.session_state.detailed_backtest_results = None
                                st.session_state.detailed_backtest_asset = None
                                st.session_state.detailed_backtest_strategy = None
                                st.rerun()

                        # Afficher warning sur les données
                        num_candles = len(backtest_results['data'])
                        num_trades = backtest_results['metrics'].get('total_trades', 0)

                        if num_trades == 0:
                            st.warning(f"⚠️ Aucun trade généré sur cette période (2024-2025, {num_candles} bougies). "
                                      f"Cela peut indiquer que les conditions de la stratégie ne sont jamais remplies, "
                                      f"ou que Yahoo Finance n'a pas assez de données pour cet actif.")

                        st.info(f"""
                        **Source des donnees:** Yahoo Finance (periode 2024-2025, {num_candles} bougies daily).
                        La bibliotheque et l'explorateur utilisent le meme moteur (BacktestEngine) et les memes donnees Yahoo Finance.
                        Les resultats sont identiques.
                        """)

                        # 1. DESCRIPTION DE LA STRATÉGIE
                        st.markdown("---")
                        st.subheader("📖 Description de la Stratégie")

                        strategy_desc = get_strategy_description(selected_strategy)

                        col_desc1, col_desc2 = st.columns([2, 1])

                        with col_desc1:
                            st.markdown(f"**{strategy_desc['name']}**")
                            st.write(strategy_desc['description'])

                            st.markdown("**Paramètres:**")
                            for param, value in strategy_desc['parameters'].items():
                                st.write(f"- {param}: {value}")

                        with col_desc2:
                            st.metric("Meilleur pour", strategy_desc['best_for'])
                            st.metric("Niveau de risque", strategy_desc['risk'])

                            st.markdown("**Indicateurs utilisés:**")
                            for indicator in strategy_desc['indicators']:
                                st.write(f"- {indicator}")

                        # 2. MÉTRIQUES DE PERFORMANCE
                        st.markdown("---")
                        st.subheader("📊 Métriques de Performance")

                        metrics = backtest_results['metrics']

                        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)

                        col_m1.metric(
                            "Rendement Total",
                            f"{metrics['total_return_pct']:.2f}%",
                            help="Rendement total sur la période"
                        )
                        col_m2.metric(
                            "Sharpe Ratio",
                            f"{metrics.get('sharpe_ratio', 0):.2f}",
                            help="Rendement ajusté au risque (>1 = bon)"
                        )
                        col_m3.metric(
                            "Max Drawdown",
                            f"{abs(metrics.get('max_drawdown', 0)):.2f}%",
                            help="Perte maximale depuis un pic"
                        )
                        col_m4.metric(
                            "Win Rate",
                            f"{metrics.get('win_rate', 0):.1f}%",
                            help="Pourcentage de trades gagnants"
                        )
                        col_m5.metric(
                            "Nombre de Trades",
                            metrics.get('total_trades', 0),
                            help="Trades exécutés sur la période"
                        )

                        # 3. GRAPHIQUE AVEC TRADES
                        st.markdown("---")
                        st.subheader("📈 Évolution du Prix avec Trades")

                        fig_backtest = create_backtest_chart(
                            data=backtest_results['data'],
                            trades=backtest_results['trades'],
                            symbol=selected_asset
                        )
                        st.plotly_chart(fig_backtest, use_container_width=True)

                        # Légende
                        st.markdown("""
                        **Légende:**
                        - 🟢 Triangle montant = Entrée LONG (achat)
                        - 🔴 Triangle descendant = Entrée SHORT (vente à découvert)
                        - ✖️ Croix verte = Sortie gagnante
                        - ✖️ Croix rouge = Sortie perdante
                        """)

                        # 4. LISTE DES TRADES
                        st.markdown("---")
                        st.subheader("📋 Liste Complète des Trades")

                        trades = backtest_results['trades']

                        if len(trades) > 0:
                            # Convertir en DataFrame pour affichage
                            trades_df = pd.DataFrame(trades)

                            # Formater les colonnes
                            trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date']).dt.strftime('%Y-%m-%d')
                            trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date']).dt.strftime('%Y-%m-%d')
                            trades_df['entry_price'] = trades_df['entry_price'].apply(lambda x: f"${_fmt_price(x)}")
                            trades_df['exit_price'] = trades_df['exit_price'].apply(lambda x: f"${_fmt_price(x)}")
                            trades_df['profit'] = trades_df['profit'].apply(lambda x: f"${x:.2f}")
                            trades_df['profit_pct'] = trades_df['profit_pct'].apply(lambda x: f"{x:.2f}%")

                            # Afficher avec style
                            st.dataframe(
                                trades_df[[
                                    'entry_date', 'exit_date', 'type',
                                    'entry_price', 'exit_price',
                                    'profit', 'profit_pct', 'duration_days'
                                ]],
                                use_container_width=True,
                                height=400
                            )

                            # Statistiques des trades
                            st.markdown("**Statistiques des Trades:**")

                            trades_raw = backtest_results['trades']
                            winning_trades = [t for t in trades_raw if t['profit'] > 0]
                            losing_trades = [t for t in trades_raw if t['profit'] <= 0]

                            col_t1, col_t2, col_t3, col_t4 = st.columns(4)

                            col_t1.metric("Trades Gagnants", len(winning_trades))
                            col_t2.metric("Trades Perdants", len(losing_trades))

                            avg_win = sum(t['profit'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
                            avg_loss = sum(t['profit'] for t in losing_trades) / len(losing_trades) if losing_trades else 0

                            col_t3.metric("Gain Moyen", f"${avg_win:.2f}")
                            col_t4.metric("Perte Moyenne", f"${avg_loss:.2f}")

                            # Profit Factor
                            total_wins = sum(t['profit'] for t in winning_trades)
                            total_losses = abs(sum(t['profit'] for t in losing_trades))
                            profit_factor = total_wins / total_losses if total_losses > 0 else 0

                            st.metric(
                                "Profit Factor",
                                f"{profit_factor:.2f}",
                                help="Ratio gains/pertes (>1 = profitable)"
                            )

                        else:
                            st.warning("Aucun trade exécuté durant cette période")

                        # 5. COURBE D'ÉQUITÉ
                        st.markdown("---")
                        st.subheader("💰 Évolution du Capital")

                        if backtest_results['equity_curve']:
                            equity_df = pd.DataFrame(backtest_results['equity_curve'])

                            fig_equity = go.Figure()

                            fig_equity.add_trace(go.Scatter(
                                x=equity_df['date'],
                                y=equity_df['equity'],
                                mode='lines',
                                name='Capital',
                                fill='tozeroy',
                                line=dict(color='blue', width=2)
                            ))

                            # Ligne de capital initial
                            fig_equity.add_hline(
                                y=10000,
                                line_dash="dash",
                                line_color="gray",
                                annotation_text="Capital Initial (€10,000)"
                            )

                            fig_equity.update_layout(
                                title="Évolution du Capital (Portfolio €10,000)",
                                xaxis_title="Date",
                                yaxis_title="Capital (€)",
                                height=400,
                                template="plotly_white",
                                hovermode='x unified'
                            )

                            st.plotly_chart(fig_equity, use_container_width=True)

                    else:
                        st.error("❌ Impossible d'exécuter le backtest. Vérifiez que l'actif et la stratégie sont valides.")

            else:
                st.warning("Aucune donnee de backtesting disponible.")
                st.info("Cliquez sur 'Calculer les Backtests' ci-dessus pour lancer le calcul via Yahoo Finance.")

        except Exception as e:
            st.error(f"Erreur chargement bibliothèque: {e}")

    # ========================================================================
    # TAB 7: Strategy Allocator
    # ========================================================================

    with tab7:
        st.header("🧠 Strategy Allocator")
        st.info("Allocation intelligente des strategies basee sur les resultats de backtesting. "
                "Filtre, score et alloue le capital aux meilleures paires actif/strategie.")

        try:
            # Charger la bibliotheque (reutiliser le cache session)
            if st.session_state.backtest_library is None:
                with st.spinner("Chargement de la bibliotheque backtests..."):
                    st.session_state.backtest_library = BacktestLibrary()

            library = st.session_state.backtest_library

            if not library.loaded:
                st.warning("Aucun backtest disponible. Allez dans l'onglet 'Bibliotheque Backtests' pour lancer le calcul.")
            else:
                # ---- Parametres dans la sidebar ----
                st.sidebar.markdown("---")
                st.sidebar.header("🧠 Allocator Config")

                alloc_capital = st.sidebar.number_input(
                    "Capital a allouer (EUR)",
                    min_value=1000, max_value=1000000,
                    value=10000, step=1000,
                    key="alloc_capital"
                )
                alloc_max_pos = st.sidebar.slider(
                    "Max positions", 3, 15, 8, key="alloc_max_pos"
                )
                alloc_method = st.sidebar.selectbox(
                    "Methode d'allocation",
                    ["score_weighted", "equal", "risk_parity"],
                    format_func=lambda x: {
                        'score_weighted': 'Score-Weighted (recommande)',
                        'equal': 'Equal Weight',
                        'risk_parity': 'Risk Parity (inverse volatilite)',
                    }[x],
                    key="alloc_method"
                )

                st.sidebar.markdown("**Filtres**")
                alloc_min_trades = st.sidebar.slider(
                    "Min trades", 5, 50, 20, key="alloc_min_trades"
                )
                alloc_min_sharpe = st.sidebar.slider(
                    "Min Sharpe", -1.0, 2.0, 0.0, 0.1, key="alloc_min_sharpe"
                )
                alloc_max_dd = st.sidebar.slider(
                    "Max Drawdown (%)", 10.0, 80.0, 50.0, 5.0, key="alloc_max_dd"
                )
                alloc_walk_forward = st.sidebar.checkbox(
                    "Walk-Forward Validation", value=False, key="alloc_wf",
                    help="Valide la robustesse sur sous-periodes (plus lent)"
                )

                # ---- Bouton de lancement ----
                run_allocator = st.button("🚀 Lancer l'allocation", type="primary", key="run_allocator_btn")

                if run_allocator:
                    with st.spinner("Allocation en cours..." + (" (walk-forward active, peut prendre quelques minutes)" if alloc_walk_forward else "")):
                        allocator = StrategyAllocator(
                            min_trades=alloc_min_trades,
                            min_sharpe=alloc_min_sharpe,
                            max_drawdown=alloc_max_dd,
                            allocation_method=alloc_method,
                            enable_walk_forward=alloc_walk_forward,
                        )
                        plan = allocator.allocate_from_library(
                            library=library,
                            total_capital=alloc_capital,
                            max_positions=alloc_max_pos,
                        )
                        st.session_state.alloc_plan = plan
                        st.session_state.alloc_allocator = allocator

                # ---- Affichage du plan ----
                if 'alloc_plan' in st.session_state and st.session_state.alloc_plan is not None:
                    plan = st.session_state.alloc_plan
                    allocator = st.session_state.alloc_allocator

                    if not plan.assignments:
                        st.warning("Aucune paire actif/strategie ne passe les filtres. Essayez d'assouplir les criteres.")
                    else:
                        # -- Metriques globales --
                        stats = plan.summary_stats
                        col_a1, col_a2, col_a3, col_a4 = st.columns(4)
                        col_a1.metric("Actifs selectionnes", stats.get('n_assets', 0))
                        col_a2.metric("Rendement moyen backtest", f"{stats.get('avg_return', 0):+.1f}%")
                        col_a3.metric("Sharpe moyen", f"{stats.get('avg_sharpe', 0):.2f}")
                        col_a4.metric("Capital alloue",
                                      f"{stats.get('total_allocation_pct', 0):.0f}%",
                                      f"Reserve: {100 - stats.get('total_allocation_pct', 0):.0f}%")

                        # -- Tableau des allocations --
                        st.subheader("Allocations")

                        table_data = []
                        for a in plan.assignments:
                            rp = plan.risk_params.get(a.asset)
                            table_data.append({
                                'Actif': a.asset,
                                'Nom': a.asset_name,
                                'Categorie': a.asset_type,
                                'Strategie': a.strategy_name,
                                'Score': round(a.score, 3),
                                'Sharpe': round(a.sharpe_ratio, 2),
                                'Return (%)': round(a.total_return_pct, 1),
                                'Drawdown (%)': round(a.max_drawdown_pct, 1),
                                'Win Rate (%)': round(a.win_rate, 1),
                                'Trades': a.num_trades,
                                'Allocation (%)': round(a.allocation_pct, 1),
                                'Capital (EUR)': round(a.allocation_pct / 100 * plan.config['total_capital'], 0),
                                'SL (%)': rp.stop_loss_pct if rp else 0,
                                'TP (%)': rp.take_profit_pct if rp else 0,
                            })
                            if a.stability_score is not None:
                                table_data[-1]['Stabilite'] = round(a.stability_score, 2)

                        alloc_df = pd.DataFrame(table_data)
                        st.dataframe(alloc_df, use_container_width=True, height=400)

                        # -- Visualisations --
                        col_chart1, col_chart2 = st.columns(2)

                        with col_chart1:
                            st.subheader("Allocation par categorie")
                            cat_data = alloc_df.groupby('Categorie')['Allocation (%)'].sum().reset_index()
                            # Ajouter reserve cash
                            cash_reserve = 100 - cat_data['Allocation (%)'].sum()
                            if cash_reserve > 0.5:
                                cat_data = pd.concat([
                                    cat_data,
                                    pd.DataFrame([{'Categorie': 'Cash reserve', 'Allocation (%)': cash_reserve}])
                                ], ignore_index=True)

                            fig_alloc_pie = px.pie(
                                cat_data,
                                values='Allocation (%)',
                                names='Categorie',
                                color_discrete_sequence=px.colors.qualitative.Set2
                            )
                            fig_alloc_pie.update_layout(height=350)
                            st.plotly_chart(fig_alloc_pie, use_container_width=True)

                        with col_chart2:
                            st.subheader("Score par actif")
                            fig_scores = go.Figure(data=[
                                go.Bar(
                                    x=[a.score for a in plan.assignments],
                                    y=[f"{a.asset}" for a in plan.assignments],
                                    orientation='h',
                                    marker=dict(
                                        color=[a.score for a in plan.assignments],
                                        colorscale='Viridis'
                                    ),
                                    text=[f"{a.score:.3f}" for a in plan.assignments],
                                    textposition='outside'
                                )
                            ])
                            fig_scores.update_layout(
                                xaxis_title="Score composite",
                                height=350,
                                margin=dict(l=100)
                            )
                            st.plotly_chart(fig_scores, use_container_width=True)

                        # -- Risk/Return scatter --
                        st.subheader("Rendement vs Risque")
                        fig_scatter = go.Figure()
                        for a in plan.assignments:
                            fig_scatter.add_trace(go.Scatter(
                                x=[a.max_drawdown_pct],
                                y=[a.total_return_pct],
                                mode='markers+text',
                                text=[a.asset],
                                textposition='top center',
                                marker=dict(
                                    size=a.allocation_pct * 2,
                                    color=a.score,
                                    colorscale='Viridis',
                                    showscale=False,
                                ),
                                name=a.asset,
                                showlegend=False,
                                hovertemplate=(
                                    f"<b>{a.asset}</b><br>"
                                    f"Return: {a.total_return_pct:+.1f}%<br>"
                                    f"Drawdown: {a.max_drawdown_pct:.1f}%<br>"
                                    f"Sharpe: {a.sharpe_ratio:.2f}<br>"
                                    f"Allocation: {a.allocation_pct:.1f}%<br>"
                                    f"Strategie: {a.strategy_name}"
                                    "<extra></extra>"
                                ),
                            ))
                        fig_scatter.update_layout(
                            xaxis_title="Max Drawdown (%)",
                            yaxis_title="Return (%)",
                            height=450,
                            template="plotly_white",
                        )
                        fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray")
                        st.plotly_chart(fig_scatter, use_container_width=True)
                        st.caption("Taille des bulles = poids de l'allocation. Couleur = score composite.")

                        # -- Walk-forward results --
                        stab_scores = [a for a in plan.assignments if a.stability_score is not None]
                        if stab_scores:
                            st.subheader("Walk-Forward Validation")
                            col_wf1, col_wf2, col_wf3 = st.columns(3)
                            all_stab = [a.stability_score for a in stab_scores]
                            col_wf1.metric("Stabilite moyenne", f"{np.mean(all_stab):.2f}")
                            col_wf2.metric("Min", f"{min(all_stab):.2f}")
                            col_wf3.metric("Max", f"{max(all_stab):.2f}")

                            fig_stab = go.Figure(data=[
                                go.Bar(
                                    x=[a.asset for a in stab_scores],
                                    y=[a.stability_score for a in stab_scores],
                                    marker=dict(
                                        color=[a.stability_score for a in stab_scores],
                                        colorscale='RdYlGn',
                                        cmin=0, cmax=1,
                                    ),
                                    text=[f"{a.stability_score:.2f}" for a in stab_scores],
                                    textposition='outside'
                                )
                            ])
                            fig_stab.update_layout(
                                yaxis_title="Stability Score",
                                height=300,
                                yaxis=dict(range=[0, 1.1])
                            )
                            st.plotly_chart(fig_stab, use_container_width=True)

                        # -- Export --
                        st.subheader("Export")
                        col_exp1, col_exp2, col_exp3 = st.columns(3)

                        with col_exp1:
                            plan_json = json.dumps({
                                'generated_at': plan.generated_at.isoformat(),
                                'config': plan.config,
                                'summary_stats': plan.summary_stats,
                                'assignments': [
                                    {
                                        'asset': a.asset, 'asset_name': a.asset_name,
                                        'asset_type': a.asset_type,
                                        'strategy': a.strategy_name,
                                        'score': a.score, 'sharpe': a.sharpe_ratio,
                                        'return_pct': a.total_return_pct,
                                        'drawdown_pct': a.max_drawdown_pct,
                                        'win_rate': a.win_rate,
                                        'allocation_pct': a.allocation_pct,
                                        'stability': a.stability_score,
                                    } for a in plan.assignments
                                ],
                                'risk_params': {
                                    k: {'sl': v.stop_loss_pct, 'tp': v.take_profit_pct,
                                         'position_size': v.position_size_pct}
                                    for k, v in plan.risk_params.items()
                                },
                            }, indent=2, default=str)

                            st.download_button(
                                "Telecharger le plan (JSON)",
                                data=plan_json,
                                file_name=f"trading_plan_{datetime.now().strftime('%Y%m%d')}.json",
                                mime="application/json",
                                key="download_plan_json"
                            )

                        with col_exp2:
                            csv_data = alloc_df.to_csv(index=False)
                            st.download_button(
                                "Telecharger allocations (CSV)",
                                data=csv_data,
                                file_name=f"allocations_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv",
                                key="download_plan_csv"
                            )

                        with col_exp3:
                            sg_format = allocator.export_for_signal_generator(plan)
                            sg_text = "# Format pour signal_generator.recommended_strategies\n"
                            sg_text += "# Genere le " + datetime.now().strftime('%Y-%m-%d %H:%M') + "\n\n"
                            for symbol, strategies in sg_format.items():
                                for name, inst in strategies:
                                    rp = plan.risk_params.get(symbol)
                                    sl_tp = f"SL={rp.stop_loss_pct}%/TP={rp.take_profit_pct}%" if rp else ""
                                    sg_text += f"{symbol:<14} -> {name:<22} {sl_tp}\n"

                            st.download_button(
                                "Telecharger mapping strategies (TXT)",
                                data=sg_text,
                                file_name=f"strategy_mapping_{datetime.now().strftime('%Y%m%d')}.txt",
                                mime="text/plain",
                                key="download_plan_txt"
                            )

        except Exception as e:
            st.error(f"Erreur Strategy Allocator: {e}")
            import traceback
            st.code(traceback.format_exc())

    # ========================================================================
    # TAB 8: Multi Paper Trading Comparison
    # ========================================================================

    with tab8:
        st.header("Multi Paper Trading - Tableau de Bord")

        state_dir = 'paper_trading_state'
        consolidated_file = os.path.join(state_dir, 'consolidated_state.json')

        if not os.path.exists(consolidated_file):
            single_state = os.path.join(state_dir, 'auto_state.json')
            if os.path.exists(single_state):
                st.warning("Mode multi-portfolio non detecte. Affichage du portfolio standalone.")
                try:
                    with open(single_state, 'r') as f:
                        state = json.load(f)
                    cash = state.get('cash', 0)
                    capital = state.get('total_capital', 10000)
                    cycle = state.get('cycle_count', 0)
                    positions = state.get('positions', {})
                    pos_val = sum(p.get('entry_price', 0) * p.get('quantity', 0) for p in positions.values())
                    total_val = cash + pos_val
                    pnl = total_val - capital
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Valeur", f"{total_val:,.0f} EUR")
                    c2.metric("P&L", f"{pnl:+,.0f} EUR")
                    c3.metric("Positions", f"{len(positions)}")
                    c4.metric("Cycles", f"{cycle}")
                except Exception as e:
                    st.error(f"Erreur: {e}")
            else:
                st.warning("Aucun etat detecte.\n\n"
                           "Lancez : `python multi_paper_trading.py --single --capital 10000`\n\n"
                           "Ou en standalone : `python auto_paper_trading.py --single --capital 10000`")
        else:
            try:
                with open(consolidated_file, 'r') as f:
                    consolidated = json.load(f)

                last_update = consolidated.get('last_update', 'N/A')
                total_capital_multi = consolidated.get('total_capital', 10000)
                capital_per = consolidated.get('capital_per_portfolio', 1000)
                n_portfolios = consolidated.get('n_portfolios', 0)
                portfolios = consolidated.get('portfolios', {})

                # --- Status banner ---
                if last_update != 'N/A':
                    try:
                        update_dt = datetime.fromisoformat(last_update)
                        age_hours = (datetime.now() - update_dt).total_seconds() / 3600
                        if age_hours < 2:
                            status_cls = "mpt-status-active"
                            status_txt = "Actif"
                        else:
                            status_cls = "mpt-status-stale"
                            status_txt = f"Derniere MAJ il y a {int(age_hours)}h"
                        update_display = update_dt.strftime('%d/%m/%Y %H:%M:%S')
                    except Exception:
                        status_cls = "mpt-status-stale"
                        status_txt = "?"
                        update_display = last_update[:19]
                else:
                    status_cls = "mpt-status-stale"
                    status_txt = "N/A"
                    update_display = "N/A"

                st.markdown(
                    f'<span class="mpt-status-badge {status_cls}">{status_txt}</span> '
                    f'&nbsp; Derniere mise a jour : **{update_display}** &mdash; '
                    f'{n_portfolios} portefeuilles, {capital_per:,.0f} EUR chacun',
                    unsafe_allow_html=True)
                st.markdown("")

                # ============================================================
                # SECTION 1 : KPIs globaux
                # ============================================================
                total_value_all = sum(p.get('value', capital_per) for p in portfolios.values())
                total_pnl_all = sum(p.get('pnl', 0) for p in portfolios.values())
                total_pnl_pct = (total_pnl_all / total_capital_multi * 100) if total_capital_multi > 0 else 0
                total_realized = sum(p.get('realized_pnl', 0) for p in portfolios.values())
                total_unrealized = total_pnl_all - total_realized
                total_positions = sum(p.get('positions', 0) for p in portfolios.values())
                total_trades = sum(p.get('trades_closed', 0) for p in portfolios.values())

                # Best and worst
                if portfolios:
                    sorted_by_pnl = sorted(portfolios.items(), key=lambda x: x[1].get('pnl_pct', 0), reverse=True)
                    best_name, best_data = sorted_by_pnl[0]
                    worst_name, worst_data = sorted_by_pnl[-1]
                    wr_active = [p.get('win_rate', 0) for p in portfolios.values() if p.get('trades_closed', 0) > 0]
                    avg_win_rate = np.mean(wr_active) if wr_active else 0
                    portfolios_with_trades = sum(1 for p in portfolios.values() if p.get('trades_closed', 0) > 0)
                else:
                    best_name = worst_name = "N/A"
                    best_data = worst_data = {}
                    avg_win_rate = 0
                    portfolios_with_trades = 0

                pnl_cls = "positive" if total_pnl_all >= 0 else "negative"
                pnl_sign = "+" if total_pnl_all >= 0 else ""

                # Row 1: Capital, Value, P&L Total
                best_pnl = best_data.get('pnl_pct', 0)
                worst_pnl = worst_data.get('pnl_pct', 0)
                st.markdown(f"""<div class="mpt-kpi-row">
                    <div class="mpt-kpi-card neutral">
                        <div class="kpi-label">Capital Investi</div>
                        <div class="kpi-value">{total_capital_multi:,.0f} EUR</div>
                        <div class="kpi-sub">{n_portfolios} portefeuilles x {capital_per:,.0f} EUR</div>
                    </div>
                    <div class="mpt-kpi-card {pnl_cls}">
                        <div class="kpi-label">Valeur Actuelle</div>
                        <div class="kpi-value">{total_value_all:,.0f} EUR</div>
                        <div class="kpi-sub">{pnl_sign}{total_pnl_all:,.0f} EUR ({pnl_sign}{total_pnl_pct:.2f}%)</div>
                    </div>
                    <div class="mpt-kpi-card {'positive' if best_pnl >= 0 else 'negative'}">
                        <div class="kpi-label">Meilleur</div>
                        <div class="kpi-value">{best_name}</div>
                        <div class="kpi-sub">{"+" if best_pnl >= 0 else ""}{best_pnl:.2f}%</div>
                    </div>
                    <div class="mpt-kpi-card {'negative' if worst_pnl < 0 else 'neutral'}">
                        <div class="kpi-label">Plus Faible</div>
                        <div class="kpi-value">{worst_name}</div>
                        <div class="kpi-sub">{"+" if worst_pnl >= 0 else ""}{worst_pnl:.2f}%</div>
                    </div>
                </div>""", unsafe_allow_html=True)

                # Row 2: Realized, Unrealized (with explicit formula), Positions, Win Rate
                st.markdown(f"""<div class="mpt-kpi-row">
                    <div class="mpt-kpi-card {'positive' if total_realized >= 0 else 'negative'}">
                        <div class="kpi-label">P&L Realise (trades fermes)</div>
                        <div class="kpi-value">{"+" if total_realized >= 0 else ""}{total_realized:,.0f} EUR</div>
                        <div class="kpi-sub">{total_trades} trades clotures</div>
                    </div>
                    <div class="mpt-kpi-card {'positive' if total_unrealized >= 0 else 'negative'}">
                        <div class="kpi-label">P&L Latent (positions ouvertes)</div>
                        <div class="kpi-value">{"+" if total_unrealized >= 0 else ""}{total_unrealized:,.0f} EUR</div>
                        <div class="kpi-sub">{total_positions} positions en cours</div>
                    </div>
                    <div class="mpt-kpi-card neutral">
                        <div class="kpi-label">Win Rate Moyen</div>
                        <div class="kpi-value">{avg_win_rate:.1f}%</div>
                        <div class="kpi-sub">{portfolios_with_trades}/{n_portfolios} ont trade</div>
                    </div>
                    <div class="mpt-kpi-card neutral">
                        <div class="kpi-label">Cycles Executes</div>
                        <div class="kpi-value">{max(p.get('cycle_count', 0) for p in portfolios.values()) if portfolios else 0}</div>
                        <div class="kpi-sub">signaux generes</div>
                    </div>
                </div>""", unsafe_allow_html=True)

                st.markdown("")

                # ============================================================
                # SECTION 2 : Leaderboard
                # ============================================================
                st.subheader("Classement des Portefeuilles")
                st.markdown('<p class="mpt-section-help">Classe par performance (P&L %). '
                            'Chaque portefeuille utilise une strategie d\'allocation differente '
                            'pour tester quel profil de risque performe le mieux.</p>',
                            unsafe_allow_html=True)

                if portfolios:
                    lb_rows = []
                    medals = {1: '1', 2: '2', 3: '3'}
                    for rank, (name, p) in enumerate(sorted_by_pnl, 1):
                        rank_display = medals.get(rank, str(rank))
                        pnl_value = p.get('pnl', 0)
                        pnl_pct_value = p.get('pnl_pct', 0)
                        realized = p.get('realized_pnl', 0)

                        # Parse config summary for cleaner display
                        config = p.get('config_summary', '')
                        alloc_method = config.split('|')[0].strip() if '|' in config else config

                        lb_rows.append({
                            '#': rank_display,
                            'Portefeuille': name,
                            'Methode': alloc_method,
                            'Description': p.get('description', ''),
                            'Valeur': round(p.get('value', 0), 0),
                            'P&L': round(pnl_value, 0),
                            'Rendement': round(pnl_pct_value, 2),
                            'Realise': round(realized, 0),
                            'Positions': f"{p.get('positions', 0)}/{p.get('max_positions', '?')}",
                            'Trades': p.get('trades_closed', 0),
                            'Win Rate': round(p.get('win_rate', 0), 1),
                        })

                    lb_df = pd.DataFrame(lb_rows)
                    st.dataframe(
                        lb_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            '#': st.column_config.TextColumn('#', width='small'),
                            'Portefeuille': st.column_config.TextColumn('Portefeuille', width='medium'),
                            'Methode': st.column_config.TextColumn('Methode', width='small',
                                help="Methode d'allocation : risk_parity, score_weighted, ou equal"),
                            'Description': st.column_config.TextColumn('Description', width='medium'),
                            'Valeur': st.column_config.NumberColumn('Valeur (EUR)', format="%.0f"),
                            'P&L': st.column_config.NumberColumn('P&L (EUR)', format="%+.0f"),
                            'Rendement': st.column_config.NumberColumn('Rendement (%)', format="%+.2f%%",
                                help="Performance totale depuis le lancement"),
                            'Realise': st.column_config.NumberColumn('Realise (EUR)', format="%+.0f",
                                help="Gains/pertes encaisses (trades fermes)"),
                            'Positions': st.column_config.TextColumn('Positions', width='small',
                                help="Nombre de positions ouvertes / maximum autorise"),
                            'Trades': st.column_config.NumberColumn('Trades', format="%d",
                                help="Nombre total de trades clotures"),
                            'Win Rate': st.column_config.ProgressColumn('Win Rate (%)',
                                format="%.1f%%", min_value=0, max_value=100,
                                help="Pourcentage de trades gagnants"),
                        },
                    )

                st.markdown("")

                # ============================================================
                # SECTION 3 : Graphiques de performance
                # ============================================================
                st.subheader("Performance Comparee")
                st.markdown('<p class="mpt-section-help">'
                            'Vue visuelle de la performance de chaque portefeuille. '
                            'Vert = gain, Rouge = perte.</p>',
                            unsafe_allow_html=True)

                if portfolios:
                    sorted_names = [n for n, _ in sorted_by_pnl]
                    chart_col1, chart_col2 = st.columns(2)

                    with chart_col1:
                        pnl_values = [portfolios[n].get('pnl', 0) for n in sorted_names]
                        colors = ['#2ca02c' if v >= 0 else '#d62728' for v in pnl_values]
                        fig_pnl = go.Figure(data=[
                            go.Bar(
                                y=sorted_names, x=pnl_values,
                                orientation='h',
                                marker_color=colors,
                                text=[f"{v:+,.0f}" for v in pnl_values],
                                textposition='outside',
                                textfont=dict(size=11),
                            )
                        ])
                        fig_pnl.update_layout(
                            title=dict(text="P&L Absolu (EUR)", font=dict(size=14)),
                            xaxis_title="EUR",
                            height=max(300, len(sorted_names) * 40),
                            template="plotly_white",
                            margin=dict(l=10, r=80, t=40, b=30),
                            yaxis=dict(autorange="reversed"),
                        )
                        fig_pnl.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
                        st.plotly_chart(fig_pnl, use_container_width=True)

                    with chart_col2:
                        pnl_pct_values = [portfolios[n].get('pnl_pct', 0) for n in sorted_names]
                        colors_pct = ['#2ca02c' if v >= 0 else '#d62728' for v in pnl_pct_values]
                        fig_pct = go.Figure(data=[
                            go.Bar(
                                y=sorted_names, x=pnl_pct_values,
                                orientation='h',
                                marker_color=colors_pct,
                                text=[f"{v:+.2f}%" for v in pnl_pct_values],
                                textposition='outside',
                                textfont=dict(size=11),
                            )
                        ])
                        fig_pct.update_layout(
                            title=dict(text="Rendement (%)", font=dict(size=14)),
                            xaxis_title="%",
                            height=max(300, len(sorted_names) * 40),
                            template="plotly_white",
                            margin=dict(l=10, r=80, t=40, b=30),
                            yaxis=dict(autorange="reversed"),
                        )
                        fig_pct.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
                        st.plotly_chart(fig_pct, use_container_width=True)

                st.markdown("")

                # ============================================================
                # SECTION 4 : Derniers trades clos + analyse
                # ============================================================
                st.subheader("Derniers Trades Clos")
                st.markdown('<p class="mpt-section-help">'
                            'Historique des trades fermes a travers tous les portefeuilles, '
                            'regroupes par operation unique (meme actif, meme direction, meme moment). '
                            'Chaque trade est accompagne d\'une analyse.</p>',
                            unsafe_allow_html=True)

                # Collect all trades from all portfolio directories
                all_closed_trades = []
                for dirname in sorted(os.listdir(state_dir)):
                    if not dirname.startswith('portfolio_'):
                        continue
                    trades_file = os.path.join(state_dir, dirname, 'auto_trades.csv')
                    if not os.path.exists(trades_file):
                        continue
                    try:
                        df_trades = pd.read_csv(trades_file)
                        if len(df_trades) == 0:
                            continue
                        # Extract portfolio name from dirname
                        parts = dirname.split('_', 2)
                        pf_name = parts[2].replace('_', ' ').title() if len(parts) > 2 else dirname
                        # Match to actual portfolio key
                        for pk in portfolios.keys():
                            if pk.lower().replace('_', '') == pf_name.lower().replace(' ', '').replace('_', ''):
                                pf_name = pk
                                break
                        df_trades['portfolio'] = pf_name
                        all_closed_trades.append(df_trades)
                    except Exception:
                        continue

                if all_closed_trades:
                    all_trades_df = pd.concat(all_closed_trades, ignore_index=True)

                    # Parse exit_time as datetime for sorting
                    all_trades_df['exit_dt'] = pd.to_datetime(all_trades_df['exit_time'], errors='coerce')
                    all_trades_df['entry_dt'] = pd.to_datetime(all_trades_df['entry_time'], errors='coerce')

                    # Group trades by unique operation: same symbol + side + strategy + close exit_time (within 5 min)
                    all_trades_df = all_trades_df.sort_values('exit_dt', ascending=False)
                    grouped_trades = []
                    used_indices = set()

                    for idx, trade in all_trades_df.iterrows():
                        if idx in used_indices:
                            continue
                        # Find all trades with same symbol, side, strategy and exit_time within 5 min
                        mask = (
                            (all_trades_df['symbol'] == trade['symbol']) &
                            (all_trades_df['side'] == trade['side']) &
                            (all_trades_df['strategy'] == trade['strategy']) &
                            (~all_trades_df.index.isin(used_indices))
                        )
                        if pd.notna(trade['exit_dt']):
                            mask = mask & (
                                (all_trades_df['exit_dt'] - trade['exit_dt']).abs() < pd.Timedelta(minutes=5)
                            )
                        group = all_trades_df[mask]
                        used_indices.update(group.index)
                        grouped_trades.append(group)

                    # Global trade stats
                    total_trade_pnl = all_trades_df['pnl'].sum()
                    total_wins = len(all_trades_df[all_trades_df['pnl'] > 0])
                    total_losses = len(all_trades_df[all_trades_df['pnl'] <= 0])
                    unique_ops = len(grouped_trades)

                    ts1, ts2, ts3, ts4 = st.columns(4)
                    ts1.metric("Trades totaux", f"{len(all_trades_df)}",
                               f"{unique_ops} operations uniques")
                    ts2.metric("P&L cumule", f"{'+'if total_trade_pnl>=0 else ''}{total_trade_pnl:.0f} EUR")
                    ts3.metric("Gagnants / Perdants", f"{total_wins}W / {total_losses}L")
                    ts4.metric("Taux de reussite",
                               f"{total_wins / len(all_trades_df) * 100:.0f}%" if len(all_trades_df) > 0 else "N/A")

                    st.markdown("")

                    # Display each unique operation
                    for group in grouped_trades:
                        ref = group.iloc[0]  # reference trade
                        symbol = ref['symbol']
                        side = ref['side']
                        strategy = ref['strategy']
                        exit_reason = ref.get('exit_reason', '?')
                        entry_price = ref['entry_price']
                        exit_price = ref['exit_price']
                        pnl_pct = ref.get('pnl_pct', 0)
                        n_portfolios_in = len(group)
                        portfolio_list = ', '.join(sorted(group['portfolio'].unique()))
                        total_group_pnl = group['pnl'].sum()

                        # Duration
                        entry_dt = ref.get('entry_dt')
                        exit_dt = ref.get('exit_dt')
                        if pd.notna(entry_dt) and pd.notna(exit_dt):
                            duration = exit_dt - entry_dt
                            days = duration.days
                            hours, remainder = divmod(duration.seconds, 3600)
                            minutes = remainder // 60
                            if days > 0:
                                duration_str = f"{days}j {hours}h {minutes}m"
                            elif hours > 0:
                                duration_str = f"{hours}h {minutes}m"
                            else:
                                duration_str = f"{minutes}m"
                            entry_str = entry_dt.strftime('%d/%m %H:%M')
                            exit_str = exit_dt.strftime('%d/%m %H:%M')
                        else:
                            duration_str = "N/A"
                            entry_str = str(ref.get('entry_time', ''))[:16]
                            exit_str = str(ref.get('exit_time', ''))[:16]

                        is_win = total_group_pnl > 0
                        icon = "W" if is_win else "L"
                        color = "#2ca02c" if is_win else "#d62728"

                        header = (
                            f"{'WIN' if is_win else 'LOSS'} | {symbol} {side} | "
                            f"{'+'if pnl_pct>=0 else ''}{pnl_pct:.2f}% | "
                            f"{strategy} | "
                            f"{'+'if total_group_pnl>=0 else ''}{total_group_pnl:.0f} EUR sur {n_portfolios_in} portefeuille{'s' if n_portfolios_in > 1 else ''} | "
                            f"{exit_str}"
                        )

                        with st.expander(header):
                            # Key metrics row
                            m1, m2, m3, m4, m5 = st.columns(5)
                            m1.metric("Symbole", f"{symbol} ({side})")
                            m2.metric("Strategie", strategy)
                            m3.metric("Duree", duration_str)
                            m4.metric("Resultat", f"{'+'if pnl_pct>=0 else ''}{pnl_pct:.2f}%")
                            m5.metric("Sortie", exit_reason)

                            m6, m7, m8, m9 = st.columns(4)
                            m6.metric("Prix entree", _fmt_price(entry_price))
                            m7.metric("Prix sortie", _fmt_price(exit_price))
                            m8.metric("Entree", entry_str)
                            m9.metric("Sortie", exit_str)

                            # Impact across portfolios
                            if n_portfolios_in > 1:
                                st.markdown(f"**Impact sur {n_portfolios_in} portefeuilles** : {portfolio_list}")
                                impact_rows = []
                                for _, t in group.iterrows():
                                    impact_rows.append({
                                        'Portefeuille': t['portfolio'],
                                        'Quantite': round(t['quantity'], 6),
                                        'P&L': round(t['pnl'], 2),
                                    })
                                st.dataframe(pd.DataFrame(impact_rows),
                                             use_container_width=True, hide_index=True,
                                             column_config={
                                                 'P&L': st.column_config.NumberColumn('P&L (EUR)', format="%+.2f"),
                                             })

                            # Analysis
                            st.markdown("---")
                            st.markdown("**Analyse**")

                            analysis_parts = []

                            # Duration analysis
                            if pd.notna(entry_dt) and pd.notna(exit_dt):
                                hold_hours = (exit_dt - entry_dt).total_seconds() / 3600
                                if hold_hours < 1:
                                    analysis_parts.append(
                                        f"Trade tres court ({duration_str}). "
                                        f"Le {'TP' if exit_reason == 'TAKE_PROFIT' else 'SL'} a ete touche rapidement "
                                        f"apres l'entree, ce qui indique un mouvement de prix fort et immediat.")
                                elif hold_hours < 24:
                                    analysis_parts.append(
                                        f"Trade intraday ({duration_str}). "
                                        f"La position a ete {'securisee par le TP' if exit_reason == 'TAKE_PROFIT' else 'coupee par le SL'} "
                                        f"dans la meme journee.")
                                else:
                                    analysis_parts.append(
                                        f"Position tenue {duration_str}. "
                                        f"Un maintien de {'plus de 2 jours' if hold_hours > 48 else 'plus de 24h'} "
                                        f"{'a permis de capturer le mouvement complet' if is_win else 'n a pas suffi a inverser la tendance'}.")

                            # Price movement
                            if entry_price > 0:
                                price_move = (exit_price - entry_price) / entry_price * 100
                                if side == 'SHORT':
                                    price_move = -price_move
                                abs_move = abs(exit_price - entry_price) / entry_price * 100

                                if exit_reason == 'STOP_LOSS':
                                    analysis_parts.append(
                                        f"Sortie par Stop Loss a {_fmt_price(exit_price)} "
                                        f"(mouvement adverse de {abs_move:.2f}%). "
                                        f"Le SL a correctement limite la perte a {abs(pnl_pct):.2f}% du capital investi.")
                                elif exit_reason == 'TAKE_PROFIT':
                                    analysis_parts.append(
                                        f"Sortie par Take Profit a {_fmt_price(exit_price)} "
                                        f"(mouvement favorable de {abs_move:.2f}%). "
                                        f"Le TP a securise un gain de {pnl_pct:.2f}% du capital investi.")
                                else:
                                    analysis_parts.append(
                                        f"Sortie par {exit_reason} a {_fmt_price(exit_price)} "
                                        f"(variation de {price_move:+.2f}%).")

                            # Strategy context
                            strat_descriptions = {
                                'RSI_14_30_70': "RSI(14) avec seuils 30/70 - detecte les zones de survente/surachat",
                                'RSI_14_35_80': "RSI(14) avec seuils 35/80 - variante asymetrique favorisant les entrees longues",
                                'ADX_Trend_14_25': "ADX(14) seuil 25 - entre en position quand la tendance est forte",
                                'Ichimoku_9_26_52': "Ichimoku classique - suit la tendance avec nuage, Tenkan et Kijun",
                                'Combined': "Strategie combinee (multi-indicateurs) - consensus de plusieurs signaux",
                                'MACD_12_26_9': "MACD classique - croisement de moyennes mobiles exponentielles",
                            }
                            strat_desc = strat_descriptions.get(strategy, f"Strategie {strategy}")
                            analysis_parts.append(f"Strategie utilisee : **{strategy}** ({strat_desc}).")

                            # Cross-portfolio insight
                            if n_portfolios_in >= 5:
                                analysis_parts.append(
                                    f"Ce trade a ete pris par **{n_portfolios_in} portefeuilles** sur {n_portfolios}, "
                                    f"ce qui montre un fort consensus entre les differents profils d'allocation. "
                                    f"Impact total : {'+'if total_group_pnl>=0 else ''}{total_group_pnl:.2f} EUR.")
                            elif n_portfolios_in > 1:
                                analysis_parts.append(
                                    f"Trade present dans {n_portfolios_in} portefeuilles ({portfolio_list}). "
                                    f"Impact total : {'+'if total_group_pnl>=0 else ''}{total_group_pnl:.2f} EUR.")

                            # Final verdict
                            if is_win:
                                analysis_parts.append(
                                    f"**Verdict** : Trade gagnant. La strategie {strategy} a correctement identifie "
                                    f"l'opportunite sur {symbol} et le TP a permis de securiser les gains.")
                            else:
                                analysis_parts.append(
                                    f"**Verdict** : Trade perdant. Le mouvement de {symbol} est alle a l'encontre "
                                    f"du signal {side}. Le SL a limite la casse a {abs(pnl_pct):.2f}% par position.")

                            for part in analysis_parts:
                                st.markdown(f"- {part}")

                else:
                    st.info("Aucun trade clos pour le moment. Les premiers trades apparaitront ici "
                            "des qu'un Stop Loss ou Take Profit sera touche.")

                st.markdown("")

                # ============================================================
                # SECTION 5 : Analyse comparative des allocations
                # ============================================================
                st.subheader("Analyse des Allocations")
                st.markdown('<p class="mpt-section-help">'
                            'Quels actifs sont selectionnes par chaque portefeuille ? '
                            'Cette matrice montre le chevauchement entre portefeuilles et '
                            'les categories d\'actifs representees.</p>',
                            unsafe_allow_html=True)

                # Load all plans
                all_plans = {}
                all_assets_set = set()
                category_counts = {}
                for dirname in sorted(os.listdir(state_dir)):
                    if not dirname.startswith('portfolio_'):
                        continue
                    plan_file = os.path.join(state_dir, dirname, 'auto_plan.json')
                    if not os.path.exists(plan_file):
                        continue
                    try:
                        with open(plan_file, 'r') as f:
                            plan = json.load(f)
                        # Extract portfolio name from dirname
                        parts = dirname.split('_', 2)
                        pname = parts[2].replace('_', ' ').title() if len(parts) > 2 else dirname
                        # Match to portfolio key
                        matched_name = None
                        for pk in portfolios.keys():
                            if pk.lower().replace('_', '') == pname.lower().replace(' ', '').replace('_', ''):
                                matched_name = pk
                                break
                        if not matched_name:
                            matched_name = pname
                        all_plans[matched_name] = plan
                        for a in plan.get('assignments', []):
                            asset = a.get('asset', '')
                            atype = a.get('asset_type', 'other')
                            all_assets_set.add(asset)
                            category_counts[atype] = category_counts.get(atype, 0) + 1
                    except Exception:
                        continue

                if all_plans:
                    alloc_col1, alloc_col2 = st.columns([3, 2])

                    with alloc_col1:
                        # Asset overlap heatmap
                        all_assets_sorted = sorted(all_assets_set)
                        plan_names = [n for n, _ in sorted_by_pnl if n in all_plans]
                        heatmap_data = []
                        for pname in plan_names:
                            row = []
                            assignments = {a['asset']: a for a in all_plans[pname].get('assignments', [])}
                            for asset in all_assets_sorted:
                                if asset in assignments:
                                    row.append(round(assignments[asset].get('allocation_pct', 0), 1))
                                else:
                                    row.append(0)
                            heatmap_data.append(row)

                        if heatmap_data:
                            fig_heatmap = go.Figure(data=go.Heatmap(
                                z=heatmap_data,
                                x=all_assets_sorted,
                                y=plan_names,
                                colorscale='Blues',
                                text=[[f"{v:.1f}%" if v > 0 else "" for v in row] for row in heatmap_data],
                                texttemplate="%{text}",
                                textfont=dict(size=9),
                                hovertemplate="Portefeuille: %{y}<br>Actif: %{x}<br>Allocation: %{z:.1f}%<extra></extra>",
                                colorbar=dict(title="Alloc %", len=0.6),
                            ))
                            fig_heatmap.update_layout(
                                title=dict(text="Matrice d'Allocation (% par actif)", font=dict(size=14)),
                                height=max(350, len(plan_names) * 35 + 100),
                                template="plotly_white",
                                margin=dict(l=10, r=10, t=40, b=80),
                                xaxis=dict(tickangle=45, tickfont=dict(size=9)),
                                yaxis=dict(tickfont=dict(size=10)),
                            )
                            st.plotly_chart(fig_heatmap, use_container_width=True)

                    with alloc_col2:
                        # Category distribution pie
                        if category_counts:
                            cat_labels = list(category_counts.keys())
                            cat_values = list(category_counts.values())
                            cat_colors = {
                                'crypto': '#f7931a', 'tech_stocks': '#0078d4',
                                'semiconductors': '#7b2d8e', 'finance': '#2ca02c',
                                'commodities': '#d4a017', 'indices': '#6c757d',
                                'healthcare': '#e74c3c', 'energy': '#ff6b35',
                            }
                            colors_pie = [cat_colors.get(c, '#adb5bd') for c in cat_labels]
                            cat_labels_display = [c.replace('_', ' ').title() for c in cat_labels]

                            fig_cat = go.Figure(data=[go.Pie(
                                labels=cat_labels_display,
                                values=cat_values,
                                marker=dict(colors=colors_pie),
                                textinfo='label+percent',
                                textfont=dict(size=11),
                                hole=0.35,
                                hovertemplate="%{label}: %{value} selections<extra></extra>",
                            )])
                            fig_cat.update_layout(
                                title=dict(text="Categories d'Actifs", font=dict(size=14)),
                                height=350,
                                margin=dict(l=10, r=10, t=40, b=10),
                                showlegend=False,
                            )
                            st.plotly_chart(fig_cat, use_container_width=True)

                        # Asset popularity (top 10)
                        asset_popularity = {}
                        for pname, plan in all_plans.items():
                            for a in plan.get('assignments', []):
                                asset = a.get('asset', '')
                                asset_name = a.get('asset_name', asset)
                                key = f"{asset_name} ({asset})"
                                asset_popularity[key] = asset_popularity.get(key, 0) + 1

                        if asset_popularity:
                            top_assets = sorted(asset_popularity.items(), key=lambda x: x[1], reverse=True)[:10]
                            st.markdown("**Actifs les plus selectionnes**")
                            for asset_label, count in top_assets:
                                bar_pct = count / n_portfolios * 100
                                st.markdown(
                                    f"<div style='display:flex;align-items:center;margin-bottom:4px;'>"
                                    f"<span style='width:180px;font-size:0.85rem;'>{asset_label}</span>"
                                    f"<div style='flex:1;background:#e9ecef;border-radius:4px;height:18px;'>"
                                    f"<div style='width:{bar_pct:.0f}%;background:#1f77b4;border-radius:4px;"
                                    f"height:18px;text-align:right;padding-right:4px;'>"
                                    f"<span style='color:white;font-size:0.75rem;line-height:18px;'>"
                                    f"{count}/{n_portfolios}</span></div></div></div>",
                                    unsafe_allow_html=True)

                st.markdown("")

                # ============================================================
                # SECTION 5 : Radar de comparaison
                # ============================================================
                if all_plans and len(all_plans) >= 2:
                    st.subheader("Profils de Risque/Rendement")
                    st.markdown('<p class="mpt-section-help">'
                                'Comparaison des profils backtest des portefeuilles. '
                                'Chaque axe represente une metrique normalisee de 0 a 100.</p>',
                                unsafe_allow_html=True)

                    radar_data = {}
                    for pname in plan_names:
                        plan = all_plans.get(pname)
                        if not plan:
                            continue
                        stats = plan.get('summary_stats', {})
                        config = plan.get('config', {})
                        assignments = plan.get('assignments', [])
                        if not assignments:
                            continue
                        avg_sharpe = stats.get('avg_sharpe', 0)
                        avg_return = stats.get('avg_return', 0)
                        n_assets = stats.get('n_assets', 0)
                        avg_wr = np.mean([a.get('win_rate', 0) for a in assignments]) if assignments else 0
                        avg_dd = np.mean([a.get('max_drawdown_pct', 0) for a in assignments]) if assignments else 0
                        n_categories = len(stats.get('categories', []))
                        radar_data[pname] = {
                            'Sharpe': avg_sharpe,
                            'Rendement': avg_return,
                            'Win Rate': avg_wr,
                            'Diversification': n_assets,
                            'Categories': n_categories,
                            'Stabilite': max(0, 100 - avg_dd * 3),
                        }

                    if radar_data:
                        # Normalize 0-100
                        metrics_keys = ['Sharpe', 'Rendement', 'Win Rate', 'Diversification', 'Categories', 'Stabilite']
                        maxvals = {}
                        for mk in metrics_keys:
                            vals = [rd[mk] for rd in radar_data.values()]
                            maxvals[mk] = max(vals) if max(vals) > 0 else 1

                        fig_radar = go.Figure()
                        radar_colors = px.colors.qualitative.Set2
                        for i, (pname, rd) in enumerate(radar_data.items()):
                            normalized = [rd[mk] / maxvals[mk] * 100 for mk in metrics_keys]
                            normalized.append(normalized[0])  # close the loop
                            fig_radar.add_trace(go.Scatterpolar(
                                r=normalized,
                                theta=metrics_keys + [metrics_keys[0]],
                                fill='toself',
                                name=pname,
                                opacity=0.3,
                                line=dict(color=radar_colors[i % len(radar_colors)], width=2),
                            ))
                        fig_radar.update_layout(
                            polar=dict(
                                radialaxis=dict(visible=True, range=[0, 100], showticklabels=False),
                            ),
                            height=450,
                            template="plotly_white",
                            margin=dict(l=60, r=60, t=30, b=30),
                            legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5,
                                        font=dict(size=10)),
                        )
                        st.plotly_chart(fig_radar, use_container_width=True)

                st.markdown("")

                # ============================================================
                # SECTION 6 : Detail par portefeuille
                # ============================================================
                st.subheader("Detail par Portefeuille")
                st.markdown('<p class="mpt-section-help">'
                            'Cliquez sur un portefeuille pour voir ses positions, '
                            'trades clos, et son plan d\'allocation complet avec '
                            'les metriques de backtest.</p>',
                            unsafe_allow_html=True)

                for rank, (name, p_info) in enumerate(sorted_by_pnl, 1):
                    pnl_val = p_info.get('pnl', 0)
                    pnl_pct_val = p_info.get('pnl_pct', 0)
                    n_pos = p_info.get('positions', 0)
                    medal = {1: ' #1', 2: ' #2', 3: ' #3'}.get(rank, '')
                    pnl_indicator = "+" if pnl_val >= 0 else ""

                    label = (f"#{rank}{medal} {name} | "
                             f"{pnl_indicator}{pnl_pct_val:.2f}% "
                             f"({pnl_indicator}{pnl_val:,.0f} EUR) | "
                             f"{n_pos} pos. | "
                             f"{p_info.get('description', '')}")

                    with st.expander(label):
                        # Row of key metrics
                        dc1, dc2, dc3, dc4, dc5 = st.columns(5)
                        dc1.metric("Valeur", f"{p_info.get('value', 0):,.0f} EUR")
                        dc2.metric("P&L Total", f"{pnl_indicator}{pnl_val:,.0f} EUR",
                                   f"{pnl_indicator}{pnl_pct_val:.2f}%")
                        dc3.metric("Realise", f"{'+'if p_info.get('realized_pnl',0)>=0 else ''}"
                                   f"{p_info.get('realized_pnl', 0):,.0f} EUR")
                        dc4.metric("Win Rate", f"{p_info.get('win_rate', 0):.1f}%",
                                   f"{p_info.get('trades_closed', 0)} trades")
                        dc5.metric("Cycles", f"{p_info.get('cycle_count', 0)}")

                        # Config summary
                        config_str = p_info.get('config_summary', '')
                        if config_str:
                            parts = [s.strip() for s in config_str.split('|')]
                            config_html = ' '.join(
                                f'<span style="display:inline-block;background:#e9ecef;'
                                f'border-radius:12px;padding:2px 10px;margin:2px;font-size:0.8rem;">'
                                f'{p}</span>' for p in parts
                            )
                            st.markdown(f"**Configuration :** {config_html}", unsafe_allow_html=True)

                        # Find portfolio directory
                        portfolio_dir = None
                        for dirname in os.listdir(state_dir):
                            if dirname.startswith('portfolio_') and name.lower() in dirname.lower():
                                portfolio_dir = os.path.join(state_dir, dirname)
                                break

                        if portfolio_dir:
                            detail_tab1, detail_tab2, detail_tab3 = st.tabs([
                                "Positions ouvertes", "Trades clos", "Plan d'allocation"
                            ])

                            # --- Positions ouvertes ---
                            with detail_tab1:
                                sub_state_file = os.path.join(portfolio_dir, 'auto_state.json')
                                if os.path.exists(sub_state_file):
                                    try:
                                        with open(sub_state_file, 'r') as f:
                                            sub_state = json.load(f)
                                        sub_positions = sub_state.get('positions', {})
                                        if sub_positions:
                                            pos_rows = []
                                            for sym, pos in sub_positions.items():
                                                entry_price = pos.get('entry_price', 0)
                                                qty = pos.get('quantity', 0)
                                                side = pos.get('side', '')
                                                entry_time_str = pos.get('entry_time', '')
                                                if entry_time_str:
                                                    try:
                                                        et = datetime.fromisoformat(entry_time_str)
                                                        date_display = et.strftime('%d/%m %H:%M')
                                                    except Exception:
                                                        date_display = entry_time_str[:16]
                                                else:
                                                    date_display = 'N/A'

                                                current_price = None
                                                try:
                                                    yahoo_sym = convert_to_yahoo_symbol(sym)
                                                    data = yf.Ticker(yahoo_sym).history(period='1d', interval='1d')
                                                    data.columns = [c.lower() for c in data.columns]
                                                    if len(data) > 0:
                                                        current_price = float(data['close'].iloc[-1])
                                                except Exception:
                                                    pass

                                                if current_price is not None and qty > 0:
                                                    if side == 'LONG':
                                                        unrealized_pnl = (current_price - entry_price) * qty
                                                        unrealized_pct = (current_price / entry_price - 1) * 100
                                                    else:
                                                        unrealized_pnl = (entry_price - current_price) * qty
                                                        unrealized_pct = (1 - current_price / entry_price) * 100
                                                else:
                                                    unrealized_pnl = None
                                                    unrealized_pct = None

                                                sl = pos.get('stop_loss', 0)
                                                tp = pos.get('take_profit', 0)
                                                # Distance to SL/TP in %
                                                if current_price and current_price > 0 and sl > 0:
                                                    if side == 'LONG':
                                                        dist_sl = (current_price - sl) / current_price * 100
                                                        dist_tp = (tp - current_price) / current_price * 100 if tp > 0 else None
                                                    else:
                                                        dist_sl = (sl - current_price) / current_price * 100
                                                        dist_tp = (current_price - tp) / current_price * 100 if tp > 0 else None
                                                else:
                                                    dist_sl = None
                                                    dist_tp = None

                                                row = {
                                                    'Symbole': sym,
                                                    'Side': side,
                                                    'Strategie': pos.get('strategy', ''),
                                                    'Entree': date_display,
                                                    'Prix entree': _fmt_price(entry_price),
                                                    'Prix actuel': _fmt_price(current_price) if current_price else 'N/A',
                                                    'P&L': f"{unrealized_pnl:+.2f}" if unrealized_pnl is not None else 'N/A',
                                                    'P&L %': f"{unrealized_pct:+.2f}%" if unrealized_pct is not None else 'N/A',
                                                    'Dist. SL': f"{dist_sl:.1f}%" if dist_sl is not None else 'N/A',
                                                    'Dist. TP': f"{dist_tp:.1f}%" if dist_tp is not None else 'N/A',
                                                }
                                                pos_rows.append(row)
                                            st.dataframe(pd.DataFrame(pos_rows),
                                                         use_container_width=True, hide_index=True)
                                        else:
                                            st.info("Aucune position ouverte")
                                    except Exception:
                                        st.info("Aucune position ouverte")
                                else:
                                    st.info("Aucune position ouverte")

                            # --- Trades clos ---
                            with detail_tab2:
                                sub_trades_file = os.path.join(portfolio_dir, 'auto_trades.csv')
                                if os.path.exists(sub_trades_file):
                                    try:
                                        sub_trades = pd.read_csv(sub_trades_file)
                                        if len(sub_trades) > 0:
                                            if 'exit_time' in sub_trades.columns:
                                                sub_trades = sub_trades.sort_values('exit_time', ascending=False)

                                            total_pnl_trades = sub_trades['pnl'].sum() if 'pnl' in sub_trades.columns else 0
                                            wins = len(sub_trades[sub_trades['pnl'] > 0]) if 'pnl' in sub_trades.columns else 0
                                            losses = len(sub_trades) - wins

                                            # Summary bar
                                            tc1, tc2, tc3, tc4 = st.columns(4)
                                            tc1.metric("Total trades", len(sub_trades))
                                            tc2.metric("P&L total", f"{'+'if total_pnl_trades>=0 else ''}{total_pnl_trades:.0f} EUR")
                                            tc3.metric("Gagnants", f"{wins}", f"{wins/(wins+losses)*100:.0f}%" if (wins+losses)>0 else "")
                                            tc4.metric("Perdants", f"{losses}")

                                            # Trades table
                                            display_cols = []
                                            for _, trade in sub_trades.iterrows():
                                                t_pnl = trade.get('pnl', 0)
                                                t_pnl_pct = trade.get('pnl_pct', 0)
                                                entry_time_raw = trade.get('entry_time', '')
                                                exit_time_raw = trade.get('exit_time', '')
                                                try:
                                                    et = datetime.fromisoformat(str(entry_time_raw))
                                                    entry_fmt = et.strftime('%d/%m %H:%M')
                                                except Exception:
                                                    entry_fmt = str(entry_time_raw)[:10]
                                                try:
                                                    xt = datetime.fromisoformat(str(exit_time_raw))
                                                    exit_fmt = xt.strftime('%d/%m %H:%M')
                                                except Exception:
                                                    exit_fmt = str(exit_time_raw)[:10]

                                                display_cols.append({
                                                    'Resultat': 'W' if t_pnl >= 0 else 'L',
                                                    'Symbole': trade.get('symbol', '?'),
                                                    'Side': trade.get('side', '?'),
                                                    'Strategie': trade.get('strategy', ''),
                                                    'Entree': entry_fmt,
                                                    'Sortie': exit_fmt,
                                                    'Raison': trade.get('exit_reason', '?'),
                                                    'P&L': round(t_pnl, 2),
                                                    'P&L %': round(t_pnl_pct, 2),
                                                })
                                            trades_df = pd.DataFrame(display_cols)
                                            st.dataframe(trades_df, use_container_width=True, hide_index=True,
                                                         column_config={
                                                             'P&L': st.column_config.NumberColumn('P&L (EUR)', format="%+.2f"),
                                                             'P&L %': st.column_config.NumberColumn('P&L %', format="%+.2f%%"),
                                                         })
                                        else:
                                            st.info("Aucun trade clos pour le moment")
                                    except Exception:
                                        st.info("Aucun trade clos pour le moment")
                                else:
                                    st.info("Aucun trade clos pour le moment")

                            # --- Plan d'allocation ---
                            with detail_tab3:
                                sub_plan_file = os.path.join(portfolio_dir, 'auto_plan.json')
                                if os.path.exists(sub_plan_file):
                                    try:
                                        with open(sub_plan_file, 'r') as f:
                                            sub_plan = json.load(f)
                                        assignments = sub_plan.get('assignments', [])
                                        summary_stats = sub_plan.get('summary_stats', {})
                                        plan_config = sub_plan.get('config', {})
                                        risk_params = sub_plan.get('risk_params', {})

                                        if assignments:
                                            # Summary stats
                                            ps1, ps2, ps3, ps4, ps5 = st.columns(5)
                                            ps1.metric("Actifs", summary_stats.get('n_assets', len(assignments)))
                                            ps2.metric("Sharpe moyen", f"{summary_stats.get('avg_sharpe', 0):.2f}")
                                            ps3.metric("Rendement moyen", f"{summary_stats.get('avg_return', 0):.1f}%")
                                            ps4.metric("Meilleur", f"{summary_stats.get('best_return', 0):.1f}%")
                                            ps5.metric("Pire", f"{summary_stats.get('worst_return', 0):.1f}%")

                                            plan_chart_col, plan_table_col = st.columns([2, 3])

                                            with plan_chart_col:
                                                # Allocation pie
                                                alloc_labels = [a.get('asset_name', a.get('asset', '')) for a in assignments]
                                                alloc_values = [a.get('allocation_pct', 0) for a in assignments]
                                                cash_reserve = plan_config.get('cash_reserve_pct', 0)
                                                total_alloc = sum(alloc_values)
                                                if total_alloc < 100:
                                                    alloc_labels.append('Cash reserve')
                                                    alloc_values.append(100 - total_alloc)

                                                fig_alloc = go.Figure(data=[go.Pie(
                                                    labels=alloc_labels,
                                                    values=alloc_values,
                                                    textinfo='label+percent',
                                                    textfont=dict(size=10),
                                                    hole=0.4,
                                                    hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
                                                )])
                                                fig_alloc.update_layout(
                                                    title=dict(text="Repartition", font=dict(size=13)),
                                                    height=300,
                                                    margin=dict(l=5, r=5, t=35, b=5),
                                                    showlegend=False,
                                                )
                                                st.plotly_chart(fig_alloc, use_container_width=True)

                                            with plan_table_col:
                                                # Full plan table with backtest metrics
                                                plan_rows = []
                                                for a in assignments:
                                                    asset_key = a.get('asset', '')
                                                    rp = risk_params.get(asset_key, {})
                                                    plan_rows.append({
                                                        'Actif': f"{a.get('asset_name', '')} ({asset_key})",
                                                        'Type': a.get('asset_type', '').replace('_', ' ').title(),
                                                        'Strategie': a.get('strategy_name', ''),
                                                        'Alloc %': round(a.get('allocation_pct', 0), 1),
                                                        'Score': round(a.get('score', 0), 3),
                                                        'Sharpe': round(a.get('sharpe_ratio', 0), 2),
                                                        'Rend. %': round(a.get('total_return_pct', 0), 1),
                                                        'DD %': round(a.get('max_drawdown_pct', 0), 1),
                                                        'WR %': round(a.get('win_rate', 0), 1),
                                                        'Trades': a.get('num_trades', 0),
                                                        'SL %': round(rp.get('stop_loss_pct', 0), 1),
                                                        'TP %': round(rp.get('take_profit_pct', 0), 1),
                                                    })
                                                plan_df = pd.DataFrame(plan_rows)
                                                st.dataframe(plan_df, use_container_width=True, hide_index=True,
                                                             column_config={
                                                                 'Alloc %': st.column_config.ProgressColumn(
                                                                     'Alloc %', format="%.1f%%",
                                                                     min_value=0, max_value=max(a.get('allocation_pct', 0) for a in assignments) * 1.2),
                                                                 'Score': st.column_config.NumberColumn('Score', format="%.3f",
                                                                     help="Score composite base sur Sharpe, rendement, drawdown et win rate"),
                                                                 'Sharpe': st.column_config.NumberColumn('Sharpe', format="%.2f",
                                                                     help="Ratio de Sharpe : rendement ajuste du risque. > 1 = bon, > 2 = excellent"),
                                                                 'Rend. %': st.column_config.NumberColumn('Rend. %', format="%.1f%%",
                                                                     help="Rendement total du backtest"),
                                                                 'DD %': st.column_config.NumberColumn('DD %', format="%.1f%%",
                                                                     help="Drawdown max : pire perte pic-a-creux en backtest"),
                                                                 'WR %': st.column_config.NumberColumn('WR %', format="%.1f%%",
                                                                     help="Win Rate : pourcentage de trades gagnants"),
                                                             })
                                    except Exception:
                                        st.info("Plan d'allocation non disponible")
                                else:
                                    st.info("Plan d'allocation non disponible")

                st.markdown("")

                # ============================================================
                # SECTION 7 : Glossaire pedagogique
                # ============================================================
                with st.expander("Glossaire - Comprendre les metriques"):
                    st.markdown("""<div class="mpt-glossary">
<dl>
<dt>P&L (Profit & Loss)</dt>
<dd>Gain ou perte total. <b>Realise</b> = trades fermes. <b>Latent</b> = positions encore ouvertes (non encaisse).</dd>

<dt>Win Rate</dt>
<dd>Pourcentage de trades gagnants. Un win rate de 60% signifie que 6 trades sur 10 sont profitables.
Un win rate eleve ne garantit pas la rentabilite si les pertes sont plus grandes que les gains.</dd>

<dt>Ratio de Sharpe</dt>
<dd>Mesure le rendement ajuste du risque. Plus il est eleve, meilleur est le rapport gain/volatilite.
< 0.5 = mediocre, 0.5-1.0 = correct, > 1.0 = bon, > 2.0 = excellent.</dd>

<dt>Drawdown (DD)</dt>
<dd>La pire perte observee entre un pic et un creux. Un DD de 15% signifie que le portefeuille a perdu
jusqu'a 15% depuis son plus haut. Plus le DD est bas, moins la strategie est volatile.</dd>

<dt>Score composite</dt>
<dd>Note de 0 a 1 combinant Sharpe (40%), rendement (25%), drawdown (20%) et win rate (15%).
Utilise pour classer et selectionner les meilleures paires actif/strategie.</dd>

<dt>Methodes d'allocation</dt>
<dd><b>risk_parity</b> : alloue plus aux actifs moins volatils (inverse de la volatilite).<br>
<b>score_weighted</b> : alloue proportionnellement au score composite.<br>
<b>equal</b> : repartition egale entre tous les actifs selectionnes.</dd>

<dt>Stop Loss (SL) / Take Profit (TP)</dt>
<dd>Niveaux automatiques de sortie. Le SL limite les pertes, le TP securise les gains.
Definit en pourcentage du prix d'entree.</dd>
</dl>
</div>""", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Erreur lecture etat multi-portfolio: {e}")
                import traceback
                st.code(traceback.format_exc())

    # ========================================================================
    # TAB 9: Signaux Macro
    # ========================================================================

    with tab9:
        st.header("📡 Signaux Macroéconomiques")

        try:
            from macro_events import MacroEventsDatabase
            from news_fetcher import MacroNewsAggregator, RSSFeedFetcher
            macro_db = MacroEventsDatabase()
            events_df = macro_db.get_events_df()
            aggregator = MacroNewsAggregator()

            # ==============================================================
            # Fetch & Score RSS signals (needed for Section 1 metrics)
            # ==============================================================
            rss_headlines = []
            scored_signals = []

            try:
                rss = RSSFeedFetcher()
                for feed_name in rss.FEEDS.keys():
                    try:
                        entries = rss.fetch_feed(feed_name, max_entries=3)
                        rss_headlines.extend(entries)
                    except Exception:
                        pass

                # Parse dates (tz-naive)
                for h in rss_headlines:
                    try:
                        parsed = pd.to_datetime(h.get('published', ''))
                        if parsed.tzinfo is not None:
                            parsed = parsed.tz_localize(None)
                        h['_parsed_date'] = parsed
                    except Exception:
                        h['_parsed_date'] = pd.Timestamp.now()
                rss_headlines.sort(key=lambda x: x['_parsed_date'], reverse=True)

                # Score each headline via MacroNewsAggregator
                for entry in rss_headlines:
                    signal = aggregator._rss_to_signal(entry, affected_assets=['all'])
                    if signal:
                        scored_signals.append(signal)
            except Exception as e:
                st.warning(f"Erreur fetch RSS : {e}")

            # ==============================================================
            # Section 1 : Contexte macro actuel (LIVE)
            # ==============================================================
            st.subheader("🌍 Contexte Macro Actuel (Live)")

            now = datetime.now()

            if scored_signals:
                # Calculate metrics from scored signals
                scores_list = [s.impact_score for s in scored_signals]
                avg_impact = np.mean(scores_list)
                positive_count = sum(1 for s in scores_list if s > 2)
                negative_count = sum(1 for s in scores_list if s < -2)
                neutral_count = len(scores_list) - positive_count - negative_count

                # Determine sentiment label
                if avg_impact > 5:
                    sentiment_label, sentiment_color = 'Tres Haussier', '#00C853'
                elif avg_impact > 2:
                    sentiment_label, sentiment_color = 'Haussier', '#66BB6A'
                elif avg_impact > -2:
                    sentiment_label, sentiment_color = 'Neutre', '#9E9E9E'
                elif avg_impact > -5:
                    sentiment_label, sentiment_color = 'Baissier', '#FF7043'
                else:
                    sentiment_label, sentiment_color = 'Tres Baissier', '#D32F2F'

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Sentiment Live", sentiment_label, f"{avg_impact:+.1f}")
                col2.metric("Signaux RSS", len(scored_signals))
                col3.metric("Haussiers / Baissiers", f"{positive_count} / {negative_count}")
                col4.metric("Neutres", neutral_count)

                # Recent signals preview (top 5)
                st.markdown("**Derniers signaux :**")
                for sig in scored_signals[:5]:
                    score = sig.impact_score
                    if score >= 5:
                        icon = "🟢"
                    elif score >= 1:
                        icon = "🔵"
                    elif score >= -1:
                        icon = "⚪"
                    elif score >= -5:
                        icon = "🟠"
                    else:
                        icon = "🔴"
                    date_str = sig.timestamp.strftime('%d/%m %H:%M') if hasattr(sig.timestamp, 'strftime') else 'N/A'
                    sentiment_fr = {'bullish': 'Haussier', 'bearish': 'Baissier', 'neutral': 'Neutre'}.get(sig.sentiment, 'Neutre')
                    st.markdown(f"{icon} **{date_str}** - {sig.title[:80]} ({sig.category}) &nbsp; Impact: **{score:+.1f}**/10 &nbsp; *{sentiment_fr}*")
            else:
                st.info("Aucun signal live disponible (connexion internet requise).")

            # ==============================================================
            # Section 2 : Signaux Live (Score Composite + Fear&Greed + FRED)
            # ==============================================================
            st.markdown("---")
            st.subheader("📡 Signaux Live")

            # --- Score Macro Composite ---
            if scored_signals:
                weighted_sum = 0.0
                total_weight = 0.0
                now_naive = datetime.now()
                for signal in scored_signals:
                    ts = signal.timestamp
                    if hasattr(ts, 'tzinfo') and ts.tzinfo is not None:
                        ts = ts.tz_localize(None) if hasattr(ts, 'tz_localize') else ts.replace(tzinfo=None)
                    hours_old = max(0, (now_naive - ts).total_seconds() / 3600)
                    time_decay = max(0.3, 1.0 - (hours_old / 48))
                    weight = signal.confidence * time_decay
                    weighted_sum += signal.impact_score * weight
                    total_weight += weight

                composite_score = (weighted_sum / total_weight) * 10 if total_weight > 0 else 0
                composite_score = np.clip(composite_score, -100, 100)

                if composite_score > 30:
                    rec_label, rec_color = "Haussier", "#00C853"
                elif composite_score > 10:
                    rec_label, rec_color = "Leg. Haussier", "#66BB6A"
                elif composite_score > -10:
                    rec_label, rec_color = "Neutre", "#9E9E9E"
                elif composite_score > -30:
                    rec_label, rec_color = "Leg. Baissier", "#FF7043"
                else:
                    rec_label, rec_color = "Baissier", "#D32F2F"

                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("Score Composite", f"{composite_score:+.1f} / 100")
                sc2.markdown(f"**Recommandation :**")
                sc2.markdown(f"<h3 style='color: {rec_color}; margin-top: -10px;'>{rec_label}</h3>", unsafe_allow_html=True)
                sc3.markdown(f"**{len(scored_signals)}** signaux analyses &nbsp; | &nbsp; "
                             f"🟢 {sum(1 for s in scored_signals if s.impact_score > 2)} &nbsp; "
                             f"🔴 {sum(1 for s in scored_signals if s.impact_score < -2)} &nbsp; "
                             f"⚪ {sum(1 for s in scored_signals if -2 <= s.impact_score <= 2)}")

                st.markdown("---")

            live_col1, live_col2 = st.columns(2)

            # --- Fear & Greed Index (gratuit, pas de cle) ---
            with live_col1:
                st.markdown("**Fear & Greed Index (Crypto)**")
                try:
                    from news_fetcher import FearGreedIndexFetcher
                    fg_fetcher = FearGreedIndexFetcher()
                    fg_data = fg_fetcher.fetch_crypto_fear_greed(limit=30)

                    if fg_data:
                        current_fg = fg_data[0]
                        fg_val = int(current_fg.get('value', 50))
                        fg_label = current_fg.get('value_classification', 'N/A')

                        # Gauge chart
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=fg_val,
                            title={'text': fg_label, 'font': {'size': 16}},
                            gauge={
                                'axis': {'range': [0, 100], 'tickwidth': 1},
                                'bar': {'color': '#1E88E5'},
                                'steps': [
                                    {'range': [0, 25], 'color': '#D32F2F'},
                                    {'range': [25, 45], 'color': '#FF7043'},
                                    {'range': [45, 55], 'color': '#9E9E9E'},
                                    {'range': [55, 75], 'color': '#66BB6A'},
                                    {'range': [75, 100], 'color': '#00C853'},
                                ],
                            },
                        ))
                        fig_gauge.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=10), template="plotly_dark")
                        st.plotly_chart(fig_gauge, use_container_width=True)

                        # Historique 30j
                        if len(fg_data) > 1:
                            fg_hist = pd.DataFrame([{
                                'date': datetime.fromtimestamp(int(d['timestamp'])),
                                'value': int(d['value']),
                            } for d in fg_data])
                            fg_hist = fg_hist.sort_values('date')

                            fig_fg_hist = go.Figure()
                            fig_fg_hist.add_trace(go.Scatter(
                                x=fg_hist['date'], y=fg_hist['value'],
                                mode='lines+markers', line=dict(color='#42A5F5', width=2),
                                fill='tozeroy', fillcolor='rgba(66,165,245,0.15)',
                            ))
                            fig_fg_hist.add_hline(y=25, line_dash="dash", line_color="#D32F2F", annotation_text="Extreme Fear")
                            fig_fg_hist.add_hline(y=75, line_dash="dash", line_color="#00C853", annotation_text="Extreme Greed")
                            fig_fg_hist.update_layout(
                                title="Historique Fear & Greed (30j)",
                                height=250, template="plotly_dark",
                                yaxis=dict(range=[0, 100]),
                                showlegend=False,
                            )
                            st.plotly_chart(fig_fg_hist, use_container_width=True)
                    else:
                        st.warning("Fear & Greed Index indisponible")
                except Exception as e:
                    st.warning(f"Fear & Greed non disponible : {e}")

            # --- FRED Indicateurs economiques ---
            with live_col2:
                st.markdown("**Indicateurs Economiques (FRED)**")
                try:
                    from news_fetcher import FredAPIFetcher
                    fred = FredAPIFetcher()

                    # Utiliser la cle de macro_data_fetcher si pas d'env var
                    if not fred.api_key:
                        fred.api_key = "480a473e9a5a6e99838252204df3cd1b"

                    if fred.api_key:
                        indicators = fred.get_latest_indicators()
                        if indicators:
                            ind_col1, ind_col2 = st.columns(2)
                            if 'fed_funds' in indicators:
                                ind_col1.metric("Taux Fed", f"{indicators['fed_funds']:.2f}%")
                            if 'cpi' in indicators:
                                ind_col2.metric("CPI (Inflation)", f"{indicators['cpi']:.1f}")
                            if 'unemployment' in indicators:
                                ind_col1.metric("Chomage", f"{indicators['unemployment']:.1f}%")
                            if 'vix' in indicators:
                                vix_val = indicators['vix']
                                ind_col2.metric("VIX", f"{vix_val:.1f}", delta=f"{'Eleve' if vix_val > 25 else 'Normal'}", delta_color=("inverse" if vix_val > 25 else "off"))
                            if 'yields_10y' in indicators:
                                ind_col1.metric("Treasury 10Y", f"{indicators['yields_10y']:.2f}%")
                        else:
                            st.info("Indicateurs FRED temporairement indisponibles")
                    else:
                        st.info("Cle API FRED non configuree")
                except Exception as e:
                    st.warning(f"FRED non disponible : {e}")

            # --- RSS News Headlines avec scoring ---
            st.markdown("---")
            st.markdown("**📰 News RSS Scorees (13 sources : Bloomberg, Reuters, CNBC, Yahoo, Investing, CoinDesk, Fed, ECB...)**")

            if scored_signals:
                with st.expander(f"📡 {len(scored_signals)} signaux RSS scores", expanded=True):
                    for signal in scored_signals[:15]:
                        # Badge sentiment
                        if signal.sentiment == 'bullish':
                            badge = "🟢 HAUSSIER"
                            badge_color = "#00C853"
                        elif signal.sentiment == 'bearish':
                            badge = "🔴 BAISSIER"
                            badge_color = "#D32F2F"
                        else:
                            badge = "⚪ NEUTRE"
                            badge_color = "#9E9E9E"

                        source_tag = signal.source.replace('_', ' ').title()
                        date_display = signal.timestamp.strftime('%d/%m %H:%M') if hasattr(signal.timestamp, 'strftime') else ''

                        st.markdown(
                            f"**[{source_tag}]** {signal.title}  \n"
                            f"<span style='background-color: {badge_color}; color: white; padding: 2px 8px; "
                            f"border-radius: 4px; font-size: 0.8em;'>{badge}</span> "
                            f"<span style='color: {badge_color}; font-weight: bold;'>Impact: {signal.impact_score:+.1f}/10</span> "
                            f"<span style='color: #888; font-size: 0.9em;'>| {signal.category.upper()} | {date_display}</span>",
                            unsafe_allow_html=True
                        )
                        st.markdown("")
            else:
                st.info("Aucun signal RSS recupere (connexion internet requise)")

            # ==============================================================
            # Section 3 : Timeline des evenements macro (archive 2024-2025)
            # ==============================================================
            st.markdown("---")
            st.subheader("📅 Evenements Macro Historiques (Archive 2024-2025)")
            st.caption("Cette section affiche les evenements macro historiques manuellement enregistres (2024-2025). Les signaux live sont dans les sections ci-dessus.")

            # Filtres
            col_f1, col_f2, col_f3 = st.columns([1, 1, 2])
            with col_f1:
                categories = sorted(events_df['category'].unique().tolist())
                selected_cats = st.multiselect("Categorie", categories, default=categories, key="macro_cat_filter")
            with col_f2:
                period_choice = st.selectbox("Periode", ["6 derniers mois", "12 derniers mois", "Tout (2024-2025)"], index=1, key="macro_period")

            # Filtrer
            if period_choice == "6 derniers mois":
                cutoff = now - timedelta(days=180)
            elif period_choice == "12 derniers mois":
                cutoff = now - timedelta(days=365)
            else:
                cutoff = pd.to_datetime('2024-01-01')

            filtered_events = events_df[
                (events_df['date'] >= cutoff) &
                (events_df['category'].isin(selected_cats))
            ].copy()

            if len(filtered_events) > 0:
                # Couleur par impact
                colors = []
                for score in filtered_events['impact_score']:
                    if score >= 5:
                        colors.append('#00C853')
                    elif score >= 1:
                        colors.append('#66BB6A')
                    elif score >= -1:
                        colors.append('#9E9E9E')
                    elif score >= -5:
                        colors.append('#FF7043')
                    else:
                        colors.append('#D32F2F')

                fig_timeline = go.Figure()
                fig_timeline.add_trace(go.Bar(
                    x=filtered_events['date'],
                    y=filtered_events['impact_score'],
                    text=filtered_events['title'],
                    hovertemplate="<b>%{text}</b><br>Date: %{x|%d/%m/%Y}<br>Impact: %{y:+.0f}/10<extra></extra>",
                    marker_color=colors,
                ))
                fig_timeline.update_layout(
                    title="Impact des evenements macro sur les marches",
                    xaxis_title="Date",
                    yaxis_title="Impact (-10 bearish / +10 bullish)",
                    yaxis=dict(range=[-11, 11], zeroline=True, zerolinewidth=2, zerolinecolor='white'),
                    height=400,
                    template="plotly_dark",
                    showlegend=False,
                )
                # Ajouter zones de reference
                fig_timeline.add_hrect(y0=6, y1=11, fillcolor="green", opacity=0.07, line_width=0,
                                       annotation_text="Zone Bullish", annotation_position="top left")
                fig_timeline.add_hrect(y0=-11, y1=-6, fillcolor="red", opacity=0.07, line_width=0,
                                       annotation_text="Zone Bearish", annotation_position="bottom left")
                st.plotly_chart(fig_timeline, use_container_width=True)

                # Tableau des evenements
                with st.expander(f"Voir les {len(filtered_events)} evenements en detail"):
                    display_df = filtered_events[['date', 'title', 'category', 'impact_score', 'affected_assets', 'description']].copy()
                    display_df['date'] = display_df['date'].dt.strftime('%d/%m/%Y')
                    display_df.columns = ['Date', 'Evenement', 'Categorie', 'Impact', 'Actifs concernes', 'Description']
                    display_df = display_df.sort_values('Date', ascending=False)
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                st.info("Aucun evenement dans la periode selectionnee.")

            # ==============================================================
            # Section 3 : Impact par classe d'actif
            # ==============================================================
            st.markdown("---")
            st.subheader("📊 Sentiment Macro par Classe d'Actif")

            asset_classes = {
                'Crypto': ['crypto', 'BTC/USDT', 'ETH/USDT'],
                'Actions US': ['stocks', 'indices', 'tech', 'NVDA', 'AAPL'],
                'Commodities': ['commodities', 'gold', 'oil'],
                'Forex': ['forex'],
            }

            rows_asset = []
            for cls_name, cls_tags in asset_classes.items():
                # Calculer sentiment 90j pour chaque classe
                cls_events = []
                for _, ev in events_df.iterrows():
                    if ev['date'] < cutoff:
                        continue
                    affected = str(ev['affected_assets']).split(',')
                    if 'all' in affected or any(tag in affected for tag in cls_tags):
                        cls_events.append(ev)

                if cls_events:
                    cls_df = pd.DataFrame(cls_events)
                    total = cls_df['impact_score'].sum()
                    avg = cls_df['impact_score'].mean()
                    pos = (cls_df['impact_score'] > 0).sum()
                    neg = (cls_df['impact_score'] < 0).sum()

                    if avg > 3:
                        sentiment_label = "Tres Haussier"
                    elif avg > 1:
                        sentiment_label = "Haussier"
                    elif avg > -1:
                        sentiment_label = "Neutre"
                    elif avg > -3:
                        sentiment_label = "Baissier"
                    else:
                        sentiment_label = "Tres Baissier"

                    rows_asset.append({
                        'Classe': cls_name,
                        'Nb evenements': len(cls_events),
                        'Impact moyen': f"{avg:+.1f}",
                        'Impact total': f"{total:+.0f}",
                        'Positifs': int(pos),
                        'Negatifs': int(neg),
                        'Sentiment': sentiment_label,
                    })

            if rows_asset:
                df_asset_cls = pd.DataFrame(rows_asset)

                def _color_sentiment(val):
                    mapping = {
                        'Tres Haussier': 'background-color: #1B5E20; color: white',
                        'Haussier': 'background-color: #388E3C; color: white',
                        'Neutre': 'background-color: #616161; color: white',
                        'Baissier': 'background-color: #E64A19; color: white',
                        'Tres Baissier': 'background-color: #B71C1C; color: white',
                    }
                    return mapping.get(val, '')

                st.dataframe(
                    df_asset_cls.style.map(_color_sentiment, subset=['Sentiment']),
                    use_container_width=True,
                    hide_index=True
                )

            # ==============================================================
            # Section 4 : Fonctionnement du filtre macro
            # ==============================================================
            st.markdown("---")
            st.subheader("🔧 Comment le Filtre Macro Impacte les Positions")

            st.markdown("""
Le filtre macro analyse le contexte macroeconomique pour **bloquer les trades qui vont contre la tendance dominante**.
Il agit comme un garde-fou entre les signaux techniques et l'execution des ordres.
            """)

            # Schema visuel avec Plotly
            fig_flow = go.Figure()

            # Boites du flow
            boxes = [
                (0.1, 0.7, "Signaux\nTechniques", "#1E88E5", "Strategies MA, RSI,\nMACD, Bollinger..."),
                (0.4, 0.7, "Filtre\nMacro", "#FF8F00", "Analyse news,\nsentiment, eco"),
                (0.7, 0.85, "LONG\nautorise", "#43A047", "Score > -60"),
                (0.7, 0.55, "TRADE\nbloque", "#E53935", "Score < -60 pour LONG\nScore > +60 pour SHORT"),
            ]

            for x, y, text, color, hover in boxes:
                fig_flow.add_trace(go.Scatter(
                    x=[x], y=[y], mode='markers+text',
                    marker=dict(size=60, color=color, symbol='square', opacity=0.85),
                    text=[text], textposition='middle center',
                    textfont=dict(size=11, color='white'),
                    hovertext=hover, hoverinfo='text',
                    showlegend=False,
                ))

            # Fleches
            annotations = [
                dict(x=0.3, y=0.7, ax=0.2, ay=0.7, arrowhead=2, arrowcolor='white', arrowwidth=2),
                dict(x=0.6, y=0.85, ax=0.5, ay=0.75, arrowhead=2, arrowcolor='#43A047', arrowwidth=2),
                dict(x=0.6, y=0.55, ax=0.5, ay=0.65, arrowhead=2, arrowcolor='#E53935', arrowwidth=2),
            ]
            fig_flow.update_layout(
                annotations=annotations,
                xaxis=dict(range=[0, 1], showgrid=False, showticklabels=False, zeroline=False),
                yaxis=dict(range=[0.3, 1.0], showgrid=False, showticklabels=False, zeroline=False),
                height=250,
                template="plotly_dark",
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig_flow, use_container_width=True)

            # Regles du filtre
            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                st.markdown("""
**Score < -60 (Bearish)**
- Signal LONG -> **BLOQUE**
- Signal SHORT -> Autorise
- *Evite d'acheter pendant un krach*
                """)
            with col_r2:
                st.markdown("""
**Score entre -60 et +60**
- Signal LONG -> Autorise
- Signal SHORT -> Autorise
- *Les signaux techniques passent*
                """)
            with col_r3:
                st.markdown("""
**Score > +60 (Bullish)**
- Signal LONG -> Autorise
- Signal SHORT -> **BLOQUE**
- *Evite de shorter un rally*
                """)

            # Exemples concrets
            st.markdown("---")
            st.markdown("**Exemples concrets d'impact :**")

            example_data = [
                {"Evenement": "Fed coupe 50bps (Sept 2024)", "Impact": "+8", "Effet": "Rally haussier -> signaux SHORT bloques", "Resultat": "Evite des pertes sur shorts contre-tendance"},
                {"Evenement": "Black Monday (Aout 2024)", "Impact": "-8", "Effet": "Krach violent -> signaux LONG bloques", "Resultat": "Evite d'acheter pendant la chute"},
                {"Evenement": "Trump Tariffs (Fev 2025)", "Impact": "-6", "Effet": "Tensions commerciales -> prudence sur longs", "Resultat": "Reduit l'exposition pendant l'incertitude"},
                {"Evenement": "Bitcoin ETF Approval (Jan 2024)", "Impact": "+9", "Effet": "Euphorie crypto -> shorts crypto bloques", "Resultat": "Evite de shorter le rally crypto"},
            ]
            st.dataframe(pd.DataFrame(example_data), use_container_width=True, hide_index=True)

            # ==============================================================
            # Section 5 : Repartition par categorie
            # ==============================================================
            st.markdown("---")
            st.subheader("📈 Repartition des Evenements par Categorie")

            cat_stats = events_df.groupby('category').agg(
                count=('impact_score', 'count'),
                avg_impact=('impact_score', 'mean'),
                total_impact=('impact_score', 'sum'),
            ).sort_values('count', ascending=False)

            col_c1, col_c2 = st.columns(2)

            with col_c1:
                fig_cat_count = go.Figure(go.Bar(
                    x=cat_stats.index,
                    y=cat_stats['count'],
                    marker_color='#42A5F5',
                    text=cat_stats['count'],
                    textposition='auto',
                ))
                fig_cat_count.update_layout(
                    title="Nombre d'evenements par categorie",
                    height=350, template="plotly_dark",
                    xaxis_title="", yaxis_title="Nombre",
                )
                st.plotly_chart(fig_cat_count, use_container_width=True)

            with col_c2:
                cat_colors = ['#43A047' if v > 0 else '#E53935' for v in cat_stats['avg_impact']]
                fig_cat_impact = go.Figure(go.Bar(
                    x=cat_stats.index,
                    y=cat_stats['avg_impact'],
                    marker_color=cat_colors,
                    text=[f"{v:+.1f}" for v in cat_stats['avg_impact']],
                    textposition='auto',
                ))
                fig_cat_impact.update_layout(
                    title="Impact moyen par categorie",
                    height=350, template="plotly_dark",
                    xaxis_title="", yaxis_title="Impact moyen",
                    yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='white'),
                )
                st.plotly_chart(fig_cat_impact, use_container_width=True)

            # ==============================================================
            # Section 6 : Sentiment par trimestre
            # ==============================================================
            st.markdown("---")
            st.subheader("📆 Evolution du Sentiment par Trimestre")

            quarters = [
                ('2024-01-01', '2024-03-31', 'Q1 2024'),
                ('2024-04-01', '2024-06-30', 'Q2 2024'),
                ('2024-07-01', '2024-09-30', 'Q3 2024'),
                ('2024-10-01', '2024-12-31', 'Q4 2024'),
                ('2025-01-01', '2025-03-31', 'Q1 2025'),
                ('2025-04-01', '2025-06-30', 'Q2 2025'),
                ('2025-07-01', '2025-09-30', 'Q3 2025'),
                ('2025-10-01', '2025-12-31', 'Q4 2025'),
            ]

            q_labels = []
            q_impacts = []
            q_counts = []
            for start, end, label in quarters:
                s = macro_db.get_sentiment_score(start, end, asset='all')
                q_labels.append(label)
                q_impacts.append(s['avg_impact'])
                q_counts.append(s['num_events'])

            q_colors = ['#43A047' if v > 0 else '#E53935' for v in q_impacts]

            fig_quarters = go.Figure()
            fig_quarters.add_trace(go.Bar(
                x=q_labels, y=q_impacts,
                marker_color=q_colors,
                text=[f"{v:+.1f}" for v in q_impacts],
                textposition='auto',
                name="Impact moyen",
            ))
            fig_quarters.update_layout(
                title="Sentiment macro moyen par trimestre",
                yaxis_title="Impact moyen",
                yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='white'),
                height=350, template="plotly_dark",
            )
            st.plotly_chart(fig_quarters, use_container_width=True)

            # Detail par trimestre
            q_rows = []
            for i, (start, end, label) in enumerate(quarters):
                s = macro_db.get_sentiment_score(start, end, asset='all')
                if s['num_events'] > 0:
                    q_rows.append({
                        'Trimestre': label,
                        'Evenements': s['num_events'],
                        'Impact moyen': f"{s['avg_impact']:+.1f}",
                        'Impact total': f"{s['total_impact']:+.0f}",
                        'Sentiment': s['sentiment'].replace('_', ' ').title(),
                    })
            if q_rows:
                st.dataframe(pd.DataFrame(q_rows), use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Erreur onglet Macro: {e}")
            import traceback
            with st.expander("Voir details erreur"):
                st.code(traceback.format_exc())

    # ========================================================================
    # Auto-refresh
    # ========================================================================

    if auto_refresh:
        time.sleep(30)
        st.rerun()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info(f"Dernière mise à jour: {datetime.now().strftime('%H:%M:%S')}")


# ============================================================================
# Point d'entrée
# ============================================================================

if __name__ == '__main__':
    main()
