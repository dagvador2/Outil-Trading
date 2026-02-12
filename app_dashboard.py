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
        st.header("Multi Paper Trading")
        st.info("Comparaison de 20 portefeuilles en parallele (10 sans + 10 avec filtre macro). "
                "Lancez : `python multi_paper_trading.py --capital 100000`")

        state_dir = 'paper_trading_state'
        consolidated_file = os.path.join(state_dir, 'consolidated_state.json')

        if not os.path.exists(consolidated_file):
            # Fallback: check single-mode state
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

                # --- Global metrics ---
                total_value_all = sum(p.get('value', capital_per) for p in portfolios.values())
                total_pnl_all = sum(p.get('pnl', 0) for p in portfolios.values())
                total_pnl_pct = (total_pnl_all / total_capital_multi * 100) if total_capital_multi > 0 else 0
                total_positions = sum(p.get('positions', 0) for p in portfolios.values())
                total_trades = sum(p.get('trades_closed', 0) for p in portfolios.values())

                col_g1, col_g2, col_g3, col_g4, col_g5 = st.columns(5)
                col_g1.metric("Valeur totale", f"{total_value_all:,.0f} EUR",
                              f"{total_pnl_all:+,.0f} ({total_pnl_pct:+.1f}%)" if total_pnl_all != 0 else None)
                col_g2.metric("Capital investi", f"{total_capital_multi:,.0f} EUR",
                              f"{n_portfolios} portefeuilles")
                col_g3.metric("Positions ouvertes", f"{total_positions}")
                col_g4.metric("Trades clos", f"{total_trades}")

                # Best portfolio
                if portfolios:
                    best_name = max(portfolios.keys(), key=lambda k: portfolios[k].get('pnl_pct', 0))
                    best_pnl = portfolios[best_name].get('pnl_pct', 0)
                    col_g5.metric("Meilleur portefeuille", best_name,
                                  f"{best_pnl:+.2f}%")

                st.caption(f"Derniere mise a jour: {last_update[:19] if last_update != 'N/A' else 'N/A'}")
                st.markdown("---")

                # --- Leaderboard ---
                st.subheader("Leaderboard")

                if portfolios:
                    lb_rows = []
                    for rank, (name, p) in enumerate(
                        sorted(portfolios.items(), key=lambda x: x[1].get('pnl_pct', 0), reverse=True), 1
                    ):
                        lb_rows.append({
                            'Rang': rank,
                            'Portefeuille': name,
                            'Valeur (EUR)': round(p.get('value', 0), 2),
                            'P&L (EUR)': round(p.get('pnl', 0), 2),
                            'P&L (%)': round(p.get('pnl_pct', 0), 2),
                            'Realise (EUR)': round(p.get('realized_pnl', 0), 2),
                            'Positions': f"{p.get('positions', 0)}/{p.get('max_positions', '?')}",
                            'Trades': p.get('trades_closed', 0),
                            'Win Rate (%)': round(p.get('win_rate', 0), 1),
                            'Actifs plan': p.get('n_assets_in_plan', 0),
                            'Config': p.get('config_summary', ''),
                        })

                    lb_df = pd.DataFrame(lb_rows)
                    st.dataframe(lb_df, use_container_width=True, hide_index=True)

                st.markdown("---")

                # --- P&L Bar Chart ---
                st.subheader("P&L par portefeuille")

                if portfolios:
                    sorted_names = sorted(portfolios.keys(),
                                          key=lambda k: portfolios[k].get('pnl', 0), reverse=True)
                    pnl_values = [portfolios[n].get('pnl', 0) for n in sorted_names]
                    colors = ['#2ca02c' if v >= 0 else '#d62728' for v in pnl_values]

                    fig_pnl = go.Figure(data=[
                        go.Bar(
                            x=sorted_names,
                            y=pnl_values,
                            marker_color=colors,
                            text=[f"{v:+.1f}" for v in pnl_values],
                            textposition='outside'
                        )
                    ])
                    fig_pnl.update_layout(
                        yaxis_title="P&L (EUR)",
                        height=400,
                        template="plotly_white",
                    )
                    fig_pnl.add_hline(y=0, line_dash="dash", line_color="gray")
                    st.plotly_chart(fig_pnl, use_container_width=True)

                    # P&L % bar chart
                    pnl_pct_values = [portfolios[n].get('pnl_pct', 0) for n in sorted_names]
                    colors_pct = ['#2ca02c' if v >= 0 else '#d62728' for v in pnl_pct_values]

                    fig_pnl_pct = go.Figure(data=[
                        go.Bar(
                            x=sorted_names,
                            y=pnl_pct_values,
                            marker_color=colors_pct,
                            text=[f"{v:+.2f}%" for v in pnl_pct_values],
                            textposition='outside'
                        )
                    ])
                    fig_pnl_pct.update_layout(
                        yaxis_title="P&L (%)",
                        height=400,
                        template="plotly_white",
                    )
                    fig_pnl_pct.add_hline(y=0, line_dash="dash", line_color="gray")
                    st.plotly_chart(fig_pnl_pct, use_container_width=True)

                st.markdown("---")

                # --- Detail par portefeuille ---
                st.subheader("Detail par portefeuille")

                for name, p_info in sorted(portfolios.items(),
                                           key=lambda x: x[1].get('pnl_pct', 0), reverse=True):
                    pnl_val = p_info.get('pnl', 0)
                    icon = "+" if pnl_val >= 0 else ""
                    n_pos = p_info.get('positions', 0)
                    pos_str = f" | {n_pos} position{'s' if n_pos != 1 else ''}" if n_pos > 0 else ""
                    label = f"{name} | P&L: {icon}{pnl_val:.2f} EUR ({icon}{p_info.get('pnl_pct', 0):.2f}%){pos_str}"

                    with st.expander(label):
                        st.caption(p_info.get('description', ''))
                        st.caption(f"Config: {p_info.get('config_summary', '')}")

                        dc1, dc2, dc3, dc4 = st.columns(4)
                        dc1.metric("Valeur", f"{p_info.get('value', 0):,.2f} EUR")
                        dc2.metric("Realise", f"{p_info.get('realized_pnl', 0):+,.2f} EUR")
                        dc3.metric("Win Rate", f"{p_info.get('win_rate', 0):.1f}%")
                        dc4.metric("Cycles", f"{p_info.get('cycle_count', 0)}")

                        # Charger trades du sous-dossier si disponibles
                        # Trouver le bon dossier
                        portfolio_dir = None
                        for dirname in os.listdir(state_dir):
                            if dirname.startswith('portfolio_') and name.lower() in dirname.lower():
                                portfolio_dir = os.path.join(state_dir, dirname)
                                break

                        if portfolio_dir:
                            # Positions ouvertes
                            sub_state_file = os.path.join(portfolio_dir, 'auto_state.json')
                            if os.path.exists(sub_state_file):
                                try:
                                    with open(sub_state_file, 'r') as f:
                                        sub_state = json.load(f)
                                    sub_positions = sub_state.get('positions', {})
                                    if sub_positions:
                                        st.write(f"**Positions ouvertes ({len(sub_positions)})**")
                                        pos_rows = []
                                        for sym, pos in sub_positions.items():
                                            entry_price = pos.get('entry_price', 0)
                                            qty = pos.get('quantity', 0)
                                            side = pos.get('side', '')

                                            # Date d'entree
                                            entry_time_str = pos.get('entry_time', '')
                                            if entry_time_str:
                                                try:
                                                    et = datetime.fromisoformat(entry_time_str)
                                                    date_display = et.strftime('%d/%m/%Y %H:%M')
                                                except Exception:
                                                    date_display = entry_time_str[:16]
                                            else:
                                                date_display = 'N/A'

                                            # Prix actuel via yfinance
                                            current_price = None
                                            try:
                                                yahoo_sym = convert_to_yahoo_symbol(sym)
                                                data = yf.Ticker(yahoo_sym).history(period='1d', interval='1d')
                                                data.columns = [c.lower() for c in data.columns]
                                                if len(data) > 0:
                                                    current_price = float(data['close'].iloc[-1])
                                            except Exception:
                                                pass

                                            # Calcul P&L non realise
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

                                            row = {
                                                'Symbole': sym,
                                                'Side': side,
                                                'Strategie': pos.get('strategy', ''),
                                                'Date entree': date_display,
                                                'Prix entree': round(entry_price, 4),
                                                'Prix actuel': round(current_price, 4) if current_price else 'N/A',
                                                'P&L': f"{unrealized_pnl:+.2f} EUR" if unrealized_pnl is not None else 'N/A',
                                                'P&L %': f"{unrealized_pct:+.2f}%" if unrealized_pct is not None else 'N/A',
                                                'SL': round(pos.get('stop_loss', 0), 4),
                                                'TP': round(pos.get('take_profit', 0), 4),
                                            }
                                            pos_rows.append(row)
                                        st.dataframe(pd.DataFrame(pos_rows),
                                                     use_container_width=True, hide_index=True)
                                except Exception:
                                    pass

                            # Trades clos
                            sub_trades_file = os.path.join(portfolio_dir, 'auto_trades.csv')
                            if os.path.exists(sub_trades_file):
                                try:
                                    sub_trades = pd.read_csv(sub_trades_file)
                                    if len(sub_trades) > 0:
                                        # Trier par date de sortie (plus recent d'abord)
                                        if 'exit_time' in sub_trades.columns:
                                            sub_trades = sub_trades.sort_values('exit_time', ascending=False)

                                        # Resume global
                                        total_pnl_trades = sub_trades['pnl'].sum() if 'pnl' in sub_trades.columns else 0
                                        wins = len(sub_trades[sub_trades['pnl'] > 0]) if 'pnl' in sub_trades.columns else 0
                                        losses = len(sub_trades) - wins
                                        pnl_icon = "+" if total_pnl_trades >= 0 else ""
                                        st.write(
                                            f"**Trades clos ({len(sub_trades)})** - "
                                            f"P&L total: {pnl_icon}{total_pnl_trades:.2f} EUR | "
                                            f"{wins}W / {losses}L"
                                        )

                                        for _, trade in sub_trades.iterrows():
                                            t_pnl = trade.get('pnl', 0)
                                            t_pnl_pct = trade.get('pnl_pct', 0)
                                            t_icon = "+" if t_pnl >= 0 else ""
                                            t_emoji = "🟢" if t_pnl >= 0 else "🔴"
                                            t_symbol = trade.get('symbol', '?')
                                            t_side = trade.get('side', '?')
                                            t_reason = trade.get('exit_reason', '?')

                                            header = (
                                                f"{t_emoji} {t_symbol} {t_side} | "
                                                f"{t_icon}{t_pnl:.2f} EUR ({t_icon}{t_pnl_pct:.2f}%) | "
                                                f"{t_reason}"
                                            )

                                            with st.expander(header):
                                                # Dates
                                                entry_time_raw = trade.get('entry_time', '')
                                                exit_time_raw = trade.get('exit_time', '')
                                                try:
                                                    et = datetime.fromisoformat(str(entry_time_raw))
                                                    entry_fmt = et.strftime('%d/%m/%Y %H:%M')
                                                except Exception:
                                                    et = None
                                                    entry_fmt = str(entry_time_raw)[:16]
                                                try:
                                                    xt = datetime.fromisoformat(str(exit_time_raw))
                                                    exit_fmt = xt.strftime('%d/%m/%Y %H:%M')
                                                except Exception:
                                                    xt = None
                                                    exit_fmt = str(exit_time_raw)[:16]

                                                # Duree
                                                if et and xt:
                                                    duration = xt - et
                                                    days = duration.days
                                                    hours, remainder = divmod(duration.seconds, 3600)
                                                    minutes = remainder // 60
                                                    if days > 0:
                                                        duration_str = f"{days}j {hours}h {minutes}m"
                                                    else:
                                                        duration_str = f"{hours}h {minutes}m"
                                                else:
                                                    duration_str = "N/A"

                                                tc1, tc2, tc3 = st.columns(3)
                                                tc1.metric("Strategie", trade.get('strategy', 'N/A'))
                                                tc2.metric("Side", t_side)
                                                tc3.metric("Raison sortie", t_reason)

                                                tc4, tc5, tc6 = st.columns(3)
                                                tc4.metric("Date entree", entry_fmt)
                                                tc5.metric("Date sortie", exit_fmt)
                                                tc6.metric("Duree", duration_str)

                                                tc7, tc8, tc9, tc10 = st.columns(4)
                                                tc7.metric("Prix entree", _fmt_price(trade.get('entry_price', 0)))
                                                tc8.metric("Prix sortie", _fmt_price(trade.get('exit_price', 0)))
                                                tc9.metric("P&L", f"{t_icon}{t_pnl:.2f} EUR")
                                                tc10.metric("P&L %", f"{t_icon}{t_pnl_pct:.2f}%")

                                except Exception:
                                    pass

                            # Plan
                            sub_plan_file = os.path.join(portfolio_dir, 'auto_plan.json')
                            if os.path.exists(sub_plan_file):
                                try:
                                    with open(sub_plan_file, 'r') as f:
                                        sub_plan = json.load(f)
                                    assignments = sub_plan.get('assignments', [])
                                    if assignments:
                                        st.write(f"**Plan d'allocation ({len(assignments)} actifs)**")
                                        plan_rows = []
                                        for a in assignments:
                                            plan_rows.append({
                                                'Actif': a.get('asset', ''),
                                                'Strategie': a.get('strategy_name', ''),
                                                'Alloc (%)': round(a.get('allocation_pct', 0), 1),
                                                'Score': round(a.get('score', 0), 3),
                                            })
                                        st.dataframe(pd.DataFrame(plan_rows),
                                                     use_container_width=True, hide_index=True)
                                except Exception:
                                    pass

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
            macro_db = MacroEventsDatabase()
            events_df = macro_db.get_events_df()

            # ==============================================================
            # Section 1 : Contexte macro actuel
            # ==============================================================
            st.subheader("🌍 Contexte Macro Actuel")

            # Sentiment sur les 90 derniers jours
            now = datetime.now()
            d90_ago = (now - timedelta(days=90)).strftime('%Y-%m-%d')
            d30_ago = (now - timedelta(days=30)).strftime('%Y-%m-%d')
            now_str = now.strftime('%Y-%m-%d')

            sentiment_90d = macro_db.get_sentiment_score(d90_ago, now_str, asset='all')
            sentiment_30d = macro_db.get_sentiment_score(d30_ago, now_str, asset='all')

            # Déterminer couleur et label du sentiment
            def _sentiment_display(sentiment_info):
                s = sentiment_info['sentiment']
                avg = sentiment_info['avg_impact']
                labels = {
                    'very_bullish': ('Tres Haussier', '#00C853'),
                    'bullish': ('Haussier', '#66BB6A'),
                    'neutral': ('Neutre', '#9E9E9E'),
                    'bearish': ('Baissier', '#FF7043'),
                    'very_bearish': ('Tres Baissier', '#D32F2F'),
                }
                label, color = labels.get(s, ('Neutre', '#9E9E9E'))
                return label, color, avg

            label_30, color_30, avg_30 = _sentiment_display(sentiment_30d)
            label_90, color_90, avg_90 = _sentiment_display(sentiment_90d)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Sentiment 30j", label_30, f"{avg_30:+.1f}")
            col2.metric("Evenements 30j", sentiment_30d['num_events'])
            col3.metric("Sentiment 90j", label_90, f"{avg_90:+.1f}")
            col4.metric("Evenements 90j", sentiment_90d['num_events'])

            # Derniers evenements
            recent_events = events_df[events_df['date'] >= d30_ago].sort_values('date', ascending=False)
            if len(recent_events) > 0:
                st.markdown("**Derniers evenements :**")
                for _, ev in recent_events.head(5).iterrows():
                    score = ev['impact_score']
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
                    date_str = ev['date'].strftime('%d/%m/%Y') if hasattr(ev['date'], 'strftime') else str(ev['date'])[:10]
                    st.markdown(f"{icon} **{date_str}** - {ev['title']} ({ev['category']}) &nbsp; Impact: **{score:+.0f}**/10")
            else:
                st.info("Aucun evenement macro dans les 30 derniers jours.")

            # ==============================================================
            # Section 2 : Signaux Live (RSS + Fear&Greed + FRED)
            # ==============================================================
            st.markdown("---")
            st.subheader("📡 Signaux Live")

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

            # --- RSS News Headlines ---
            st.markdown("**Dernieres News (RSS - Bloomberg, Reuters, Fed, CoinDesk)**")
            try:
                from news_fetcher import RSSFeedFetcher
                rss = RSSFeedFetcher()

                # Recuperer les news de 2-3 feeds principaux (pas tous pour eviter la lenteur)
                rss_headlines = []
                for feed_name in ['fed_news', 'coindesk', 'cnbc_markets']:
                    try:
                        entries = rss.fetch_feed(feed_name, max_entries=5)
                        rss_headlines.extend(entries)
                    except Exception:
                        pass

                if rss_headlines:
                    # Trier par date (forcer tz-naive pour eviter erreur de comparaison)
                    for h in rss_headlines:
                        try:
                            parsed = pd.to_datetime(h.get('published', ''))
                            if parsed.tzinfo is not None:
                                parsed = parsed.tz_localize(None)
                            h['_parsed_date'] = parsed
                        except Exception:
                            h['_parsed_date'] = pd.Timestamp.now()
                    rss_headlines.sort(key=lambda x: x['_parsed_date'], reverse=True)

                    with st.expander(f"📰 {len(rss_headlines)} headlines recentes", expanded=False):
                        for h in rss_headlines[:15]:
                            source_tag = h.get('source', 'rss').replace('_', ' ').title()
                            date_display = h['_parsed_date'].strftime('%d/%m %H:%M') if hasattr(h['_parsed_date'], 'strftime') else ''
                            st.markdown(f"- **[{source_tag}]** {h.get('title', 'N/A')} _{date_display}_")
                else:
                    st.info("Aucune headline RSS recuperee (connexion internet requise)")
            except Exception as e:
                st.warning(f"RSS non disponible : {e}")

            # ==============================================================
            # Section 3 : Timeline des evenements macro (historiques)
            # ==============================================================
            st.markdown("---")
            st.subheader("📅 Timeline des Evenements Macro (Historique)")

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
