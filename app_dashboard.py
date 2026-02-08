"""
Dashboard Web Interactif - Streamlit
Interface de monitoring en temps réel pour 50 actifs
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import os
from typing import Dict, List

from live_data_feed import LiveDataFeed
from signal_generator import LiveSignalGenerator
from assets_config import MONITORED_ASSETS, get_all_symbols, get_asset_type, get_asset_info
from macro_data_fetcher import MacroDataFetcher
import pytz


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

def get_all_current_signals(generator: LiveSignalGenerator,
                             asset_filter: str = "Tous") -> List[Dict]:
    """
    Récupère tous les signaux actuels

    Args:
        generator: Instance LiveSignalGenerator
        asset_filter: Filtre par catégorie ('Tous', 'Crypto', 'Stocks', etc.)

    Returns:
        Liste de signaux
    """
    signals = []

    # Filtrer symboles selon catégorie
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

    # Récupérer signaux
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, symbol in enumerate(symbols):
        try:
            status_text.text(f"Chargement {symbol}... ({i+1}/{len(symbols)})")

            asset_type = get_asset_type(symbol)
            signal = generator.get_current_signal(symbol, asset_type)

            if signal:
                signals.append(signal)

            progress_bar.progress((i + 1) / len(symbols))

        except Exception as e:
            st.warning(f"Erreur pour {symbol}: {str(e)}")
            continue

    progress_bar.empty()
    status_text.empty()

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

    # Initialiser session state pour la bibliothèque backtests (évite rechargement)
    if 'backtest_library' not in st.session_state:
        st.session_state.backtest_library = None

    # Sidebar
    st.sidebar.title("⚙️ Paramètres")

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
    refresh_signals = st.sidebar.button("🔄 Rafraîchir Signaux", type="primary")

    # Détecter si la catégorie a changé (invalider le cache si oui)
    if st.session_state.last_category != selected_category:
        st.session_state.signals_loaded = False
        st.session_state.last_category = selected_category

    # Charger les signaux seulement si demandé ou si catégorie a changé
    if refresh_signals or not st.session_state.signals_loaded:
        with st.spinner("Chargement des signaux..."):
            st.session_state.all_signals = get_all_current_signals(generator, selected_category)
            st.session_state.signals_loaded = True
            st.sidebar.success("✅ Signaux chargés!")

    # Utiliser les signaux en cache
    all_signals = st.session_state.all_signals

    # ========================================================================
    # Onglets
    # ========================================================================

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Vue d'ensemble",
        "🎯 Signaux actifs",
        "📈 Analyse détaillée",
        "📜 Historique",
        "📚 Bibliothèque Backtests",
        "💼 Paper Trading"
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

        # Filtrer selon paramètres
        filtered_signals = [
            s for s in all_signals
            if s.get('confidence', 0) >= min_confidence
            and s.get('signal') in signal_filter
        ]

        st.info(f"{len(filtered_signals)} signaux correspondent aux critères")

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
    # TAB 4: Historique
    # ========================================================================

    with tab4:
        st.header("📜 Historique des signaux")

        # Charger historique
        if os.path.exists('signals_history.csv'):
            history_df = pd.read_csv('signals_history.csv')

            st.info(f"Total signaux enregistrés : {len(history_df)}")

            # Filtres
            col_f1, col_f2 = st.columns(2)

            with col_f1:
                symbols_hist = history_df['symbol'].unique().tolist()
                selected_symbol_hist = st.selectbox("Filtre par actif", ["Tous"] + symbols_hist)

            with col_f2:
                signals_hist = history_df['signal'].unique().tolist()
                selected_signal_hist = st.selectbox("Filtre par signal", ["Tous"] + signals_hist)

            # Appliquer filtres
            filtered_hist = history_df.copy()

            if selected_symbol_hist != "Tous":
                filtered_hist = filtered_hist[filtered_hist['symbol'] == selected_symbol_hist]

            if selected_signal_hist != "Tous":
                filtered_hist = filtered_hist[filtered_hist['signal'] == selected_signal_hist]

            # Afficher tableau
            st.dataframe(
                filtered_hist[['symbol', 'signal', 'confidence', 'current_price',
                               'change_24h', 'consensus', 'checked_at']].tail(50),
                use_container_width=True
            )

            # Statistiques historiques
            st.subheader("Statistiques")

            col_s1, col_s2, col_s3 = st.columns(3)

            col_s1.metric("Total BUY",
                          len(filtered_hist[filtered_hist['signal'] == 'BUY']))
            col_s2.metric("Total SELL",
                          len(filtered_hist[filtered_hist['signal'] == 'SELL']))
            col_s3.metric("Confiance Moyenne",
                          f"{filtered_hist['confidence'].mean():.1%}")

        else:
            st.warning("Aucun historique disponible. Lancez monitor_continuous.py pour commencer l'enregistrement.")

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

            else:
                st.warning("Aucune donnée de backtesting disponible.")
                st.info("Générez des données avec: python generate_demo_backtests.py")

        except Exception as e:
            st.error(f"Erreur chargement bibliothèque: {e}")

    # ========================================================================
    # TAB 6: Paper Trading
    # ========================================================================

    with tab6:
        st.header("💼 Paper Trading Portfolio")

        try:
            from paper_trading import PaperTradingPortfolio

            # Initialiser portfolio dans session state
            if 'portfolio' not in st.session_state:
                st.session_state.portfolio = PaperTradingPortfolio(
                    initial_capital=10000.0,
                    min_confidence=0.70,
                    position_size_pct=20.0,
                    max_positions=5
                )

            portfolio = st.session_state.portfolio

            # Configuration sidebar
            st.sidebar.markdown("---")
            st.sidebar.header("⚙️ Paper Trading Config")

            # Afficher configuration actuelle
            st.sidebar.info(f"""
            **Configuration Active:**
            - Capital: €{portfolio.initial_capital:,.0f}
            - Confiance: {portfolio.min_confidence:.0%}
            - Position: {portfolio.position_size_pct:.0f}%
            - Max Pos: {portfolio.max_positions}
            """)

            st.sidebar.write("**Nouvelle Configuration (appliquer avec Reset):**")

            config_capital = st.sidebar.number_input(
                "Capital Initial (€)",
                min_value=1000,
                max_value=100000,
                value=int(portfolio.initial_capital),
                step=1000
            )

            config_confidence = st.sidebar.slider(
                "Confiance Min",
                0.5, 0.9, portfolio.min_confidence, 0.05
            )

            config_position_size = st.sidebar.slider(
                "Taille Position (%)",
                10, 50, int(portfolio.position_size_pct), 5
            )

            config_max_positions = st.sidebar.slider(
                "Max Positions",
                1, 10, portfolio.max_positions
            )

            if st.sidebar.button("🔄 Reset Portfolio", help="Applique la nouvelle configuration et réinitialise le portfolio"):
                st.session_state.portfolio = PaperTradingPortfolio(
                    initial_capital=float(config_capital),
                    min_confidence=config_confidence,
                    position_size_pct=float(config_position_size),
                    max_positions=config_max_positions
                )
                st.sidebar.success(f"✅ Portfolio réinitialisé avec €{config_capital:,.0f}")
                st.rerun()

            # Récupérer prix actuels (signaux existants)
            current_prices = {}
            if len(all_signals) > 0:
                for signal in all_signals:
                    current_prices[signal['symbol']] = signal.get('current_price', 0)

            # Stats portfolio
            stats_portfolio = portfolio.get_statistics(current_prices)

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Valeur Portfolio", f"€{stats_portfolio['portfolio_value']:,.2f}")
            col2.metric("Cash Disponible", f"€{stats_portfolio['cash']:,.2f}")
            col3.metric("P&L Total", f"€{stats_portfolio['total_pnl']:,.2f}",
                        f"{stats_portfolio['total_pnl_pct']:+.2f}%")
            col4.metric("Positions Ouvertes", f"{stats_portfolio['open_positions']}/{portfolio.max_positions}")

            # Positions ouvertes
            st.subheader("🔓 Positions Ouvertes")

            if len(portfolio.positions) > 0:
                positions_data = []

                for symbol, pos in portfolio.positions.items():
                    current_price = current_prices.get(symbol, pos.entry_price)
                    pnl = pos.current_pnl(current_price)
                    pnl_pct = pos.current_pnl_pct(current_price)

                    positions_data.append({
                        'Symbol': symbol,
                        'Side': pos.side,
                        'Entry Price': f"€{pos.entry_price:,.2f}",
                        'Current Price': f"€{current_price:,.2f}",
                        'Quantity': f"{pos.quantity:.4f}",
                        'P&L': f"€{pnl:,.2f}",
                        'P&L %': f"{pnl_pct:+.2f}%",
                        'SL': f"€{pos.stop_loss:,.2f}",
                        'TP': f"€{pos.take_profit:,.2f}"
                    })

                positions_df = pd.DataFrame(positions_data)
                st.dataframe(positions_df, use_container_width=True)

            else:
                st.info("Aucune position ouverte")

            # Stats trades fermés
            if stats_portfolio['closed_trades'] > 0:
                st.subheader("📊 Performance")

                col_p1, col_p2, col_p3, col_p4 = st.columns(4)

                col_p1.metric("Trades Fermés", stats_portfolio['closed_trades'])
                col_p2.metric("Win Rate", f"{stats_portfolio['win_rate']:.1f}%")
                col_p3.metric("Gain Moyen", f"€{stats_portfolio['avg_win']:,.2f}")
                col_p4.metric("Profit Factor", f"{stats_portfolio['profit_factor']:.2f}")

                # Historique trades
                st.subheader("📜 Historique Trades")

                if os.path.exists('paper_trading_trades.csv'):
                    trades_df = pd.read_csv('paper_trading_trades.csv')

                    st.dataframe(
                        trades_df[['symbol', 'side', 'entry_price', 'exit_price',
                                   'pnl', 'pnl_pct', 'exit_reason']].tail(20),
                        use_container_width=True
                    )

            else:
                st.info("Aucun trade fermé pour le moment")

        except Exception as e:
            st.error(f"Erreur Paper Trading: {e}")

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
