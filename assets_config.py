"""
Configuration des 50 actifs à monitorer
Organisés par catégorie avec métadonnées
"""

MONITORED_ASSETS = {
    # ============================================================================
    # CRYPTO (15 actifs)
    # ============================================================================
    'crypto': [
        {'symbol': 'BTC/USDT', 'name': 'Bitcoin', 'market_cap_rank': 1},
        {'symbol': 'ETH/USDT', 'name': 'Ethereum', 'market_cap_rank': 2},
        {'symbol': 'BNB/USDT', 'name': 'Binance Coin', 'market_cap_rank': 4},
        {'symbol': 'SOL/USDT', 'name': 'Solana', 'market_cap_rank': 5},
        {'symbol': 'XRP/USDT', 'name': 'Ripple', 'market_cap_rank': 6},
        {'symbol': 'ADA/USDT', 'name': 'Cardano', 'market_cap_rank': 7},
        {'symbol': 'AVAX/USDT', 'name': 'Avalanche', 'market_cap_rank': 10},
        {'symbol': 'DOT/USDT', 'name': 'Polkadot', 'market_cap_rank': 11},
        {'symbol': 'LINK/USDT', 'name': 'Chainlink', 'market_cap_rank': 13},
        {'symbol': 'MATIC/USDT', 'name': 'Polygon', 'market_cap_rank': 14},
        {'symbol': 'UNI/USDT', 'name': 'Uniswap', 'market_cap_rank': 15},
        {'symbol': 'LTC/USDT', 'name': 'Litecoin', 'market_cap_rank': 16},
        {'symbol': 'ATOM/USDT', 'name': 'Cosmos', 'market_cap_rank': 18},
        {'symbol': 'ARB/USDT', 'name': 'Arbitrum', 'market_cap_rank': 20},
        {'symbol': 'OP/USDT', 'name': 'Optimism', 'market_cap_rank': 25},
    ],

    # ============================================================================
    # TECH STOCKS (10 actifs)
    # ============================================================================
    'tech_stocks': [
        {'symbol': 'NVDA', 'name': 'Nvidia', 'sector': 'Semiconductors'},
        {'symbol': 'AAPL', 'name': 'Apple', 'sector': 'Consumer Electronics'},
        {'symbol': 'MSFT', 'name': 'Microsoft', 'sector': 'Software'},
        {'symbol': 'GOOGL', 'name': 'Alphabet (Google)', 'sector': 'Internet'},
        {'symbol': 'AMZN', 'name': 'Amazon', 'sector': 'E-commerce'},
        {'symbol': 'META', 'name': 'Meta (Facebook)', 'sector': 'Social Media'},
        {'symbol': 'TSLA', 'name': 'Tesla', 'sector': 'Electric Vehicles'},
        {'symbol': 'AMD', 'name': 'AMD', 'sector': 'Semiconductors'},
        {'symbol': 'NFLX', 'name': 'Netflix', 'sector': 'Streaming'},
        {'symbol': 'COIN', 'name': 'Coinbase', 'sector': 'Crypto Exchange'},
    ],

    # ============================================================================
    # COMMODITIES (8 actifs)
    # ============================================================================
    'commodities': [
        {'symbol': 'GC=F', 'name': 'Gold', 'unit': 'oz'},
        {'symbol': 'SI=F', 'name': 'Silver', 'unit': 'oz'},
        {'symbol': 'CL=F', 'name': 'Crude Oil WTI', 'unit': 'barrel'},
        {'symbol': 'NG=F', 'name': 'Natural Gas', 'unit': 'MMBtu'},
        {'symbol': 'HG=F', 'name': 'Copper', 'unit': 'lb'},
        {'symbol': 'PL=F', 'name': 'Platinum', 'unit': 'oz'},
        {'symbol': 'PA=F', 'name': 'Palladium', 'unit': 'oz'},
        {'symbol': 'ZC=F', 'name': 'Corn', 'unit': 'bushel'},
    ],

    # ============================================================================
    # INDICES (7 actifs)
    # ============================================================================
    'indices': [
        {'symbol': '^GSPC', 'name': 'S&P 500', 'region': 'US'},
        {'symbol': '^IXIC', 'name': 'Nasdaq', 'region': 'US'},
        {'symbol': '^DJI', 'name': 'Dow Jones', 'region': 'US'},
        {'symbol': '^FTSE', 'name': 'FTSE 100', 'region': 'UK'},
        {'symbol': '^GDAXI', 'name': 'DAX', 'region': 'Germany'},
        {'symbol': '^FCHI', 'name': 'CAC 40', 'region': 'France'},
        {'symbol': '^N225', 'name': 'Nikkei 225', 'region': 'Japan'},
    ],

    # ============================================================================
    # FOREX (5 actifs)
    # ============================================================================
    'forex': [
        {'symbol': 'EURUSD=X', 'name': 'EUR/USD', 'pair': 'EUR/USD'},
        {'symbol': 'GBPUSD=X', 'name': 'GBP/USD', 'pair': 'GBP/USD'},
        {'symbol': 'USDJPY=X', 'name': 'USD/JPY', 'pair': 'USD/JPY'},
        {'symbol': 'USDCHF=X', 'name': 'USD/CHF', 'pair': 'USD/CHF'},
        {'symbol': 'AUDUSD=X', 'name': 'AUD/USD', 'pair': 'AUD/USD'},
    ],

    # ============================================================================
    # DEFENSIVE STOCKS (5 actifs)
    # ============================================================================
    'defensive': [
        {'symbol': 'JNJ', 'name': 'Johnson & Johnson', 'sector': 'Healthcare'},
        {'symbol': 'PG', 'name': 'Procter & Gamble', 'sector': 'Consumer Staples'},
        {'symbol': 'KO', 'name': 'Coca-Cola', 'sector': 'Beverages'},
        {'symbol': 'PEP', 'name': 'PepsiCo', 'sector': 'Beverages'},
        {'symbol': 'WMT', 'name': 'Walmart', 'sector': 'Retail'},
    ]
}


def get_all_symbols():
    """Retourne la liste de tous les symboles à monitorer (strings uniquement)"""
    all_symbols = []
    for category, assets in MONITORED_ASSETS.items():
        for asset in assets:
            all_symbols.append(asset['symbol'])
    return all_symbols


def get_all_assets_with_metadata():
    """Retourne tous les actifs avec leurs métadonnées complètes"""
    all_assets = []
    for category, assets in MONITORED_ASSETS.items():
        for asset in assets:
            asset_copy = asset.copy()
            asset_copy['category'] = category
            all_assets.append(asset_copy)
    return all_assets


def get_asset_type(symbol):
    """Détermine le type d'actif pour les APIs"""
    # Gérer les cas où symbol est un dict (pour compatibilité)
    if isinstance(symbol, dict):
        symbol = symbol['symbol']

    if 'USDT' in symbol:
        return 'crypto'
    elif '=F' in symbol or '=X' in symbol:
        return 'stock'  # Utilise même API que stocks
    elif symbol.startswith('^'):
        return 'stock'  # Indices
    else:
        return 'stock'  # Actions


def get_category_from_symbol(symbol):
    """Trouve la catégorie d'un symbole"""
    for category, assets in MONITORED_ASSETS.items():
        for asset in assets:
            if asset['symbol'] == symbol:
                return category
    return None


def get_asset_info(symbol):
    """Récupère toutes les infos d'un actif"""
    for category, assets in MONITORED_ASSETS.items():
        for asset in assets:
            if asset['symbol'] == symbol:
                asset_info = asset.copy()
                asset_info['category'] = category
                return asset_info
    return None


def get_recommended_strategy(symbol, category=None):
    """
    Retourne la stratégie recommandée basée sur les backtests

    Args:
        symbol: Symbole de l'actif
        category: Catégorie (auto-détectée si None)

    Returns:
        Tuple (strategy_name, strategy_params)
    """
    # Auto-détecter catégorie si non fournie
    if category is None:
        category = get_category_from_symbol(symbol)

    # Crypto
    if category == 'crypto':
        if symbol == 'BTC/USDT':
            return ('Bollinger 20/2', {'period': 20, 'num_std': 2})
        elif symbol == 'ETH/USDT':
            return ('MA Crossover 10/30', {'fast': 10, 'slow': 30})
        else:
            return ('RSI 14/35/80', {'period': 14, 'oversold': 35, 'overbought': 80})

    # Tech stocks
    elif category == 'tech_stocks':
        if symbol == 'NVDA':
            return ('RSI 14/35/80', {'period': 14, 'oversold': 35, 'overbought': 80})
        else:
            return ('Combined', {})

    # Commodities
    elif category == 'commodities':
        if symbol == 'SI=F':
            return ('Event-Aware Combined', {'event_sensitivity': 1.5})
        elif symbol == 'NG=F':
            return ('Bollinger 20/2', {'period': 20, 'num_std': 2})
        else:
            return ('Combined', {})

    # Indices
    elif category == 'indices':
        return ('MA Crossover 20/50', {'fast': 20, 'slow': 50})

    # Forex
    elif category == 'forex':
        return ('Event-Aware MA 20/50', {'fast': 20, 'slow': 50, 'event_sensitivity': 1.0})

    # Defensive
    else:
        return ('Combined', {})


def print_config_stats():
    """Affiche les statistiques de configuration"""
    print(f"✅ Configuration chargée:")
    print(f"   Total actifs: {sum(len(assets) for assets in MONITORED_ASSETS.values())}")
    for category, assets in MONITORED_ASSETS.items():
        print(f"   - {category}: {len(assets)} actifs")


# Afficher au chargement (optionnel)
if __name__ == '__main__':
    print_config_stats()
