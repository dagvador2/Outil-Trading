"""
Configuration des 100 actifs a monitorer
Organises par categorie avec metadonnees
"""

MONITORED_ASSETS = {
    # ============================================================================
    # CRYPTO (20 actifs)
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
        {'symbol': 'TRX/USDT', 'name': 'Tron', 'market_cap_rank': 9},
        {'symbol': 'UNI/USDT', 'name': 'Uniswap', 'market_cap_rank': 15},
        {'symbol': 'LTC/USDT', 'name': 'Litecoin', 'market_cap_rank': 16},
        {'symbol': 'ATOM/USDT', 'name': 'Cosmos', 'market_cap_rank': 18},
        {'symbol': 'ARB/USDT', 'name': 'Arbitrum', 'market_cap_rank': 20},
        {'symbol': 'OP/USDT', 'name': 'Optimism', 'market_cap_rank': 25},
        {'symbol': 'DOGE/USDT', 'name': 'Dogecoin', 'market_cap_rank': 8},
        {'symbol': 'SHIB/USDT', 'name': 'Shiba Inu', 'market_cap_rank': 12},
        {'symbol': 'NEAR/USDT', 'name': 'NEAR Protocol', 'market_cap_rank': 19},
        {'symbol': 'SUI/USDT', 'name': 'Sui', 'market_cap_rank': 17},
        {'symbol': 'ALGO/USDT', 'name': 'Algorand', 'market_cap_rank': 50},
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
    # SEMICONDUCTORS (3 actifs)
    # ============================================================================
    'semiconductors': [
        {'symbol': 'AVGO', 'name': 'Broadcom', 'sector': 'Semiconductors'},
        {'symbol': 'INTC', 'name': 'Intel', 'sector': 'Semiconductors'},
        {'symbol': 'QCOM', 'name': 'Qualcomm', 'sector': 'Semiconductors'},
    ],

    # ============================================================================
    # FINANCE (5 actifs)
    # ============================================================================
    'finance': [
        {'symbol': 'JPM', 'name': 'JPMorgan Chase', 'sector': 'Banking'},
        {'symbol': 'GS', 'name': 'Goldman Sachs', 'sector': 'Investment Banking'},
        {'symbol': 'V', 'name': 'Visa', 'sector': 'Payments'},
        {'symbol': 'MA', 'name': 'Mastercard', 'sector': 'Payments'},
        {'symbol': 'BAC', 'name': 'Bank of America', 'sector': 'Banking'},
    ],

    # ============================================================================
    # HEALTHCARE (5 actifs)
    # ============================================================================
    'healthcare': [
        {'symbol': 'UNH', 'name': 'UnitedHealth', 'sector': 'Health Insurance'},
        {'symbol': 'ABBV', 'name': 'AbbVie', 'sector': 'Pharma'},
        {'symbol': 'LLY', 'name': 'Eli Lilly', 'sector': 'Pharma'},
        {'symbol': 'MRK', 'name': 'Merck', 'sector': 'Pharma'},
        {'symbol': 'PFE', 'name': 'Pfizer', 'sector': 'Pharma'},
    ],

    # ============================================================================
    # ENERGY (5 actifs)
    # ============================================================================
    'energy': [
        {'symbol': 'XOM', 'name': 'ExxonMobil', 'sector': 'Oil & Gas'},
        {'symbol': 'CVX', 'name': 'Chevron', 'sector': 'Oil & Gas'},
        {'symbol': 'COP', 'name': 'ConocoPhillips', 'sector': 'Oil & Gas'},
        {'symbol': 'SLB', 'name': 'Schlumberger', 'sector': 'Oilfield Services'},
        {'symbol': 'EOG', 'name': 'EOG Resources', 'sector': 'Oil & Gas'},
    ],

    # ============================================================================
    # CONSUMER (5 actifs)
    # ============================================================================
    'consumer': [
        {'symbol': 'NKE', 'name': 'Nike', 'sector': 'Sportswear'},
        {'symbol': 'SBUX', 'name': 'Starbucks', 'sector': 'Coffee & Restaurants'},
        {'symbol': 'MCD', 'name': "McDonald's", 'sector': 'Fast Food'},
        {'symbol': 'HD', 'name': 'Home Depot', 'sector': 'Home Improvement'},
        {'symbol': 'COST', 'name': 'Costco', 'sector': 'Retail'},
    ],

    # ============================================================================
    # COMMODITIES (12 actifs)
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
        {'symbol': 'ZW=F', 'name': 'Wheat', 'unit': 'bushel'},
        {'symbol': 'ZS=F', 'name': 'Soybeans', 'unit': 'bushel'},
        {'symbol': 'KC=F', 'name': 'Coffee', 'unit': 'lb'},
        {'symbol': 'CT=F', 'name': 'Cotton', 'unit': 'lb'},
    ],

    # ============================================================================
    # INDICES (10 actifs)
    # ============================================================================
    'indices': [
        {'symbol': '^GSPC', 'name': 'S&P 500', 'region': 'US'},
        {'symbol': '^IXIC', 'name': 'Nasdaq', 'region': 'US'},
        {'symbol': '^DJI', 'name': 'Dow Jones', 'region': 'US'},
        {'symbol': '^FTSE', 'name': 'FTSE 100', 'region': 'UK'},
        {'symbol': '^GDAXI', 'name': 'DAX', 'region': 'Germany'},
        {'symbol': '^FCHI', 'name': 'CAC 40', 'region': 'France'},
        {'symbol': '^N225', 'name': 'Nikkei 225', 'region': 'Japan'},
        {'symbol': '^RUT', 'name': 'Russell 2000', 'region': 'US'},
        {'symbol': '^HSI', 'name': 'Hang Seng', 'region': 'Hong Kong'},
        {'symbol': '^STOXX50E', 'name': 'Euro Stoxx 50', 'region': 'Europe'},
    ],

    # ============================================================================
    # FOREX (10 actifs)
    # ============================================================================
    'forex': [
        {'symbol': 'EURUSD=X', 'name': 'EUR/USD', 'pair': 'EUR/USD'},
        {'symbol': 'GBPUSD=X', 'name': 'GBP/USD', 'pair': 'GBP/USD'},
        {'symbol': 'USDJPY=X', 'name': 'USD/JPY', 'pair': 'USD/JPY'},
        {'symbol': 'USDCHF=X', 'name': 'USD/CHF', 'pair': 'USD/CHF'},
        {'symbol': 'AUDUSD=X', 'name': 'AUD/USD', 'pair': 'AUD/USD'},
        {'symbol': 'NZDUSD=X', 'name': 'NZD/USD', 'pair': 'NZD/USD'},
        {'symbol': 'USDCAD=X', 'name': 'USD/CAD', 'pair': 'USD/CAD'},
        {'symbol': 'EURGBP=X', 'name': 'EUR/GBP', 'pair': 'EUR/GBP'},
        {'symbol': 'EURJPY=X', 'name': 'EUR/JPY', 'pair': 'EUR/JPY'},
        {'symbol': 'GBPJPY=X', 'name': 'GBP/JPY', 'pair': 'GBP/JPY'},
    ],

    # ============================================================================
    # ETF (10 actifs)
    # ============================================================================
    'etf': [
        {'symbol': 'SPY', 'name': 'SPDR S&P 500 ETF', 'tracks': 'S&P 500'},
        {'symbol': 'QQQ', 'name': 'Invesco QQQ (Nasdaq)', 'tracks': 'Nasdaq 100'},
        {'symbol': 'IWM', 'name': 'iShares Russell 2000', 'tracks': 'Small Caps US'},
        {'symbol': 'EEM', 'name': 'iShares Emerging Markets', 'tracks': 'Emerging Markets'},
        {'symbol': 'GLD', 'name': 'SPDR Gold Shares', 'tracks': 'Gold'},
        {'symbol': 'SLV', 'name': 'iShares Silver Trust', 'tracks': 'Silver'},
        {'symbol': 'TLT', 'name': 'iShares 20+ Year Treasury', 'tracks': 'US Bonds'},
        {'symbol': 'XLF', 'name': 'Financial Select SPDR', 'tracks': 'US Financials'},
        {'symbol': 'XLE', 'name': 'Energy Select SPDR', 'tracks': 'US Energy'},
        {'symbol': 'VNQ', 'name': 'Vanguard Real Estate', 'tracks': 'US REIT'},
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
    ],
}


def get_all_symbols():
    """Retourne la liste de tous les symboles a monitorer"""
    all_symbols = []
    for category, assets in MONITORED_ASSETS.items():
        for asset in assets:
            all_symbols.append(asset['symbol'])
    return all_symbols


def get_all_assets_with_metadata():
    """Retourne tous les actifs avec leurs metadonnees completes"""
    all_assets = []
    for category, assets in MONITORED_ASSETS.items():
        for asset in assets:
            asset_copy = asset.copy()
            asset_copy['category'] = category
            all_assets.append(asset_copy)
    return all_assets


def get_asset_type(symbol):
    """Determine le type d'actif pour les APIs"""
    if isinstance(symbol, dict):
        symbol = symbol['symbol']

    if 'USDT' in symbol:
        return 'crypto'
    elif '=F' in symbol or '=X' in symbol:
        return 'stock'
    elif symbol.startswith('^'):
        return 'stock'
    else:
        return 'stock'


def get_category_from_symbol(symbol):
    """Trouve la categorie d'un symbole"""
    for category, assets in MONITORED_ASSETS.items():
        for asset in assets:
            if asset['symbol'] == symbol:
                return category
    return None


def get_asset_info(symbol):
    """Recupere toutes les infos d'un actif"""
    for category, assets in MONITORED_ASSETS.items():
        for asset in assets:
            if asset['symbol'] == symbol:
                asset_info = asset.copy()
                asset_info['category'] = category
                return asset_info
    return None


def get_recommended_strategy(symbol, category=None):
    """
    Retourne la strategie recommandee basee sur les backtests

    Args:
        symbol: Symbole de l'actif
        category: Categorie (auto-detectee si None)

    Returns:
        Tuple (strategy_name, strategy_params)
    """
    if category is None:
        category = get_category_from_symbol(symbol)

    if category == 'crypto':
        if symbol == 'BTC/USDT':
            return ('Bollinger 20/2', {'period': 20, 'num_std': 2})
        elif symbol == 'ETH/USDT':
            return ('MA Crossover 10/30', {'fast': 10, 'slow': 30})
        else:
            return ('RSI 14/35/80', {'period': 14, 'oversold': 35, 'overbought': 80})

    elif category == 'tech_stocks':
        if symbol == 'NVDA':
            return ('RSI 14/35/80', {'period': 14, 'oversold': 35, 'overbought': 80})
        else:
            return ('Combined', {})

    elif category == 'commodities':
        if symbol == 'SI=F':
            return ('Event-Aware Combined', {'event_sensitivity': 1.5})
        elif symbol == 'NG=F':
            return ('Bollinger 20/2', {'period': 20, 'num_std': 2})
        else:
            return ('Combined', {})

    elif category == 'indices':
        return ('MA Crossover 20/50', {'fast': 20, 'slow': 50})

    elif category == 'forex':
        return ('Event-Aware MA 20/50', {'fast': 20, 'slow': 50, 'event_sensitivity': 1.0})

    else:
        return ('Combined', {})


def print_config_stats():
    """Affiche les statistiques de configuration"""
    total = sum(len(assets) for assets in MONITORED_ASSETS.values())
    print(f"Configuration chargee: {total} actifs")
    for category, assets in MONITORED_ASSETS.items():
        print(f"  {category}: {len(assets)} actifs")


if __name__ == '__main__':
    print_config_stats()
