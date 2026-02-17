"""
Configuration des 200 actifs a monitorer
Organises par categorie avec metadonnees
100 actifs originaux + 100 nouveaux (fort accent europeen)
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
    # CRYPTO SUPPLEMENTAIRES (10 actifs) [NOUVEAU]
    # ============================================================================
    'crypto_extra': [
        {'symbol': 'MATIC/USDT', 'name': 'Polygon', 'market_cap_rank': 14},
        {'symbol': 'FTM/USDT', 'name': 'Fantom', 'market_cap_rank': 55},
        {'symbol': 'AAVE/USDT', 'name': 'Aave', 'market_cap_rank': 35},
        {'symbol': 'MKR/USDT', 'name': 'Maker', 'market_cap_rank': 40},
        {'symbol': 'CRV/USDT', 'name': 'Curve DAO', 'market_cap_rank': 80},
        {'symbol': 'APE/USDT', 'name': 'ApeCoin', 'market_cap_rank': 90},
        {'symbol': 'FIL/USDT', 'name': 'Filecoin', 'market_cap_rank': 30},
        {'symbol': 'SAND/USDT', 'name': 'The Sandbox', 'market_cap_rank': 60},
        {'symbol': 'MANA/USDT', 'name': 'Decentraland', 'market_cap_rank': 65},
        {'symbol': 'AXS/USDT', 'name': 'Axie Infinity', 'market_cap_rank': 70},
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
    # TECH & GROWTH US SUPPLEMENTAIRES (15 actifs) [NOUVEAU]
    # ============================================================================
    'tech_growth_extra': [
        {'symbol': 'CRM', 'name': 'Salesforce', 'sector': 'Cloud/CRM'},
        {'symbol': 'ORCL', 'name': 'Oracle', 'sector': 'Enterprise Software'},
        {'symbol': 'IBM', 'name': 'IBM', 'sector': 'IT Services'},
        {'symbol': 'DIS', 'name': 'Walt Disney', 'sector': 'Entertainment'},
        {'symbol': 'UBER', 'name': 'Uber Technologies', 'sector': 'Mobility'},
        {'symbol': 'SQ', 'name': 'Block (Square)', 'sector': 'Fintech'},
        {'symbol': 'PYPL', 'name': 'PayPal', 'sector': 'Fintech'},
        {'symbol': 'PLTR', 'name': 'Palantir Technologies', 'sector': 'Data Analytics'},
        {'symbol': 'RIVN', 'name': 'Rivian Automotive', 'sector': 'Electric Vehicles'},
        {'symbol': 'BABA', 'name': 'Alibaba Group', 'sector': 'E-commerce China'},
        {'symbol': 'SHOP', 'name': 'Shopify', 'sector': 'E-commerce SaaS'},
        {'symbol': 'SNAP', 'name': 'Snap Inc', 'sector': 'Social Media'},
        {'symbol': 'ROKU', 'name': 'Roku', 'sector': 'Streaming Tech'},
        {'symbol': 'ZM', 'name': 'Zoom Video', 'sector': 'Video Communications'},
        {'symbol': 'ABNB', 'name': 'Airbnb', 'sector': 'Travel Tech'},
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
    # COMMODITIES SUPPLEMENTAIRES (5 actifs) [NOUVEAU]
    # ============================================================================
    'commodities_extra': [
        {'symbol': 'SB=F', 'name': 'Sugar', 'unit': 'lb'},
        {'symbol': 'CC=F', 'name': 'Cocoa', 'unit': 'ton'},
        {'symbol': 'OJ=F', 'name': 'Orange Juice', 'unit': 'lb'},
        {'symbol': 'LBS=F', 'name': 'Lumber', 'unit': 'board ft'},
        {'symbol': 'LE=F', 'name': 'Live Cattle', 'unit': 'lb'},
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
    # INDICES EUROPEENS & ASIATIQUES SUPPLEMENTAIRES (5 actifs) [NOUVEAU]
    # ============================================================================
    'indices_extra': [
        {'symbol': '^IBEX', 'name': 'IBEX 35', 'region': 'Spain'},
        {'symbol': '^AEX', 'name': 'AEX', 'region': 'Netherlands'},
        {'symbol': '^SSMI', 'name': 'SMI', 'region': 'Switzerland'},
        {'symbol': '^KS11', 'name': 'KOSPI', 'region': 'South Korea'},
        {'symbol': '^BSESN', 'name': 'BSE Sensex', 'region': 'India'},
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
    # FOREX SUPPLEMENTAIRES (5 actifs) [NOUVEAU]
    # ============================================================================
    'forex_extra': [
        {'symbol': 'EURCHF=X', 'name': 'EUR/CHF', 'pair': 'EUR/CHF'},
        {'symbol': 'EURAUD=X', 'name': 'EUR/AUD', 'pair': 'EUR/AUD'},
        {'symbol': 'GBPCHF=X', 'name': 'GBP/CHF', 'pair': 'GBP/CHF'},
        {'symbol': 'USDNOK=X', 'name': 'USD/NOK', 'pair': 'USD/NOK'},
        {'symbol': 'USDSEK=X', 'name': 'USD/SEK', 'pair': 'USD/SEK'},
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
    # ETF EUROPEENS (10 actifs) [NOUVEAU]
    # ============================================================================
    'etf_europe': [
        {'symbol': 'EWG', 'name': 'iShares MSCI Germany', 'tracks': 'Germany Large Cap'},
        {'symbol': 'EWU', 'name': 'iShares MSCI United Kingdom', 'tracks': 'UK Large Cap'},
        {'symbol': 'EWQ', 'name': 'iShares MSCI France', 'tracks': 'France Large Cap'},
        {'symbol': 'EWP', 'name': 'iShares MSCI Spain', 'tracks': 'Spain Large Cap'},
        {'symbol': 'EWI', 'name': 'iShares MSCI Italy', 'tracks': 'Italy Large Cap'},
        {'symbol': 'EWL', 'name': 'iShares MSCI Switzerland', 'tracks': 'Switzerland Large Cap'},
        {'symbol': 'EWN', 'name': 'iShares MSCI Netherlands', 'tracks': 'Netherlands Large Cap'},
        {'symbol': 'IEUR', 'name': 'iShares Core MSCI Europe', 'tracks': 'Europe Large Cap'},
        {'symbol': 'VGK', 'name': 'Vanguard FTSE Europe', 'tracks': 'Europe All Cap'},
        {'symbol': 'EZU', 'name': 'iShares MSCI Eurozone', 'tracks': 'Eurozone Large Cap'},
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

    # ============================================================================
    # ACTIONS FRANCAISES - CAC 40 (10 actifs) [NOUVEAU]
    # ============================================================================
    'france_cac40': [
        {'symbol': 'MC.PA', 'name': 'LVMH', 'sector': 'Luxury', 'region': 'France'},
        {'symbol': 'TTE.PA', 'name': 'TotalEnergies', 'sector': 'Energy', 'region': 'France'},
        {'symbol': 'SAN.PA', 'name': 'Sanofi', 'sector': 'Pharma', 'region': 'France'},
        {'symbol': 'AI.PA', 'name': 'Air Liquide', 'sector': 'Chemicals', 'region': 'France'},
        {'symbol': 'SU.PA', 'name': 'Schneider Electric', 'sector': 'Electrical Equipment', 'region': 'France'},
        {'symbol': 'OR.PA', 'name': "L'Oreal", 'sector': 'Cosmetics', 'region': 'France'},
        {'symbol': 'RMS.PA', 'name': 'Hermes', 'sector': 'Luxury', 'region': 'France'},
        {'symbol': 'DSY.PA', 'name': 'Dassault Systemes', 'sector': 'Software', 'region': 'France'},
        {'symbol': 'BNP.PA', 'name': 'BNP Paribas', 'sector': 'Banking', 'region': 'France'},
        {'symbol': 'CS.PA', 'name': 'AXA', 'sector': 'Insurance', 'region': 'France'},
    ],

    # ============================================================================
    # ACTIONS ALLEMANDES - DAX (10 actifs) [NOUVEAU]
    # ============================================================================
    'germany_dax': [
        {'symbol': 'SAP.DE', 'name': 'SAP', 'sector': 'Enterprise Software', 'region': 'Germany'},
        {'symbol': 'SIE.DE', 'name': 'Siemens', 'sector': 'Industrial Conglomerate', 'region': 'Germany'},
        {'symbol': 'ALV.DE', 'name': 'Allianz', 'sector': 'Insurance', 'region': 'Germany'},
        {'symbol': 'BAS.DE', 'name': 'BASF', 'sector': 'Chemicals', 'region': 'Germany'},
        {'symbol': 'DTE.DE', 'name': 'Deutsche Telekom', 'sector': 'Telecom', 'region': 'Germany'},
        {'symbol': 'BMW.DE', 'name': 'BMW', 'sector': 'Automotive', 'region': 'Germany'},
        {'symbol': 'ADS.DE', 'name': 'Adidas', 'sector': 'Sportswear', 'region': 'Germany'},
        {'symbol': 'MBG.DE', 'name': 'Mercedes-Benz Group', 'sector': 'Automotive', 'region': 'Germany'},
        {'symbol': 'IFX.DE', 'name': 'Infineon Technologies', 'sector': 'Semiconductors', 'region': 'Germany'},
        {'symbol': 'BAYN.DE', 'name': 'Bayer', 'sector': 'Pharma/Chemicals', 'region': 'Germany'},
    ],

    # ============================================================================
    # ACTIONS BRITANNIQUES - FTSE (10 actifs) [NOUVEAU]
    # ============================================================================
    'uk_ftse': [
        {'symbol': 'SHEL.L', 'name': 'Shell', 'sector': 'Oil & Gas', 'region': 'UK'},
        {'symbol': 'AZN.L', 'name': 'AstraZeneca', 'sector': 'Pharma', 'region': 'UK'},
        {'symbol': 'HSBA.L', 'name': 'HSBC Holdings', 'sector': 'Banking', 'region': 'UK'},
        {'symbol': 'ULVR.L', 'name': 'Unilever', 'sector': 'Consumer Goods', 'region': 'UK'},
        {'symbol': 'RIO.L', 'name': 'Rio Tinto', 'sector': 'Mining', 'region': 'UK'},
        {'symbol': 'BP.L', 'name': 'BP', 'sector': 'Oil & Gas', 'region': 'UK'},
        {'symbol': 'GSK.L', 'name': 'GSK', 'sector': 'Pharma', 'region': 'UK'},
        {'symbol': 'DGE.L', 'name': 'Diageo', 'sector': 'Beverages', 'region': 'UK'},
        {'symbol': 'BARC.L', 'name': 'Barclays', 'sector': 'Banking', 'region': 'UK'},
        {'symbol': 'RR.L', 'name': 'Rolls-Royce', 'sector': 'Aerospace & Defense', 'region': 'UK'},
    ],

    # ============================================================================
    # ACTIONS EUROPE NORD & BENELUX (10 actifs) [NOUVEAU]
    # ============================================================================
    'europe_north_benelux': [
        {'symbol': 'ASML.AS', 'name': 'ASML Holding', 'sector': 'Semiconductors', 'region': 'Netherlands'},
        {'symbol': 'PHIA.AS', 'name': 'Koninklijke Philips', 'sector': 'Health Tech', 'region': 'Netherlands'},
        {'symbol': 'INGA.AS', 'name': 'ING Group', 'sector': 'Banking', 'region': 'Netherlands'},
        {'symbol': 'NESN.SW', 'name': 'Nestle', 'sector': 'Food & Beverage', 'region': 'Switzerland'},
        {'symbol': 'NOVN.SW', 'name': 'Novartis', 'sector': 'Pharma', 'region': 'Switzerland'},
        {'symbol': 'ROG.SW', 'name': 'Roche', 'sector': 'Pharma', 'region': 'Switzerland'},
        {'symbol': 'UBSG.SW', 'name': 'UBS Group', 'sector': 'Banking', 'region': 'Switzerland'},
        {'symbol': 'ITX.MC', 'name': 'Inditex (Zara)', 'sector': 'Retail/Fashion', 'region': 'Spain'},
        {'symbol': 'SAN.MC', 'name': 'Banco Santander', 'sector': 'Banking', 'region': 'Spain'},
        {'symbol': 'IBE.MC', 'name': 'Iberdrola', 'sector': 'Utilities/Renewables', 'region': 'Spain'},
    ],

    # ============================================================================
    # ACTIONS ASIATIQUES (10 actifs) [NOUVEAU]
    # ============================================================================
    'asia_stocks': [
        {'symbol': '7203.T', 'name': 'Toyota Motor', 'sector': 'Automotive', 'region': 'Japan'},
        {'symbol': '6758.T', 'name': 'Sony Group', 'sector': 'Electronics/Entertainment', 'region': 'Japan'},
        {'symbol': '9984.T', 'name': 'SoftBank Group', 'sector': 'Investment/Tech', 'region': 'Japan'},
        {'symbol': '6861.T', 'name': 'Keyence', 'sector': 'Sensors/Automation', 'region': 'Japan'},
        {'symbol': '8306.T', 'name': 'Mitsubishi UFJ Financial', 'sector': 'Banking', 'region': 'Japan'},
        {'symbol': '005930.KS', 'name': 'Samsung Electronics', 'sector': 'Electronics', 'region': 'South Korea'},
        {'symbol': '9988.HK', 'name': 'Alibaba Group (HK)', 'sector': 'E-commerce', 'region': 'Hong Kong'},
        {'symbol': '0700.HK', 'name': 'Tencent Holdings', 'sector': 'Tech/Gaming', 'region': 'Hong Kong'},
        {'symbol': '2330.TW', 'name': 'Taiwan Semiconductor (TSMC)', 'sector': 'Semiconductors', 'region': 'Taiwan'},
        {'symbol': 'RELIANCE.NS', 'name': 'Reliance Industries', 'sector': 'Conglomerate', 'region': 'India'},
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

    if category == 'crypto' or category == 'crypto_extra':
        if symbol == 'BTC/USDT':
            return ('Bollinger 20/2', {'period': 20, 'num_std': 2})
        elif symbol == 'ETH/USDT':
            return ('MA Crossover 10/30', {'fast': 10, 'slow': 30})
        else:
            return ('RSI 14/35/80', {'period': 14, 'oversold': 35, 'overbought': 80})

    elif category in ('tech_stocks', 'tech_growth_extra'):
        if symbol == 'NVDA':
            return ('RSI 14/35/80', {'period': 14, 'oversold': 35, 'overbought': 80})
        else:
            return ('Combined', {})

    elif category in ('commodities', 'commodities_extra'):
        if symbol == 'SI=F':
            return ('Event-Aware Combined', {'event_sensitivity': 1.5})
        elif symbol == 'NG=F':
            return ('Bollinger 20/2', {'period': 20, 'num_std': 2})
        else:
            return ('Combined', {})

    elif category in ('indices', 'indices_extra'):
        return ('MA Crossover 20/50', {'fast': 20, 'slow': 50})

    elif category in ('forex', 'forex_extra'):
        return ('Event-Aware MA 20/50', {'fast': 20, 'slow': 50, 'event_sensitivity': 1.0})

    elif category in ('france_cac40', 'germany_dax', 'uk_ftse', 'europe_north_benelux'):
        return ('Ichimoku 9/26/52', {'tenkan': 9, 'kijun': 26, 'senkou_b': 52})

    elif category == 'asia_stocks':
        return ('ADX Trend 14/25', {'adx_period': 14, 'adx_threshold': 25})

    elif category in ('etf', 'etf_europe'):
        return ('MA Crossover 20/50', {'fast': 20, 'slow': 50})

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
