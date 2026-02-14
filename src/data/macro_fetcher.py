"""
Système automatisé de récupération d'événements macroéconomiques
Sources: FRED API, Trading Economics, News APIs, Economic Calendars
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json


class MacroDataFetcher:
    """
    Récupère automatiquement les événements macro depuis diverses sources
    """

    def __init__(self):
        # Configuration des APIs (à remplir avec vos clés)
        self.fred_api_key = "480a473e9a5a6e99838252204df3cd1b"  # Federal Reserve Economic Data (gratuit)
        self.alpha_vantage_key = "0BATHZR3JLRYZ5ND"  # Alpha Vantage (gratuit)
        self.trading_economics_key = None  # Trading Economics (payant)
        self.news_api_key = "17f1a8482231420db05fcea7ce4db490"  # NewsAPI (gratuit avec limite)

        # Mapping actifs → catégories d'événements
        self.asset_categories = {
            # Natural Gas
            'NG=F': {
                'keywords': ['natural gas', 'lng', 'energy', 'pipeline', 'storage'],
                'indicators': ['NATURALGAS', 'DHHNGSP'],  # FRED codes
                'events': ['EIA Natural Gas Storage', 'Weather Forecasts', 'Russia Gas Supply']
            },

            # Crude Oil
            'CL=F': {
                'keywords': ['crude oil', 'wti', 'opec', 'petroleum', 'refinery'],
                'indicators': ['DCOILWTICO', 'WTISPLC'],
                'events': ['EIA Crude Oil Inventory', 'OPEC Meeting', 'Middle East Tensions']
            },

            # Gold
            'GC=F': {
                'keywords': ['gold', 'precious metals', 'safe haven', 'central bank reserves'],
                'indicators': ['GOLDAMGBD228NLBM'],
                'events': ['Central Bank Gold Purchases', 'USD Weakness', 'Geopolitical Risk']
            },

            # EUR/USD
            'EURUSD=X': {
                'keywords': ['ecb', 'european central bank', 'eurozone', 'interest rate'],
                'indicators': ['DEXUSEU'],
                'events': ['ECB Rate Decision', 'Eurozone CPI', 'Lagarde Speech']
            },

            # GBP/USD
            'GBPUSD=X': {
                'keywords': ['bank of england', 'boe', 'uk inflation', 'interest rate'],
                'indicators': ['DEXUSUK'],
                'events': ['BoE Rate Decision', 'UK CPI', 'Bailey Speech']
            },

            # Bitcoin
            'BTC/USDT': {
                'keywords': ['bitcoin', 'crypto', 'sec', 'etf', 'regulation', 'halving'],
                'indicators': [],
                'events': ['Bitcoin ETF', 'Halving', 'SEC Regulation', 'Institutional Adoption']
            },

            # Ethereum
            'ETH/USDT': {
                'keywords': ['ethereum', 'eth', 'defi', 'smart contracts', 'merge'],
                'indicators': [],
                'events': ['Ethereum Upgrade', 'DeFi TVL', 'Gas Fees']
            },

            # S&P 500
            '^GSPC': {
                'keywords': ['s&p 500', 'federal reserve', 'interest rates', 'inflation'],
                'indicators': ['SP500', 'FEDFUNDS'],
                'events': ['FOMC Meeting', 'CPI Release', 'Jobs Report', 'Powell Speech']
            },

            # Nvidia
            'NVDA': {
                'keywords': ['nvidia', 'ai', 'gpu', 'semiconductor', 'data center'],
                'indicators': [],
                'events': ['Nvidia Earnings', 'GTC Conference', 'AI Breakthroughs', 'Chip Demand']
            },

            # Apple
            'AAPL': {
                'keywords': ['apple', 'iphone', 'services', 'china sales'],
                'indicators': [],
                'events': ['Apple Earnings', 'iPhone Launch', 'Services Growth', 'China Risk']
            }
        }

    def get_fred_data(self, series_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Récupère des données FRED (Federal Reserve Economic Data)

        API gratuite: https://fred.stlouisfed.org/docs/api/fred/
        Exemples de séries utiles:
        - DCOILWTICO: WTI Crude Oil Price
        - GOLDAMGBD228NLBM: Gold Price
        - DEXUSEU: EUR/USD Exchange Rate
        - SP500: S&P 500 Index
        - FEDFUNDS: Federal Funds Rate
        """

        if not self.fred_api_key:
            print("⚠️  FRED API key non configurée")
            print("   → Obtenez une clé gratuite sur: https://fred.stlouisfed.org/docs/api/api_key.html")
            return pd.DataFrame()

        url = f"https://api.stlouisfed.org/fred/series/observations"
        params = {
            'series_id': series_id,
            'api_key': self.fred_api_key,
            'file_type': 'json',
            'observation_start': start_date,
            'observation_end': end_date
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if 'observations' in data:
                df = pd.DataFrame(data['observations'])
                df['date'] = pd.to_datetime(df['date'])
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                return df[['date', 'value']]
            else:
                return pd.DataFrame()

        except Exception as e:
            print(f"❌ Erreur FRED API pour {series_id}: {e}")
            return pd.DataFrame()

    def detect_significant_changes(self, df: pd.DataFrame, threshold_pct: float = 5.0) -> List[Dict]:
        """
        Détecte les changements significatifs dans une série temporelle

        Args:
            df: DataFrame avec colonnes 'date' et 'value'
            threshold_pct: Seuil de changement en % pour considérer un événement

        Returns:
            Liste d'événements détectés
        """

        if len(df) < 2:
            return []

        events = []
        df = df.sort_values('date').reset_index(drop=True)

        for i in range(1, len(df)):
            prev_value = df.loc[i-1, 'value']
            curr_value = df.loc[i, 'value']

            if pd.isna(prev_value) or pd.isna(curr_value) or prev_value == 0:
                continue

            pct_change = ((curr_value - prev_value) / abs(prev_value)) * 100

            if abs(pct_change) >= threshold_pct:
                events.append({
                    'date': df.loc[i, 'date'].strftime('%Y-%m-%d'),
                    'type': 'price_movement' if pct_change > 0 else 'price_drop',
                    'change_pct': pct_change,
                    'value': curr_value,
                    'prev_value': prev_value
                })

        return events

    def get_economic_calendar_events(self, start_date: str, end_date: str,
                                    country: str = 'US') -> List[Dict]:
        """
        Récupère les événements du calendrier économique

        Sources possibles:
        - Trading Economics API (payant mais complet)
        - Investing.com scraping (gratuit mais peut bloquer)
        - ForexFactory scraping (gratuit)
        """

        # Pour démo, voici les événements Fed typiques
        fed_events = [
            # 2024
            {'date': '2024-01-31', 'event': 'FOMC Meeting', 'impact': 'high'},
            {'date': '2024-03-20', 'event': 'FOMC Meeting', 'impact': 'high'},
            {'date': '2024-05-01', 'event': 'FOMC Meeting', 'impact': 'high'},
            {'date': '2024-06-12', 'event': 'FOMC Meeting', 'impact': 'high'},
            {'date': '2024-07-31', 'event': 'FOMC Meeting', 'impact': 'high'},
            {'date': '2024-09-18', 'event': 'FOMC Meeting + Rate Cut 50bps', 'impact': 'very_high'},
            {'date': '2024-11-07', 'event': 'FOMC Meeting + Rate Cut 25bps', 'impact': 'high'},
            {'date': '2024-12-18', 'event': 'FOMC Meeting + Rate Cut 25bps', 'impact': 'high'},

            # 2025
            {'date': '2025-01-29', 'event': 'FOMC Meeting', 'impact': 'high'},
            {'date': '2025-03-19', 'event': 'FOMC Meeting', 'impact': 'high'},
            {'date': '2025-05-07', 'event': 'FOMC Meeting', 'impact': 'high'},
            {'date': '2025-06-18', 'event': 'FOMC Meeting + Surprise Cut', 'impact': 'very_high'},
            {'date': '2025-07-30', 'event': 'FOMC Meeting', 'impact': 'high'},
            {'date': '2025-09-17', 'event': 'FOMC Meeting + Rate Cut 25bps', 'impact': 'high'},
            {'date': '2025-11-06', 'event': 'FOMC Meeting + Rate Cut 25bps', 'impact': 'high'},
            {'date': '2025-12-17', 'event': 'FOMC Meeting', 'impact': 'high'},
        ]

        # CPI Releases (important pour tous les actifs)
        cpi_events = []
        for year in [2024, 2025]:
            for month in range(1, 13):
                # CPI sort généralement le 13-15 du mois suivant
                release_date = f"{year}-{month:02d}-13"
                try:
                    date_obj = pd.to_datetime(release_date)
                    if start_date <= release_date <= end_date:
                        cpi_events.append({
                            'date': release_date,
                            'event': f'US CPI Release {year}-{month:02d}',
                            'impact': 'high'
                        })
                except:
                    pass

        # Combiner
        all_events = fed_events + cpi_events

        # Filtrer par dates
        filtered = [
            e for e in all_events
            if start_date <= e['date'] <= end_date
        ]

        return filtered

    def get_commodity_specific_events(self, commodity: str,
                                     start_date: str, end_date: str) -> List[Dict]:
        """
        Récupère les événements spécifiques aux commodités
        """

        events = []

        if commodity == 'NG=F':  # Natural Gas
            # EIA Natural Gas Storage Reports (tous les jeudis)
            current_date = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)

            while current_date <= end:
                # Storage report le jeudi
                if current_date.weekday() == 3:  # Thursday
                    events.append({
                        'date': current_date.strftime('%Y-%m-%d'),
                        'event': 'EIA Natural Gas Storage Report',
                        'impact': 'medium',
                        'category': 'energy'
                    })
                current_date += timedelta(days=1)

        elif commodity == 'CL=F':  # Crude Oil
            # EIA Petroleum Status Report (tous les mercredis)
            current_date = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)

            while current_date <= end:
                # Report le mercredi
                if current_date.weekday() == 2:  # Wednesday
                    events.append({
                        'date': current_date.strftime('%Y-%m-%d'),
                        'event': 'EIA Petroleum Status Report',
                        'impact': 'medium',
                        'category': 'energy'
                    })
                current_date += timedelta(days=1)

            # OPEC Meetings (trimestriels)
            opec_meetings = [
                {'date': '2024-06-02', 'event': 'OPEC+ Meeting', 'impact': 'very_high'},
                {'date': '2024-12-05', 'event': 'OPEC+ Meeting', 'impact': 'very_high'},
                {'date': '2025-06-01', 'event': 'OPEC+ Meeting', 'impact': 'very_high'},
                {'date': '2025-12-04', 'event': 'OPEC+ Meeting', 'impact': 'very_high'},
            ]

            events.extend([
                e for e in opec_meetings
                if start_date <= e['date'] <= end_date
            ])

        elif commodity == 'GC=F':  # Gold
            # Central Bank meetings (impact sur safe-haven)
            pass  # Déjà couvert par Fed events

        return events

    def get_crypto_specific_events(self, crypto: str,
                                   start_date: str, end_date: str) -> List[Dict]:
        """
        Récupère les événements crypto spécifiques
        """

        events = []

        if crypto == 'BTC/USDT':
            # Bitcoin-specific events
            btc_events = [
                {'date': '2024-01-10', 'event': 'Bitcoin ETF Approval (SEC)', 'impact': 'very_high'},
                {'date': '2024-04-19', 'event': 'Bitcoin Halving', 'impact': 'very_high'},
                {'date': '2025-07-15', 'event': 'Trump Bitcoin Reserve Announcement', 'impact': 'very_high'},
            ]

            events.extend([
                e for e in btc_events
                if start_date <= e['date'] <= end_date
            ])

        elif crypto == 'ETH/USDT':
            # Ethereum-specific events
            eth_events = [
                {'date': '2024-03-13', 'event': 'Dencun Upgrade (Ethereum)', 'impact': 'high'},
                {'date': '2024-05-23', 'event': 'Ethereum ETF Approval', 'impact': 'very_high'},
            ]

            events.extend([
                e for e in eth_events
                if start_date <= e['date'] <= end_date
            ])

        # Événements régulation crypto génériques
        crypto_regulation_events = [
            {'date': '2024-01-10', 'event': 'SEC Crypto Regulation Update', 'impact': 'medium'},
            {'date': '2024-06-06', 'event': 'MiCA Regulation EU Passed', 'impact': 'high'},
            {'date': '2025-02-10', 'event': 'SEC vs Crypto Exchanges Settlements', 'impact': 'medium'},
        ]

        events.extend([
            e for e in crypto_regulation_events
            if start_date <= e['date'] <= end_date
        ])

        return events

    def get_stock_specific_events(self, ticker: str,
                                  start_date: str, end_date: str) -> List[Dict]:
        """
        Récupère les événements spécifiques aux actions (earnings, annonces produits)
        """

        events = []

        # Earnings (trimestriels)
        earnings_months = {
            'NVDA': [(2, 21), (5, 22), (8, 28), (11, 20)],  # Nvidia earnings dates typiques
            'AAPL': [(2, 1), (5, 2), (8, 1), (11, 1)],      # Apple earnings
            'MSFT': [(1, 30), (4, 25), (7, 30), (10, 29)],  # Microsoft earnings
            'GOOGL': [(2, 1), (4, 25), (7, 25), (10, 29)],  # Google earnings
            'AMZN': [(2, 1), (4, 30), (7, 30), (10, 31)],   # Amazon earnings
            'TSLA': [(1, 24), (4, 19), (7, 23), (10, 18)],  # Tesla earnings
        }

        if ticker in earnings_months:
            for year in [2024, 2025]:
                for month, day in earnings_months[ticker]:
                    date_str = f"{year}-{month:02d}-{day:02d}"
                    if start_date <= date_str <= end_date:
                        events.append({
                            'date': date_str,
                            'event': f'{ticker} Q{(month-1)//3 + 1} Earnings',
                            'impact': 'high',
                            'category': 'earnings'
                        })

        # Événements spécifiques
        if ticker == 'NVDA':
            nvidia_events = [
                {'date': '2024-03-18', 'event': 'Nvidia GTC Conference 2024', 'impact': 'very_high'},
                {'date': '2025-03-17', 'event': 'Nvidia GTC Conference 2025', 'impact': 'very_high'},
            ]
            events.extend([e for e in nvidia_events if start_date <= e['date'] <= end_date])

        elif ticker == 'AAPL':
            apple_events = [
                {'date': '2024-09-09', 'event': 'iPhone 16 Launch Event', 'impact': 'high'},
                {'date': '2025-09-08', 'event': 'iPhone 17 Launch Event', 'impact': 'high'},
            ]
            events.extend([e for e in apple_events if start_date <= e['date'] <= end_date])

        return events

    def get_forex_specific_events(self, pair: str,
                                  start_date: str, end_date: str) -> List[Dict]:
        """
        Récupère les événements Forex (décisions banques centrales)
        """

        events = []

        if 'EUR' in pair:
            # ECB Events
            ecb_meetings = [
                {'date': '2024-01-25', 'event': 'ECB Rate Decision', 'impact': 'high'},
                {'date': '2024-03-07', 'event': 'ECB Rate Decision', 'impact': 'high'},
                {'date': '2024-04-11', 'event': 'ECB Rate Decision', 'impact': 'high'},
                {'date': '2024-06-06', 'event': 'ECB Rate Cut -25bps', 'impact': 'very_high'},
                {'date': '2024-07-18', 'event': 'ECB Rate Decision', 'impact': 'high'},
                {'date': '2024-09-12', 'event': 'ECB Rate Cut -25bps', 'impact': 'very_high'},
                {'date': '2024-10-17', 'event': 'ECB Rate Decision', 'impact': 'high'},
                {'date': '2024-12-12', 'event': 'ECB Rate Cut -25bps', 'impact': 'very_high'},

                {'date': '2025-01-30', 'event': 'ECB Rate Decision', 'impact': 'high'},
                {'date': '2025-03-06', 'event': 'ECB Rate Cut -25bps', 'impact': 'very_high'},
                {'date': '2025-04-17', 'event': 'ECB Rate Decision', 'impact': 'high'},
                {'date': '2025-06-05', 'event': 'ECB Rate Cut -25bps', 'impact': 'very_high'},
            ]
            events.extend([e for e in ecb_meetings if start_date <= e['date'] <= end_date])

        if 'GBP' in pair:
            # BoE Events
            boe_meetings = [
                {'date': '2024-02-01', 'event': 'BoE Rate Decision', 'impact': 'high'},
                {'date': '2024-03-21', 'event': 'BoE Rate Decision', 'impact': 'high'},
                {'date': '2024-05-09', 'event': 'BoE Rate Decision', 'impact': 'high'},
                {'date': '2024-06-20', 'event': 'BoE Rate Decision', 'impact': 'high'},
                {'date': '2024-08-01', 'event': 'BoE Rate Cut -25bps', 'impact': 'very_high'},
                {'date': '2024-09-19', 'event': 'BoE Rate Decision', 'impact': 'high'},
                {'date': '2024-11-07', 'event': 'BoE Rate Cut -25bps', 'impact': 'very_high'},
                {'date': '2024-12-19', 'event': 'BoE Rate Decision', 'impact': 'high'},

                {'date': '2025-02-06', 'event': 'BoE Rate Cut -25bps', 'impact': 'very_high'},
                {'date': '2025-03-20', 'event': 'BoE Rate Decision', 'impact': 'high'},
            ]
            events.extend([e for e in boe_meetings if start_date <= e['date'] <= end_date])

        return events

    def generate_comprehensive_events(self, asset: str,
                                     start_date: str = '2024-01-01',
                                     end_date: str = '2025-12-31') -> pd.DataFrame:
        """
        Génère une liste complète d'événements pour un actif donné

        Combine:
        - Événements globaux (Fed, CPI)
        - Événements spécifiques à l'actif
        - Données FRED si disponibles
        """

        all_events = []

        # 1. Événements globaux (Fed, CPI) - impact tous les actifs
        global_events = self.get_economic_calendar_events(start_date, end_date)
        for event in global_events:
            all_events.append({
                'date': event['date'],
                'title': event['event'],
                'category': 'Fed',
                'impact_score': 5 if event['impact'] == 'high' else 7,
                'source': 'economic_calendar',
                'asset': 'all'
            })

        # 2. Événements spécifiques selon le type d'actif
        if asset in ['NG=F', 'CL=F', 'GC=F']:
            # Commodities
            commodity_events = self.get_commodity_specific_events(asset, start_date, end_date)
            for event in commodity_events:
                impact_map = {'very_high': 8, 'high': 6, 'medium': 4}
                all_events.append({
                    'date': event['date'],
                    'title': event['event'],
                    'category': event.get('category', 'Commodities'),
                    'impact_score': impact_map.get(event['impact'], 5),
                    'source': 'commodity_calendar',
                    'asset': asset
                })

        elif 'USDT' in asset or 'USD' in asset:
            # Crypto
            crypto_events = self.get_crypto_specific_events(asset, start_date, end_date)
            for event in crypto_events:
                impact_map = {'very_high': 9, 'high': 7, 'medium': 5}
                all_events.append({
                    'date': event['date'],
                    'title': event['event'],
                    'category': 'Crypto',
                    'impact_score': impact_map.get(event['impact'], 5),
                    'source': 'crypto_calendar',
                    'asset': asset
                })

        elif asset in ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X']:
            # Forex
            forex_events = self.get_forex_specific_events(asset, start_date, end_date)
            for event in forex_events:
                impact_map = {'very_high': 8, 'high': 6, 'medium': 4}
                all_events.append({
                    'date': event['date'],
                    'title': event['event'],
                    'category': 'Central Banks',
                    'impact_score': impact_map.get(event['impact'], 5),
                    'source': 'forex_calendar',
                    'asset': asset
                })

        elif asset in ['NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']:
            # Tech Stocks
            stock_events = self.get_stock_specific_events(asset, start_date, end_date)
            for event in stock_events:
                impact_map = {'very_high': 8, 'high': 6, 'medium': 4}
                all_events.append({
                    'date': event['date'],
                    'title': event['event'],
                    'category': event.get('category', 'Earnings'),
                    'impact_score': impact_map.get(event['impact'], 5),
                    'source': 'stock_calendar',
                    'asset': asset
                })

        # Convertir en DataFrame
        df = pd.DataFrame(all_events)

        if len(df) > 0:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)

        return df


# ============================================================================
# Script principal
# ============================================================================

if __name__ == '__main__':
    fetcher = MacroDataFetcher()

    print("="*80)
    print("🌍 SYSTÈME AUTOMATISÉ D'ÉVÉNEMENTS MACROÉCONOMIQUES")
    print("="*80)

    # Test pour quelques actifs clés
    test_assets = [
        ('NG=F', 'Natural Gas'),
        ('BTC/USDT', 'Bitcoin'),
        ('NVDA', 'Nvidia'),
        ('EURUSD=X', 'EUR/USD'),
    ]

    for symbol, name in test_assets:
        print(f"\n{'='*80}")
        print(f"📊 {name} ({symbol})")
        print('='*80)

        events_df = fetcher.generate_comprehensive_events(
            symbol,
            start_date='2024-01-01',
            end_date='2025-12-31'
        )

        if len(events_df) > 0:
            print(f"✅ {len(events_df)} événements récupérés")

            # Résumé par catégorie
            print("\nPar catégorie:")
            category_counts = events_df['category'].value_counts()
            for cat, count in category_counts.items():
                avg_impact = events_df[events_df['category'] == cat]['impact_score'].mean()
                print(f"  {cat:20} : {count:3d} événements (impact moyen: {avg_impact:.1f})")

            # Top 5 événements les plus impactants
            print("\n🔥 Top 5 événements les plus impactants:")
            top_5 = events_df.nlargest(5, 'impact_score')
            for _, row in top_5.iterrows():
                print(f"  [{row['date'].strftime('%Y-%m-%d')}] {row['title']} (impact: +{row['impact_score']})")

            # Sauvegarder
            filename = f"events_{symbol.replace('/', '_')}_2024_2025.csv"
            events_df.to_csv(filename, index=False)
            print(f"\n💾 Sauvegardé: {filename}")

        else:
            print("⚠️  Aucun événement généré")

    print("\n" + "="*80)
    print("💡 PROCHAINES ÉTAPES:")
    print("="*80)
    print("1. Configurer les API keys (FRED, Alpha Vantage) pour données en temps réel")
    print("2. Implémenter scraping calendriers économiques (Investing.com, ForexFactory)")
    print("3. Ajouter détection automatique événements géopolitiques (news APIs)")
    print("4. Intégrer dans strategies_event_aware.py pour trading en direct")
    print("="*80)
