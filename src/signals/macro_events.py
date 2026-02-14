"""
Système de gestion des événements macroéconomiques
Scoring d'impact : -10 (très baissier) à +10 (très haussier)
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional

class MacroEvent:
    """Représente un événement macroéconomique"""

    def __init__(self, date: str, title: str, category: str,
                 impact_score: float, description: str = "",
                 affected_assets: List[str] = None):
        """
        Args:
            date: Date de l'événement (YYYY-MM-DD)
            title: Titre court de l'événement
            category: Catégorie (Fed, Trump, Géopolitique, Earnings, etc.)
            impact_score: Score d'impact de -10 (très baissier) à +10 (très haussier)
            description: Description détaillée
            affected_assets: Liste des actifs impactés (ex: ['crypto', 'stocks', 'gold'])
        """
        self.date = pd.to_datetime(date)
        self.title = title
        self.category = category
        self.impact_score = max(-10, min(10, impact_score))  # Clamp entre -10 et 10
        self.description = description
        self.affected_assets = affected_assets or ['all']

    def to_dict(self):
        return {
            'date': self.date,
            'title': self.title,
            'category': self.category,
            'impact_score': self.impact_score,
            'description': self.description,
            'affected_assets': ','.join(self.affected_assets)
        }


class MacroEventsDatabase:
    """Base de données des événements macroéconomiques"""

    def __init__(self):
        self.events: List[MacroEvent] = []
        self._load_2024_2025_events()

    def _load_2024_2025_events(self):
        """Charge les événements historiques 2024-2025"""

        # ============================================================================
        # 2024
        # ============================================================================

        # JANVIER 2024
        self.add_event('2024-01-10', 'Bitcoin ETF Approval', 'Crypto', +9,
                      'SEC approuve les ETF Bitcoin spot - Historique pour les cryptos',
                      ['crypto', 'BTC/USDT', 'ETH/USDT'])

        self.add_event('2024-01-31', 'Fed Holds Rates at 5.25-5.50%', 'Fed', +3,
                      'La Fed maintient les taux, signaux de pause dans les hausses',
                      ['all'])

        # FÉVRIER 2024
        self.add_event('2024-02-13', 'Hot CPI Print', 'Économie', -4,
                      'Inflation reste élevée à 3.1%, retarde les baisses de taux',
                      ['stocks', 'indices'])

        # MARS 2024
        self.add_event('2024-03-20', 'Fed Maintains Hawkish Stance', 'Fed', -3,
                      'Powell signale moins de baisses de taux en 2024',
                      ['all'])

        self.add_event('2024-03-22', 'Nvidia GTC Conference', 'Tech', +8,
                      'Nvidia annonce nouvelles GPU Blackwell - Boom de l\'IA',
                      ['stocks', 'NVDA', 'tech'])

        # AVRIL 2024
        self.add_event('2024-04-19', 'Bitcoin Halving', 'Crypto', +7,
                      'Halving du Bitcoin - Réduction de moitié des récompenses minières',
                      ['crypto', 'BTC/USDT'])

        # MAI 2024
        self.add_event('2024-05-01', 'Fed Holds Steady', 'Fed', +2,
                      'Fed maintient les taux mais reste vigilante sur l\'inflation',
                      ['all'])

        # JUIN 2024
        self.add_event('2024-06-12', 'Fed Signals Potential Cuts', 'Fed', +5,
                      'Powell signale possibles baisses de taux fin 2024',
                      ['stocks', 'indices', 'crypto'])

        # JUILLET 2024
        self.add_event('2024-07-11', 'CPI Cools to 3.0%', 'Économie', +6,
                      'Inflation ralentit significativement - Bon signe',
                      ['all'])

        self.add_event('2024-07-13', 'Trump Assassination Attempt', 'Géopolitique', -5,
                      'Tentative d\'assassinat contre Trump - Incertitude politique',
                      ['all'])

        self.add_event('2024-07-31', 'Bank of Japan Hikes Rates', 'Global', -6,
                      'BoJ hausse les taux inopinément - Choc sur les marchés',
                      ['forex', 'indices', 'stocks'])

        # AOÛT 2024
        self.add_event('2024-08-05', 'Black Monday Sell-off', 'Marchés', -8,
                      'Krach global - Nikkei -12%, S&P -3% suite décision BoJ',
                      ['all'])

        self.add_event('2024-08-23', 'Fed Signals September Cut', 'Fed', +7,
                      'Powell à Jackson Hole : Baisses de taux imminentes',
                      ['all'])

        # SEPTEMBRE 2024
        self.add_event('2024-09-18', 'Fed Cuts 50bps', 'Fed', +8,
                      'Fed coupe de 50bps - Début du cycle de baisse',
                      ['all'])

        # OCTOBRE 2024
        self.add_event('2024-10-07', 'Israel-Iran Tensions', 'Géopolitique', -7,
                      'Escalade Iran-Israël - Risque de conflit régional',
                      ['commodities', 'gold', 'oil'])

        self.add_event('2024-10-30', 'Nvidia Earnings Beat', 'Tech', +7,
                      'Nvidia explose les attentes - IA en plein boom',
                      ['stocks', 'NVDA', 'tech'])

        # NOVEMBRE 2024
        self.add_event('2024-11-05', 'Trump Wins 2024 Election', 'Politique', +6,
                      'Trump remporte l\'élection - Promesses de dérégulation',
                      ['stocks', 'crypto', 'indices'])

        self.add_event('2024-11-07', 'Fed Cuts Another 25bps', 'Fed', +5,
                      'Fed poursuit les baisses - Taux à 4.50-4.75%',
                      ['all'])

        self.add_event('2024-11-22', 'Nvidia Earnings Again', 'Tech', +8,
                      'Nvidia continue de surperformer - Guidance solide',
                      ['stocks', 'NVDA'])

        # DÉCEMBRE 2024
        self.add_event('2024-12-18', 'Fed Cuts 25bps But Hawkish', 'Fed', -4,
                      'Fed coupe mais prévoit moins de baisses en 2025 - Déception',
                      ['all'])

        # ============================================================================
        # 2025
        # ============================================================================

        # JANVIER 2025
        self.add_event('2025-01-20', 'Trump Inauguration', 'Politique', +7,
                      'Trump prend ses fonctions - Promesses pro-business',
                      ['stocks', 'crypto'])

        self.add_event('2025-01-29', 'Fed Holds Rates', 'Fed', +2,
                      'Fed en pause après 3 coupes consécutives',
                      ['all'])

        # FÉVRIER 2025
        self.add_event('2025-02-05', 'Trump Tariffs Announcement', 'Commerce', -6,
                      'Trump annonce tarifs sur Chine/Europe - Tensions commerciales',
                      ['stocks', 'indices', 'forex'])

        # MARS 2025
        self.add_event('2025-03-19', 'Fed Signals Patience', 'Fed', -2,
                      'Fed signale qu\'elle va rester en pause plus longtemps',
                      ['all'])

        # AVRIL 2025
        self.add_event('2025-04-10', 'China Retaliates on Tariffs', 'Commerce', -7,
                      'Chine riposte aux tarifs de Trump - Guerre commerciale',
                      ['stocks', 'indices', 'commodities'])

        # MAI 2025
        self.add_event('2025-05-07', 'Fed Holds, Inflation Sticky', 'Fed', -3,
                      'Inflation reste collante à 2.7% - Fed prudente',
                      ['all'])

        # JUIN 2025
        self.add_event('2025-06-18', 'Fed Cuts 25bps Surprise', 'Fed', +6,
                      'Fed coupe de manière inattendue suite ralentissement',
                      ['all'])

        # JUILLET 2025
        self.add_event('2025-07-15', 'Trump Bitcoin Reserve', 'Crypto', +9,
                      'Trump annonce réserve stratégique de Bitcoin pour les USA',
                      ['crypto', 'BTC/USDT', 'ETH/USDT'])

        # AOÛT 2025
        self.add_event('2025-08-23', 'Jackson Hole Dovish', 'Fed', +5,
                      'Powell à Jackson Hole : Fed prête à soutenir l\'économie',
                      ['all'])

        # SEPTEMBRE 2025
        self.add_event('2025-09-18', 'Fed Cuts Another 25bps', 'Fed', +5,
                      'Deuxième baisse 2025 - Cycle de détente continue',
                      ['all'])

        # OCTOBRE 2025
        self.add_event('2025-10-15', 'Middle East Conflict Escalates', 'Géopolitique', -8,
                      'Escalade majeure Moyen-Orient - Prix pétrole explose',
                      ['commodities', 'oil', 'gold'])

        # NOVEMBRE 2025
        self.add_event('2025-11-07', 'Fed Cuts 25bps', 'Fed', +4,
                      'Troisième baisse consécutive - Taux à 4.00%',
                      ['all'])

        self.add_event('2025-11-20', 'AI Breakthrough', 'Tech', +8,
                      'Percée majeure en IA - OpenAI GPT-5, nouvelles capacités',
                      ['stocks', 'tech', 'NVDA'])

        # DÉCEMBRE 2025
        self.add_event('2025-12-18', 'Fed Holds, Upbeat Outlook', 'Fed', +5,
                      'Fed en pause mais optimiste sur 2026',
                      ['all'])

    def add_event(self, date: str, title: str, category: str,
                  impact_score: float, description: str = "",
                  affected_assets: List[str] = None):
        """Ajoute un événement à la base"""
        event = MacroEvent(date, title, category, impact_score, description, affected_assets)
        self.events.append(event)

    def get_events_df(self) -> pd.DataFrame:
        """Retourne tous les événements sous forme de DataFrame"""
        return pd.DataFrame([e.to_dict() for e in self.events])

    def get_events_for_date_range(self, start_date: str, end_date: str,
                                  category: Optional[str] = None) -> List[MacroEvent]:
        """Récupère les événements dans une période donnée"""
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        filtered = [e for e in self.events if start <= e.date <= end]

        if category:
            filtered = [e for e in filtered if e.category == category]

        return filtered

    def get_impact_for_date(self, date: str, asset: str = 'all',
                           window_days: int = 7) -> float:
        """
        Calcule l'impact cumulé des événements autour d'une date

        Args:
            date: Date cible
            asset: Type d'actif ('all', 'crypto', 'stocks', etc.)
            window_days: Fenêtre de jours avant/après la date

        Returns:
            Score d'impact cumulé (peut être < -10 ou > 10 si plusieurs événements)
        """
        target_date = pd.to_datetime(date)
        start = target_date - timedelta(days=window_days)
        end = target_date + timedelta(days=window_days)

        impact = 0.0
        for event in self.events:
            if start <= event.date <= end:
                # Vérifier si l'événement affecte cet actif
                if 'all' in event.affected_assets or asset in event.affected_assets:
                    # Diminuer l'impact selon la distance temporelle
                    days_distance = abs((event.date - target_date).days)
                    decay_factor = 1.0 - (days_distance / window_days) * 0.5  # Max 50% decay
                    impact += event.impact_score * decay_factor

        return impact

    def get_sentiment_score(self, start_date: str, end_date: str,
                           asset: str = 'all') -> Dict:
        """
        Calcule le sentiment global sur une période

        Returns:
            Dict avec 'total_impact', 'avg_impact', 'num_events', 'sentiment'
        """
        events = self.get_events_for_date_range(start_date, end_date)

        # Filtrer par actif
        relevant_events = [
            e for e in events
            if 'all' in e.affected_assets or asset in e.affected_assets
        ]

        if not relevant_events:
            return {
                'total_impact': 0.0,
                'avg_impact': 0.0,
                'num_events': 0,
                'sentiment': 'neutral'
            }

        total_impact = sum(e.impact_score for e in relevant_events)
        avg_impact = total_impact / len(relevant_events)

        # Déterminer le sentiment
        if avg_impact > 3:
            sentiment = 'very_bullish'
        elif avg_impact > 1:
            sentiment = 'bullish'
        elif avg_impact > -1:
            sentiment = 'neutral'
        elif avg_impact > -3:
            sentiment = 'bearish'
        else:
            sentiment = 'very_bearish'

        return {
            'total_impact': total_impact,
            'avg_impact': avg_impact,
            'num_events': len(relevant_events),
            'sentiment': sentiment
        }

    def export_to_csv(self, filename: str = 'macro_events_2024_2025.csv'):
        """Exporte les événements en CSV"""
        df = self.get_events_df()
        df.to_csv(filename, index=False)
        print(f"✅ {len(df)} événements exportés vers {filename}")

    def print_summary(self):
        """Affiche un résumé des événements"""
        df = self.get_events_df()

        print("="*80)
        print("📊 RÉSUMÉ DES ÉVÉNEMENTS MACROÉCONOMIQUES 2024-2025")
        print("="*80)
        print(f"Total événements: {len(df)}")
        print()

        # Par catégorie
        print("Par catégorie:")
        category_counts = df['category'].value_counts()
        for cat, count in category_counts.items():
            avg_impact = df[df['category'] == cat]['impact_score'].mean()
            print(f"  {cat:15} : {count:2d} événements (impact moyen: {avg_impact:+.1f})")

        print()

        # Top 5 plus bullish
        print("🟢 Top 5 événements les plus BULLISH:")
        top_bullish = df.nlargest(5, 'impact_score')
        for _, row in top_bullish.iterrows():
            print(f"  [{row['date'].strftime('%Y-%m-%d')}] {row['title']:50} (+{row['impact_score']:.0f})")

        print()

        # Top 5 plus bearish
        print("🔴 Top 5 événements les plus BEARISH:")
        top_bearish = df.nsmallest(5, 'impact_score')
        for _, row in top_bearish.iterrows():
            print(f"  [{row['date'].strftime('%Y-%m-%d')}] {row['title']:50} ({row['impact_score']:.0f})")


if __name__ == '__main__':
    # Test du système
    db = MacroEventsDatabase()

    # Afficher résumé
    db.print_summary()

    # Exporter
    db.export_to_csv()

    # Test de calcul d'impact
    print("\n" + "="*80)
    print("📈 EXEMPLES DE CALCUL D'IMPACT")
    print("="*80)

    # Bitcoin ETF approval
    impact = db.get_impact_for_date('2024-01-10', asset='crypto', window_days=7)
    print(f"\nImpact autour Bitcoin ETF approval (2024-01-10) pour crypto: {impact:+.2f}")

    # Black Monday
    impact = db.get_impact_for_date('2024-08-05', asset='stocks', window_days=7)
    print(f"Impact autour Black Monday (2024-08-05) pour stocks: {impact:+.2f}")

    # Trump inauguration
    impact = db.get_impact_for_date('2025-01-20', asset='crypto', window_days=7)
    print(f"Impact autour Trump inauguration (2025-01-20) pour crypto: {impact:+.2f}")

    # Sentiment Q1 2024
    print("\n" + "="*80)
    print("🎭 SENTIMENT PAR PÉRIODE")
    print("="*80)

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

    for start, end, label in quarters:
        sentiment = db.get_sentiment_score(start, end, asset='all')
        print(f"\n{label}:")
        print(f"  Événements: {sentiment['num_events']}")
        print(f"  Impact total: {sentiment['total_impact']:+.1f}")
        print(f"  Impact moyen: {sentiment['avg_impact']:+.1f}")
        print(f"  Sentiment: {sentiment['sentiment'].upper()}")
