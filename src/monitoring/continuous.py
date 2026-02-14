"""
Système de monitoring continu 24/7
Génère des signaux toutes les 5-15 minutes
Enregistre l'historique et envoie des alertes
"""

import pandas as pd
import time
from datetime import datetime
import os
from typing import Dict, List
from src.data.feed import LiveDataFeed
from src.signals.generator import LiveSignalGenerator
from src.config.assets import get_all_symbols, get_asset_type, MONITORED_ASSETS
from src.data.macro_fetcher import MacroDataFetcher
from src.db.database import TradingDatabase


class ContinuousMonitor:
    """
    Monitore en continu les signaux de trading
    """

    def __init__(self, check_interval_minutes: int = 5,
                 min_confidence: float = 0.60,
                 alert_confidence: float = 0.75):
        """
        Args:
            check_interval_minutes: Intervalle entre chaque vérification
            min_confidence: Confiance minimale pour logger un signal
            alert_confidence: Confiance minimale pour déclencher une alerte
        """
        # Configuration
        self.check_interval = check_interval_minutes * 60  # en secondes
        self.min_confidence = min_confidence
        self.alert_confidence = alert_confidence

        # Initialiser API keys
        fetcher = MacroDataFetcher()
        av_key = fetcher.alpha_vantage_key
        fred_key = fetcher.fred_api_key

        # Créer live feed et générateur de signaux
        self.feed = LiveDataFeed(av_key, fred_key)
        self.generator = LiveSignalGenerator(self.feed)

        # Database for signal logging
        self.db = TradingDatabase()

        # Historique des signaux (in-memory for current session, persisted to DB)
        self.signals_history = []

        print("\n" + "="*80)
        print("🔄 MONITORING CONTINU INITIALISÉ")
        print("="*80)
        print(f"Intervalle        : {check_interval_minutes} minutes")
        print(f"Confiance min     : {min_confidence:.0%}")
        print(f"Alerte si         : ≥ {alert_confidence:.0%}")
        print(f"Assets monitorés  : {len(get_all_symbols())}")
        print("="*80 + "\n")

    def get_all_signals(self) -> List[Dict]:
        """
        Récupère les signaux pour tous les actifs monitorés

        Returns:
            Liste de signaux avec métadonnées
        """
        all_symbols = get_all_symbols()
        signals = []

        for symbol in all_symbols:
            try:
                asset_type = get_asset_type(symbol)

                # Récupérer signal
                signal = self.generator.get_current_signal(symbol, asset_type)

                # Ajouter metadata
                signal['asset_type'] = asset_type
                signal['checked_at'] = datetime.now()

                signals.append(signal)

                # Petit délai pour éviter rate limits
                time.sleep(0.5)

            except Exception as e:
                print(f"⚠️  Erreur pour {symbol}: {e}")
                continue

        return signals

    def filter_strong_signals(self, signals: List[Dict]) -> List[Dict]:
        """
        Filtre les signaux avec confiance >= min_confidence

        Args:
            signals: Liste de tous les signaux

        Returns:
            Liste filtrée des signaux forts
        """
        return [
            s for s in signals
            if s.get('confidence', 0) >= self.min_confidence
            and s.get('signal') in ['BUY', 'SELL']
        ]

    def get_alert_signals(self, signals: List[Dict]) -> List[Dict]:
        """
        Récupère les signaux qui déclenchent une alerte

        Args:
            signals: Liste de signaux forts

        Returns:
            Liste des signaux à alerter
        """
        return [
            s for s in signals
            if s.get('confidence', 0) >= self.alert_confidence
        ]

    def log_signals(self, signals: List[Dict]):
        """
        Enregistre les signaux dans l'historique (DB + in-memory)

        Args:
            signals: Liste de signaux à enregistrer
        """
        if len(signals) == 0:
            return

        # Ajouter à l'historique en mémoire
        self.signals_history.extend(signals)

        # Persist to DB
        for signal_dict in signals:
            try:
                self.db.insert_signal(signal_dict)
            except Exception:
                pass

    def display_signals(self, signals: List[Dict], title: str = "SIGNAUX"):
        """
        Affiche les signaux dans la console

        Args:
            signals: Liste de signaux à afficher
            title: Titre de la section
        """
        if len(signals) == 0:
            return

        print("\n" + "="*80)
        print(f"📊 {title}")
        print("="*80)

        for signal in signals:
            symbol = signal.get('symbol', 'N/A')
            signal_type = signal.get('signal', 'N/A')
            confidence = signal.get('confidence', 0)
            price = signal.get('current_price', 0)
            change_24h = signal.get('change_24h', 0)
            consensus = signal.get('consensus', 'N/A')

            # Emoji selon signal
            emoji = "🟢" if signal_type == "BUY" else "🔴" if signal_type == "SELL" else "⚪"

            print(f"\n{emoji} {symbol}")
            print(f"  Signal      : {signal_type}")
            print(f"  Confiance   : {confidence:.1%}")
            print(f"  Prix actuel : ${price:,.2f}")
            print(f"  Change 24h  : {change_24h:+.2f}%")
            print(f"  Consensus   : {consensus}")

            # Indicateurs si disponibles
            if 'indicators' in signal and signal['indicators']:
                ind = signal['indicators']
                print(f"  RSI         : {ind.get('rsi', 0):.1f}")
                print(f"  BB Position : {ind.get('bb_position', 0):.1f}%")

        print("\n" + "="*80)

    def run_single_check(self):
        """
        Effectue une vérification unique de tous les actifs
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"\n🔍 Vérification à {timestamp}")
        print("-"*80)

        # Récupérer tous les signaux
        all_signals = self.get_all_signals()

        print(f"✅ {len(all_signals)} actifs vérifiés")

        # Filtrer signaux forts
        strong_signals = self.filter_strong_signals(all_signals)

        if len(strong_signals) > 0:
            print(f"💪 {len(strong_signals)} signaux forts (confiance ≥ {self.min_confidence:.0%})")

            # Afficher signaux forts
            self.display_signals(strong_signals, f"SIGNAUX FORTS (≥ {self.min_confidence:.0%})")

            # Logger
            self.log_signals(strong_signals)

            # Vérifier alertes
            alert_signals = self.get_alert_signals(strong_signals)

            if len(alert_signals) > 0:
                print(f"\n🚨 {len(alert_signals)} ALERTES à déclencher (confiance ≥ {self.alert_confidence:.0%})")

                # Envoyer alertes Discord (sera implémenté ensuite)
                try:
                    from src.monitoring.discord import DiscordAlerter
                    alerter = DiscordAlerter()
                    alerter.send_batch_alerts(alert_signals)
                except ImportError:
                    print("⚠️  Module discord_alerts non disponible")
                except Exception as e:
                    print(f"⚠️  Erreur alertes Discord: {e}")
        else:
            print("ℹ️  Aucun signal fort pour le moment")

        print(f"\n⏰ Prochaine vérification dans {self.check_interval // 60} minutes")

    def run_continuous(self):
        """
        Lance le monitoring en continu (boucle infinie)
        """
        print("\n🚀 Démarrage du monitoring continu...")
        print("   (Ctrl+C pour arrêter)\n")

        iteration = 0

        try:
            while True:
                iteration += 1
                print(f"\n{'='*80}")
                print(f"ITÉRATION #{iteration}")
                print(f"{'='*80}")

                # Effectuer vérification
                self.run_single_check()

                # Attendre intervalle
                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            print("\n\n🛑 Monitoring arrêté par l'utilisateur")
            print(f"📊 Total signaux enregistrés : {len(self.signals_history)}")
            print(f"💾 Historique sauvegardé dans {self.signals_log_file}")

    def get_summary_stats(self) -> Dict:
        """
        Statistiques sur l'historique des signaux

        Returns:
            Dict avec statistiques
        """
        if len(self.signals_history) == 0:
            return {}

        df = pd.DataFrame(self.signals_history)

        stats = {
            'total_signals': len(df),
            'buy_signals': len(df[df['signal'] == 'BUY']),
            'sell_signals': len(df[df['signal'] == 'SELL']),
            'avg_confidence': df['confidence'].mean(),
            'top_assets': df['symbol'].value_counts().head(5).to_dict(),
        }

        return stats

    def display_summary(self):
        """
        Affiche un résumé des statistiques
        """
        stats = self.get_summary_stats()

        if not stats:
            print("⚠️  Pas encore de données historiques")
            return

        print("\n" + "="*80)
        print("📈 STATISTIQUES MONITORING")
        print("="*80)
        print(f"Total signaux      : {stats['total_signals']}")
        print(f"  BUY              : {stats['buy_signals']}")
        print(f"  SELL             : {stats['sell_signals']}")
        print(f"Confiance moyenne  : {stats['avg_confidence']:.1%}")
        print("\nTop 5 actifs les plus signalés:")
        for symbol, count in stats['top_assets'].items():
            print(f"  {symbol:15} : {count} signaux")
        print("="*80 + "\n")


# ============================================================================
# Script principal
# ============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Monitoring continu de signaux de trading')
    parser.add_argument('--interval', type=int, default=5,
                        help='Intervalle entre vérifications (minutes, défaut: 5)')
    parser.add_argument('--min-confidence', type=float, default=0.60,
                        help='Confiance minimale pour logger (défaut: 0.60)')
    parser.add_argument('--alert-confidence', type=float, default=0.75,
                        help='Confiance minimale pour alerter (défaut: 0.75)')
    parser.add_argument('--single', action='store_true',
                        help='Effectuer une seule vérification puis arrêter')
    parser.add_argument('--stats', action='store_true',
                        help='Afficher les statistiques historiques')

    args = parser.parse_args()

    # Créer monitor
    monitor = ContinuousMonitor(
        check_interval_minutes=args.interval,
        min_confidence=args.min_confidence,
        alert_confidence=args.alert_confidence
    )

    # Mode stats
    if args.stats:
        monitor.display_summary()
    # Mode single check
    elif args.single:
        monitor.run_single_check()
        monitor.display_summary()
    # Mode continu
    else:
        monitor.run_continuous()
