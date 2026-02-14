"""
Système d'alertes Discord via Webhook
Envoie des notifications enrichies pour signaux forts
"""

import requests
import json
from datetime import datetime
from typing import Dict, List, Optional


class DiscordAlerter:
    """
    Envoie des alertes Discord pour signaux de trading
    """

    def __init__(self, webhook_url: Optional[str] = None):
        """
        Args:
            webhook_url: URL du webhook Discord
                        Si None, lit depuis discord_webhook.txt
        """
        if webhook_url:
            self.webhook_url = webhook_url
        else:
            # Lire depuis fichier de config
            try:
                with open('discord_webhook.txt', 'r') as f:
                    self.webhook_url = f.read().strip()
            except FileNotFoundError:
                print("⚠️  Fichier discord_webhook.txt non trouvé")
                print("   Créez-le avec votre URL de webhook Discord")
                self.webhook_url = None

        self.enabled = self.webhook_url is not None

        if not self.enabled:
            print("⚠️  Alertes Discord désactivées (pas de webhook configuré)")

    def create_embed(self, signal: Dict) -> Dict:
        """
        Crée un embed Discord enrichi pour un signal

        Args:
            signal: Dictionnaire du signal

        Returns:
            Dict embed Discord
        """
        symbol = signal.get('symbol', 'N/A')
        signal_type = signal.get('signal', 'N/A')
        confidence = signal.get('confidence', 0)
        price = signal.get('current_price', 0)
        change_24h = signal.get('change_24h', 0)
        consensus = signal.get('consensus', 'N/A')
        macro_impact = signal.get('macro_impact', 0)

        # Couleur selon signal
        if signal_type == 'BUY':
            color = 0x00FF00  # Vert
            emoji = "🟢"
        elif signal_type == 'SELL':
            color = 0xFF0000  # Rouge
            emoji = "🔴"
        else:
            color = 0x808080  # Gris
            emoji = "⚪"

        # Titre
        title = f"{emoji} {signal_type} Signal - {symbol}"

        # Description
        description = f"Confiance : **{confidence:.1%}**"

        # Champs
        fields = []

        # Prix et variation
        fields.append({
            'name': '💰 Prix actuel',
            'value': f"${price:,.2f}",
            'inline': True
        })

        fields.append({
            'name': '📊 Change 24h',
            'value': f"{change_24h:+.2f}%",
            'inline': True
        })

        # Consensus stratégies
        fields.append({
            'name': '🎯 Consensus',
            'value': consensus,
            'inline': False
        })

        # Impact macro
        macro_emoji = "📈" if macro_impact > 0 else "📉" if macro_impact < 0 else "➡️"
        fields.append({
            'name': f'{macro_emoji} Impact Macro',
            'value': f"{macro_impact:+.2f}",
            'inline': True
        })

        # Indicateurs techniques
        if 'indicators' in signal and signal['indicators']:
            ind = signal['indicators']

            rsi = ind.get('rsi', 0)
            bb_position = ind.get('bb_position', 0)
            macd_histogram = ind.get('macd_histogram', 0)

            # RSI
            rsi_status = "🔥 Survente" if rsi < 35 else "❄️ Surachat" if rsi > 70 else "⚖️ Neutre"
            fields.append({
                'name': 'RSI',
                'value': f"{rsi:.1f} - {rsi_status}",
                'inline': True
            })

            # Bollinger position
            bb_status = "🔽 Bande basse" if bb_position < 20 else "🔼 Bande haute" if bb_position > 80 else "➡️ Milieu"
            fields.append({
                'name': 'Bollinger',
                'value': f"{bb_position:.0f}% - {bb_status}",
                'inline': True
            })

            # MACD
            macd_status = "📈 Positif" if macd_histogram > 0 else "📉 Négatif"
            fields.append({
                'name': 'MACD',
                'value': f"{macd_histogram:.2f} - {macd_status}",
                'inline': True
            })

        # Recommandations SL/TP (si BUY)
        if signal_type == 'BUY' and price > 0:
            # Déterminer SL/TP selon actif
            if 'BTC' in symbol or 'ETH' in symbol:
                sl_pct = 8.0
                tp_pct = 15.0
            else:
                sl_pct = 3.0
                tp_pct = 6.0

            sl_price = price * (1 - sl_pct / 100)
            tp_price = price * (1 + tp_pct / 100)

            fields.append({
                'name': '🛡️ Stop-Loss',
                'value': f"${sl_price:,.2f} (-{sl_pct}%)",
                'inline': True
            })

            fields.append({
                'name': '🎯 Take-Profit',
                'value': f"${tp_price:,.2f} (+{tp_pct}%)",
                'inline': True
            })

            risk_reward = tp_pct / sl_pct
            fields.append({
                'name': '⚖️ Risk/Reward',
                'value': f"1:{risk_reward:.1f}",
                'inline': True
            })

        # Embed
        embed = {
            'title': title,
            'description': description,
            'color': color,
            'fields': fields,
            'timestamp': datetime.utcnow().isoformat(),
            'footer': {
                'text': 'Trading Signal Generator'
            }
        }

        return embed

    def send_alert(self, signal: Dict) -> bool:
        """
        Envoie une alerte Discord pour un signal

        Args:
            signal: Dictionnaire du signal

        Returns:
            True si succès, False sinon
        """
        if not self.enabled:
            return False

        try:
            embed = self.create_embed(signal)

            payload = {
                'embeds': [embed]
            }

            response = requests.post(
                self.webhook_url,
                json=payload,
                headers={'Content-Type': 'application/json'}
            )

            if response.status_code == 204:
                print(f"✅ Alerte Discord envoyée pour {signal.get('symbol', 'N/A')}")
                return True
            else:
                print(f"❌ Erreur Discord: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            print(f"❌ Erreur envoi Discord: {e}")
            return False

    def send_batch_alerts(self, signals: List[Dict]) -> int:
        """
        Envoie plusieurs alertes en batch

        Args:
            signals: Liste de signaux

        Returns:
            Nombre d'alertes envoyées avec succès
        """
        if not self.enabled:
            print("⚠️  Alertes Discord désactivées")
            return 0

        success_count = 0

        for signal in signals:
            if self.send_alert(signal):
                success_count += 1

        print(f"\n📤 {success_count}/{len(signals)} alertes Discord envoyées")

        return success_count

    def send_summary(self, signals: List[Dict], title: str = "Résumé des signaux"):
        """
        Envoie un résumé groupé de plusieurs signaux

        Args:
            signals: Liste de signaux
            title: Titre du résumé
        """
        if not self.enabled or len(signals) == 0:
            return

        # Créer embed de résumé
        buy_count = sum(1 for s in signals if s.get('signal') == 'BUY')
        sell_count = sum(1 for s in signals if s.get('signal') == 'SELL')
        avg_confidence = sum(s.get('confidence', 0) for s in signals) / len(signals)

        description = f"**{len(signals)} signaux détectés**\n"
        description += f"🟢 {buy_count} BUY | 🔴 {sell_count} SELL\n"
        description += f"Confiance moyenne : {avg_confidence:.1%}"

        # Top signaux
        top_signals = sorted(signals, key=lambda s: s.get('confidence', 0), reverse=True)[:5]

        fields = []
        for i, signal in enumerate(top_signals, 1):
            emoji = "🟢" if signal.get('signal') == 'BUY' else "🔴"
            fields.append({
                'name': f"#{i} {emoji} {signal.get('symbol', 'N/A')}",
                'value': f"{signal.get('signal')} - {signal.get('confidence', 0):.1%} - ${signal.get('current_price', 0):,.2f}",
                'inline': False
            })

        embed = {
            'title': f"📊 {title}",
            'description': description,
            'color': 0x3498DB,  # Bleu
            'fields': fields,
            'timestamp': datetime.utcnow().isoformat(),
            'footer': {
                'text': 'Trading Signal Generator'
            }
        }

        try:
            response = requests.post(
                self.webhook_url,
                json={'embeds': [embed]},
                headers={'Content-Type': 'application/json'}
            )

            if response.status_code == 204:
                print(f"✅ Résumé Discord envoyé ({len(signals)} signaux)")
            else:
                print(f"❌ Erreur envoi résumé: {response.status_code}")

        except Exception as e:
            print(f"❌ Erreur résumé Discord: {e}")

    def test_webhook(self) -> bool:
        """
        Teste la connexion au webhook Discord

        Returns:
            True si webhook fonctionnel, False sinon
        """
        if not self.enabled:
            print("❌ Webhook non configuré")
            return False

        test_embed = {
            'title': '🧪 Test Webhook',
            'description': 'Connexion Discord OK ✅',
            'color': 0x00FF00,
            'timestamp': datetime.utcnow().isoformat(),
        }

        try:
            response = requests.post(
                self.webhook_url,
                json={'embeds': [test_embed]},
                headers={'Content-Type': 'application/json'}
            )

            if response.status_code == 204:
                print("✅ Webhook Discord fonctionnel!")
                return True
            else:
                print(f"❌ Erreur webhook: {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ Erreur test webhook: {e}")
            return False


# ============================================================================
# Configuration du webhook
# ============================================================================

def setup_webhook():
    """
    Assistant de configuration du webhook Discord
    """
    print("\n" + "="*80)
    print("🔧 CONFIGURATION WEBHOOK DISCORD")
    print("="*80)
    print("\nÉtapes pour obtenir votre webhook URL:")
    print("1. Ouvrez Discord et allez sur votre serveur")
    print("2. Clic droit sur le salon où vous voulez les alertes")
    print("3. Paramètres du salon > Intégrations > Webhooks")
    print("4. Créer un nouveau webhook")
    print("5. Copier l'URL du webhook")
    print("\n" + "="*80)

    webhook_url = input("\nCollez votre URL de webhook Discord: ").strip()

    if not webhook_url.startswith('https://discord.com/api/webhooks/'):
        print("❌ URL invalide. Elle doit commencer par https://discord.com/api/webhooks/")
        return

    # Sauvegarder dans fichier
    with open('discord_webhook.txt', 'w') as f:
        f.write(webhook_url)

    print("✅ Webhook sauvegardé dans discord_webhook.txt")

    # Tester
    print("\n🧪 Test de connexion...")
    alerter = DiscordAlerter(webhook_url)
    alerter.test_webhook()


# ============================================================================
# Script de test
# ============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Système d\'alertes Discord')
    parser.add_argument('--setup', action='store_true',
                        help='Configurer le webhook Discord')
    parser.add_argument('--test', action='store_true',
                        help='Tester le webhook')
    parser.add_argument('--demo', action='store_true',
                        help='Envoyer une alerte de démonstration')

    args = parser.parse_args()

    if args.setup:
        setup_webhook()

    elif args.test:
        alerter = DiscordAlerter()
        alerter.test_webhook()

    elif args.demo:
        # Créer alerte de démo
        alerter = DiscordAlerter()

        demo_signal = {
            'symbol': 'BTC/USDT',
            'signal': 'BUY',
            'confidence': 0.85,
            'current_price': 45678.90,
            'change_24h': 3.45,
            'consensus': '2 BUY / 0 SELL / 1 HOLD',
            'macro_impact': 2.5,
            'indicators': {
                'rsi': 32.5,
                'bb_position': 15.8,
                'macd_histogram': 0.123
            }
        }

        alerter.send_alert(demo_signal)

    else:
        print("Utilisez --setup pour configurer, --test pour tester, ou --demo pour une démo")
        print("Exemple: python discord_alerts.py --setup")
