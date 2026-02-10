# 🚀 COMMENCER ICI

## ⚡ DÉMARRAGE RAPIDE (Crypto Sans Limites)

**Si vous avez 36 actifs sur 50 qui retournent NO_DATA :**

✅ **SOLUTION IMMÉDIATE : Mode Crypto**

```bash
./start_crypto_monitoring.sh
```

ou pour le dashboard :

```bash
./start_dashboard.sh
# Puis sélectionnez "Crypto" dans la sidebar
```

→ **15 cryptos fonctionnels**, données temps réel, **AUCUN rate limit**

📖 **Explication complète** : Voir [SOLUTIONS_NO_DATA.md](SOLUTIONS_NO_DATA.md)

---

## ✅ Bugs Corrigés

**Bug 1** : `'dict' object has no attribute 'startswith'` ✅ Corrigé
**Bug 2** : 36 actifs retournent NO_DATA → **Solution : Mode Crypto ci-dessus**

---

## 🎯 Lancement Options

### Option 1 : Dashboard Web (Interface visuelle)

**Double-cliquez sur** : [`start_dashboard.sh`](start_dashboard.sh)

ou dans le terminal :
```bash
./start_dashboard.sh
```

→ Le dashboard s'ouvre automatiquement dans votre navigateur

### Option 2 : Monitoring Automatique (Arrière-plan)

**Double-cliquez sur** : [`start_monitoring.sh`](start_monitoring.sh)

ou dans le terminal :
```bash
./start_monitoring.sh
```

→ Vérification automatique toutes les 5 minutes + alertes Discord

### Option 3 : Monitoring Crypto Seulement (✅ Recommandé)

**Double-cliquez sur** : [`start_crypto_monitoring.sh`](start_crypto_monitoring.sh)

ou dans le terminal :
```bash
./start_crypto_monitoring.sh
```

→ Monitoring de 15 cryptos (Binance API - Sans limites)

### Option 4 : Vérification Rapide (Test unique)

**Double-cliquez sur** : [`quick_check.sh`](quick_check.sh)

ou dans le terminal :
```bash
./quick_check.sh
```

→ Check unique de tous les actifs (~2-3 minutes)

---

## 📱 Configurer Discord (Optionnel mais Recommandé)

Si vous voulez recevoir des alertes Discord :

```bash
source venv/bin/activate
python3 discord_alerts.py --setup
```

Suivez les instructions pour coller votre webhook URL.

**Tester** :
```bash
source venv/bin/activate
python3 discord_alerts.py --test
```

---

## 🔧 Commandes Utiles

### Lancement Manuel

```bash
# Activer l'environnement
source venv/bin/activate

# Dashboard Web
streamlit run app_dashboard.py

# Monitoring continu
python3 monitor_continuous.py

# Check unique
python3 monitor_continuous.py --single

# Voir les stats historiques
python3 monitor_continuous.py --stats
```

### Configuration Avancée

```bash
# Intervalle de 10 minutes au lieu de 5
python3 monitor_continuous.py --interval 10

# Confiance minimale 70% pour alerts
python3 monitor_continuous.py --alert-confidence 0.70

# Confiance minimale 65% pour logger
python3 monitor_continuous.py --min-confidence 0.65
```

---

## 📊 Utilisation du Dashboard

Une fois lancé (`./start_dashboard.sh`) :

1. **Sidebar (gauche)** :
   - Sélectionnez une catégorie (ex: "Crypto")
   - Ajustez la confiance minimale (slider)
   - Activez "Auto-refresh" pour actualisation toutes les 30s

2. **4 Onglets** :
   - **Vue d'ensemble** : Métriques globales + Top 10
   - **Signaux actifs** : Liste filtrée avec détails par actif
   - **Analyse détaillée** : Graphiques et indicateurs pour 1 actif
   - **Historique** : Tous les signaux enregistrés

3. **Interpréter un signal** :
   - 🟢 **BUY** + confiance ≥ 70% = Signal fort d'achat
   - 🔴 **SELL** + confiance ≥ 70% = Signal fort de vente
   - Prix, SL, TP affichés automatiquement

---

## 🎯 Workflows Recommandés

### Pour Day Trading / Scalping

```bash
# Terminal 1 : Monitoring ultra-réactif
source venv/bin/activate
python3 monitor_continuous.py --interval 2 --alert-confidence 0.80

# Terminal 2 : Dashboard avec auto-refresh
./start_dashboard.sh
```

Puis dans le dashboard :
- Catégorie : Crypto
- Auto-refresh : ON
- Confiance min : 75%

### Pour Swing Trading

```bash
# Monitoring modéré (15 min)
source venv/bin/activate
python3 monitor_continuous.py --interval 15 --alert-confidence 0.70
```

Consultez le dashboard ponctuellement pour confirmer les signaux.

### Pour Trading Positionnel

```bash
# Check quotidien le matin
./quick_check.sh
```

Regardez les signaux avec confiance ≥ 65% et analysez dans le dashboard.

---

## 📁 Fichiers Importants

| Fichier | Description |
|---------|-------------|
| `start_dashboard.sh` | Lance le dashboard web |
| `start_monitoring.sh` | Lance le monitoring 24/7 |
| `quick_check.sh` | Vérification unique rapide |
| `assets_config.py` | Configuration des 50 actifs |
| `strategies_extended.py` | 10 stratégies disponibles |
| `discord_alerts.py` | Système d'alertes Discord |
| `signals_history.csv` | Historique des signaux (auto-généré) |

---

## 📚 Documentation Complète

- **QUICKSTART_LIVE.md** : Guide détaillé d'utilisation (15 pages)
- **MONITORING_GUIDE.md** : Guide complet du monitoring (20 pages)
- **EXTENSION_COMPLETE.md** : Résumé de tout ce qui a été livré
- **INSTALL.md** : Guide d'installation des dépendances

---

## 🐛 Problème ?

**Le script ne se lance pas** :
```bash
chmod +x start_dashboard.sh
./start_dashboard.sh
```

**Erreur "venv not found"** :
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Erreur "module not found"** :
```bash
source venv/bin/activate
pip install -r requirements.txt
```

---

## ✅ Test Rapide (30 secondes)

Vérifiez que tout fonctionne :

```bash
source venv/bin/activate
python3 -c "
from assets_config import get_all_symbols
print(f'✅ {len(get_all_symbols())} actifs configurés')
print('🎯 Système prêt!')
"
```

Si vous voyez "✅ 50 actifs configurés" → **Tout est OK !**

---

## 🚀 Prochaines Étapes

1. ✅ Lancez le dashboard : `./start_dashboard.sh`
2. ✅ Configurez Discord : `python3 discord_alerts.py --setup`
3. ✅ Lancez le monitoring : `./start_monitoring.sh`
4. ✅ Commencez à recevoir des signaux !

**Questions ?** Consultez [QUICKSTART_LIVE.md](QUICKSTART_LIVE.md) pour le guide complet.

---

**Bon trading ! 📊🚀**
