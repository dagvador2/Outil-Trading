# Guide de Déploiement - Intégration Signaux Macro

## 📋 Résumé des Modifications

### Nouveaux Fichiers Créés

1. **[news_fetcher.py](news_fetcher.py)** - Module de récupération de signaux macro (850 lignes)
2. **[macro_signal_scorer.py](macro_signal_scorer.py)** - Système de scoring composite (650 lignes)
3. **[macro_integration.py](macro_integration.py)** - Filtre macro pour paper trading (400 lignes)
4. **[.env](.env)** - Configuration API avec clé FRED
5. **[requirements_macro.txt](requirements_macro.txt)** - Dépendances additionnelles

### Fichiers Modifiés

1. **[auto_paper_trading.py](auto_paper_trading.py)**
   - Ajout paramètres `enable_macro_filter` et `macro_threshold`
   - Intégration du MacroFilter dans `_generate_signal()`
   - Stockage des infos macro dans les signaux

2. **[multi_paper_trading.py](multi_paper_trading.py)**
   - **20 portfolios** au lieu de 10 (10 sans macro + 10 avec macro)
   - Passage automatique des paramètres macro aux portfolios
   - Noms suffixés `_Macro` pour les portfolios avec filtre

3. **[app_dashboard.py](app_dashboard.py)**
   - Nouvel onglet "📡 Signaux Macro" (tab9)
   - Affichage score marché général
   - Scores par actif
   - Comparaison portfolios avec/sans macro

### Documentation

- **[MACRO_SIGNALS_README.md](MACRO_SIGNALS_README.md)** - Guide de démarrage rapide
- **[MACRO_SIGNALS_GUIDE.md](MACRO_SIGNALS_GUIDE.md)** - Guide complet des APIs
- **[MACRO_INTEGRATION_PLAN.md](MACRO_INTEGRATION_PLAN.md)** - Plan d'intégration détaillé
- **[.env.example](.env.example)** - Template de configuration

---

## 🚀 Déploiement sur le Serveur

### Étape 1 : Commit et Push Local

```bash
cd ~/Desktop/Outil\ trading/Outil-Trading/

# Vérifier les changements
git status

# Ajouter tous les nouveaux fichiers
git add .

# Commit
git commit -m "Intégration signaux macro - 20 portfolios (10 sans + 10 avec filtre)

- Ajout news_fetcher.py pour récupération signaux (RSS, FRED, APIs)
- Ajout macro_signal_scorer.py pour scoring composite
- Ajout macro_integration.py avec MacroFilter
- Modification auto_paper_trading.py pour intégrer le filtre
- Modification multi_paper_trading.py pour 20 portfolios
- Ajout onglet Macro au dashboard
- Configuration FRED API

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"

# Push
git push origin main
```

### Étape 2 : Connexion au Serveur

```bash
ssh root@188.245.184.69
```

### Étape 3 : Update du Code sur le Serveur

```bash
# Aller dans le répertoire du projet
cd /opt/trading/

# Fetch et reset (préserve les positions actuelles car elles sont dans des fichiers séparés)
git fetch origin
git reset --hard origin/main

# Vérifier que les nouveaux fichiers sont présents
ls -la news_fetcher.py macro_signal_scorer.py macro_integration.py
```

### Étape 4 : Installer les Dépendances

```bash
# Activer l'environnement virtuel
source venv/bin/activate

# Installer les nouvelles dépendances
pip install feedparser requests python-dotenv

# Vérifier l'installation
python3 -c "import feedparser; import requests; print('✅ Dépendances OK')"
```

### Étape 5 : Configurer .env sur le Serveur

```bash
# Créer/éditer .env
nano .env
```

Ajouter :
```bash
# FRED API Key
FRED_KEY=480a473e9a5a6e99838252204df3cd1b
```

Sauvegarder : `Ctrl+O`, `Enter`, `Ctrl+X`

### Étape 6 : Test Rapide des Modules

```bash
# Test du fetcher (devrait fonctionner avec RSS + FRED)
python3 news_fetcher.py

# Si ça affiche des signaux récupérés : ✅
# Si erreur : vérifier les dépendances
```

### Étape 7 : Vérifier l'État Actuel des Portfolios

```bash
# Vérifier que les états existent
ls -la paper_trading_state/

# Vérifier le consolidated state
cat paper_trading_state/consolidated_state.json | python3 -m json.tool | head -50
```

**IMPORTANT:** Les positions actuelles sont préservées car elles sont stockées dans `paper_trading_state/portfolio_XX_*/auto_state.json`. Le code va simplement créer 10 nouveaux répertoires pour les portfolios avec macro.

### Étape 8 : Restart du Service Paper Trading

```bash
# Arrêter le service
systemctl stop paper-trading

# Vérifier le statut
systemctl status paper-trading

# Redémarrer
systemctl restart paper-trading

# Vérifier que ça démarre bien
systemctl status paper-trading

# Suivre les logs en temps réel
journalctl -u paper-trading -f
```

**Vous devriez voir :**
```
MULTI PAPER TRADING - 20 portefeuilles
Capital total: 100,000 EUR (5,000 EUR/portefeuille)
...
[01] Conservative: Filtres stricts, faible risque
  ✅ Macro filter disabled
  OK: 5 actifs dans le plan
[02] Balanced: Configuration equilibree
  ✅ Macro filter disabled
  OK: 8 actifs dans le plan
...
[11] Conservative_Macro: Filtres stricts, faible risque + Filtre macro
  ✅ Macro filter enabled (threshold=60)
  OK: 5 actifs dans le plan
...
```

### Étape 9 : Restart du Dashboard (optionnel)

```bash
# Si le dashboard tournait déjà
systemctl restart trading-dashboard

# Vérifier
systemctl status trading-dashboard
```

### Étape 10 : Vérification

1. **Logs du paper trading**
   ```bash
   journalctl -u paper-trading -f
   ```

   Vous devriez voir :
   - Setup de 20 portfolios (au lieu de 10)
   - Les 10 premiers avec "Macro filter disabled"
   - Les 10 suivants avec "Macro filter enabled (threshold=60)"
   - Génération de signaux
   - Application du filtre macro sur les portfolios _Macro

2. **Dashboard**
   - Ouvrir http://188.245.184.69:8501
   - Onglet "🖥️ Multi Paper Trading" → Devrait afficher 20 portfolios
   - Nouvel onglet "📡 Signaux Macro" → Affiche les scores macro

3. **États des portfolios**
   ```bash
   ls -la paper_trading_state/
   ```

   Vous devriez voir :
   - `portfolio_01_conservative/` à `portfolio_10_crypto_commodities/` (baseline sans macro)
   - `portfolio_11_conservative_macro/` à `portfolio_20_crypto_commodities_macro/` (avec macro)

---

## 📊 Vérifications Post-Déploiement

### Check 1 : Nombre de Portfolios

```bash
# Compter les répertoires de portfolios
ls -d paper_trading_state/portfolio_* | wc -l
# Devrait afficher : 20
```

### Check 2 : Signaux Macro Générés

```bash
# Vérifier le cache des signaux
cat macro_signals_cache.json | python3 -m json.tool | head -20
```

### Check 3 : Dashboard Accessible

```bash
# Vérifier que le dashboard tourne
curl -s http://localhost:8501 | grep "Outil de Trading" && echo "✅ Dashboard OK"
```

### Check 4 : Logs Sans Erreur

```bash
# Dernières lignes des logs
journalctl -u paper-trading -n 100 --no-pager
```

Chercher :
- ✅ "Macro filter enabled" pour les portfolios 11-20
- ✅ Génération de signaux
- ✅ Pas d'erreur Python
- ❌ Aucune ligne "ERROR" ou "CRITICAL"

---

## 🔧 Troubleshooting

### Problème : Import Error sur le serveur

```bash
# Vérifier les dépendances
source /opt/trading/venv/bin/activate
pip list | grep -E "feedparser|requests|dotenv"

# Réinstaller si nécessaire
pip install --upgrade feedparser requests python-dotenv
```

### Problème : Macro filter ne s'active pas

```bash
# Vérifier les logs
journalctl -u paper-trading -n 200 | grep -i macro

# Devrait voir :
# "Macro filter enabled (threshold=60)" pour portfolios 11-20
# "Macro filter disabled" pour portfolios 1-10
```

### Problème : FRED API ne fonctionne pas

```bash
# Tester la clé FRED
python3 -c "
import os
from dotenv import load_dotenv
load_dotenv()
key = os.getenv('FRED_KEY')
print(f'FRED_KEY: {key}')
import requests
r = requests.get(f'https://api.stlouisfed.org/fred/series/observations?series_id=DFF&api_key={key}&file_type=json&limit=1')
print(f'Status: {r.status_code}')
print(r.json())
"
```

### Problème : Positions existantes perdues

**Ne devrait PAS arriver** car les positions sont dans des fichiers séparés par portfolio.

Si ça arrive quand même :
```bash
# Vérifier les backups
ls -la paper_trading_state/portfolio_01_conservative/
cat paper_trading_state/portfolio_01_conservative/auto_state.json
```

Les positions sont dans `auto_state.json` de chaque portfolio et ne sont pas touchées par le git pull.

---

## 📈 Comparaison des Performances

Après quelques jours/semaines de trading, comparer :

### Méthode 1 : Via Dashboard

1. Ouvrir onglet "🖥️ Multi Paper Trading"
2. Comparer les PnL des portfolios 1-10 (sans macro) vs 11-20 (avec macro)
3. Regarder notamment :
   - Total PnL %
   - Win rate
   - Nombre de trades
   - Drawdown max

### Méthode 2 : Via Logs

```bash
# Voir le consolidated state
cat paper_trading_state/consolidated_state.json | python3 -m json.tool
```

### Méthode 3 : Analyse CSV

```bash
# Exporter les trades de chaque portfolio
for i in {01..20}; do
    portfolio=$(ls -d paper_trading_state/portfolio_${i}_* 2>/dev/null | head -1)
    if [ -d "$portfolio" ]; then
        name=$(basename $portfolio)
        echo "=== $name ==="
        if [ -f "$portfolio/auto_trades.csv" ]; then
            tail -5 "$portfolio/auto_trades.csv"
        fi
    fi
done
```

---

## 🎯 Métriques à Suivre

### Semaine 1-2 : Phase de Validation

- [ ] Les 20 portfolios démarrent correctement
- [ ] Les portfolios avec macro génèrent bien des signaux
- [ ] Le filtre macro annule des trades (vérifier dans les logs "LONG annulé" ou "SHORT annulé")
- [ ] Pas d'erreurs critiques dans les logs
- [ ] Dashboard affiche correctement l'onglet Macro

### Semaine 3-4 : Première Analyse

Comparer **10 portfolios sans macro** vs **10 portfolios avec macro** :

| Métrique | Sans Macro (1-10) | Avec Macro (11-20) | Différence |
|----------|-------------------|-------------------|------------|
| PnL total moyen | ? | ? | ? |
| Win rate moyen | ? | ? | ? |
| Trades totaux | ? | ? | ? |
| Trades filtrés | - | ? | - |
| Drawdown max moyen | ? | ? | ? |

**Hypothèse** : Le filtre macro devrait :
- ✅ Réduire le nombre de trades (filtrage)
- ✅ Augmenter le win rate (éviter mauvais trades)
- ✅ Réduire le drawdown (protection macro)
- ⚠️ Potentiellement réduire le PnL total (moins de trades = moins d'opportunités)

---

## 📝 Rollback (si nécessaire)

Si le déploiement pose problème, revenir en arrière :

```bash
cd /opt/trading/

# Revenir au commit précédent
git log --oneline | head -5
git reset --hard <commit_hash_avant_macro>

# Restart
systemctl restart paper-trading
```

Les positions actuelles seront préservées car elles sont dans des fichiers locaux non versionnés.

---

## ✅ Checklist de Déploiement

- [ ] Code commité et pushé sur main
- [ ] Connexion SSH au serveur OK
- [ ] Git pull sur /opt/trading/ OK
- [ ] Dépendances installées (feedparser, requests, python-dotenv)
- [ ] .env créé avec FRED_KEY
- [ ] Test news_fetcher.py OK
- [ ] Service paper-trading restart OK
- [ ] 20 portfolios visibles dans les logs
- [ ] Dashboard accessible
- [ ] Onglet "Signaux Macro" visible
- [ ] Pas d'erreurs critiques dans journalctl
- [ ] Positions actuelles préservées

---

## 🎉 Après Déploiement

**Félicitations !** Le système tourne maintenant avec 20 portfolios :
- **10 portfolios baseline** (techniques pures) → Référence
- **10 portfolios avec macro** (techniques + filtre macro) → Expérimental

Tu peux suivre en temps réel :
- **Dashboard** : http://188.245.184.69:8501
- **Logs** : `journalctl -u paper-trading -f`

Les prochaines semaines permettront de valider si le filtre macro améliore effectivement les performances ! 📈
