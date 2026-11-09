# Trading intraday piloté par les news : recherche et backtest

Projet personnel de recherche quantitative. Je teste si le **sentiment des news financières**, combiné à des indicateurs techniques, peut produire une stratégie rentable sur 9 actions US en bougies d'1 minute. Le travail va de la collecte des données à l'optimisation de plusieurs centaines de milliers de stratégies, avec validation croisée et coûts de transaction.

**Univers :** AAPL, AMD, AMZN, GOOGL, META, MSFT, NFLX, NVDA, TSLA
**Données :** 2,4 M de bougies 1 min (juillet 2023 → juillet 2026, Alpaca) et 43 385 news, dont 16 607 mono-ticker, scorées par un LLM.

## Pipeline

| Étape | Script | Rôle |
|---|---|---|
| 1 | `src/news.py` | Télécharge ~900 jours de news Alpaca pour les tickers ciblés |
| 2 | `src/stock_prices.py` | Télécharge les bougies 1 min, une par ticker, puis les fusionne |
| 3 | `src/gemini_score_news.py` | Score le sentiment (`news_score` ∈ [-1, 1]) et l'impact de chaque titre avec Gemini, une seule fois, en cache et avec reprise |
| 4 | `src/rth.py`, `src/audit_rth.py` | Masque des heures de séance (voir plus bas) et audit de son effet |
| 5 | `src/backtest_search_v10.py` | Random search parallélisé (6 workers) avec validation croisée glissante |
| 6 | `src/repair_results_csv.py` | Répare les CSV de résultats corrompus à la reprise d'une session |

## La stratégie

Un score composite est calculé à chaque bougie à partir de quatre facteurs normalisés en z-score :

```
score = w_trend · ROC(trend_window) + w_vol · volume/MA20 + w_volat · σ(volat_window) + w_news · sentiment_des_news
```

Le score déclenche un achat s'il dépasse `buy_threshold`, une vente à découvert s'il passe sous `sell_threshold`. Chaque position a un stop-loss et un take-profit. Les 10 hyperparamètres (4 poids, 2 fenêtres, SL, TP, 2 seuils) sont tirés au hasard.

Le sentiment des news décroît avec une demi-vie de 15 min. Une news publiée hors séance est portée par la première bougie qui suit sa publication.

**Hypothèses du simulateur, volontairement conservatrices :**
- exécution à l'ouverture de la bougie **t+1**, sans look-ahead ;
- coûts de 5 bps par côté, majorés quand la volatilité est élevée ;
- stop et take-profit exécutés au pire prix entre le niveau visé et l'ouverture de la bougie, donc avec gap ;
- PnL = somme des rendements nets par trade, pas un rendement composé.

**Validation :** 183 jours de warm-up, 243 jours de train et 122 jours de validation, puis des fenêtres glissantes en validation croisée.
`Robust_Score = moyenne(PnL validation) − écart-type entre fenêtres`. Il pénalise les stratégies qui ne marchent que sur une période ou un ticker.

## Résultats

| Version | Changement | Essais | Meilleur Robust_Score | Essais avec score > 0 |
|---|---|---:|---:|---:|
| v9 | Base : validation sur les 9 tickers, signaux hors séance autorisés | 42 108 | 0,018 | 2 |
| v10 | Masque RTH et validation croisée glissante | 85 000 | 0,055 | 26 |
| v11 | Nouveau run | 60 000 | 0,070 | 39 |
| v12 | Run long | 453 340 | **0,103** | 109 |
| v13 | Run long | 425 000 | 0,079 | 115 |

Au total, plus de **1 million de stratégies** ont été évaluées. Les 50 meilleures de chaque version sont dans [`results/`](results/). Les fichiers complets pèsent plus de 1 Go.

### Ce que j'en retiens

1. **Presque aucune stratégie aléatoire ne survit à la validation.** Le PnL médian en validation est d'environ −41 %, et seuls 0,3 % des essais ont un Robust_Score positif. C'est un signal clair de surapprentissage : avec assez de tirages, on trouve toujours quelque chose de brillant en train.
2. **Le hors-séance produisait de faux gains.** `audit_rth.py` montre que 1,3 % des bougies (hors séance) généraient 8 à 16 % des trades. Les bougies y sont éparses, le ROC absorbe le gap overnight et le z-score s'envole. Ces ordres ne sont de plus pas exécutables : Alpaca les met en file jusqu'à l'ouverture, à un prix sans rapport. Sur la stratégie « Sécurité », **36 % du PnL venait de ces trades**, et le classement s'inverse quand on les retire. D'où le masque RTH de la v10.
3. **Ne pas filtrer les bougies, filtrer les signaux.** Supprimer les bougies hors séance créerait un trou de 17 h 30 : le ROC de 9 h 30 comparerait au prix de la veille et fabriquerait un faux signal à l'ouverture, là où l'ordre part vraiment. Les indicateurs sont donc calculés sur la série complète, et seuls les signaux sont annulés (`rth.py`).
4. **Le meilleur candidat reste fragile.** Le trial 808651 (v12) est rentable sur les 9 tickers (+17,2 % de PnL moyen en validation, écart-type 6,8 %). Cela reste un maximum sélectionné parmi près d'un demi-million d'essais, donc un test hors échantillon est nécessaire avant d'en tirer une conclusion.
5. **Un bug de reprise corrompait les CSV.** Un `Trial_ID` textuel faisait croire au script qu'il repartait de zéro, ce qui écrivait un en-tête au milieu des données. Il est corrigé dans la v10, et `repair_results_csv.py` répare les fichiers abîmés.

### Stratégies de référence (`results/champions.csv`)

Deux stratégies issues d'une première recherche, retestées après l'ajout du masque RTH : « Sécurité » (trial 47006) et « Fréquence » (trial 27405).

## Bilan : ce que j'ai appris et ce que j'ai construit

### Ce que j'ai réussi à mettre en œuvre
- **Une chaîne de données complète**, de bout en bout : collecte via l'API Alpaca avec reprise sur incident, scoring de 16 607 titres par LLM (Gemini) en une seule passe mise en cache, et fusion avec 2,4 M de bougies à la minute.
- **Un simulateur de backtest réaliste** : exécution en t+1 sans look-ahead, coûts dépendant de la volatilité, gaps sur les stops, fuseau horaire New York.
- **Un moteur de recherche massivement parallèle** : plus d'1 million de stratégies évaluées, avec validation croisée glissante et un score de robustesse qui pénalise l'instabilité.
- **La détection et la correction de mes propres biais** : j'ai mesuré que le hors-séance gonflait artificiellement les gains (jusqu'à 36 % du PnL d'une stratégie), compris le mécanisme et corrigé sans introduire de nouveau biais. J'ai aussi corrigé un bug de reprise qui corrompait mes fichiers de résultats.

### Pourquoi ça n'a pas donné de stratégie exploitable
Ce sont des hypothèses, que je n'ai pas toutes testées.

1. **Du bruit sélectionné.** Avec 10 paramètres libres et plus d'1 million d'essais, le meilleur résultat est surtout un maximum statistique. Le PnL médian en validation est de −41 %. Mon Robust_Score pénalise l'instabilité mais ne corrige pas le biais de sélection : aucune correction pour tests multiples (Deflated Sharpe Ratio, test de White).
2. **Pas de test hors échantillon.** Les fenêtres de validation ont servi à classer les stratégies. Elles ne sont donc plus indépendantes, et il manque une période finale jamais touchée.
3. **Un signal probablement déjà dans le prix.** Les titres d'actions très liquides comme celles de mon univers sont lus et traités par des acteurs bien plus rapides que moi. Une news à la minute près arrive peut-être trop tard.
4. **Un score de sentiment grossier.** Je ne score que le titre, sans le corps de l'article, la surprise par rapport aux attentes ni la nouveauté de l'information. Un modèle léger et un seul prompt ajoutent du bruit.
5. **Un score linéaire trop simple.** Une somme pondérée de quatre z-scores avec seuils fixes ne capture ni les interactions (une bonne news en forte volatilité) ni les régimes de marché.
6. **Des coûts sous-estimés.** 5 bps par côté est optimiste pour du trading à haute fréquence : le slippage et l'impact sur le prix peuvent annuler un avantage de quelques dizaines de bps.
7. **Un univers étroit.** Neuf méga-caps très corrélées, sur une période qui n'a connu qu'un seul type de marché.

### Pistes de recherche
- **Évaluer correctement** : réserver une période finale intouchée, appliquer le Deflated Sharpe Ratio et utiliser une validation walk-forward purgée avec embargo.
- **Réduire l'espace de recherche** : passer à une optimisation bayésienne, avec une régularisation ou un nombre de paramètres limité.
- **Mesurer d'abord, optimiser ensuite** : une étude d'événements (rendement moyen après une news positive ou négative, à plusieurs horizons) dirait si un signal existe avant de chercher des paramètres.
- **Améliorer le signal** : scorer le texte complet, mesurer la surprise et la nouveauté, tester des modèles plus capables et comparer plusieurs prompts.
- **Élargir l'univers** : small et mid caps moins couvertes, où l'information circule plus lentement, et périodes plus variées.
- **Modéliser les coûts de façon réaliste** : spread, slippage et impact sur le prix, avec des données bid/ask.
- **Passer à des modèles non linéaires** : gradient boosting ou méthodes par régimes, avec une validation stricte.

### En résumé
Je n'ai pas trouvé de stratégie rentable, et ce dépôt ne prétend pas le contraire. Je l'ai surtout mené comme un exercice de rigueur : construire un banc d'essai honnête, repérer ce qui le faussait, et reconnaître qu'un bon score en recherche ne prouve rien. Le résultat le plus utile est la méthode, et la liste de ce qu'il faudrait faire pour conclure sérieusement.

---

## Utilisation

```bash
pip install -r requirements.txt
cp .env.example .env      # puis renseigner les clés Alpaca et Gemini

python src/news.py                 # news
python src/stock_prices.py         # prix
python src/gemini_score_news.py    # sentiment
python src/audit_rth.py            # impact du hors-séance
python src/backtest_search_v10.py  # recherche (reprend là où elle s'est arrêtée)
```

Les scripts s'exécutent depuis le dossier qui contient les fichiers de données. Les paramètres (nombre d'essais, plages, coûts) sont en tête de `backtest_search_v10.py`.

> Le module `cv_splits.py` (génération des fenêtres de validation croisée) n'est pas inclus dans ce dépôt, il faut donc le recréer pour relancer `backtest_search_v10.py`.

*Projet de recherche et d'apprentissage, pas un conseil en investissement.*
