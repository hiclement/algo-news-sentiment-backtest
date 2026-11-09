"""
backtest_search_v10.py — Random Search, PARALLÉLISÉ, VALIDATION CROISÉE
[v10 — v9 + masque RTH + reprise de session corrigée]

═══════════════════════════════════════════════════════════════════════════════
CE QUI CHANGE PAR RAPPORT À v9 (et pourquoi)
═══════════════════════════════════════════════════════════════════════════════

[1] MASQUE RTH — aucun signal hors séance                        ← LE CHANGEMENT
    Mesuré sur vos données (audit_rth.py) : 1,3 % des bougies (le hors-séance)
    produisaient 8 à 16 % des trades — une densité 10× supérieure. Ce n'est pas
    du signal, c'est du bruit : hors séance les bougies sont ÉPARSES (Alpaca
    n'en émet que s'il y a des transactions), donc remonter 34 bougies peut
    couvrir une nuit entière. Le ROC absorbe le gap overnight et le z-score
    s'envole.

    Pire : ces trades n'étaient pas exécutables. Un ordre au marché passé hors
    séance n'est pas rejeté par Alpaca — il est MIS EN FILE jusqu'à l'ouverture
    suivante, à un prix sans rapport avec celui supposé ici. Le Robust_Score les
    comptait pourtant comme de vrais gains : 36 % du PnL du Trial 47006 en
    provenait, et le classement s'inverse une fois qu'on les retire.

    ⚠️ ON NE FILTRE PAS LES BOUGIES. Les indicateurs sont calculés sur la série
    COMPLÈTE (continuité préservée) ; seuls les SIGNAUX sont annulés hors séance.
    Supprimer les bougies créerait un trou de 17h30 entre 16h00 et 09h30 : le ROC
    de la bougie de 09h30 comparerait alors au prix de la veille 15h10, absorbant
    tout le gap overnight → un z_trend aberrant À L'OUVERTURE, c'est-à-dire là où
    l'ordre PART vraiment. On remplacerait un faux signal inoffensif par un faux
    signal coûteux.

[2] REPRISE DE SESSION — le bug qui corrompait le CSV
    resume_trial_id() faisait `int(tail[-1].split(",")[0])`. Si la dernière ligne
    était une stratégie IMPORTÉE (Trial_ID = un nom, "trial_47006"), ValueError
    → `except: return 0` → le script croyait repartir de zéro → `writeheader()`
    EN PLEIN MILIEU du fichier. D'où le fameux :
        DtypeWarning: Columns (0: Trial_ID) have mixed types
    Et les stratégies importées étaient ré-évaluées et dupliquées à chaque
    reprise ratée. Corrigé ci-dessous ; utilisez repair_results_csv.py pour les
    fichiers déjà abîmés.

[3] FICHIERS DE SORTIE RENOMMÉS (v10)
    Indispensable : un CSV mélangeant des essais AVEC et SANS masque RTH serait
    trié ensemble par Robust_Score — des résultats incomparables, classés côte à
    côte, sans qu'aucune colonne ne permette de les distinguer.

Nécessite rth.py dans le même dossier.
═══════════════════════════════════════════════════════════════════════════════
"""

import csv
import time
import numpy as np
import pandas as pd
import multiprocessing as mp
from pathlib import Path

import rth
import cv_splits as cv

# ==============================================================================
# CONFIGURATION
# ==============================================================================
PRICES_FILE = "alpaca_stock_1min.parquet"
NEWS_FILE = "alpaca_news_test.parquet"
SCORES_FILE = "news_scores.parquet"

# ⚠️ NOUVEAUX NOMS : ne mélangez pas des essais avec et sans masque RTH.
RESULTS_CSV = "backtest_results_v10.csv"
DETAIL_CSV = "backtest_results_v10_detail.csv"
SORTED_CSV = "backtest_results_sorted_v10.csv"
IMPORT_CSV = "strategies_to_test.csv"

# True  : aucun signal hors séance (09h30–16h00 NY). ← recommandé
# False : comportement v9 (tradait hors séance). À n'utiliser que pour reproduire
#         les anciens résultats — et alors, écrivez dans d'AUTRES fichiers.
RTH_ONLY = True

WARMUP_DAYS = 183
TRAIN_DAYS = 243
VAL_DAYS = 122

COST_BPS = 5.0
VOLUME_MA = 20
NEWS_DECAY_MINUTES = 30

N_TRIALS = 450_000
PROGRESS_EVERY = 200
SORT_EVERY = 5000
N_WORKERS = 6
SEED = None

TREND_RANGE = (5, 60)
VOLAT_RANGE = (10, 100)
SL_RANGE = (0.005, 0.05)
RR_RANGE = (1.5, 3.0)
BUY_RANGE = (0.1, 2.0)
SELL_RANGE = (-2.0, -0.1)

# En-têtes STRICTEMENT d'origine (21 colonnes pour AGG)
AGG_FIELDNAMES = ["Trial_ID", "w_trend", "w_vol", "w_volat", "w_news",
                  "trend_window", "volat_window", "stop_loss", "take_profit",
                  "buy_threshold", "sell_threshold",
                  "N_tickers", "N_profitable",
                  "Train_PnL_mean", "Val_PnL_mean", "Val_PnL_std", "Val_PnL_worst",
                  "Win_Rate_mean", "Max_Drawdown_mean", "Max_Drawdown_worst",
                  "Robust_Score"]

DETAIL_FIELDNAMES = ["Trial_ID", "ticker", "w_trend", "w_vol", "w_volat", "w_news",
                     "trend_window", "volat_window", "stop_loss", "take_profit",
                     "buy_threshold", "sell_threshold",
                     "Train_PnL", "Val_PnL", "Train_Trades", "Val_Trades",
                     "Win_Rate", "Max_Drawdown"]

_CACHES = None
_WINDOWS = []
_COST_FRAC = 2 * COST_BPS / 1e4


# ==============================================================================
# 1. CHARGEMENT & PRÉCALCULS
# ==============================================================================
def load_market_data():
    prices = pd.read_parquet(PRICES_FILE).reset_index()
    ts = "timestamp" if "timestamp" in prices.columns else prices.columns[0]
    prices = prices.rename(columns={ts: "timestamp"})
    prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True)

    sent_map = {}
    if Path(NEWS_FILE).exists() and Path(SCORES_FILE).exists():
        news = pd.read_parquet(NEWS_FILE)
        news = news[news["num_symbols"] == 1].copy()
        news["ticker"] = news["symbols"].apply(lambda s: str(s[0]))
        news["created_at"] = pd.to_datetime(news["created_at"], utc=True, format="mixed")
        sc = pd.read_parquet(SCORES_FILE)
        cols = ["id", "news_score"] + (["impact"] if "impact" in sc.columns else [])
        news = news.merge(sc[cols], on="id", how="inner")

        if "impact" in news.columns:
            news["impact"] = news["impact"].clip(0.5, 2.0)
            news["news_score"] = news["news_score"] * news["impact"]

        for tk, g in news.groupby("ticker"):
            sent_map[tk] = g[["created_at", "news_score"]].sort_values("created_at")

    frames = {}
    for tk, g in prices.groupby("ticker"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        g["news_sentiment"] = 0.0
        if tk in sent_map:
            idx = g["timestamp"].values
            # side="left" : la news est portée par la PREMIÈRE bougie ≥ à son
            # heure de parution. Une news du soir attend donc la réouverture.
            pos = np.searchsorted(idx, sent_map[tk]["created_at"].values, side="left")
            for p, v in zip(pos, sent_map[tk]["news_score"].values):
                if p < len(g):
                    g.at[p, "news_sentiment"] += v
            if NEWS_DECAY_MINUTES > 1:
                halflife = NEWS_DECAY_MINUTES / 2
                kernel = np.exp(-np.log(2) / halflife * np.arange(NEWS_DECAY_MINUTES))
                raw = g["news_sentiment"].to_numpy(float)
                g["news_sentiment"] = np.convolve(raw, kernel, mode="full")[:len(raw)]
        frames[tk] = g
    return frames


def compute_split_bounds(frames):
    t0 = min(f["timestamp"].min() for f in frames.values())
    t_max = max(f["timestamp"].max() for f in frames.values())
    train_start = t0 + pd.Timedelta(days=WARMUP_DAYS)
    train_end = train_start + pd.Timedelta(days=TRAIN_DAYS)
    val_end = train_end + pd.Timedelta(days=VAL_DAYS)
    val_end = min(val_end, t_max)
    return (int(train_start.value), int(train_end.value), int(val_end.value),
            train_start, train_end, val_end)


class TickerCache:
    def __init__(self, df):
        self.open = df["open"].to_numpy(float)
        self.close = df["close"].to_numpy(float)
        self.high = df["high"].to_numpy(float)
        self.low = df["low"].to_numpy(float)
        self.news = df["news_sentiment"].to_numpy(float)
        # Timestamps entiers : les masques train/val sont dérivés PAR FENÊTRE de
        # validation croisée (voir _eval_trial), plus figés une fois pour toutes.
        self.ts_int = df["timestamp"].dt.tz_convert("UTC").astype(
            "datetime64[ns, UTC]").astype("int64").to_numpy()
        n = len(self.close)

        # ── [1] MASQUE DES HEURES DE SÉANCE ────────────────────────────────
        # Calculé UNE fois par ticker. Les indicateurs ci-dessous continuent
        # d'être calculés sur TOUTES les bougies : seule la génération de
        # signaux sera bridée (voir _eval_trial).
        #
        # `rth_exec` : la bougie d'EXÉCUTION (T+1, car simulate() entre à
        # open[t+1]) doit elle aussi être en séance. Sans ça, un signal émis à
        # 15h59 s'exécuterait à 16h00 — hors séance, donc non exécutable en réel.
        if RTH_ONLY:
            self.rth = rth.rth_mask(df["timestamp"])
            self.rth_exec = np.zeros(n, dtype=bool)
            self.rth_exec[:-1] = self.rth[1:]
        else:
            self.rth = np.ones(n, dtype=bool)
            self.rth_exec = np.ones(n, dtype=bool)

        vol = df["volume"].to_numpy(float)
        vol_ma = pd.Series(vol).rolling(VOLUME_MA, min_periods=VOLUME_MA).mean().to_numpy()
        self.vol_ratio_z = self._z(np.divide(vol, vol_ma, out=np.full(n, np.nan), where=vol_ma > 0))

        self.roc = {}
        for w in range(TREND_RANGE[0], TREND_RANGE[1] + 1):
            roc = np.full(n, np.nan)
            roc[w:] = self.close[w:] / self.close[:-w] - 1.0
            self.roc[w] = self._z(roc)

        logret = np.concatenate([[np.nan], np.diff(np.log(np.maximum(self.close, 1e-10)))])
        self.volat = {}
        for w in range(VOLAT_RANGE[0], VOLAT_RANGE[1] + 1):
            v = pd.Series(logret).rolling(w, min_periods=w).std().to_numpy()
            self.volat[w] = self._z(v)

    def _z(self, x):
        # Calibration GLOBALE (sur toutes les données) et non plus sur un TRAIN
        # figé : avec la validation croisée, le TRAIN change à chaque fenêtre.
        # Recalibrer par fenêtre coûterait N_SPLITS fois plus cher, pour un effet
        # négligeable — la normalisation ne fait que fixer l'échelle des features,
        # elle ne « voit » pas la cible. σ n'est jamais nul (garde-fou), donc
        # aucune division par zéro possible.
        mu = np.nanmean(x); sd = np.nanstd(x)
        if not np.isfinite(sd) or sd == 0: sd = 1.0
        return np.nan_to_num((x - mu) / sd, nan=0.0)


def _worker_init():
    global _CACHES, _WINDOWS
    frames = load_market_data()
    t0 = min(f["timestamp"].min() for f in frames.values())
    t_max = max(f["timestamp"].max() for f in frames.values())
    _WINDOWS = cv.make_windows(t0, t_max)
    _CACHES = {}
    for tk, df in frames.items():
        if len(df) < 500:
            continue
        _CACHES[tk] = TickerCache(df)


# ==============================================================================
# 2. MOTEUR DE SIMULATION
# ==============================================================================
def simulate(open_p, close, high, low, sig, sl, tp, base_cost, volat_z):
    n = len(close)
    rets = []

    vol_penalty = np.maximum(0.0, volat_z) * 0.5
    dynamic_costs = base_cost * (1.0 + vol_penalty)

    t = 0
    while t < n - 2:
        s = sig[t]
        if s == 0:
            t += 1; continue

        e = t + 1
        entry = open_p[e]
        if not np.isfinite(entry) or entry <= 0:
            t += 1; continue

        fwd = slice(e + 1, n)
        if s == 1:
            sl_hit = low[fwd] <= entry * (1 - sl)
            tp_hit = high[fwd] >= entry * (1 + tp)
        else:
            sl_hit = high[fwd] >= entry * (1 + sl)
            tp_hit = low[fwd] <= entry * (1 - tp)

        exit_cond = sl_hit | tp_hit | (sig[fwd] == -s)
        rel = int(np.argmax(exit_cond)) if exit_cond.any() else len(exit_cond) - 1
        b = e + 1 + rel

        if sl_hit[rel]:
            o_price = open_p[b]
            if s == 1:
                exit_price = float(np.minimum(entry * (1 - sl), o_price))
            else:
                exit_price = float(np.maximum(entry * (1 + sl), o_price))
        elif tp_hit[rel]:
            o_price = open_p[b]
            if s == 1:
                exit_price = float(np.maximum(entry * (1 + tp), o_price))
            else:
                exit_price = float(np.minimum(entry * (1 - tp), o_price))
        else:
            exit_price = close[b]

        rets.append(s * (exit_price - entry) / entry - dynamic_costs[e])
        t = b
    return rets


def metrics(rets):
    if not rets: return {"pnl": 0.0, "win": 0.0, "maxdd": 0.0, "trades": 0}
    r = np.array(rets)
    eq = np.cumsum(r)
    peak = np.maximum.accumulate(eq)
    return {"pnl": float(r.sum()), "win": float((r > 0).mean()),
            "maxdd": float(np.max(peak - eq)), "trades": len(r)}


def _slice_from_mask(mask):
    idx = np.flatnonzero(mask)
    return (0, 0) if len(idx) == 0 else (int(idx[0]), int(idx[-1]) + 1)


def _eval_trial(task):
    tid, p = task
    display_id = p.get("_import_name", tid)   # nom si importé, sinon ID numérique

    # Signal figé par ticker (indépendant de la fenêtre)
    signals = {}
    for tk, cache in _CACHES.items():
        score = (p["w"][0] * cache.roc[p["trend_window"]] +
                 p["w"][1] * cache.vol_ratio_z +
                 p["w"][2] * cache.volat[p["volat_window"]] +
                 p["w"][3] * cache.news)
        sig = np.where(score > p["buy"], 1, np.where(score < p["sell"], -1, 0)).astype(np.int8)
        # ── AUCUN SIGNAL HORS SÉANCE (bougie ET exécution T+1 en séance). Les
        #    sorties restent libres : un stop doit pouvoir se déclencher. ──
        sig[~(cache.rth & cache.rth_exec)] = 0
        signals[tk] = sig

    # ── VALIDATION CROISÉE : une évaluation par fenêtre glissante ──────────────
    per_window, per_ticker = [], []
    for w in _WINDOWS:
        tr_pnls, va_pnls, wins, dds = [], [], [], []
        for tk, cache in _CACHES.items():
            train_mask, val_mask = cv.masks_for_window(cache.ts_int, w)
            i0, i1 = _slice_from_mask(train_mask)
            j0, j1 = _slice_from_mask(val_mask)
            if i1 - i0 < 100 or j1 - j0 < 100:
                continue
            sig = signals[tk]
            vol_arr = cache.volat[p["volat_window"]]
            tr = metrics(simulate(cache.open[i0:i1], cache.close[i0:i1], cache.high[i0:i1],
                                  cache.low[i0:i1], sig[i0:i1], p["sl"], p["tp"], _COST_FRAC, vol_arr[i0:i1]))
            va = metrics(simulate(cache.open[j0:j1], cache.close[j0:j1], cache.high[j0:j1],
                                  cache.low[j0:j1], sig[j0:j1], p["sl"], p["tp"], _COST_FRAC, vol_arr[j0:j1]))
            tr_pnls.append(tr["pnl"]); va_pnls.append(va["pnl"])
            wins.append(va["win"]); dds.append(va["maxdd"])
            per_ticker.append({
                "Trial_ID": display_id, "ticker": f"{tk}@F{w['k'] + 1}",
                "w_trend": round(p["w"][0], 4), "w_vol": round(p["w"][1], 4),
                "w_volat": round(p["w"][2], 4), "w_news": round(p["w"][3], 4),
                "trend_window": p["trend_window"], "volat_window": p["volat_window"],
                "stop_loss": round(p["sl"], 4), "take_profit": round(p["tp"], 4),
                "buy_threshold": round(p["buy"], 4), "sell_threshold": round(p["sell"], 4),
                "Train_PnL": round(tr["pnl"], 5), "Val_PnL": round(va["pnl"], 5),
                "Train_Trades": tr["trades"], "Val_Trades": va["trades"],
                "Win_Rate": round(va["win"], 4), "Max_Drawdown": round(va["maxdd"], 5),
            })
        if va_pnls:
            per_window.append({
                "train_mean": float(np.mean(tr_pnls)), "val_mean": float(np.mean(va_pnls)),
                "val_worst": float(np.min(va_pnls)),
                "n_profitable": int((np.array(va_pnls) > 0).sum()), "n_tickers": len(va_pnls),
                "win": float(np.mean(wins)), "dd_mean": float(np.mean(dds)),
                "dd_worst": float(np.max(dds)),
            })

    if not per_window:
        return None, []

    win_val = np.array([w["val_mean"] for w in per_window])
    win_train = np.array([w["train_mean"] for w in per_window])
    train_mean = float(win_train.mean())
    val_mean = float(win_val.mean())
    val_std = float(win_val.std())     # dispersion ENTRE FENÊTRES : cœur de la CV

    agg = {
        "Trial_ID": display_id,
        "w_trend": round(p["w"][0], 4), "w_vol": round(p["w"][1], 4),
        "w_volat": round(p["w"][2], 4), "w_news": round(p["w"][3], 4),
        "trend_window": p["trend_window"], "volat_window": p["volat_window"],
        "stop_loss": round(p["sl"], 4), "take_profit": round(p["tp"], 4),
        "buy_threshold": round(p["buy"], 4), "sell_threshold": round(p["sell"], 4),
        "N_tickers": int(round(np.mean([w["n_tickers"] for w in per_window]))),
        "N_profitable": int(round(np.mean([w["n_profitable"] for w in per_window]))),
        "Train_PnL_mean": round(train_mean, 5),
        "Val_PnL_mean": round(val_mean, 5),
        "Val_PnL_std": round(val_std, 5),
        "Val_PnL_worst": round(float(min(w["val_worst"] for w in per_window)), 5),
        "Win_Rate_mean": round(float(np.mean([w["win"] for w in per_window])), 4),
        "Max_Drawdown_mean": round(float(np.mean([w["dd_mean"] for w in per_window])), 5),
        "Max_Drawdown_worst": round(float(max(w["dd_worst"] for w in per_window)), 5),
        "Robust_Score": round(val_mean - val_std, 5),
    }
    return agg, per_ticker


# ==============================================================================
# 3. ORCHESTRATION & IMPORT CSV
# ==============================================================================
def sample_params(rng):
    sl = float(rng.uniform(*SL_RANGE))
    w_others = rng.uniform(-1, 1, 3)
    w_news = float(rng.uniform(-1.0, 2.0))
    w_combined = np.append(w_others, w_news)

    return {
        "w": w_combined,
        "trend_window": int(rng.integers(*TREND_RANGE)),
        "volat_window": int(rng.integers(*VOLAT_RANGE)),
        "buy": float(rng.uniform(*BUY_RANGE)),
        "sell": float(rng.uniform(*SELL_RANGE)),
        "sl": sl,
        "tp": sl * float(rng.uniform(*RR_RANGE)),
    }


def load_imported_strategies():
    imported_tasks = []
    import_path = Path(IMPORT_CSV)

    print("\n" + "=" * 80)
    print("🔍 DIAGNOSTIC D'IMPORTATION DES STRATÉGIES")
    print("=" * 80)

    if not import_path.exists():
        print(f"❌ Fichier '{IMPORT_CSV}' introuvable. Lancement en Random Search.")
        print("=" * 80 + "\n")
        return imported_tasks

    try:
        # sep=None + engine="python" : détecte automatiquement la virgule, le
        # point-virgule (Excel FR) ou la tabulation.
        df = pd.read_csv(import_path, sep=None, engine="python")

        if "active" in df.columns:
            active_mask = df["active"].astype(str).str.lower().isin(
                ["true", "1", "t", "active", "oui", "yes"])
            ignored_count = len(df) - active_mask.sum()
            df = df[active_mask]
        else:
            ignored_count = 0

        if df.empty:
            print(f"⚠️ Le fichier '{IMPORT_CSV}' est vide ou inactif.")
            print("=" * 80 + "\n")
            return imported_tasks

        print(f"✅ Fichier '{IMPORT_CSV}' détecté !")
        print(f"📥 Chargement de {len(df)} stratégie(s) active(s) "
              f"(et {ignored_count} ignorée(s)) :\n")

        for idx, row in df.iterrows():
            name = str(row.get("name", row.get("id", f"Import_{idx}")))
            print(f"   👉 [{idx}] Trial_ID : {name:<25} | SL: {float(row['stop_loss']):.3f}")

            p = {
                "_import_name": name,   # variable interne, jamais écrite dans le CSV
                "w": np.array([
                    float(row["w_trend"]),
                    float(row["w_vol"]),
                    float(row["w_volat"]),
                    float(row["w_news"])
                ]),
                "trend_window": int(row["trend_window"]),
                "volat_window": int(row["volat_window"]),
                "buy": float(row["buy_threshold"]),
                "sell": float(row["sell_threshold"]),
                "sl": float(row["stop_loss"]),
                "tp": float(row["take_profit"]),
            }
            imported_tasks.append(p)

    except Exception as e:
        print(f"❌ Erreur de lecture CSV : {e}")

    print("=" * 80 + "\n")
    return imported_tasks


# ==============================================================================
# [2] REPRISE DE SESSION — CORRIGÉE
# ==============================================================================
def resume_trial_id():
    """Prochain Trial_ID numérique, en tolérant les Trial_ID TEXTUELS.

    L'ancienne version faisait `int(tail[-1].split(",")[0])` : si la dernière
    ligne était une stratégie importée (Trial_ID = "trial_47006"), ValueError →
    `except: return 0` → le script croyait démarrer une session neuve, réécrivait
    un en-tête au milieu du CSV et ré-évaluait les stratégies importées.

    En cas de fichier illisible, on S'ARRÊTE : mieux vaut refuser de démarrer que
    de corrompre des jours de calcul.
    """
    if not Path(RESULTS_CSV).exists() or Path(RESULTS_CSV).stat().st_size == 0:
        return 0
    try:
        ids = pd.read_csv(RESULTS_CSV, usecols=["Trial_ID"],
                          dtype={"Trial_ID": str}, low_memory=False)["Trial_ID"]
        ids = ids[ids != "Trial_ID"].dropna()          # en-têtes parasites éventuels
        num = pd.to_numeric(ids, errors="coerce").dropna()
        return int(num.max()) + 1 if len(num) else len(ids)
    except Exception as e:
        raise SystemExit(
            f"❌ Reprise impossible depuis {RESULTS_CSV} : {e}\n"
            f"   Lancez `python repair_results_csv.py {RESULTS_CSV}` puis relancez.")


def _open_writers(start_id):
    """Ouvre les CSV en append. L'en-tête n'est écrit QUE si le fichier est vide.

    Le test doit se faire AVANT l'ouverture en mode 'a' : ouvrir CRÉE le fichier,
    donc `Path(...).exists()` serait toujours vrai ensuite. C'était le second
    maillon du bug de corruption.
    """
    agg_vide = (not Path(RESULTS_CSV).exists()
                or Path(RESULTS_CSV).stat().st_size == 0)
    det_vide = (not Path(DETAIL_CSV).exists()
                or Path(DETAIL_CSV).stat().st_size == 0)

    fa = open(RESULTS_CSV, "a", newline="", encoding="utf-8-sig")
    wa = csv.DictWriter(fa, fieldnames=AGG_FIELDNAMES)
    if agg_vide:
        wa.writeheader()

    fd = open(DETAIL_CSV, "a", newline="", encoding="utf-8-sig")
    wd = csv.DictWriter(fd, fieldnames=DETAIL_FIELDNAMES)
    if det_vide:
        wd.writeheader()

    if start_id > 0:
        print(f"♻️  Reprise : {start_id} essai(s) déjà présents, "
              f"on continue à partir du Trial_ID {start_id}.")
    return fa, wa, fd, wd


def _write_sorted():
    try:
        df = pd.read_csv(RESULTS_CSV, dtype={"Trial_ID": str}, low_memory=False)
        df = df[df["Trial_ID"] != "Trial_ID"]           # ceinture et bretelles
        df["Robust_Score"] = pd.to_numeric(df["Robust_Score"], errors="coerce")
        df.dropna(subset=["Robust_Score"]).sort_values(
            "Robust_Score", ascending=False).to_csv(
            SORTED_CSV, index=False, encoding="utf-8-sig")
    except Exception:
        pass


def main():
    if not Path(PRICES_FILE).exists():
        print(f"❌ Fichier de prix {PRICES_FILE} introuvable.")
        return

    print(f"🕒 Masque RTH : {'ACTIF — aucun signal hors séance' if RTH_ONLY else 'DÉSACTIVÉ (comportement v9)'}")
    if not RTH_ONLY:
        print("   ⚠️ Sans masque, ~15 % des trades sont pris hors séance et ne "
              "seraient PAS exécutables en réel (ordre mis en file jusqu'à "
              "l'ouverture suivante).")

    frames = load_market_data()
    t0 = min(f["timestamp"].min() for f in frames.values())
    t_max = max(f["timestamp"].max() for f in frames.values())
    windows = cv.make_windows(t0, t_max)
    print("🗓️  " + cv.describe(windows))
    if len(windows) < 2:
        print("   ⚠️ Une seule fenêtre : historique trop court pour une vraie "
              "validation croisée (le score ne mesurera pas la régularité "
              "inter-périodes).")
    del frames

    start_id = resume_trial_id()
    fa, wa, fd, wd = _open_writers(start_id)

    imported_configs = load_imported_strategies() if start_id == 0 else []
    n_imported = len(imported_configs)

    rng = np.random.default_rng(SEED)

    def task_generator():
        current_id = start_id
        for p_import in imported_configs:
            yield (current_id, p_import)
            current_id += 1

        remaining_trials = max(0, N_TRIALS - n_imported)
        for _ in range(remaining_trials):
            yield (current_id, sample_params(rng))
            current_id += 1

    tasks = task_generator()

    t0 = time.time(); done = 0; best_score = -1e9
    print(f"⚙️  Démarrage de {N_WORKERS} workers pour {N_TRIALS} essais...")

    with mp.Pool(N_WORKERS, initializer=_worker_init) as pool:
        for agg, detail_rows in pool.imap_unordered(_eval_trial, tasks, chunksize=4):
            if not agg: continue
            wa.writerow(agg); fa.flush()
            for row in detail_rows: wd.writerow(row)
            fd.flush()
            done += 1
            if agg["Robust_Score"] > best_score: best_score = agg["Robust_Score"]

            if done % PROGRESS_EVERY == 0:
                rate = done / (time.time() - t0)
                eta = (N_TRIALS - done) / rate if rate else 0
                print(f"  {done:>7}/{N_TRIALS} | {rate:.1f} essais/s | "
                      f"Robust_Score = {best_score:+.4f} | ETA {eta / 3600:.1f}h")
            if SORT_EVERY and done % SORT_EVERY == 0: _write_sorted()

    _write_sorted(); fa.close(); fd.close()
    print(f"\n✅ Session terminée. {done} essais évalués.")


if __name__ == "__main__":
    main()
