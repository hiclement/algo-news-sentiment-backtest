"""
audit_rth.py — Combien ton backtest doit-il au hors-séance ?

À lancer AVANT de relancer 450 000 essais. Il répond à la seule question qui
compte : « si j'interdis le hors-séance, qu'est-ce que je perds ? »

    python audit_rth.py

Utilise tes propres fichiers (alpaca_stock_1min.parquet + strategies_to_test.csv,
ou à défaut les deux Champions codés en dur ci-dessous).
"""

import numpy as np
import pandas as pd
from pathlib import Path

import rth

PRICES_FILE = "alpaca_stock_1min.parquet"
STRATS_FILE = "strategies_to_test.csv"

COST_BPS = 5.0
VOLUME_MA = 20
_COST_FRAC = 2 * COST_BPS / 1e4

CHAMPIONS = [
    dict(name="Sécurité", w_trend=0.2622, w_vol=-0.185, w_volat=0.2041,
         w_news=0.1433, trend_window=34, volat_window=46, stop_loss=0.0404,
         take_profit=0.1104, buy_threshold=1.0399, sell_threshold=-1.5828),
    dict(name="Fréquence", w_trend=0.1419, w_vol=0.0091, w_volat=-0.1333,
         w_news=0.0633, trend_window=52, volat_window=27, stop_loss=0.0192,
         take_profit=0.0483, buy_threshold=0.2452, sell_threshold=-1.5862),
]


def _z(x, mask):
    tr = x[mask]
    mu, sd = np.nanmean(tr), np.nanstd(tr)
    if not np.isfinite(sd) or sd == 0:
        sd = 1.0
    return np.nan_to_num((x - mu) / sd, nan=0.0)


def simulate(open_p, close, high, low, sig, sl, tp, cost):
    """Copie de benchmark_v9.simulate (sans le coût dynamique, non nécessaire ici).
    Retourne les trades : (indice d'entrée, indice de sortie, rendement)."""
    n = len(close)
    out, t = [], 0
    while t < n - 2:
        s = sig[t]
        if s == 0:
            t += 1
            continue
        e = t + 1
        entry = open_p[e]
        if not np.isfinite(entry) or entry <= 0:
            t += 1
            continue
        fwd = slice(e + 1, n)
        if s == 1:
            hit_sl = low[fwd] <= entry * (1 - sl)
            hit_tp = high[fwd] >= entry * (1 + tp)
        else:
            hit_sl = high[fwd] >= entry * (1 + sl)
            hit_tp = low[fwd] <= entry * (1 - tp)
        cond = hit_sl | hit_tp | (sig[fwd] == -s)
        rel = int(np.argmax(cond)) if cond.any() else len(cond) - 1
        b = e + 1 + rel
        out.append((e, b, s * (close[b] - entry) / entry - cost))
        t = b
    return out


def main():
    if not Path(PRICES_FILE).exists():
        raise SystemExit(f"❌ {PRICES_FILE} introuvable (lancez ce script dans le "
                         "dossier de vos données).")

    prices = pd.read_parquet(PRICES_FILE).reset_index()
    ts_col = "timestamp" if "timestamp" in prices.columns else prices.columns[0]
    prices = prices.rename(columns={ts_col: "timestamp"})
    prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True)

    strategies = CHAMPIONS
    if Path(STRATS_FILE).exists():
        d = pd.read_csv(STRATS_FILE, sep=None, engine="python")
        strategies = d.to_dict("records")
        print(f"📥 {len(strategies)} stratégie(s) lue(s) dans {STRATS_FILE}\n")

    # ── 1) Part des bougies hors séance ──
    m = rth.rth_mask(prices["timestamp"])
    print("=" * 78)
    print(f"BOUGIES : {len(prices):,}".replace(",", " "))
    print(f"  en séance (RTH 09h30–16h00 NY) : {m.sum():>10,}  ({m.mean():6.1%})"
          .replace(",", " "))
    print(f"  HORS séance                    : {(~m).sum():>10,}  ({(~m).mean():6.1%})"
          .replace(",", " "))
    print("=" * 78)

    total = {"trades": 0, "hors": 0, "pnl": 0.0, "pnl_hors": 0.0}

    for strat in strategies:
        p = {k: strat[k] for k in
             ("w_trend", "w_vol", "w_volat", "w_news", "trend_window",
              "volat_window", "stop_loss", "take_profit", "buy_threshold",
              "sell_threshold")}
        nom = strat.get("name", strat.get("id", "?"))
        print(f"\n── {nom} " + "─" * (72 - len(str(nom))))

        for tk, g in prices.groupby("ticker"):
            g = g.sort_values("timestamp").reset_index(drop=True)
            n = len(g)
            if n < 5000:
                continue
            close = g["close"].to_numpy(float)
            openp = g["open"].to_numpy(float)
            high = g["high"].to_numpy(float)
            low = g["low"].to_numpy(float)
            vol = g["volume"].to_numpy(float)
            mk = rth.rth_mask(g["timestamp"])
            train = np.zeros(n, bool)
            train[: int(n * 0.66)] = True        # approximation du split TRAIN

            w = int(p["trend_window"])
            roc = np.full(n, np.nan)
            roc[w:] = close[w:] / close[:-w] - 1.0
            z_trend = _z(roc, train)

            vma = pd.Series(vol).rolling(VOLUME_MA, min_periods=VOLUME_MA).mean().to_numpy()
            ratio = np.divide(vol, vma, out=np.full(n, np.nan), where=vma > 0)
            z_vol = _z(ratio, train)

            logret = np.r_[np.nan, np.diff(np.log(np.maximum(close, 1e-10)))]
            v = pd.Series(logret).rolling(int(p["volat_window"]),
                                          min_periods=int(p["volat_window"])).std().to_numpy()
            z_volat = _z(v, train)

            score = (p["w_trend"] * z_trend + p["w_vol"] * z_vol
                     + p["w_volat"] * z_volat)          # news ignorées ici
            sig = np.where(score > p["buy_threshold"], 1,
                           np.where(score < p["sell_threshold"], -1, 0)).astype(np.int8)

            trades = simulate(openp, close, high, low, sig,
                              p["stop_loss"], p["take_profit"], _COST_FRAC)
            if not trades:
                continue
            entrees = np.array([t[0] for t in trades])
            rets = np.array([t[2] for t in trades])
            hors = ~mk[entrees]

            total["trades"] += len(trades)
            total["hors"] += int(hors.sum())
            total["pnl"] += float(rets.sum())
            total["pnl_hors"] += float(rets[hors].sum())

            print(f"  {tk:6} {len(trades):>5} trades | "
                  f"{hors.sum():>4} hors séance ({hors.mean():5.1%}) | "
                  f"PnL {rets.sum():+7.3f} dont {rets[hors].sum():+7.3f} hors séance")

            # Amplitude des z-scores : en séance vs hors séance
            if tk == list(prices["ticker"].unique())[0]:
                print(f"         z_vol |max| en séance {np.abs(z_vol[mk]).max():6.1f} | "
                      f"hors séance {np.abs(z_vol[~mk]).max():6.1f}")

    print("\n" + "=" * 78)
    t, h = total["trades"], total["hors"]
    print(f"TOTAL : {t} trades, dont {h} entrés HORS SÉANCE "
          f"({h / t:.1%})" if t else "aucun trade")
    if t:
        print(f"PnL total {total['pnl']:+.3f} — dont {total['pnl_hors']:+.3f} "
              f"provenant du hors-séance "
              f"({total['pnl_hors'] / total['pnl']:.1%} du résultat)")
        print("\nCE QUE ÇA VEUT DIRE :")
        print("  • Ces trades hors séance, ton courtier ne les aurait PAS exécutés")
        print("    au prix supposé : un ordre au marché passé hors séance chez Alpaca")
        print("    est mis en file et exécuté à L'OUVERTURE suivante.")
        print("  • Ton Robust_Score les compte pourtant comme de vrais gains.")
        print("  • Si la part ci-dessus est significative, tes paramètres sont")
        print("    optimisés sur des trades IMPOSSIBLES → il faut relancer la")
        print("    recherche avec le masque RTH (voir patch_benchmark_v9.py).")
    print("=" * 78)


if __name__ == "__main__":
    main()
