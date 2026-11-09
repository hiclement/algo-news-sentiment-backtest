import os
import time
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dotenv import load_dotenv

from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed, Adjustment

# ==============================================================================
# CONFIGURATION
# ==============================================================================
load_dotenv()
ALPACA_KEY = os.getenv("ALPACA_API_KEY_ID")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY")

NEWS_FILE = "alpaca_news_test.parquet"      # source des tickers
OUTPUT_DIR = Path("stock_data")             # un fichier .parquet par ticker
OUTPUT_FILE = "alpaca_stock_1min.parquet"   # fichier fusionné final

DAYS_BACK = 900 + 183  # ~548 jours (365 news + 6 mois de contexte pré-news)

# Ne garder que les tickers ayant AU MOINS ce nombre de news MONO-TICKER.
# On ne sélectionne QUE sur les news à un seul symbole : c'est la population
# exploitable pour une étude d'influence, et ça exclut d'office les ETF
# (SPY, QQQ, SMH...) qui sont presque toujours co-tagués.
MIN_NEWS_COUNT = 20

# ==============================================================================
# SÉLECTION DES TICKERS (news mono-ticker uniquement)
# ==============================================================================
def monoticker_counts(news_file):
    """Series {ticker: nb de news MONO-ticker}, triée desc."""
    df = pd.read_parquet(news_file, columns=["symbols"])
    n_symbols = df["symbols"].apply(lambda s: len(s) if s is not None else 0)
    mono = df[n_symbols == 1].copy()
    mono["ticker"] = mono["symbols"].apply(lambda s: str(s[0]).strip())
    mono = mono[mono["ticker"] != ""]
    return mono["ticker"].value_counts()


def preview_thresholds(counts, total_news, mono_news):
    pct = 100 * mono_news / total_news if total_news else 0
    print(f"   News mono-ticker : {mono_news:,}/{total_news:,} ({pct:.0f}%)")
    print("   Distribution (aide au choix de MIN_NEWS_COUNT) :")
    for thr in [1, 5, 10, 20, 50, 100]:
        n = int((counts >= thr).sum())
        est = n * 390 * int(DAYS_BACK * 5 / 7)
        print(f"      >= {thr:>3} news mono : {n:>4} tickers  (~{est:>12,} barres)")


# ==============================================================================
# TÉLÉCHARGEMENT
# ==============================================================================
def download_ticker(client, ticker, start_date, end_date):
    all_dfs = []
    current_start = start_date

    # Découpage par blocs de 30 jours pour éviter les Timeouts de l'API
    while current_start < end_date:
        current_end = min(current_start + timedelta(days=30), end_date)
        
        request = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Minute,
            start=current_start,
            end=current_end,
            adjustment=Adjustment.ALL,
            feed=DataFeed.IEX  # À passer en DataFeed.SIP si vous prenez l'abonnement
        )
        
        try:
            bars = client.get_stock_bars(request)
            if not bars.df.empty:
                df_block = bars.df.xs(ticker, level=0)
                all_dfs.append(df_block)
        except Exception as e:
            # On ignore silencieusement les erreurs de type "pas de données sur cette période"
            if "no data" not in str(e).lower():
                print(f"\n   ❌ Erreur {ticker} ({current_start.strftime('%Y-%m')}): {e}")
        
        current_start = current_end
        
        # SÉCURITÉ PLAN GRATUIT : 0.35s = ~171 requêtes/minute (Reste sous la barre des 200)
        time.sleep(0.35) 

    if not all_dfs:
        return pd.DataFrame()

    df = pd.concat(all_dfs)
    # Nettoyage des éventuels doublons aux limites des blocs de 30 jours
    df = df[~df.index.duplicated(keep='first')]
    df['ticker'] = ticker
    return df


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    if not ALPACA_KEY or not ALPACA_SECRET:
        print("❌ Clés Alpaca manquantes dans .env")
        return
    if not Path(NEWS_FILE).exists():
        print(f"❌ Fichier news introuvable : {NEWS_FILE}. Lance d'abord le scraper de news.")
        return

    # 1. Sélection sur les news mono-ticker
    total_news = len(pd.read_parquet(NEWS_FILE, columns=["symbols"]))
    counts = monoticker_counts(NEWS_FILE)
    print(f"🎯 {len(counts)} tickers ont au moins une news mono-ticker")
    preview_thresholds(counts, total_news, int(counts.sum()))

    tickers = counts[counts >= MIN_NEWS_COUNT].index.tolist()
    if not tickers:
        print(f"\n⚠️ Aucun ticker avec >= {MIN_NEWS_COUNT} news mono. Baisse MIN_NEWS_COUNT.")
        return
    print(f"\n✅ Seuil >= {MIN_NEWS_COUNT} → {len(tickers)} tickers à télécharger")

    # 2. Reprise : un fichier par ticker, on saute ceux déjà présents
    OUTPUT_DIR.mkdir(exist_ok=True)
    todo = [t for t in tickers if not (OUTPUT_DIR / f"{t}.parquet").exists()]
    done = len(tickers) - len(todo)
    if done:
        print(f"   ({done} déjà téléchargés, {len(todo)} restants)")

    client = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=DAYS_BACK)
    print(f"   Période : {start_date:%Y-%m-%d} → {end_date:%Y-%m-%d} UTC\n")

    # 3. Téléchargement : chaque ticker écrit dans SON fichier (coût constant)
    for i, ticker in enumerate(todo, 1):
        t0 = time.time()
        print(f"[{i}/{len(todo)}] {ticker:>6s} ...", end="", flush=True)
        df = download_ticker(client, ticker, start_date, end_date)
        if df.empty:
            print(" ⚠️ aucune donnée (absent d'IEX ou delisté)")
            continue
        df.to_parquet(OUTPUT_DIR / f"{ticker}.parquet", index=True, compression="zstd")
        print(f" ✅ {len(df):>8,} barres | "
              f"{df.index.min():%Y-%m-%d} → {df.index.max():%Y-%m-%d} | {time.time()-t0:.0f}s")

    # 4. Fusion finale en une seule passe (fichier compatible build_labels.py)
    print(f"\n🔀 Fusion des fichiers par ticker...")
    parts = []
    for t in tickers:
        p = OUTPUT_DIR / f"{t}.parquet"
        if p.exists():
            parts.append(pd.read_parquet(p))
    if not parts:
        print("⚠️ Rien à fusionner.")
        return
    df_total = pd.concat(parts)
    df_total.to_parquet(OUTPUT_FILE, index=True, compression="zstd")

    print(f"{'='*60}")
    print(f"✅ {len(df_total):,} barres | {df_total['ticker'].nunique()} tickers → '{OUTPUT_FILE}'")
    print(f"📅 {df_total.index.min()} → {df_total.index.max()}")


if __name__ == "__main__":
    main()