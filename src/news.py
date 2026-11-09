import os
import time
import html
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest
from alpaca.common.exceptions import APIError

# ==============================================================================
# CONFIGURATION
# ==============================================================================
load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY_ID")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")

OUTPUT_FILE = "alpaca_news_test.parquet"
DAYS_BACK = 900 
WINDOW_DAYS = 30          # découpage pour sauvegarde incrémentale + reprise
MAX_RETRIES = 5

# Export de la liste des tickers impactés (pour alimenter le scraper de prix)
TICKERS_CSV = "tickers_impactes.csv"
MIN_NEWS_COUNT = 1        # ne garder que les tickers cités dans >= N news
                          # (monte à 5-10 pour éliminer le bruit illiquide)

TICKERS_CIBLES = ["AAPL", "TSLA", "MSFT", "NVDA", "NFLX", "AMD", "AMZN", "GOOGL", "META"]

MARKET_OPEN_UTC = 14 * 60 + 30
MARKET_CLOSE_UTC = 21 * 60

# ==============================================================================
# TRAITEMENT
# ==============================================================================
def get_market_session(dt):
    minutes_day = dt.hour * 60 + dt.minute
    if dt.weekday() >= 5:
        return "weekend"
    elif minutes_day < MARKET_OPEN_UTC:
        return "pre-market"
    elif minutes_day >= MARKET_CLOSE_UTC:
        return "after-hours"
    else:
        return "regular"

def process_article(article):
    # html.unescape : Benzinga renvoie des entités HTML (&#8217; -> ')
    headline = html.unescape(getattr(article, "headline", "") or "")
    content = html.unescape(getattr(article, "content", "") or "")
    summary = html.unescape(getattr(article, "summary", "") or "")
    created_at = getattr(article, "created_at", None)
    symbols = getattr(article, "symbols", []) or []

    published_timestamp = None
    weekday = month = quarter = None
    market_session = "unknown"
    minutes_to_market_open = minutes_to_market_close = None

    if created_at:
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        elif isinstance(created_at, datetime) and created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)

        published_timestamp = int(created_at.timestamp())
        weekday = created_at.weekday()
        month = created_at.month
        quarter = (created_at.month - 1) // 3 + 1
        market_session = get_market_session(created_at)
        minutes_day = created_at.hour * 60 + created_at.minute
        minutes_to_market_open = MARKET_OPEN_UTC - minutes_day
        minutes_to_market_close = MARKET_CLOSE_UTC - minutes_day

    return {
        "id": getattr(article, "id", None),
        "headline": headline,
        "summary": summary,
        "content": content,
        "author": getattr(article, "author", None),
        "created_at": str(created_at) if created_at else None,
        "published_timestamp": published_timestamp,
        "updated_at": str(getattr(article, "updated_at", None)),
        "url": getattr(article, "url", None),
        "source": getattr(article, "source", None),
        "symbols": list(symbols),
        "num_symbols": len(symbols),
        "headline_length": len(headline),
        "content_length": len(content),
        "has_content": len(content.strip()) > 0,
        "weekday": weekday,
        "month": month,
        "quarter": quarter,
        "market_session": market_session,
        "minutes_to_market_open": minutes_to_market_open,
        "minutes_to_market_close": minutes_to_market_close,
    }

def save_to_parquet(rows, output_path):
    df_new = pd.DataFrame(rows)
    if Path(output_path).exists():
        df_old = pd.read_parquet(output_path)
        df_total = pd.concat([df_old, df_new], ignore_index=True)
        df_total = df_total.drop_duplicates(subset="id", keep="first")
    else:
        df_total = df_new
    df_total.to_parquet(output_path, index=False, compression="zstd")
    return len(df_total)

def export_impacted_tickers(parquet_path, csv_path, min_count=MIN_NEWS_COUNT):
    """Explose le champ `symbols` de toutes les news et produit un CSV listant
    chaque ticker impacté, avec le nombre de news qui le mentionnent et la
    plage de dates. Trié par fréquence pour pouvoir seuiller facilement."""
    df = pd.read_parquet(parquet_path)

    # Une ligne par (news, ticker)
    exploded = df[["id", "symbols", "created_at"]].explode("symbols")
    exploded = exploded.dropna(subset=["symbols"])
    exploded = exploded[exploded["symbols"].astype(str).str.strip() != ""]

    agg = (
        exploded.groupby("symbols")
        .agg(
            news_count=("id", "nunique"),
            first_seen=("created_at", "min"),
            last_seen=("created_at", "max"),
        )
        .reset_index()
        .rename(columns={"symbols": "ticker"})
    )
    agg = agg[agg["news_count"] >= min_count]
    agg = agg.sort_values("news_count", ascending=False).reset_index(drop=True)
    agg.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return agg


def fetch_news(client, request, max_retries=MAX_RETRIES):
    """get_news gère DÉJÀ toute la pagination interne (pages de 50) jusqu'à
    épuisement de la fenêtre, tant qu'on ne passe PAS de `limit`.
    Ici on ne gère que le retry sur erreur transitoire (429 / 5xx)."""
    for attempt in range(max_retries):
        try:
            return client.get_news(request)
        except APIError as e:
            code = getattr(e, "status_code", None)
            if code == 429 or (code is not None and 500 <= code < 600):
                wait = 2 ** attempt
                print(f" [rate limit/{code}, retry {wait}s]", end="", flush=True)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"Échec après {max_retries} tentatives")

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    if not API_KEY or not API_SECRET:
        print("❌ Clés Alpaca manquantes.")
        return

    if Path(OUTPUT_FILE).exists():
        Path(OUTPUT_FILE).unlink()   # repartir propre

    client = NewsClient(API_KEY, API_SECRET)

    global_end = datetime.now(timezone.utc)
    global_start = global_end - timedelta(days=DAYS_BACK)

    # Fenêtres de WINDOW_DAYS jours
    windows = []
    w_start = global_start
    while w_start < global_end:
        w_end = min(w_start + timedelta(days=WINDOW_DAYS), global_end)
        windows.append((w_start, w_end))
        w_start = w_end

    print(f"🚀 {DAYS_BACK} jours de news en {len(windows)} fenêtres de ~{WINDOW_DAYS}j")
    print(f"   Tickers : {', '.join(TICKERS_CIBLES)}")
    print(f"   Période : {global_start:%Y-%m-%d} → {global_end:%Y-%m-%d} UTC\n")

    total = 0
    oldest_seen = newest_seen = None

    for i, (w_start, w_end) in enumerate(windows, 1):
        print(f"[{i:>2}/{len(windows)}] {w_start:%Y-%m-%d} → {w_end:%Y-%m-%d} ...", end="", flush=True)

        request = NewsRequest(
            symbols=",".join(TICKERS_CIBLES),
            start=w_start,
            end=w_end,
            # PAS de `limit` -> le SDK pagine jusqu'au bout de la fenêtre.
            include_content=True,
            exclude_contentless=False,
            sort="desc",
        )

        try:
            response = fetch_news(client, request)
        except Exception as e:
            print(f" ❌ {e}")
            continue

        news_list = response.data.get("news", []) if hasattr(response, "data") else []
        print(f" {len(news_list)} articles", end="")

        rows = []
        for article in news_list:
            rows.append(process_article(article))
            ca = getattr(article, "created_at", None)
            if ca is not None:
                if isinstance(ca, str):
                    ca = datetime.fromisoformat(ca.replace("Z", "+00:00"))
                elif ca.tzinfo is None:
                    ca = ca.replace(tzinfo=timezone.utc)
                oldest_seen = ca if oldest_seen is None or ca < oldest_seen else oldest_seen
                newest_seen = ca if newest_seen is None or ca > newest_seen else newest_seen

        if rows:
            n_in_file = save_to_parquet(rows, OUTPUT_FILE)
            total += len(rows)
            print(f" 💾 ({n_in_file:,} au total)")
        else:
            print()

        time.sleep(0.35)

    print(f"\n✅ Terminé ! {total:,} articles téléchargés (avant dédup).")
    if Path(OUTPUT_FILE).exists():
        df = pd.read_parquet(OUTPUT_FILE)
        print(f"📊 {len(df):,} articles UNIQUES dans '{OUTPUT_FILE}'.")
    if oldest_seen and newest_seen:
        span = (newest_seen - oldest_seen).days
        print(f"📅 Plage récupérée : {oldest_seen:%Y-%m-%d} → {newest_seen:%Y-%m-%d} (~{span} jours).")

    # --- Export de la liste des tickers impactés (pour le scraper de prix) ---
    if Path(OUTPUT_FILE).exists():
        print(f"\n🎯 Extraction des tickers impactés (min {MIN_NEWS_COUNT} news)...")
        tickers_df = export_impacted_tickers(OUTPUT_FILE, TICKERS_CSV)
        print(f"   {len(tickers_df):,} tickers uniques → '{TICKERS_CSV}'")
        print("   Top 15 par nombre de news :")
        for _, r in tickers_df.head(15).iterrows():
            print(f"     {r['ticker']:>6s} : {r['news_count']:>6,} news")
        top = tickers_df["ticker"].tolist()
        apercu = top[:50] + (["..."] if len(top) > 50 else [])
        print(f"\n   À coller dans le scraper de prix (TICKERS = [...]) :")
        print(f"   {apercu}")

if __name__ == "__main__":
    main()