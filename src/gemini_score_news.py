"""
gemini_score_news.py — Scoring de sentiment des news, UNE seule passe (caché).

Idée clé : le score d'une news ne dépend PAS des paramètres de trading, donc on
le calcule une fois pour toutes et on le stocke. La boucle d'optimisation lira ce
fichier sans jamais rappeler Gemini.

Sortie : news_scores.parquet  (colonnes : id, news_score [-1..1], impact [0..1], model)

Économie de tokens :
  - modèle léger 'gemini-2.5-flash-lite' (gratuit avec rate limits, ou ~0.10$/1M in)
  - on n'envoie que le HEADLINE (court) -> quelques dizaines de tokens par news
  - reprise : on ne re-score jamais un id déjà présent dans la sortie

Prérequis :  pip install google-genai
             variable d'environnement GEMINI_API_KEY
"""

import os
import re
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from dotenv import load_dotenv

from google import genai
from google.genai import types
from google.genai.errors import APIError

# ==============================================================================
# CONFIGURATION
# ==============================================================================
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

NEWS_FILE = "alpaca_news_test.parquet"
OUTPUT_FILE = "news_scores.parquet"     # pour le code (rapide)
OUTPUT_CSV = "news_scores.csv"          # miroir lisible (avec le titre)
# Modèle léger par défaut. Google retire ses modèles souvent : si celui-ci
# n'est plus dispo, pick_model() en choisit automatiquement un autre Flash-Lite.
PREFERRED_MODEL = "gemini-3.1-flash-lite"

MONO_ONLY = True          # ne scorer que les news mono-ticker (celles exploitables)
N_WORKERS = 8             # appels Gemini en parallèle (tier payant -> monte à 16-32)
SAVE_EVERY = 100          # sauvegarde incrémentale
MAX_RETRIES = 5

SYSTEM = (
    "Tu es un analyste financier. On te donne le titre d'une actualité boursière "
    "concernant une action. Évalue son impact ATTENDU sur le cours à court terme. "
    "Réponds UNIQUEMENT en JSON avec deux champs : "
    "'sentiment' (nombre de -1.0 = très baissier à +1.0 = très haussier, 0 = neutre) "
    "et 'impact' (0.0 = non-événement à 1.0 = catalyseur majeur). Aucun autre texte."
)

# ==============================================================================
# SÉLECTION AUTOMATIQUE DU MODÈLE (robuste aux retraits fréquents de Google)
# ==============================================================================
def pick_model(client, preferred=PREFERRED_MODEL):
    """Choisit un Flash-Lite réellement disponible pour cette clé (évite les 404)."""
    try:
        models = list(client.models.list())
    except Exception as e:
        print(f"   (listing impossible : {e} — on tente {preferred})")
        return preferred

    def usable(m):
        acts = getattr(m, "supported_actions", None)
        return (not acts) or ("generateContent" in acts)

    names = [m.name.split("/")[-1] for m in models if usable(m)]
    exclude = ("image", "tts", "live", "audio", "robotics", "embedding", "vision")

    def clean(pool):
        return sorted([n for n in pool if not any(x in n for x in exclude)], reverse=True)

    if preferred in names:
        return preferred
    for pool in (
        clean([n for n in names if "flash-lite" in n and "preview" not in n]),
        clean([n for n in names if "flash-lite" in n]),
        clean([n for n in names if "flash" in n and "preview" not in n]),
    ):
        if pool:
            return pool[0]
    raise RuntimeError(f"Aucun modèle Flash utilisable. Dispo : {names[:15]}")


# ==============================================================================
# SCORING
# ==============================================================================
def parse_score(text):
    """Extrait {sentiment, impact}. Lève ValueError sur texte vide (jamais TypeError)."""
    if not text:
        raise ValueError("réponse vide")
    text = re.sub(r"```(json)?", "", text).strip()
    data = json.loads(text)
    sent = float(data.get("sentiment", 0.0))
    impact = float(data.get("impact", 0.0))
    return max(-1.0, min(1.0, sent)), max(0.0, min(1.0, impact))


def _generate(client, model, headline, thinking_level):
    cfg = dict(
        system_instruction=SYSTEM,
        temperature=0.0,
        max_output_tokens=512,        # assez pour le "thinking" 3.x + le JSON court
        response_mime_type="application/json",
    )
    if thinking_level is not None:
        cfg["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)
    return client.models.generate_content(
        model=model, contents=f"Titre : {headline}",
        config=types.GenerateContentConfig(**cfg),
    )


def score_headline(client, headline, model):
    """Un appel Gemini -> (sentiment, impact). INCASSABLE : ne lève jamais sur une
    réponse vide ou non-JSON ; retente les erreurs transitoires ; (0,0) en dernier recours."""
    thinking = "low"                  # réduit le raisonnement interne (économie de tokens)
    for attempt in range(MAX_RETRIES):
        try:
            resp = _generate(client, model, headline, thinking)
            text = getattr(resp, "text", None)
            if not text:              # thinking a tout consommé, ou blocage safety
                if attempt < MAX_RETRIES - 1:
                    time.sleep(0.3); continue
                return 0.0, 0.0
            return parse_score(text)
        except APIError as e:
            code = getattr(e, "code", None)
            if code == 429 or (isinstance(code, int) and code >= 500):
                print(f" [rate limit/{code}, retry]", end="", flush=True)
                time.sleep(2 ** attempt); continue
            if code == 400 and "thinking" in str(e).lower() and thinking is not None:
                thinking = None       # ce modèle ne gère pas thinking_level -> on l'enlève
                continue
            raise
        except (json.JSONDecodeError, ValueError):
            if attempt < MAX_RETRIES - 1:
                time.sleep(0.3); continue
            return 0.0, 0.0
    return 0.0, 0.0


# ==============================================================================
# MAIN
# ==============================================================================
def load_news_to_score():
    df = pd.read_parquet(NEWS_FILE, columns=["id", "headline", "symbols", "num_symbols"])
    if MONO_ONLY:
        df = df[df["num_symbols"] == 1]
    df = df.dropna(subset=["headline"]).drop_duplicates(subset="id")
    return df[["id", "headline"]].reset_index(drop=True)


def already_done():
    if not Path(OUTPUT_FILE).exists():
        return pd.DataFrame(columns=["id", "news_score", "impact", "model"])
    return pd.read_parquet(OUTPUT_FILE)


# mapping id -> titre, pour enrichir le CSV lisible (rempli dans main)
_HEADLINES = {}


def _flush(buffer):
    """Écrit le parquet (pour le code) + le CSV lisible (avec titre)."""
    new = pd.DataFrame(buffer)
    if Path(OUTPUT_FILE).exists():
        old = pd.read_parquet(OUTPUT_FILE)
        new = pd.concat([old, new], ignore_index=True).drop_duplicates(subset="id", keep="last")
    new.to_parquet(OUTPUT_FILE, index=False, compression="zstd")

    csv = new.copy()
    csv["headline"] = csv["id"].map(_HEADLINES)
    csv = csv[["id", "news_score", "impact", "headline", "model"]]
    csv.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    return len(new)


def main():
    if not API_KEY:
        print("❌ GEMINI_API_KEY manquante (mets-la dans .env).")
        return
    if not Path(NEWS_FILE).exists():
        print(f"❌ {NEWS_FILE} introuvable.")
        return

    news = load_news_to_score()
    _HEADLINES.update(dict(zip(news["id"], news["headline"])))
    done_ids = set(already_done()["id"])
    todo = news[~news["id"].isin(done_ids)].reset_index(drop=True)
    print(f"📰 {len(news):,} news | {len(done_ids):,} déjà faites | {len(todo):,} restantes")
    if todo.empty:
        print("✅ Rien à faire, tout est déjà scoré.")
        return

    client = genai.Client(api_key=API_KEY)
    model = pick_model(client)
    print(f"🤖 Modèle : {model} | {N_WORKERS} workers en parallèle")

    rows = list(todo.to_dict("records"))
    buffer, done_n, t0 = [], 0, time.time()

    # Appels Gemini en parallèle ; la collecte + la sauvegarde restent dans le
    # thread principal (pas de lock nécessaire, un seul écrivain).
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(score_headline, client, r["headline"], model): r for r in rows}
        for fut in as_completed(futs):
            r = futs[fut]
            try:
                sent, impact = fut.result()
            except Exception as e:
                print(f"\n   ⚠️ {r['id']} : {e} -> score neutre")
                sent, impact = 0.0, 0.0
            buffer.append({"id": r["id"], "news_score": sent, "impact": impact, "model": model})
            done_n += 1

            if done_n % 50 == 0 or done_n == len(rows):
                rate = done_n / max(time.time() - t0, 1e-9)
                eta = (len(rows) - done_n) / max(rate, 1e-9)
                print(f"  {done_n}/{len(rows)}  ({rate:.1f} news/s, ETA {eta/60:.1f} min)")

            if len(buffer) >= SAVE_EVERY:
                total = _flush(buffer); buffer = []
                print(f"   💾 {total:,} scores enregistrés")

    if buffer:
        total = _flush(buffer)
        print(f"   💾 {total:,} scores enregistrés")

    final = pd.read_parquet(OUTPUT_FILE)
    print(f"\n✅ {len(final):,} news scorées → '{OUTPUT_FILE}' (+ '{OUTPUT_CSV}' lisible)")
    print(f"   sentiment moyen : {final['news_score'].mean():+.3f} | "
          f"impact moyen : {final['impact'].mean():.3f}")


if __name__ == "__main__":
    main()