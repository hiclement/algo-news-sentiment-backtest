"""
repair_results_csv.py — Répare un backtest_results_*.csv abîmé.

Corrige ce que le bug de reprise de benchmark_v9 a produit :
  • lignes d'en-tête réinsérées au milieu des données (→ DtypeWarning,
    « Columns (0: Trial_ID) have mixed types ») ;
  • essais dupliqués (les stratégies importées réévaluées à chaque reprise ratée).

    python repair_results_csv.py backtest_results_v10.csv

L'original est sauvegardé en .bak avant toute écriture.
"""

import shutil
import sys
from pathlib import Path

import pandas as pd


def repair(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"❌ {path} introuvable.")

    df = pd.read_csv(path, dtype=str, low_memory=False)
    n0 = len(df)

    # 1) Lignes d'en-tête réinsérées : leur Trial_ID vaut littéralement "Trial_ID"
    entetes = df["Trial_ID"] == "Trial_ID"
    df = df[~entetes]

    # 2) Lignes vides / tronquées (écriture interrompue)
    df = df.dropna(subset=["Trial_ID", "Robust_Score"])

    # 3) Doublons d'essais. Attention : on dédoublonne sur les HYPERPARAMÈTRES,
    #    pas sur le Trial_ID — après une reprise ratée, deux essais distincts ont
    #    pu recevoir le même id, et un même essai importé a pu être réévalué.
    params = ["w_trend", "w_vol", "w_volat", "w_news", "trend_window",
              "volat_window", "stop_loss", "take_profit", "buy_threshold",
              "sell_threshold"]
    params = [c for c in params if c in df.columns]
    n_avant = len(df)
    df = df.drop_duplicates(subset=params, keep="first")
    n_dup = n_avant - len(df)

    # 4) Recasting numérique (Trial_ID reste en texte : les essais importés
    #    portent un NOM, pas un entier — c'est légitime.)
    for c in df.columns:
        if c != "Trial_ID":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Robust_Score"])

    shutil.copy(path, path.with_suffix(path.suffix + ".bak"))
    df.to_csv(path, index=False, encoding="utf-8-sig")

    print(f"✅ {path.name} réparé")
    print(f"   {n0} lignes lues")
    print(f"   − {int(entetes.sum())} en-tête(s) réinséré(s) au milieu du fichier")
    print(f"   − {n_dup} essai(s) dupliqué(s) (mêmes hyperparamètres)")
    print(f"   = {len(df)} essais valides")
    print(f"   Sauvegarde de l'original : {path.name}.bak")

    num = pd.to_numeric(df["Trial_ID"], errors="coerce").dropna()
    nommes = len(df) - len(num)
    if nommes:
        print(f"   ℹ️  {nommes} essai(s) ont un Trial_ID textuel (stratégies "
              f"importées) — c'est normal, le correctif de resume_trial_id() "
              f"les gère désormais.")


if __name__ == "__main__":
    cible = Path(sys.argv[1] if len(sys.argv) > 1 else "backtest_results_v10.csv")
    repair(cible)
    trie = Path(str(cible).replace("results", "results_sorted"))
    if trie.exists() and trie != cible:
        repair(trie)
