"""
rth.py — Heures de séance régulières (Regular Trading Hours).

Partagé par le backtest ET le moteur live : c'est la SEULE façon de garantir
que les deux voient exactement les mêmes bougies « tradables ».

RTH = 09h30 → 16h00 heure de New York. On travaille en fuseau America/New_York
et non en UTC ou en heure de Paris : les États-Unis et l'Europe ne changent pas
d'heure les mêmes semaines, donc « 15h30 Paris » vaut 09h30 NY la plupart de
l'année, mais 10h30 NY pendant les semaines de décalage. Un filtre écrit en
heure de Paris se décale donc deux fois par an — silencieusement.

Principe retenu (et c'est le point crucial) :
    → les indicateurs sont calculés sur TOUTES les bougies (continuité préservée) ;
    → seuls les SIGNAUX sont annulés hors séance.

Ne PAS filtrer les bougies avant de calculer les indicateurs : cela créerait un
trou de 17h30 entre 16h00 et 09h30, et le ROC de la première bougie de séance
absorberait tout le gap overnight → un z_trend aberrant à l'ouverture, c'est-à-
dire exactement au moment où l'ordre partira vraiment. On remplacerait un faux
signal inoffensif (hors séance, non exécutable) par un faux signal coûteux.
"""

import numpy as np
import pandas as pd

NY = "America/New_York"
OPEN_H, OPEN_M = 9, 30
CLOSE_H, CLOSE_M = 16, 0

# Clôtures anticipées (13h00 NY) : veille de Thanksgiving, 24/12, 3/07…
EARLY_CLOSES = {
    "2025-11-28", "2025-12-24",
    "2026-07-03", "2026-11-27", "2026-12-24",
    "2027-11-26", "2027-12-23",
}


def rth_mask(timestamps) -> np.ndarray:
    """Booléen par bougie : True si la bougie est dans la séance régulière.

    `timestamps` : série/index datetime tz-aware (UTC ou autre).
    La borne haute est EXCLUE : une bougie estampillée 16h00 ouvre à 16h00 et
    n'appartient donc plus à la séance (les bougies Alpaca portent leur heure
    d'OUVERTURE).
    """
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True)).tz_convert(NY)
    minutes = ts.hour * 60 + ts.minute

    debut = OPEN_H * 60 + OPEN_M
    fin = np.full(len(ts), CLOSE_H * 60 + CLOSE_M)

    # Clôtures anticipées à 13h00
    jours = ts.strftime("%Y-%m-%d")
    early = np.isin(jours, list(EARLY_CLOSES))
    fin = np.where(early, 13 * 60, fin)

    ouvre = (minutes >= debut) & (minutes < fin)
    semaine = ts.dayofweek < 5              # lundi=0 … vendredi=4
    return np.asarray(ouvre & semaine)


def apply_rth(signal: np.ndarray, timestamps) -> np.ndarray:
    """Annule les signaux hors séance. Les indicateurs, eux, restent calculés
    sur toutes les bougies (continuité)."""
    return np.where(rth_mask(timestamps), signal, 0).astype(signal.dtype)
