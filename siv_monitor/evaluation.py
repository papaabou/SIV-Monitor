"""Mesure des performances des détecteurs sur des données étiquetées."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def incidents(df: pd.DataFrame) -> list[tuple[int, int, str]]:
    """Liste des incidents (début, fin exclue, type) à partir de la colonne `label`."""
    labels = df["label"].to_numpy()
    out, start = [], None
    for i, value in enumerate(np.append(labels, 0)):
        if value and start is None:
            start = i
        elif not value and start is not None:
            out.append((start, i, df["incident"].iloc[start]))
            start = None
    return out


def false_alarm_episodes(df: pd.DataFrame, pred: np.ndarray) -> int:
    """Nombre d'épisodes d'alerte qui ne recouvrent aucun incident réel."""
    labels = df["label"].to_numpy()
    count, i, n = 0, 0, len(pred)
    while i < n:
        if pred[i]:
            j = i
            while j < n and pred[j]:
                j += 1
            if not labels[i:j].any():
                count += 1
            i = j
        else:
            i += 1
    return count


def evaluate(df: pd.DataFrame, pred: np.ndarray) -> dict:
    """
    - taux de détection : part des incidents dont au moins une minute est signalée
    - précision / rappel / F1 : calculés minute par minute
    - fausses alertes : épisodes d'alerte sans incident réel
    """
    y = df["label"].to_numpy()
    events = incidents(df)
    detected = [bool(pred[s:e].any()) for s, e, _ in events]

    by_type = {}
    for (s, e, kind), hit in zip(events, detected):
        total, ok = by_type.get(kind, (0, 0))
        by_type[kind] = (total + 1, ok + int(hit))

    return {
        "incidents": len(events),
        "incidents_detectes": int(sum(detected)),
        "taux_detection": float(np.mean(detected)) if events else 0.0,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "rappel": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "fausses_alertes": false_alarm_episodes(df, pred),
        "par_type": {k: v[1] / v[0] for k, v in sorted(by_type.items())},
    }
