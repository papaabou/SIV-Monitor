"""
Simulation de la télémétrie d'un Système d'Information Voyageurs (SIV) embarqué.

Le simulateur produit une série temporelle (1 mesure / minute) avec des
comportements normaux réalistes (cycle jour/nuit, charge CPU aux heures de
pointe, passages en tunnel qui dégradent le GPS, régulation du ventilateur)
et injecte des incidents étiquetés, ce qui permet d'évaluer objectivement
les méthodes de détection.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SENSORS = ["temperature", "cpu_load", "fan_rpm", "voltage", "latency_ms", "gps_snr"]

INCIDENT_TYPES = {
    "surchauffe": "Pic de température brutal",
    "panne_ventilateur": "Ventilateur défaillant, échauffement progressif",
    "alimentation_instable": "Oscillations de la tension d'alimentation",
    "degradation_reseau": "Latence réseau qui se dégrade",
    "capteur_bloque": "Capteur de température figé sur une valeur",
}


@dataclass
class SimulationConfig:
    n_minutes: int = 3 * 24 * 60
    incident_rate_per_day: float = 12.0
    seed: int | None = 42
    start: str = "2026-01-05 00:00"


def _normal_behaviour(n: int, rng: np.random.Generator) -> pd.DataFrame:
    minutes = np.arange(n)
    hour = (minutes / 60) % 24

    # Température ambiante : cycle jour/nuit entre ~18 et ~30 °C
    ambient = 24 + 6 * np.sin((hour - 9) / 24 * 2 * np.pi) + rng.normal(0, 0.4, n)

    # Charge CPU plus forte aux heures de pointe (7-9h, 17-19h)
    rush = np.exp(-((hour - 8) ** 2) / 2) + np.exp(-((hour - 18) ** 2) / 2)
    cpu = np.clip(30 + 25 * rush + rng.normal(0, 5, n), 5, 100)

    # Température interne : dépend de l'ambiant et de la charge, lissée (inertie)
    raw_temp = ambient + 0.45 * cpu + 8
    temperature = pd.Series(raw_temp).ewm(alpha=0.15).mean().to_numpy() + rng.normal(0, 0.6, n)

    # Régulation : le ventilateur accélère avec la température
    fan = np.clip(900 + 45 * (temperature - 35), 600, 4000) + rng.normal(0, 60, n)

    voltage = 24 + rng.normal(0, 0.15, n)
    latency = rng.lognormal(mean=np.log(45), sigma=0.25, size=n)
    gps = 36 + rng.normal(0, 1.5, n)

    # Passages en tunnel : chute du GPS normale, ce n'est pas une panne
    for start in rng.choice(n, size=max(1, n // 240), replace=False):
        length = rng.integers(2, 6)
        gps[start:start + length] = rng.normal(14, 2, size=len(gps[start:start + length]))

    return pd.DataFrame({
        "temperature": temperature,
        "cpu_load": cpu,
        "fan_rpm": fan,
        "voltage": voltage,
        "latency_ms": latency,
        "gps_snr": gps,
    })


def _inject(df: pd.DataFrame, kind: str, start: int, length: int, rng: np.random.Generator) -> None:
    end = min(start + length, len(df))
    idx = df.index[start:end]
    k = np.arange(len(idx))

    if kind == "surchauffe":
        df.loc[idx, "temperature"] += rng.uniform(18, 30)
    elif kind == "panne_ventilateur":
        df.loc[idx, "fan_rpm"] = rng.normal(350, 40, len(idx))
        df.loc[idx, "temperature"] += 0.6 * k
    elif kind == "alimentation_instable":
        df.loc[idx, "voltage"] = 24 + 1.6 * np.sin(k * 1.7) + rng.normal(0, 0.3, len(idx))
    elif kind == "degradation_reseau":
        df.loc[idx, "latency_ms"] *= np.linspace(2.0, 6.0, len(idx))
    elif kind == "capteur_bloque":
        df.loc[idx, "temperature"] = df.loc[idx[0], "temperature"]

    df.loc[idx, "label"] = 1
    df.loc[idx, "incident"] = kind


def simulate(config: SimulationConfig | None = None) -> pd.DataFrame:
    """Génère la télémétrie avec des incidents étiquetés (colonnes `label` et `incident`)."""
    config = config or SimulationConfig()
    rng = np.random.default_rng(config.seed)
    n = config.n_minutes

    df = _normal_behaviour(n, rng)
    df["label"] = 0
    df["incident"] = "aucun"

    n_incidents = rng.poisson(config.incident_rate_per_day * n / (24 * 60))
    kinds = list(INCIDENT_TYPES)
    for _ in range(n_incidents):
        kind = kinds[rng.integers(len(kinds))]
        length = int(rng.integers(8, 30))
        start = int(rng.integers(60, max(61, n - length)))
        if df["label"].iloc[start:start + length].any():
            continue
        _inject(df, kind, start, length, rng)

    df.insert(0, "timestamp", pd.date_range(config.start, periods=n, freq="min"))
    df[SENSORS] = df[SENSORS].round(2)
    return df
