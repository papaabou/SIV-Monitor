"""
Deux approches de détection d'anomalies comparées :

- `RuleDetector`  : seuils fixes par capteur (approche classique de supervision)
- `MLDetector`    : Isolation Forest sur des variables construites
                    (moyennes/écarts-types glissants, variations, relations
                    entre capteurs), entraîné sans étiquettes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .simulator import SENSORS

DEFAULT_THRESHOLDS = {
    "temperature": (None, 75.0),
    "fan_rpm": (500.0, None),
    "voltage": (22.0, 26.0),
    "latency_ms": (None, 300.0),
    "gps_snr": (20.0, None),
}


@dataclass
class RuleDetector:
    thresholds: dict = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        alert = np.zeros(len(df), dtype=bool)
        for sensor, (low, high) in self.thresholds.items():
            if low is not None:
                alert |= df[sensor].to_numpy() < low
            if high is not None:
                alert |= df[sensor].to_numpy() > high
        return alert.astype(int)

    def reasons(self, row: pd.Series) -> list[str]:
        out = []
        for sensor, (low, high) in self.thresholds.items():
            if low is not None and row[sensor] < low:
                out.append(f"{sensor} < {low:g}")
            if high is not None and row[sensor] > high:
                out.append(f"{sensor} > {high:g}")
        return out


def build_features(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Variables temporelles et relationnelles utilisées par le modèle."""
    x = pd.DataFrame(index=df.index)
    for col in ["temperature", "cpu_load", "fan_rpm", "voltage", "latency_ms"]:
        x[col] = df[col]
        # Volatilité sur la fenêtre, en échelle log : un capteur figé
        # (écart-type nul) ou une tension qui oscille ressortent nettement.
        std = df[col].rolling(window, min_periods=2).std().bfill()
        x[f"{col}_log_std"] = np.log10(std + 1e-3)
    # Le GPS chute brièvement dans les tunnels : on utilise le signal lissé
    x["gps_snr_median"] = df["gps_snr"].rolling(window, min_periods=1).median()
    # Relations physiques : le ventilateur suit la température,
    # la température suit la charge CPU.
    x["fan_per_degree"] = df["fan_rpm"] / df["temperature"].clip(lower=1)
    x["temp_vs_cpu"] = df["temperature"] - 0.45 * df["cpu_load"].rolling(window, min_periods=1).mean()
    return x


@dataclass
class MLDetector:
    contamination: float = 0.03
    n_estimators: int = 300
    random_state: int = 0
    quantile: float = 0.97
    min_duration: int = 2

    def fit(self, df: pd.DataFrame) -> "MLDetector":
        x = build_features(df)
        self.columns_ = list(x.columns)
        self.scaler_ = StandardScaler().fit(x)
        self.model_ = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        ).fit(self.scaler_.transform(x))
        # Seuil de décision fixé sur la période d'apprentissage
        self.threshold_ = np.quantile(self._raw_scores(x), self.quantile)
        return self

    def _raw_scores(self, x: pd.DataFrame) -> np.ndarray:
        return -self.model_.score_samples(self.scaler_.transform(x[self.columns_]))

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Score d'anomalie : plus il est élevé, plus la mesure est suspecte."""
        return self._raw_scores(build_features(df))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Alerte si le score dépasse le seuil pendant au moins `min_duration` minutes."""
        above = pd.Series(self.score(df) > self.threshold_)
        if self.min_duration <= 1:
            return above.astype(int).to_numpy()
        run_id = (above != above.shift()).cumsum()
        run_len = above.groupby(run_id).transform("size")
        return (above & (run_len >= self.min_duration)).astype(int).to_numpy()
