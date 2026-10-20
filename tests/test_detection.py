import numpy as np
import pandas as pd

from siv_monitor import MLDetector, RuleDetector, SimulationConfig, evaluate, simulate


def test_rules_flag_out_of_range_values():
    df = pd.DataFrame({
        "temperature": [50, 90], "cpu_load": [30, 30], "fan_rpm": [1500, 1500],
        "voltage": [24, 24], "latency_ms": [40, 40], "gps_snr": [35, 35],
    })
    assert RuleDetector().predict(df).tolist() == [0, 1]


def test_ml_beats_fixed_thresholds():
    history = simulate(SimulationConfig(n_minutes=7 * 24 * 60, seed=900, incident_rate_per_day=2))
    test = simulate(SimulationConfig(seed=500))
    ml = MLDetector().fit(history)
    res_ml = evaluate(test, ml.predict(test))
    res_rules = evaluate(test, RuleDetector().predict(test))
    assert res_ml["taux_detection"] > res_rules["taux_detection"]
    assert res_ml["f1"] > res_rules["f1"]


def test_evaluate_counts_false_alarms():
    df = pd.DataFrame({"label": [0, 0, 1, 1, 0, 0], "incident": ["aucun", "aucun", "surchauffe", "surchauffe", "aucun", "aucun"]})
    res = evaluate(df, np.array([1, 0, 1, 0, 0, 1]))
    assert res["incidents_detectes"] == 1
    assert res["fausses_alertes"] == 2
