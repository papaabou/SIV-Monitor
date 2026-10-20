"""
Point d'entrée en ligne de commande.

    python main.py              # génère les données, entraîne le modèle, écrit le rapport
    python main.py --benchmark  # compare seuils fixes et ML sur 20 scénarios indépendants
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from siv_monitor import INCIDENT_TYPES, MLDetector, RuleDetector, SimulationConfig, evaluate, simulate

OUTPUT_DIR = Path("output")
LOG_FILE = OUTPUT_DIR / "logs_systeme.csv"
REPORT_FILE = OUTPUT_DIR / "rapport.md"


def train_detector(seed: int = 1000) -> MLDetector:
    """Entraîne l'Isolation Forest sur une semaine d'historique (sans utiliser les étiquettes)."""
    history = simulate(SimulationConfig(n_minutes=7 * 24 * 60, seed=seed, incident_rate_per_day=2))
    return MLDetector().fit(history)


def run(seed: int = 7) -> tuple[pd.DataFrame, dict, dict]:
    df = simulate(SimulationConfig(seed=seed))
    rules, ml = RuleDetector(), train_detector()
    df["alerte_seuils"] = rules.predict(df)
    df["score_ml"] = ml.score(df).round(4)
    df["alerte_ml"] = ml.predict(df)
    return df, evaluate(df, df["alerte_seuils"].to_numpy()), evaluate(df, df["alerte_ml"].to_numpy())


def benchmark(n_runs: int = 20) -> pd.DataFrame:
    """Moyenne des performances sur `n_runs` scénarios jamais utilisés pendant la mise au point."""
    rows = []
    for i in range(n_runs):
        test = simulate(SimulationConfig(seed=500 + i))
        history = simulate(SimulationConfig(n_minutes=7 * 24 * 60, seed=900 + i, incident_rate_per_day=2))
        ml = MLDetector().fit(history)
        for name, pred in [("Seuils fixes", RuleDetector().predict(test)), ("Isolation Forest", ml.predict(test))]:
            res = evaluate(test, pred)
            rows.append({"methode": name, **{k: v for k, v in res.items() if k != "par_type"},
                         **{f"type_{k}": v for k, v in res["par_type"].items()}})
    return pd.DataFrame(rows).groupby("methode").mean(numeric_only=True)


def write_report(rules: dict, ml: dict) -> str:
    def pct(x: float) -> str:
        return f"{100 * x:.0f} %"

    lines = [
        "# Rapport de supervision — SIV Monitor",
        "",
        f"Incidents simulés : **{rules['incidents']}** sur 3 jours de télémétrie.",
        "",
        "| Indicateur | Seuils fixes | Isolation Forest |",
        "|---|---|---|",
        f"| Incidents détectés | {rules['incidents_detectes']} ({pct(rules['taux_detection'])}) "
        f"| {ml['incidents_detectes']} ({pct(ml['taux_detection'])}) |",
        f"| Précision | {pct(rules['precision'])} | {pct(ml['precision'])} |",
        f"| Rappel | {pct(rules['rappel'])} | {pct(ml['rappel'])} |",
        f"| F1 | {rules['f1']:.2f} | {ml['f1']:.2f} |",
        f"| Fausses alertes | {rules['fausses_alertes']} | {ml['fausses_alertes']} |",
        "",
        "## Détection par type d'incident",
        "",
        "| Type | Seuils fixes | Isolation Forest |",
        "|---|---|---|",
    ]
    for kind, label in INCIDENT_TYPES.items():
        if kind in rules["par_type"]:
            lines.append(f"| {label} | {pct(rules['par_type'][kind])} | {pct(ml['par_type'][kind])} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="SIV Monitor")
    parser.add_argument("--benchmark", action="store_true", help="comparer les méthodes sur 20 scénarios")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.benchmark:
        with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
            print(benchmark().T)
        return

    OUTPUT_DIR.mkdir(exist_ok=True)
    df, rules, ml = run(args.seed)
    df.to_csv(LOG_FILE, index=False)
    report = write_report(rules, ml)
    REPORT_FILE.write_text(report, encoding="utf-8")
    print(report)
    print(f"Logs : {LOG_FILE}  |  Rapport : {REPORT_FILE}")


if __name__ == "__main__":
    main()
