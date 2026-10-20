"""Génère les illustrations du README : `python docs/generate_figures.py`"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import benchmark, run  # noqa: E402
from siv_monitor import incidents  # noqa: E402

OUT = Path(__file__).parent
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

# 1. Télémétrie : incidents réels et alertes des deux méthodes
df, _, _ = run(seed=7)
day = df.iloc[: 24 * 60]
fig, ax = plt.subplots(figsize=(11, 3.6))
ax.plot(day["timestamp"], day["temperature"], lw=0.8, color="#3b6ea5", label="Température (°C)")
for i, (s, e, _) in enumerate(incidents(day)):
    ax.axvspan(day["timestamp"].iloc[s], day["timestamp"].iloc[e - 1], color="#d62728", alpha=0.15,
               label="Incident réel" if i == 0 else None)
ml_hits, rule_hits = day[day["alerte_ml"] == 1], day[day["alerte_seuils"] == 1]
ax.scatter(ml_hits["timestamp"], ml_hits["temperature"], s=10, color="#ff7f0e", label="Alerte Isolation Forest", zorder=3)
ax.scatter(rule_hits["timestamp"], [ax.get_ylim()[0] + 1] * len(rule_hits), s=8, marker="|", color="#555",
           label="Alerte seuils fixes")
ax.set_title("24 h de télémétrie : incidents injectés et alertes levées")
ax.legend(loc="upper left", ncol=4, fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(OUT / "telemetrie.png")

# 2. Comparaison moyenne sur 20 scénarios
res = benchmark(20)
metrics = ["taux_detection", "precision", "rappel", "f1"]
labels = ["Incidents détectés", "Précision", "Rappel", "F1"]
fig, ax = plt.subplots(figsize=(8, 3.6))
width = 0.38
for k, (name, color) in enumerate([("Seuils fixes", "#9aa5b1"), ("Isolation Forest", "#3b6ea5")]):
    vals = res.loc[name, metrics].to_numpy()
    xs = [i + (k - 0.5) * width for i in range(len(metrics))]
    bars = ax.bar(xs, vals, width, label=name, color=color)
    ax.bar_label(bars, labels=[f"{v:.0%}" if m != "f1" else f"{v:.2f}" for v, m in zip(vals, metrics)], fontsize=8)
ax.set_xticks(range(len(metrics)), labels)
ax.set_ylim(0, 1.1)
ax.set_yticks([])
ax.set_title("Moyenne sur 20 scénarios indépendants")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / "comparaison.png")
print(res[metrics + ["fausses_alertes"]].round(3))
