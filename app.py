"""Dashboard Streamlit : `streamlit run app.py`"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from siv_monitor import INCIDENT_TYPES, SENSORS, MLDetector, RuleDetector, SimulationConfig, evaluate, incidents, simulate

st.set_page_config(page_title="SIV Monitor", page_icon="🚆", layout="wide")

SENSOR_LABELS = {
    "temperature": "Température (°C)",
    "cpu_load": "Charge CPU (%)",
    "fan_rpm": "Ventilateur (tr/min)",
    "voltage": "Alimentation (V)",
    "latency_ms": "Latence réseau (ms)",
    "gps_snr": "Signal GPS (dB)",
}


@st.cache_resource
def trained_detector(seed: int) -> MLDetector:
    history = simulate(SimulationConfig(n_minutes=7 * 24 * 60, seed=seed, incident_rate_per_day=2))
    return MLDetector().fit(history)


@st.cache_data
def scenario(seed: int, days: int, rate: float) -> pd.DataFrame:
    return simulate(SimulationConfig(n_minutes=days * 24 * 60, seed=seed, incident_rate_per_day=rate))


st.title("🚆 SIV Monitor")
st.caption("Supervision d'un système d'information voyageurs embarqué — seuils fixes vs machine learning (Isolation Forest)")

with st.sidebar:
    st.header("Scénario")
    seed = st.number_input("Graine aléatoire", 0, 10_000, 7)
    days = st.slider("Durée simulée (jours)", 1, 7, 3)
    rate = st.slider("Incidents par jour", 2, 25, 12)
    st.divider()
    st.markdown(
        "Le modèle est entraîné **sans étiquettes** sur une semaine d'historique, "
        "puis appliqué au scénario. Les incidents injectés servent uniquement à mesurer les performances."
    )

df = scenario(seed, days, rate).copy()
ml = trained_detector(1000)
df["alerte_seuils"] = RuleDetector().predict(df)
df["score_ml"] = ml.score(df)
df["alerte_ml"] = ml.predict(df)
res_rules = evaluate(df, df["alerte_seuils"].to_numpy())
res_ml = evaluate(df, df["alerte_ml"].to_numpy())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Incidents simulés", res_ml["incidents"])
c2.metric("Détectés par le ML", f"{100 * res_ml['taux_detection']:.0f} %",
          f"{100 * (res_ml['taux_detection'] - res_rules['taux_detection']):+.0f} pts vs seuils")
c3.metric("Fausses alertes ML", res_ml["fausses_alertes"],
          f"{res_ml['fausses_alertes'] - res_rules['fausses_alertes']:+d} vs seuils", delta_color="inverse")
c4.metric("F1-score ML", f"{res_ml['f1']:.2f}", f"{res_ml['f1'] - res_rules['f1']:+.2f} vs seuils")

tab_live, tab_compare, tab_alerts = st.tabs(["📈 Télémétrie", "⚖️ Seuils vs ML", "🚨 Journal des alertes"])

with tab_live:
    sensor = st.selectbox("Capteur", SENSORS, format_func=SENSOR_LABELS.get)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df[sensor], name=SENSOR_LABELS[sensor], line=dict(width=1)))
    for start, end, kind in incidents(df):
        fig.add_vrect(x0=df["timestamp"].iloc[start], x1=df["timestamp"].iloc[end - 1],
                      fillcolor="red", opacity=0.12, line_width=0)
    hits = df[df["alerte_ml"] == 1]
    fig.add_trace(go.Scatter(x=hits["timestamp"], y=hits[sensor], mode="markers", name="Alerte ML",
                             marker=dict(size=5, color="orange")))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")
    st.caption("Zones rouges : incidents réels injectés. Points orange : alertes levées par le modèle.")

    fig_score = go.Figure()
    fig_score.add_trace(go.Scatter(x=df["timestamp"], y=df["score_ml"], name="Score d'anomalie", line=dict(width=1)))
    fig_score.add_hline(y=ml.threshold_, line_dash="dash", line_color="red", annotation_text="seuil")
    fig_score.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10), title="Score d'anomalie (Isolation Forest)")
    st.plotly_chart(fig_score, width="stretch")

with tab_compare:
    metrics = ["taux_detection", "precision", "rappel", "f1"]
    names = ["Incidents détectés", "Précision", "Rappel", "F1"]
    fig_cmp = go.Figure([
        go.Bar(name="Seuils fixes", x=names, y=[res_rules[m] for m in metrics]),
        go.Bar(name="Isolation Forest", x=names, y=[res_ml[m] for m in metrics]),
    ])
    fig_cmp.update_layout(barmode="group", height=380, yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_cmp, width="stretch")

    by_type = pd.DataFrame({
        "Incident": [INCIDENT_TYPES[k] for k in res_ml["par_type"]],
        "Seuils fixes": [f"{100 * res_rules['par_type'][k]:.0f} %" for k in res_ml["par_type"]],
        "Isolation Forest": [f"{100 * v:.0f} %" for v in res_ml["par_type"].values()],
    })
    st.subheader("Taux de détection par type d'incident")
    st.dataframe(by_type, hide_index=True, width="stretch")

with tab_alerts:
    rules = RuleDetector()
    alerts = df[(df["alerte_ml"] == 1) | (df["alerte_seuils"] == 1)].copy()
    alerts["source"] = alerts.apply(
        lambda r: " + ".join(s for s, flag in [("ML", r["alerte_ml"]), ("Seuils", r["alerte_seuils"])] if flag), axis=1)
    alerts["motif seuils"] = alerts.apply(lambda r: ", ".join(rules.reasons(r)) or "—", axis=1)
    alerts["incident réel"] = alerts["incident"].map(lambda k: INCIDENT_TYPES.get(k, "Aucun (fausse alerte)"))
    alerts[["score_ml", *SENSORS]] = alerts[["score_ml", *SENSORS]].round(2)
    st.dataframe(
        alerts[["timestamp", "source", "score_ml", "motif seuils", "incident réel", *SENSORS]],
        hide_index=True, width="stretch", height=460,
    )
    st.download_button("Exporter le journal (CSV)", alerts.to_csv(index=False).encode("utf-8"),
                       "alertes_siv.csv", "text/csv")
