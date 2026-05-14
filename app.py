import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import subprocess
# Configuration de la pagest.set_page_config(
    page_title="SIV Monitor",
    page_icon="�",
    layout="wide"
)

st.title("SIV Monitor")
st.subheader("Dashboard de supervision d’un système d’information voyageurs embarqué")

st.markdown("""
Ce projet simule la supervision d’un système embarqué ferroviaire :  
température, connexion, alimentation, écran voyageurs, GPS et annonces sonores.
""")

# Bouton pour générer de nouvelles données
if st.button("Générer de nouvelles données"):
    try:
        subprocess.run(["python", "main.py"], check=True)
        st.success("Nouvelles données générées avec succès.")
    except subprocess.CalledProcessError:
        st.error("Erreur lors de la génération des données.")

# Vérifier que le fichier de données existe
if not os.path.exists("logs_systeme.csv"):
    st.warning("Aucune donnée disponible. Clique sur le bouton pour générer les données.")
    st.stop()

# Charger les données
try:
    df = pd.read_csv("logs_systeme.csv")
except Exception as e:
    st.error(f"Erreur lors de la lecture des données : {e}")
    st.stop()

# Calcul des métriques
total_tests = len(df)
total_alertes = len(df[df["statut"] == "ALERTE"])
total_ok = len(df[df["statut"] == "OK"])
taux_fiabilite = (total_ok / total_tests) * 100 if total_tests > 0 else 0

# Affichage des métriques dans des colonnes
col1, col2, col3, col4 = st.columns(4)

col1.metric("Tests effectués", total_tests)
col2.metric("Systèmes OK", total_ok)
col3.metric("Alertes détectées", total_alertes)
col4.metric("Fiabilité", f"{taux_fiabilite:.2f}%")

st.divider()

st.subheader("Évolution de la température")

# Graphique de température
fig, ax = plt.subplots()
ax.plot(df["temperature"], label="Température mesurée")
ax.axhline(75, linestyle="--", color="red", label="Seuil critique 75°C")
ax.set_xlabel("Test")
ax.set_ylabel("Température (°C)")
ax.set_title("Température du système embarqué")
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# Graphique de répartition
status_counts = df["statut"].value_counts()

fig2, ax2 = plt.subplots()
colors = ["green" if status == "OK" else "red" for status in status_counts.index]
ax2.bar(status_counts.index, status_counts.values, color=colors, alpha=0.7)
ax2.set_xlabel("Statut")
ax2.set_ylabel("Nombre")
ax2.set_title("OK vs Alertes")
ax2.grid(True, axis="y", alpha=0.3
ax2.set_ylabel("Nombre")
ax2.set_title("OK vs Alertes")
st.pyplot(fig2)

st.divider()

st.subheader("Derniers logs système")
st.dataframe(df.tail(15), width='stretch')

st.subheader("Alertes détectées")
alertes = df[df["statut"] == "ALERTE"]

if alertes.empty:
    st.success("Aucune alerte détectée.")
else:
    st.dataframe(alertes, width='stretch')

st.divider()

st.stry:
        with open("rapport.txt", "r", encoding="utf-8") as file:
            st.text(file.read())
    except Exception as e:
        st.error(f"Erreur lors de la lecture du rapport : {e}"
if os.path.exists("rapport.txt"):
    with open("rapport.txt", "r", encoding="utf-8") as file:
        st.text(file.read())
else:
    st.info("Rapport non généré.")