import csv
import random
from datetime import datetime

LOG_FILE = "logs_systeme.csv"
REPORT_FILE = "rapport.txt"

# Configuration des seuils
TEMP_THRESHOLD = 75
TOTAL_TESTS = 50


def generate_data():
    """
    Génère une ligne de données simulant un test du système.
    
    Returns:
        dict: Dictionnaire contenant les données du test avec clés:
              - date, temperature, connexion, alimentation, ecran, gps, audio
              - statut (OK ou ALERTE), anomalies
    """
    temperature = random.randint(35, 95)
    connexion = random.choice(["OK", "OK", "OK", "PERDUE"])
    alimentation = random.randint(20, 100)
    ecran = random.choice(["OK", "OK", "OK", "ERREUR"])
    gps = random.choice(["OK", "OK", "OK", "SIGNAL_FAIBLE"])
    audio = random.choice(["OK", "OK", "ERREUR"])

    anomalies = []

    if temperature > TEMP_THRESHOLD:
        anomalies.append("Température élevée")
    if connexion == "PERDUE":
        anomalies.append("Connexion perdue")
    if ecran == "ERREUR":
        anomalies.append("Erreur écran voyageurs")
    if gps == "SIGNAL_FAIBLE":
        anomalies.append("Signal GPS faible")
    if audio == "ERREUR":
        anomalies.append("Erreur annonce sonore")

    statut = "ALERTE" if anomalies else "OK"

    return {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "temperature": temperature,
        "connexion": connexion,
        "alimentation": alimentation,
        "ecran": ecran,
        "gps": gps,
        "audio": audio,
        "statut": statut,
        "anomalies": ", ".join(anomalies) if anomalies else "Aucune"
    }


def main():
    """
    Génère les données de test, les enregistre dans un CSV
    et crée un rapport de fiabilité en fichier texte.
    """
    total_alertes = 0

    try:
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=[
                "date", "temperature", "connexion", "alimentation",
                "ecran", "gps", "audio", "statut", "anomalies"
            ])
            writer.writeheader()

            for _ in range(TOTAL_TESTS):
                data = generate_data()
                writer.writerow(data)

                if data["statut"] == "ALERTE":
                    total_alertes += 1

        taux_fiabilite = ((TOTAL_TESTS - total_alertes) / TOTAL_TESTS) * 100

        with open(REPORT_FILE, "w", encoding="utf-8") as report:
            report.write("Rapport de supervision - SIV Monitor\n")
            report.write("====================================\n\n")
            report.write(f"Nombre total de tests : {TOTAL_TESTS}\n")
            report.write(f"Nombre total d'alertes : {total_alertes}\n")
            report.write(f"Taux de bon fonctionnement : {taux_fiabilite:.2f}%\n")

        print("Données générées avec succès.")
        print(f"Fichier {LOG_FILE} créé.")
        print(f"Fichier {REPORT_FILE} créé.")
        
    except IOError as e:
        print(f"Erreur lors de l'écriture des fichiers : {e}")
        raise
    except Exception as e:
        print(f"Erreur inattendue : {e}")
        raise


if __name__ == "__main__":
    main()