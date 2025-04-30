# Importation des bibliothèques nécessaires
from pathlib import Path
from flask import Flask, render_template, request
from kedro.framework.startup import bootstrap_project
from kedro.framework.session import KedroSession
import pandas as pd
import csv

# Création de l'application Flask
app = Flask(__name__)

# Initialisation du projet Kedro (permet de charger le contexte du projet)
bootstrap_project(Path.cwd())

# Page d’accueil, simplement le formulaire HTML
@app.route("/", methods=["GET"])
def index():
    return render_template('index.html')

# Route pour faire une prédiction (GET pour affichage, POST pour traitement des données)
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        input_type = request.form.get('inputType')  # Récupère le type de données (csv ou json)
        filepath = "data/generated_csv.csv"  # Fichier temporaire utilisé pour stocker les données

        if input_type == "csv":
            file = request.files.get('file')  # Récupère le fichier envoyé
            if file and file.filename != '':
                file.save(filepath)  # Sauvegarde le fichier dans le dossier data
            else:
                return "Aucun fichier CSV envoyé.", 400

        elif input_type == "json":
            json_text = request.form.get('json')  # Récupère les données JSON en texte
            if json_text:
                try:
                    data = pd.read_json(json_text)  # Lecture du JSON sous forme de DataFrame
                    data.to_csv(filepath, index=False)  # Enregistrement en CSV pour être utilisé dans Kedro
                except ValueError as e:
                    return f"Erreur de parsing JSON : {e}", 400
            else:
                return "Aucun JSON envoyé.", 400

        else:
            return "Type d'entrée non reconnu.", 400

        # Exécution des pipelines transform_data (prétraitement) et prediction
        run_pipelines(["transform_data", "prediction"])

        # Lecture et affichage du fichier résultat (les prédictions)
        output = pd.read_csv("data/predictions.csv")
        return render_template('predict.html', tables=output.to_html(classes='data', index=False))

    return render_template('predict.html')

# Route pour entraîner le modèle
@app.route("/train", methods=["GET", "POST"])
def train():
    if request.method == "POST":
        input_type = request.form.get('inputType')  # Récupère le type de données fourni
        filepath = "data/generated_csv.csv"

        if input_type == "csv":
            file = request.files.get('file')
            if file and file.filename != '':
                file.save(filepath)  # Sauvegarde du fichier fourni
            else:
                return "Aucun fichier CSV envoyé.", 400

        elif input_type == "json":
            json_text = request.form.get('json')
            if json_text:
                try:
                    data = pd.read_json(json_text)  # Convertit le JSON en DataFrame
                    data.to_csv(filepath, index=False)  # Sauvegarde en CSV
                except ValueError as e:
                    return f"Erreur de parsing JSON : {e}", 400
            else:
                return "Aucun JSON envoyé.", 400

        # Exécute les pipelines nécessaires à l'entraînement
        run_pipelines(["transform_data", "train_data"])

        # Retourne un tableau HTML des données utilisées
        output = pd.read_csv(filepath)
        html_table = output.to_html(classes='data', index=False).replace("\n", "")
        return render_template('train.html', tables=html_table)

    return render_template('train.html')

# Fonction pour exécuter une ou plusieurs pipelines Kedro
def run_pipelines(pipelines):
    for pipeline_name in pipelines:
        print(f"Running pipeline: {pipeline_name}")
        with KedroSession.create("./", Path.cwd()) as session:
            session.run(pipeline_name=pipeline_name)

# Démarrage de l'application en local sur le port 8080
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080)