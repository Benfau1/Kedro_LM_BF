# app.py
import json
from pathlib import Path
from flask import Flask, Request, request, jsonify
from kedro.framework.startup import bootstrap_project
from kedro.framework.session import KedroSession
import pandas as pd

app = Flask(__name__)
bootstrap_project(Path.cwd())

def save_from_post_request(request: Request, filepath: str):
    try:
        data = request.get_json()
        if data is None:
            raise ValueError("Aucune donnée JSON reçue.")

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        print(f"Données sauvegardées dans {filepath}")

    except Exception as e:
        print(f"Erreur lors de la sauvegarde des données : {e}")
        raise

# Define Flask route for POST requests
@app.route("/predict", methods=["POST"])
def predict():
    # Set the filepath to save the POST request data
    filepath = "data/predict_api.json"

    # Save data from POST request to JSON file
    save_from_post_request(request, filepath)
    with KedroSession.create("./", Path.cwd()) as session:
        processed_data = session.run(
        pipeline_name ="predict_api", # Le nom du pipe à exécuter
    )

    output = pd.read_csv(filepath)
    return output.to_json(orient='records')


@app.route("/train", methods=["POST"])
def train():
    # Set the filepath to save the POST request data
    filepath = "data/train_api.json"

    # Save data from POST request to JSON file
    save_from_post_request(request, filepath)
    with KedroSession.create("./", Path.cwd()) as session:
        processed_data = session.run(
        pipeline_name ="train_data_api", # Le nom du pipe à exécuter
    )

    output = pd.read_csv(filepath)
    return output.to_json(orient='records')

@app.route("/", methods=["POST","GET"])
def default():
    return

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080)