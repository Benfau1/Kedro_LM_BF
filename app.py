# app.py
from pathlib import Path
from flask import Flask, Request, request
from kedro.framework.startup import bootstrap_project
from kedro.framework.session import KedroSession
import pandas as pd
from kedro.framework.startup import bootstrap_project

app = Flask(__name__)
bootstrap_project(Path.cwd())

import csv
from flask import Request

def save_from_post_request(request: Request, filepath: str):
    try:
        data = request.get_json()
        if data is None:
            raise ValueError("Aucune donnée JSON reçue.")

        if not isinstance(data, list):
            data = [data]

        fieldnames = data[0].keys()

        with open(filepath, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

        print(f"Données sauvegardées en CSV dans {filepath}")

    except Exception as e:
        print(f"Erreur lors de la sauvegarde des données : {e}")
        raise

# Define Flask route for POST requests
@app.route("/predict", methods=["POST"])
def predict():
    filepath = "data/generated_csv.csv"
    save_from_post_request(request, filepath)

    run_pipelines(["transform_data","prediction"])

    output = pd.read_csv(filepath)
    return output.to_json(orient='records')


@app.route("/train", methods=["POST"])
def train():
    filepath = "data/generated_csv.csv"
    save_from_post_request(request, filepath)

    run_pipelines(["transform_data","train_data"])

    output = pd.read_csv(filepath)
    return output.to_json(orient='records')

@app.route("/", methods=["POST","GET"])
def default():
    return

def run_pipelines(pipelines):
    for pipeline_name in pipelines:
        print(f"Running pipeline: {pipeline_name}")
        with KedroSession.create("./", Path.cwd()) as session:
            session.run(pipeline_name=pipeline_name)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080)