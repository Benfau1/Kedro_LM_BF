from pathlib import Path
from flask import Flask, render_template, request
from kedro.framework.startup import bootstrap_project
from kedro.framework.session import KedroSession
import pandas as pd
import csv

app = Flask(__name__)
bootstrap_project(Path.cwd())

def save_from_post_request(request, filepath: str):
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

@app.route("/", methods=["GET"])
def index():
    return render_template('index.html')

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        file = request.files['file']
        filepath = "data/generated_csv.csv"
        file.save(filepath)

        run_pipelines(["transform_data", "prediction"])

        output = pd.read_csv("data/predictions.csv")
        return render_template('predict.html', tables=output.to_html(classes='data', index=False))

    return render_template('predict.html')

@app.route("/train", methods=["GET", "POST"])
def train():
    if request.method == "POST":
        file = request.files['file']
        filepath = "data/generated_csv.csv"
        file.save(filepath)

        run_pipelines(["transform_data", "train_data"])

        output = pd.read_csv(filepath)
        html_table = output.to_html(classes='data', index=False).replace("\n", "")
        return render_template('train.html', tables=html_table)

    return render_template('train.html')

def run_pipelines(pipelines):
    for pipeline_name in pipelines:
        print(f"Running pipeline: {pipeline_name}")
        with KedroSession.create("./", Path.cwd()) as session:
            session.run(pipeline_name=pipeline_name)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080)
