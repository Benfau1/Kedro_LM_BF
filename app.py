from pathlib import Path
from flask import Flask, render_template, request
from kedro.framework.startup import bootstrap_project
from kedro.framework.session import KedroSession
import pandas as pd
import csv

app = Flask(__name__)
bootstrap_project(Path.cwd())

@app.route("/", methods=["GET"])
def index():
    return render_template('index.html')

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        input_type = request.form.get('inputType')

        filepath = "data/generated_csv.csv"

        if input_type == "csv":
            file = request.files.get('file')
            if file and file.filename != '':
                file.save(filepath)
            else:
                return "Aucun fichier CSV envoyé.", 400

        elif input_type == "json":
            json_text = request.form.get('json')
            if json_text:
                try:
                    data = pd.read_json(json_text)
                    data.to_csv(filepath, index=False)
                except ValueError as e:
                    return f"Erreur de parsing JSON : {e}", 400
            else:
                return "Aucun JSON envoyé.", 400

        else:
            return "Type d'entrée non reconnu.", 400

        # Lancer les pipelines
        run_pipelines(["transform_data", "prediction"])

        # Lire le résultat
        output = pd.read_csv("data/predictions.csv")
        return render_template('predict.html', tables=output.to_html(classes='data', index=False))

    return render_template('predict.html')


@app.route("/train", methods=["GET", "POST"])
def train():
    if request.method == "POST":
        input_type = request.form.get('inputType')
        filepath = "data/generated_csv.csv"

        if input_type == "csv":
            file = request.files.get('file')
            if file and file.filename != '':
                file.save(filepath)
            else:
                return "Aucun fichier CSV envoyé.", 400

        elif input_type == "json":
            json_text = request.form.get('json')
            if json_text:
                try:
                    data = pd.read_json(json_text)
                    data.to_csv(filepath, index=False)
                except ValueError as e:
                    return f"Erreur de parsing JSON : {e}", 400
            else:
                return "Aucun JSON envoyé.", 400

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
