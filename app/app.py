from flask import Flask, render_template, request, redirect, url_for
import os
from src.inference import load_model, predict_disease
from src.explainer import get_explanation
import matplotlib.pyplot as plt
import pandas as pd

app = Flask(__name__)

# Cargar el modelo una sola vez al iniciar la app
MODEL_PATH = os.path.join("models", "model.pkl")
model, scaler, features = load_model(MODEL_PATH)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Obtener datos del formulario
        input_data = {}
        for feat in features:
            val = request.form.get(feat, None)
            if val is not None:
                val = float(val)
            else:
                val = 0.0
            input_data[feat] = val

        # Redirigir a la página de resultados con los datos
        return redirect(url_for("result", **input_data))
    else:
        return render_template("index.html", features=features)

@app.route("/result")
def result():
    # Obtener datos de la query
    input_data = {feat: float(request.args.get(feat, 0.0)) for feat in features}
    diagnosis = predict_disease(input_data, model, scaler, features)

    # Generar el gráfico explicativo
    fig = get_explanation(input_data, model, scaler, features)
    img_path = os.path.join("app", "static", "shap_plot.png")
    fig.savefig(img_path)
    plt.close(fig)

    # Renderizar la página con el diagnóstico y el gráfico
    return render_template("result.html", diagnosis=diagnosis, input_data=input_data)

@app.route("/mass_predictions", methods=["GET", "POST"])
def mass_predictions():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded", 400
        file = request.files["file"]
        if file.filename == "":
            return "No file selected", 400

        # Leer el archivo CSV cargado
        try:
            data = pd.read_csv(file)
        except Exception as e:
            return f"Error reading file: {e}", 400

        # Verificar que las columnas coincidan con las características esperadas
        if not set(features).issubset(data.columns):
            return "Uploaded file does not contain the required columns.", 400

        # Procesar cada fila para realizar predicciones
        predictions = []
        for _, row in data.iterrows():
            input_data = row[features].to_dict()
            diagnosis = predict_disease(input_data, model, scaler, features)
            predictions.append({"input_data": input_data, "diagnosis": diagnosis})

        return render_template("mass_predictions.html", predictions=predictions)

    return render_template("mass_predictions.html")

if __name__ == "__main__":
    # Ejecutar el servidor Flask
    app.run(debug=True)
