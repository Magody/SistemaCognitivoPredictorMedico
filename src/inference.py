import joblib
import numpy as np

def load_model(model_path: str):
    data = joblib.load(model_path)
    model = data["model"]
    scaler = data["scaler"]
    features = data["features"]
    return model, scaler, features

def predict_disease(input_data: dict, model, scaler, features):
    # input_data es un diccionario con los valores de las características
    # Ordenar las características según 'features'
    X = np.array([input_data[feat] for feat in features]).reshape(1, -1)
    # Escalar
    X_scaled = scaler.transform(X)
    # Predecir
    pred = model.predict(X_scaled)[0]
    return pred
