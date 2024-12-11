import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from src.data_preprocessing import load_and_preprocess_data

def train_model(data_path: str, model_path: str):
    # Cargar y preprocesar datos
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess_data(data_path)
    
    # Entrenar el modelo
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluar el modelo
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # Guardar el modelo y el scaler
    joblib.dump({"model": clf, "scaler": scaler, "features": feature_names}, model_path)

if __name__ == "__main__":
    train_model("data/dataset.csv", "models/model.pkl")
