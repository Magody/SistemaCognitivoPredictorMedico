import shap
import numpy as np
import matplotlib.pyplot as plt

def get_explanation(input_data: dict, model, scaler, features):
    # Convertir los datos de entrada a un array escalado
    X = np.array([input_data[feat] for feat in features]).reshape(1, -1)
    X_scaled = scaler.transform(X)

    # Crear el explicador
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    # Seleccionar la clase a explicar
    class_to_explain = 0  # Cambiar al índice de la clase que quieras explicar
    shap_values_for_instance = shap_values[class_to_explain][0]
    expected_value = explainer.expected_value[class_to_explain]

    # Crear el gráfico tipo waterfall
    plt.figure(figsize=(10, 8))  # Ajustar el tamaño de la figura
    shap.plots._waterfall.waterfall_legacy(
        expected_value,  # Seleccionar el valor esperado para la clase elegida
        shap_values_for_instance,
        feature_names=features,
        show=False
    )

    # Ajustar los márgenes automáticamente
    plt.tight_layout()

    # Obtener la figura actual
    fig = plt.gcf()
    return fig
