import joblib
from pathlib import Path
import pandas as pd
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

def make_predictions(
    X_test: pd.DataFrame,
    model_path: Path
) -> pd.Series:
    """
    Carga un modelo entrenado y realiza predicciones en datos no vistos.

    Args:
        X_test (pd.DataFrame): DataFrame de características del conjunto de prueba.
        model_path (Path): Ruta al modelo entrenado (.joblib).

    Returns:
        pd.Series: Serie con las predicciones.
    """
    print(f"--- Cargando modelo desde {model_path} para realizar predicciones... ---")
    
    # Cargar el pipeline del modelo
    model_pipeline = joblib.load(model_path)
    
    # Realizar predicciones
    predictions = model_pipeline.predict(X_test)
    
    return pd.Series(predictions, index=X_test.index)


def evaluate_model_on_test_set(
    y_true: pd.Series,
    y_pred: pd.Series
) -> dict:
    """
    Calcula las métricas de rendimiento en el conjunto de prueba.

    Args:
        y_true (pd.Series): Valores reales del target.
        y_pred (pd.Series): Predicciones del modelo.

    Returns:
        dict: Diccionario con las métricas calculadas (RMSE, MAE, R²).
    """
    print("--- Evaluando el rendimiento del modelo en el conjunto de prueba... ---")
    
    metrics = {
        "rmse_test": root_mean_squared_error(y_true, y_pred),
        "mae_test": mean_absolute_error(y_true, y_pred),
        "r2_test": r2_score(y_true, y_pred)
    }
    
    print(f"Resultados en datos no vistos: {metrics}")
    return metrics