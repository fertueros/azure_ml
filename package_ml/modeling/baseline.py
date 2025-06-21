from __future__ import annotations
import mlflow
import numpy as np
import pandas as pd
import gc
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_validate, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, mean_absolute_error, root_mean_squared_error, median_absolute_error

# Diccionario de modelos base.
MODELS = {
    "ridge": Ridge(random_state=42),
    "lasso": Lasso(random_state=42, max_iter=5000),
    "random_forest": RandomForestRegressor(random_state=42, n_jobs=-1),
    "hgb": HistGradientBoostingRegressor(random_state=42),
    "xgb": XGBRegressor(random_state=42),
    "lgbm": LGBMRegressor(random_state=42, verbose=-1),
    "knn": KNeighborsRegressor(n_jobs=-1),
}

# Diccionario de scorers
# Las métricas se calcularán sobre las predicciones ya transformadas inversamente.
SCORERS = {
    "val_rmse": make_scorer(root_mean_squared_error, greater_is_better=False),
    "val_mae":  make_scorer(mean_absolute_error, greater_is_better=False),
    "val_medae": make_scorer(median_absolute_error, greater_is_better=False)
}

def _build_pipeline(name: str, transform_target: bool = True) -> Pipeline | TransformedTargetRegressor:
    """
    Construye un pipeline de Scikit-learn para un modelo dado.
    
    Si transform_target es True, envuelve el pipeline en un TransformedTargetRegressor
    para manejar la transformación logarítmica del target.
    """

    model = MODELS[name]

    # Define el pipeline de preprocesamiento y modelo
    if name in {"ridge","lasso","knn"}:
        base_pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
    else:
        base_pipe = Pipeline([("model", model)])
    
    # si se transforma el target completamos el pipe
    if transform_target:
        # aplicamos np.log1p(x) por ser mas estable, revertimos con np.expm1(x)
        return TransformedTargetRegressor(
            regressor=base_pipe,
            func=np.log1p,
            inverse_func=np.expm1
        )
    
    return base_pipe

def train_baselines_with_mlflow(
        X: pd.DataFrame, y: pd.Series,
        cv_splits: int=10,
        transform_target: bool = True,
        experiment_name: str = "baseline_models"
) -> pd.DataFrame:
    """
    Entrena modelos base, registra resultados con MLflow y devuelve un resumen de métricas.
    Usa autolog para parámetros/tags y logging manual para métricas de CV para mayor fiabilidad.
    """
    mlflow.set_experiment(experiment_name)
    
    # Usaremos autolog, pero solo para lo que funciona bien (parámetros, tags).
    # Desactivamos el logging de modelos para hacerlo manualmente y tener más control.
    mlflow.sklearn.autolog(log_models=False, log_input_examples=False, log_model_signatures=False, silent=True)

    input_example = X.head(10)

    metrics_rows = []
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)

    for name in MODELS:
        print(f"--- Entrenando modelo: {name} ---")
        with mlflow.start_run(run_name=f"{name}") as run:
            
            # Autolog se encargará de los parámetros del pipeline cuando llamemos a cross_validate.
            # Añadimos nuestros tags personalizados manualmente.
            mlflow.set_tag("model_name", name)
            mlflow.set_tag("target_transformed", str(transform_target))
            
            pipe = _build_pipeline(name, transform_target=transform_target)

            scores = cross_validate(
                pipe, X, y,
                scoring=SCORERS,
                cv=cv,
                return_estimator=True # Necesitamos el estimador para guardarlo
            )
            
            # Extraemos y calculamos el promedio de las métricas de los folds
            val_metrics = {}
            for metric_name, value_list in scores.items():
                if "test_" in metric_name:
                    # Limpiamos el nombre: 'test_val_rmse' -> 'val_rmse'
                    clean_metric_name = metric_name.replace("test_", "")
                    val_metrics[clean_metric_name] = round(np.mean(value_list),4)
            
            # Registramos las métricas manualmente
            print(f"Registrando métricas para {name}: {val_metrics}")
            mlflow.log_metrics(val_metrics)
            
            # --- LOGGING MANUAL DEL MODELO ---
            # Guardamos el primer estimador del CV como el artefacto del modelo
            first_estimator = scores["estimator"][0]
            mlflow.sklearn.log_model(
                sk_model=first_estimator,
                name="model", # MLflow lo guardará en la carpeta 'artifacts/model'
                input_example=input_example
            )

            # Preparamos la fila para nuestro DataFrame de resumen
            row = {"model": name}
            row.update({key.replace('val_', ''): value for key, value in val_metrics.items()})
            metrics_rows.append(row)
            
    # Desactivamos autologging al final
    mlflow.sklearn.autolog(disable=True)
            
    return pd.DataFrame(metrics_rows)