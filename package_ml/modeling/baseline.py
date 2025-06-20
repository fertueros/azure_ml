from __future__ import annotations
import mlflow
import numpy as np
import pandas as pd
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
    "knn": KNeighborsRegressor(n_jobs=-1),
    "random_forest": RandomForestRegressor(random_state=42, n_jobs=-1),
    "hgb": HistGradientBoostingRegressor(random_state=42),
    "xgb": XGBRegressor(random_state=42),
    "lgbm": LGBMRegressor(random_state=42, verbose=-1),
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
    Entrena modelos base, registra los resultados con MLflow y devuelve las métricas.
    """
    mlflow.set_experiment(experiment_name)
    mlflow.sklearn.autolog(log_models=True, silent=True)
    # out_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows = []
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)

    for name in MODELS:
        print(f"--- Entrenando modelo: {name} ---")
        # MLflow autologging crea la ejecución automáticamente al llamar a una función de sklearn
        # como cross_validate, pero usar un 'with' nos da más control sobre los tags.
        with mlflow.start_run(run_name=f"{name}_baseline") as run:
            
            mlflow.set_tag("model_name", name)
            mlflow.set_tag("target_transformed", str(transform_target))
            
            pipe = _build_pipeline(name, transform_target=transform_target)

            # cross_validate activará autologging. Registrará los parámetros del pipe,
            # las métricas promedio de CV y el primer estimador.
            scores = cross_validate(
                pipe, X, y,
                scoring=SCORERS,
                cv=cv,
                n_jobs=-1,
                return_estimator=False  # Ya no necesitamos devolver el estimador
            )
            
            # Extraemos las métricas para nuestro DataFrame resumen
            # Autolog ya las ha guardado en MLflow, esto es solo para la salida de la función
            row = {"model": name}
            for metric_name, value in scores.items():
                if "test_" in metric_name:
                    # Limpiamos el nombre para la tabla: 'test_val_rmse' -> 'rmse'
                    clean_metric_name = metric_name.replace("test_val_", "")
                    row[clean_metric_name] = np.mean(value)
            
            metrics_rows.append(row)
            
    # Desactivamos autologging al final para no interferir con otros notebooks
    mlflow.sklearn.autolog(disable=True)
            
    return pd.DataFrame(metrics_rows)