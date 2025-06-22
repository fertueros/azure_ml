import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import mlflow
import joblib
from pathlib import Path
from sklearn.compose import TransformedTargetRegressor
# from optuna.pruners import MedianPruner
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import root_mean_squared_error

def optimize_lgbm_hyperparameters(
    X: pd.DataFrame,
    y: pd.Series,
    stratify_on: pd.Series,
    n_trials: int = 100,
    cv_splits: int = 5,
    experiment_name: str = "lgbm_optimization"
) -> optuna.study.Study:
    """
    Optimiza los hiperparámetros de un LGBMRegressor usando Optuna y registra
    cada trial en MLflow como una ejecución anidada.

    Args:
        X (pd.DataFrame): DataFrame con las características seleccionadas.
        y (pd.Series): Serie del target.
        n_trials (int): Número de trials que ejecutará Optuna.
        cv_splits (int): Número de folds para la validación cruzada.
        experiment_name (str): Nombre del experimento en MLflow.

    Returns:
        optuna.study.Study: El objeto de estudio de Optuna completado.
    """
    mlflow.set_experiment(experiment_name)

    def objective(trial: optuna.Trial) -> float:
        # Iniciar una ejecución ANIDADA para este trial de Optuna
        with mlflow.start_run(nested=True):
            # 1. Definir y registrar el espacio de búsqueda de parámetros
            params = {
                "objective": "regression_l1",
                "metric": "rmse",
                "verbosity": -1,
                "random_state": 42,
                "n_estimators":        trial.suggest_int("n_estimators", 1000, 4000),
                "learning_rate":       trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "num_leaves":          trial.suggest_int("num_leaves", 31, 512),
                "max_depth":           trial.suggest_int("max_depth", -1, 15),
                "min_child_samples":   trial.suggest_int("min_child_samples", 5, 100),
                "subsample":           trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree":    trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "lambda_l1":           trial.suggest_float("lambda_l1", 1e-4, 100.0, log=True),
                "lambda_l2":           trial.suggest_float("lambda_l2", 1e-4, 100.0, log=True),
            }
            mlflow.log_params(params)

            # 2. Ejecutar la validación cruzada
            skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
            rmses = []

            for tr_idx, val_idx in skf.split(X, stratify_on):
                X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
                y_tr, y_val = np.log1p(y.iloc[tr_idx]), y.iloc[val_idx]

                model = lgb.LGBMRegressor(**params)
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, np.log1p(y_val))],
                    callbacks=[lgb.early_stopping(100, verbose=False)]
                )

                preds_log = model.predict(X_val)
                preds_original = np.expm1(preds_log)
                
                rmse_fold = root_mean_squared_error(y_val, preds_original)
                rmses.append(rmse_fold)
            
            # 3. Registrar la métrica promedio y devolverla a Optuna
            avg_rmse = float(np.mean(rmses))
            mlflow.log_metric("avg_rmse_cv", avg_rmse)
            
        return avg_rmse
    
    # Inicia la ejecución principal de MLflow que englobará todo el estudio
    with mlflow.start_run(run_name="Optuna_Study_LGBM") as parent_run:
        print(f"Iniciando estudio de Optuna bajo la ejecución principal de MLflow: {parent_run.info.run_id}")

        # pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=3)
        study = optuna.create_study(direction="minimize", study_name="calories_lgbm")
        study.optimize(objective, n_trials=n_trials)

        # 4. Registrar los resultados finales del estudio en la ejecución principal
        print("Optimización completada. Registrando los mejores resultados.")
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_rmse_cv", study.best_value)
        mlflow.set_tag("n_trials", str(n_trials))

    return study

def train_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: dict,
    model_path: Path
) -> None:
    """
    Entrena el modelo final con los mejores hiperparámetros y todos los datos
    de entrenamiento, y lo guarda en disco.

    Args:
        X_train (pd.DataFrame): DataFrame de características de entrenamiento.
        y_train (pd.Series): Serie del target de entrenamiento.
        best_params (dict): Diccionario con los mejores hiperparámetros de Optuna.
        model_path (Path): Ruta donde se guardará el modelo entrenado.
    """
    print("--- Entrenando el modelo final con todos los datos de entrenamiento... ---")
    
    # Asegurarse de que el directorio del modelo exista
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Definir el modelo con los mejores parámetros
    final_model_base = lgb.LGBMRegressor(random_state=42, **best_params, verbosity=-1)

    # 2. Envolverlo en el transformador de target para que el pipeline sea consistente
    final_model_pipeline = TransformedTargetRegressor(
        regressor=final_model_base,
        func=np.log1p,
        inverse_func=np.expm1
    )

    # 3. Entrenar el pipeline con todos los datos
    final_model_pipeline.fit(X_train, y_train)

    # 4. Guardar (serializar) el pipeline completo
    joblib.dump(final_model_pipeline, model_path)
    
    print(f"Modelo final entrenado y guardado en: {model_path}")