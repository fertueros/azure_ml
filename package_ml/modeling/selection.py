from __future__ import annotations
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.compose import TransformedTargetRegressor

def get_permutation_importance(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    n_repeats: int = 10
) -> pd.DataFrame:
    """
    Calcula la importancia de las características mediante el método de permutación.

    Entrena un modelo LGBM con transformación de target en un subconjunto de
    entrenamiento y evalúa la importancia de cada característica midiendo la caída
    del rendimiento en un subconjunto de validación.

    Args:
        X (pd.DataFrame): DataFrame de características.
        y (pd.Series): Serie del target.
        test_size (float, optional): Proporción de datos a usar para validación. Defaults to 0.2.
        n_repeats (int, optional): Número de veces que se permuta cada característica. Defaults to 10.

    Returns:
        pd.DataFrame: DataFrame con las características y su importancia media y desviación estándar,
        ordenado de mayor a menor importancia.
    """
    # 1. Dividir los datos
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    print(f"División de datos: {len(X_train)} para entrenar, {len(X_val)} para validar.")

    # 2. Definir y entrenar el modelo con transformación de target
    model_base = lgb.LGBMRegressor(random_state=42)
    model_with_transform = TransformedTargetRegressor(
        regressor=model_base,
        func=np.log1p,
        inverse_func=np.expm1
    )
    
    print("Entrenando modelo base para la evaluación de características...")
    model_with_transform.fit(X_train, y_train)

    # 3. Calcular la importancia por permutación
    print("Calculando importancias en el set de validación...")
    result = permutation_importance(
        model_with_transform, 
        X_val, 
        y_val, 
        n_repeats=n_repeats,
        random_state=42, 
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )

    # 4. Crear y devolver el DataFrame de resultados
    perm_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    print("--- Cálculo de Permutación de Importancia completado ---")
    return perm_importance_df