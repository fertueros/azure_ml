"""
package_ml.features
===================

Funciones reutilizables para la generación de variables del
proyecto *Calorías quemadas*.

El notebook **2-feature-engineering.ipynb** importa estas
funciones —no contiene lógica pesada—, lo que facilita
testing, serialización en pipelines de scikit-learn y
reutilización en otros flujos.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Listas de columnas que se usan en varios lugares
# ---------------------------------------------------------------------
NUM_COLS: List[str] = [
    "Age",
    "Height",
    "Weight",
    "Duration",
    "Heart_Rate",
    "Body_Temp",
]


# ---------------------------------------------------------------------
# Funciones de transformación
# ---------------------------------------------------------------------
def add_feature_cross_terms(
    df: pd.DataFrame, numerical_features: Iterable[str] = NUM_COLS
) -> pd.DataFrame:
    """
    Genera términos de interacción multiplicativos de orden 2 y 3.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset de entrada.
    numerical_features : Iterable[str], default NUM_COLS
        Columnas numéricas sobre las que generar los cruces.

    Returns
    -------
    pd.DataFrame
        Copia del *DataFrame* con las nuevas columnas.
    """
    out = df.copy()
    for f1, f2 in combinations(numerical_features, 2):
        out[f"{f1}_x_{f2}"] = out[f1] * out[f2]

    for f1, f2, f3 in combinations(numerical_features, 3):
        out[f"{f1}_x_{f2}_x_{f3}"] = out[f1] * out[f2] * out[f3]

    return out


def add_interaction_features(
    df: pd.DataFrame, features: Iterable[str] = NUM_COLS
) -> pd.DataFrame:
    """
    Suma, resta y división entre pares de variables numéricas.
    """
    out = df.copy()
    for f1, f2 in combinations(features, 2):
        out[f"{f1}_plus_{f2}"] = out[f1] + out[f2]
        out[f"{f1}_minus_{f2}"] = out[f1] - out[f2]
        out[f"{f2}_minus_{f1}"] = out[f2] - out[f1]
        out[f"{f1}_div_{f2}"] = out[f1] / (out[f2] + 1e-5)
        out[f"{f2}_div_{f1}"] = out[f2] / (out[f1] + 1e-5)
    return out


def squares(df: pd.DataFrame, features: Iterable[str] = NUM_COLS) -> pd.DataFrame:
    """Añade el cuadrado de cada columna numérica."""
    out = df.copy()
    for f in features:
        out[f"{f}_2"] = np.square(out[f])
    return out


def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera *features* de dominio (BMI, %FCMT, etc.) y flags categóricos.
    """
    out = df.copy()

    # Conversión de género
    out["Gender"] = (out["Gender"] == "male").astype(bool)

    # BMI
    out["BMI"] = out["Weight"] / np.square(out["Height"] / 100)

    # Frecuencias cardíacas teóricas
    out["FCMT_simple"] = 220.0 - out["Age"]
    out["FCMT_tanaka"] = 208 - 0.7 * out["Age"]

    # Porcentaje sobre FC máx.
    for col in ["FCMT_simple", "FCMT_tanaka"]:
        out[f"Percent_{col}"] = np.clip(
            100 * out["Heart_Rate"] / out[col].clip(lower=1), 0, 150
        )
    
    # Desviación de temperatura corporal
    out["Body_Temp_Deviation"] = out["Body_Temp"] - 37.0
    
    # cudrado y cubo de FCMT
    out['Pct_FCMT_sq'] = out['Percent_FCMT_simple']**2
    out['Pct_FCMT_cu'] = out['Percent_FCMT_simple']**3

    # Logs (ejemplo)
    for col in ["Duration", "Heart_Rate", "Body_Temp", "Weight", "Duration_x_Heart_Rate"]:
        out[f"{col}_log"] = np.log1p(out[col])
   
    # Bandera de temperatura alta y sobrepeso
    out["is_temp_high"] = (out["Body_Temp"] > 39).astype(bool)
    out["is_overweight"] = (out["BMI"] > 27).astype(bool)

    out["feno_var"] = np.where(
        out["Gender"] == 1,
        -55.0969 * out["Duration"]
        + 0.6309 * out["Duration_x_Heart_Rate"]
        + 0.1988 * out["Weight_x_Duration"]
        + 0.2017 * out["Age_x_Duration"],
        -20.4022 * out["Duration"]
        + 0.4472 * out["Duration_x_Heart_Rate"]
        + 0.1263 * out["Weight_x_Duration"]
        + 0.074 * out["Age_x_Duration"],
    )

    return out
