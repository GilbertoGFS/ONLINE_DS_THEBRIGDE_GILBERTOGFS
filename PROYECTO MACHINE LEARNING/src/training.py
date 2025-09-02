# src/training.py
"""
Funciones de entrenamiento y guardado de modelos supervisados:
- ETS (Holt-Winters)
- SARIMA (mini-búsqueda)
- Ridge
- Random Forest
- XGBoost (opcional)
- LightGBM (opcional)
"""

from __future__ import annotations

from pathlib import Path
import pickle
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

# Modelos sklearn
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

# Modelos de series temporales (statsmodels)
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import statsmodels.api as sm


# ===== Opcionales: XGBoost / LightGBM =====
try:
    from xgboost import XGBRegressor  # pip install xgboost
except Exception:
    XGBRegressor = None  # type: ignore

try:
    from lightgbm import LGBMRegressor  # pip install lightgbm
except Exception:
    LGBMRegressor = None  # type: ignore


# =========================================================
# Utilidad: guardar modelo en pickle
# =========================================================
def save_model(model: Any, path: str | Path) -> None:
    """Guarda un objeto de modelo en formato pickle."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(model, f)


# =========================================================
# ETS (Holt-Winters)
# =========================================================
def train_ets(
    y_train: pd.Series,
    seasonal: str = "add",
    seasonal_periods: int = 7,
    trend: Optional[str] = None,
):
    """
    Entrena un ETS sencillo para frecuencia diaria.
    - y_train: Serie diaria con índice datetime (asfreq('D') recomendado).
    """
    y_train = y_train.asfreq("D")
    model = ExponentialSmoothing(
        y_train,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
    ).fit(optimized=True)
    return model


# =========================================================
# SARIMA (mini-grid muy simple)
# =========================================================
def train_sarima(
    y_train: pd.Series,
    p_list=(0, 1, 2, 3),
    d: int = 0,
    q_list=(0, 1, 2),
    P_list=(0, 1),
    D: int = 1,
    Q_list=(1,),
    s: int = 7,
) -> Tuple[sm.tsa.statespace.sarimax.SARIMAXResults, Tuple[int, int, int], Tuple[int, int, int, int]]:
    """
    Hace una búsqueda mínima de SARIMA sobre combinaciones provistas.
    Devuelve: (modelo_ajustado, order, seasonal_order)
    Nota: No valida aquí; la validación externa (test) es la que decide.
    """
    y_train = y_train.asfreq("D")
    best = None

    for p in p_list:
        for q in q_list:
            order = (p, d, q)
            for P in P_list:
                for Q in Q_list:
                    seasonal_order = (P, D, Q, s)
                    try:
                        m = sm.tsa.statespace.SARIMAX(
                            y_train,
                            order=order,
                            seasonal_order=seasonal_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        ).fit(disp=False)
                        if best is None:
                            best = (m, order, seasonal_order)
                    except Exception:
                        continue

    if best is None:
        raise RuntimeError("No se pudo ajustar ningún SARIMA con las combinaciones dadas.")

    return best  # (model, order, seasonal_order)


# =========================================================
# Modelos ML clásicos
# =========================================================
def train_ridge(X_train: pd.DataFrame, y_train: pd.Series, alpha: float = 1.0) -> Ridge:
    """Entrena Ridge Regression sencillo."""
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model


def train_rf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 500,
    max_depth: int | None = 10,
    min_samples_leaf: int = 1,
    random_state: int = 42,
) -> RandomForestRegressor:
    """Entrena RandomForestRegressor con hiperparámetros básicos."""
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_xgb(X_train: pd.DataFrame, y_train: pd.Series, cfg: Dict[str, Any]):
    """Entrena XGBoost con un diccionario de hiperparámetros."""
    if XGBRegressor is None:
        raise ImportError("xgboost no está instalado. Instala con: pip install xgboost")
    model = XGBRegressor(random_state=42, n_jobs=-1, **cfg)
    model.fit(X_train, y_train)
    return model


def train_lgbm(X_train: pd.DataFrame, y_train: pd.Series, cfg: Dict[str, Any]):
    """Entrena LightGBM con un diccionario de hiperparámetros."""
    if LGBMRegressor is None:
        raise ImportError("lightgbm no está instalado. Instala con: pip install lightgbm")
    model = LGBMRegressor(random_state=42, n_jobs=-1, **cfg)
    model.fit(X_train, y_train)
    return model
