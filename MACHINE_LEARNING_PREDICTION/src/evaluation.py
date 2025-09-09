# src/evaluation.py
"""
evaluation.py
Métricas y utilidades para evaluar modelos de regresión temporal.
Incluye helpers para guardar resultados.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------------
# Métricas
# -----------------------------

def regression_metrics(y_true, y_pred) -> dict:
    """
    Devuelve un dict con MAE y RMSE.
    Alinea índices si y_true es Series y y_pred es ndarray/Series.
    """
    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred, index=y_true.index)
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"MAE": mae, "RMSE": rmse}

def smape(y_true, y_pred, eps: float = 1e-8) -> float:
    """
    sMAPE (%) robusto a ceros: 200*|y - yhat| / (|y| + |yhat| + eps)
    """
    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred, index=y_true.index)
    val = 200.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + eps)
    return float(val.mean())

def nmae(y_true, y_pred, eps: float = 1e-8) -> float:
    """
    MAE normalizado por la media del valor real (en %).
    """
    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred, index=y_true.index)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return float(100.0 * mae / (np.mean(np.abs(y_true)) + eps))

# -----------------------------
# Reportes
# -----------------------------

def compare_models(results: dict) -> pd.DataFrame:
    """
    results: {"NombreModelo": {"MAE": float, "RMSE": float, ...}, ...}
    Retorna DataFrame ordenado por MAE ascendente.
    """
    df = pd.DataFrame(results).T
    # columnas numéricas primero por orden
    cols = [c for c in ["MAE", "RMSE", "sMAPE", "nMAE"] if c in df.columns] + \
           [c for c in df.columns if c not in {"MAE","RMSE","sMAPE","nMAE"}]
    df = df[cols] if cols else df
    return df.sort_values(by="MAE") if "MAE" in df.columns else df

def evaluate_and_save(y_true, y_pred, out_dir: str | Path,
                      save_predictions: bool = True,
                      save_metrics: bool = True,
                      prefix: str = "final") -> dict:
    """
    Calcula métricas (MAE, RMSE, sMAPE, nMAE) y guarda:
      - {out_dir}/metricas_{prefix}.csv
      - {out_dir}/predicciones_{prefix}.csv
    Devuelve dict con métricas.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    y_true_s = pd.Series(y_true)
    y_pred_s = pd.Series(y_pred, index=y_true_s.index)

    met = {
        "MAE":  float(mean_absolute_error(y_true_s, y_pred_s)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true_s, y_pred_s))),
        "sMAPE": float(smape(y_true_s, y_pred_s)),
        "nMAE": float(nmae(y_true_s, y_pred_s)),
    }

    if save_metrics:
        pd.DataFrame([met]).to_csv(out / f"metricas_{prefix}.csv", index=False)

    if save_predictions:
        pd.DataFrame({"y_real": y_true_s, "y_pred": y_pred_s}, index=y_true_s.index)\
          .to_csv(out / f"predicciones_{prefix}.csv")

    return met
