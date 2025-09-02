from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

# === Helpers (idénticos a los que usaste al entrenar) ===

def build_base_features(series: pd.Series, lags=[1,7,14], ma_windows=[7,14]) -> pd.DataFrame:
    s = series.asfreq("D").copy()
    df = pd.DataFrame({"y": s})
    for L in lags:
        df[f"lag{L}"] = df["y"].shift(L)
    for W in ma_windows:
        df[f"ma{W}"] = df["y"].rolling(W, min_periods=W).mean().shift(1)
    df["weekday"] = df.index.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    return df

def _km_iso_for_date(date, y_guess, km_scaler, km_model, km_cols, iso_model, iso_cols):
    weekday = date.weekday(); is_weekend = int(weekday >= 5)
    row = {"TotalPrice": float(y_guess), "weekday": weekday, "is_weekend": is_weekend}
    df_tmp = pd.DataFrame([row], index=[date])
    clus = int(km_model.predict(km_scaler.transform(df_tmp[km_cols]))[0])
    anom = int(iso_model.predict(df_tmp[iso_cols])[0])
    return clus, anom

# === Carga/uso del modelo final ===

def load_final_model(path: str | Path):
    """Carga el diccionario de artifacts guardado como final_model.pkl."""
    with open(Path(path), "rb") as f:
        return pickle.load(f)

def predict_final_model(X: pd.DataFrame, final_model: dict):
    """Hace predicciones con el RF del modelo final, alineando columnas."""
    rf = final_model["rf_model"]
    cols = final_model["feature_columns"]
    X_aligned = X.reindex(columns=cols, fill_value=0)
    return rf.predict(X_aligned)

def forecast_next_n(series: pd.Series, final_model: dict, N=7) -> pd.Series:
    """
    Forecast recursivo para N días usando:
    - lags/MA (base_features)
    - cluster/anomaly (KMeans + IsolationForest) estimados con y_guess/y_hat
    """
    rf_model  = final_model["rf_model"]
    rf_cols   = final_model["feature_columns"]
    cfg       = final_model["feature_config"]
    km_scaler = final_model["kmeans"]["scaler"]
    km_model  = final_model["kmeans"]["model"]
    km_cols   = final_model["kmeans"]["feature_cols"]
    iso_model = final_model["isoforest"]["model"]
    iso_cols  = final_model["isoforest"]["feature_cols"]

    s_hist = series.asfreq("D").copy()
    preds = []
    current_date = s_hist.index[-1]

    for _ in range(N):
        next_date = current_date + pd.Timedelta(days=1)
        df_base = build_base_features(s_hist, **cfg).dropna()
        last_row = df_base.iloc[[-1]].copy()

        # y_guess inicial
        ma7 = last_row["ma7"].values[0] if "ma7" in last_row else np.nan
        lag1 = last_row["lag1"].values[0] if "lag1" in last_row else np.nan
        y_guess = ma7 if np.isfinite(ma7) else (lag1 if np.isfinite(lag1) else float(s_hist.iloc[-1]))

        clus, anom = _km_iso_for_date(next_date, y_guess, km_scaler, km_model, km_cols, iso_model, iso_cols)

        # features futuras
        x_future = last_row.drop(columns=["y"], errors="ignore").copy()
        x_future["weekday"] = next_date.weekday()
        x_future["is_weekend"] = int(next_date.weekday() >= 5)
        x_future["cluster"] = clus
        x_future["anomaly"] = anom
        x_future = x_future.reindex(columns=rf_cols, fill_value=0)

        # primera predicción + refinamiento
        y_hat1 = float(rf_model.predict(x_future)[0])
        clus2, anom2 = _km_iso_for_date(next_date, y_hat1, km_scaler, km_model, km_cols, iso_model, iso_cols)
        x_future["cluster"] = clus2
        x_future["anomaly"] = anom2
        y_hat = float(rf_model.predict(x_future)[0])

        preds.append(y_hat)
        s_hist = pd.concat([s_hist, pd.Series([y_hat], index=[next_date])])
        current_date = next_date

    future_index = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=N, freq="D")
    return pd.Series(preds, index=future_index)

def forecast_and_save(series: pd.Series, final_model: dict, N: int, out_csv: str | Path) -> pd.Series:
    """
    Hace forecast N días y guarda un CSV con una columna 'forecast'.
    Devuelve la Serie predicha (índice = fechas futuras).
    """
    out_csv = Path(out_csv)
    preds = forecast_next_n(series, final_model, N=N)
    preds.to_frame("forecast").to_csv(out_csv)
    return preds
