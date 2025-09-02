import os, sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# --- asegurar que podemos importar "src.*" al ejecutar desde la raíz
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- módulos propios
from src.forecast import load_final_model, forecast_next_n
from src.data_processing import create_features
from src.evaluation import regression_metrics, smape, nmae

DATA_SERIES_PATH = PROJECT_ROOT / "data" / "processed" / "ventas_diarias.csv"
METRICS_PATH     = PROJECT_ROOT / "data" / "test" / "metricas_final_model.csv"
PRED_PATH        = PROJECT_ROOT / "data" / "test" / "predicciones_final_model.csv"
FINAL_MODEL_PATH = PROJECT_ROOT / "models" / "final_model.pkl"
CLUSTER_CSV      = PROJECT_ROOT / "data" / "processed" / "cluster_labels_por_dia.csv"
ANOMALY_CSV      = PROJECT_ROOT / "data" / "processed" / "anomalies.csv"

st.set_page_config(page_title="Predicción de Ventas", layout="wide")

# ======================
# CACHÉS
# ======================
@st.cache_data
def load_series(path: Path) -> pd.Series:
    s = pd.read_csv(path, parse_dates=["InvoiceDate"], index_col="InvoiceDate")["TotalPrice"]
    return s.asfreq("D").fillna(0.0)

@st.cache_resource
def load_model(path: Path):
    return load_final_model(path)

@st.cache_data
def load_csv_if_exists(path: Path, col_name: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=[0], index_col=0)
    if col_name not in df.columns and df.shape[1] == 1:
        df.columns = [col_name]
    df.index.name = "InvoiceDate"
    return df[[col_name]]

# ======================
# FUNCIONES AUXILIARES
# ======================
def build_features_with_unsup(series: pd.Series, final_model: dict) -> pd.DataFrame:
    """
    Reconstruye features base (lags/MA/calendario) y añade cluster/anomaly
    usando, si hay, los CSV guardados. Si faltan fechas, los infiere con
    KMeans/IsolationForest incluidos en final_model.
    """
    df_feat = create_features(series)  # y, lags, MA, weekday, is_weekend

    # 1) intentar leer labels guardados
    clusters = load_csv_if_exists(CLUSTER_CSV, "cluster")
    anomalies = load_csv_if_exists(ANOMALY_CSV, "anomaly")
    if clusters is not None:
        df_feat = df_feat.join(clusters, how="left")
    if anomalies is not None:
        df_feat = df_feat.join(anomalies, how="left")

    # 2) completar faltantes con modelos del final_model
    km_scaler = final_model["kmeans"]["scaler"]
    km_model  = final_model["kmeans"]["model"]
    km_cols   = final_model["kmeans"]["feature_cols"]      # ["TotalPrice","weekday","is_weekend"]
    iso_model = final_model["isoforest"]["model"]
    iso_cols  = final_model["isoforest"]["feature_cols"]   # ["TotalPrice","weekday","is_weekend"]

    def infer_row(idx, y_val):
        weekday    = idx.weekday()
        is_weekend = int(weekday >= 5)
        row = pd.DataFrame({"TotalPrice":[float(y_val)],
                            "weekday":[weekday],
                            "is_weekend":[is_weekend]}, index=[idx])
        clus = int(km_model.predict(km_scaler.transform(row[km_cols]))[0])
        anom = int(iso_model.predict(row[iso_cols])[0])
        return clus, anom

    if "cluster" not in df_feat.columns:
        df_feat["cluster"] = np.nan
    if "anomaly" not in df_feat.columns:
        df_feat["anomaly"] = np.nan

    missing_mask = df_feat[["cluster","anomaly"]].isna().any(axis=1)
    if missing_mask.any():
        for idx, row in df_feat.loc[missing_mask].iterrows():
            clus, anom = infer_row(idx, row["y"])
            df_feat.at[idx, "cluster"] = clus
            df_feat.at[idx, "anomaly"] = anom

    # ordenar columnas según el modelo
    feat_cols = final_model["feature_columns"]
    X = df_feat.drop(columns="y").reindex(columns=feat_cols, fill_value=0)
    y = df_feat["y"]
    return X, y

def compute_test_metrics(series: pd.Series, final_model: dict, cutoff: pd.Timestamp) -> tuple[dict, pd.DataFrame]:
    """
    Si existe metricas_final_model.csv la muestra; si no, calcula métricas
    sobre TEST reconstruyendo features con cluster/anomaly.
    """
    if METRICS_PATH.exists() and PRED_PATH.exists():
        met = pd.read_csv(METRICS_PATH).iloc[0].to_dict()
        preds = pd.read_csv(PRED_PATH, parse_dates=[0], index_col=0)
        preds.columns = ["y_real", "y_pred"]
        return met, preds

    # calcular en caliente
    X, y = build_features_with_unsup(series, final_model)
    X_test = X.loc[X.index >= cutoff]
    y_test = y.loc[y.index >= cutoff]

    rf = final_model["rf_model"]
    y_pred = rf.predict(X_test)

    met = regression_metrics(y_test, y_pred)
    met["sMAPE"] = smape(y_test, y_pred)
    met["nMAE"]  = nmae(y_test, y_pred)

    preds = pd.DataFrame({"y_real": y_test, "y_pred": pd.Series(y_pred, index=y_test.index)})
    return met, preds

def plot_last30_plus_forecast(series: pd.Series, forecast: pd.Series, title: str):
    last_30 = series.tail(30)
    fig, ax = plt.subplots(figsize=(11,3.8))
    ax.plot(last_30.index, last_30.values, label="Últimos 30 (real)")
    ax.plot(forecast.index, forecast.values, "--", label=f"Próximos {len(forecast)} (pred)")
    ax.axvline(last_30.index[-1], linestyle="--", color="gray")
    ax.set_title(title); ax.set_ylabel("Ventas (£)"); ax.legend()
    st.pyplot(fig)

def plot_test_vs_pred(preds: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(11,3.8))
    ax.plot(preds.index, preds["y_real"], label="Real (TEST)")
    ax.plot(preds.index, preds["y_pred"], "--", label="Predicción")
    ax.set_title(title); ax.set_ylabel("Ventas (£)"); ax.legend()
    st.pyplot(fig)

# ======================
# UI
# ======================
st.title("📈 Predicción de Ventas – Modelo Final")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuración")
    cutoff_str = st.text_input("Fecha de corte TEST (YYYY-MM-DD)", value="2011-03-01")
    try:
        cutoff = pd.to_datetime(cutoff_str)
    except Exception:
        st.error("Fecha de corte inválida. Usa formato YYYY-MM-DD.")
        st.stop()

    n_days = st.slider("Días a pronosticar", min_value=1, max_value=90, value=7, step=1)

# Cargar serie y modelo
if not DATA_SERIES_PATH.exists():
    st.error(f"No se encontró {DATA_SERIES_PATH}.")
    st.stop()
series = load_series(DATA_SERIES_PATH)

if not FINAL_MODEL_PATH.exists():
    st.error(f"No se encontró {FINAL_MODEL_PATH}.")
    st.stop()
final_model = load_model(FINAL_MODEL_PATH)

# Métricas
met, preds = compute_test_metrics(series, final_model, cutoff)
c1, c2, c3, c4 = st.columns(4)
c1.metric("MAE", f"{met['MAE']:.0f}")
c2.metric("RMSE", f"{met['RMSE']:.0f}")
c3.metric("sMAPE", f"{met['sMAPE']:.1f}%")
c4.metric("nMAE", f"{met['nMAE']:.1f}%")

st.subheader("🧪 Test: Real vs Predicción (modelo final)")
plot_test_vs_pred(preds, "Test (marzo 2011) – Real vs Predicho")

# Forecast N días
st.subheader(f"🔮 Forecast próximo {n_days} días")
forecast_n = forecast_next_n(series, final_model, N=n_days)
plot_last30_plus_forecast(series, forecast_n, f"Últimos 30 + {n_days} predichos")

# Descargar forecast
csv_bytes = forecast_n.to_frame("forecast").to_csv().encode("utf-8")
st.download_button(
    label="Descargar forecast en CSV",
    data=csv_bytes,
    file_name=f"forecast_next_{n_days}_days.csv",
    mime="text/csv"
)