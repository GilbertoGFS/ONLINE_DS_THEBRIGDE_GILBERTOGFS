import pandas as pd

# -----------------------------
# Carga y filtrado
# -----------------------------

def load_raw(path: str) -> pd.DataFrame:
    """Carga el CSV original (data/raw)."""
    return pd.read_csv(path, encoding="latin1")

def filter_country(df: pd.DataFrame, country: str = "United Kingdom") -> pd.DataFrame:
    """Filtra el dataset por país."""
    return df[df["Country"] == country].copy()

def clean_retail(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza básica:
    - Convertir fechas
    - Quitar facturas de cancelación (Invoice con 'C')
    - Filtrar cantidad > 0 y precio > 0
    - Crear columna TotalPrice
    """
    df = df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    return df

# -----------------------------
# Series temporales y features
# -----------------------------

def build_daily_series(df: pd.DataFrame) -> pd.Series:
    """Agrupa a nivel diario por fecha."""
    return df.groupby(df["InvoiceDate"].dt.date)["TotalPrice"].sum()\
             .rename_axis("InvoiceDate")\
             .rename("TotalPrice")\
             .asfreq("D")

def create_features(series: pd.Series) -> pd.DataFrame:
    """Genera features de lags, medias móviles, calendario."""
    df_feat = pd.DataFrame({"y": series})

    # Lags
    df_feat["lag1"] = df_feat["y"].shift(1)
    df_feat["lag7"] = df_feat["y"].shift(7)
    df_feat["lag14"] = df_feat["y"].shift(14)

    # Medias móviles
    df_feat["ma7"] = df_feat["y"].rolling(7, min_periods=1).mean().shift(1)
    df_feat["ma14"] = df_feat["y"].rolling(14, min_periods=1).mean().shift(1)

    # Calendario
    df_feat["weekday"] = df_feat.index.weekday
    df_feat["is_weekend"] = (df_feat["weekday"] >= 5).astype(int)

    return df_feat.dropna()

def time_split(series: pd.Series, split_date: str):
    """Divide serie en train y test según fecha."""
    train = series[series.index < split_date]
    test = series[series.index >= split_date]
    return train, test