Predicción de Ventas

Proyecto final del Bootcamp de Data Science.  
El objetivo es predecir las ventas diarias de un e-commerce utilizando técnicas de series temporales y modelos de machine learning supervisado y no supervisado.

# Estructura del proyecto
prediccion_ventas/

│-- data/
│ │-- raw/ # Datos originales (dataset completo, filtrado UK)
│ │-- processed/ # Datos procesados (ventas diarias, features, etc.)
│ │-- train/ # Conjuntos de entrenamiento
│ │-- test/ # Conjuntos de test
│
│-- notebooks/
│ │-- 01_Fuentes.ipynb
│ │-- 02_LimpiezaEDA.ipynb
│ │-- 03_Entrenamiento_Evaluacion.ipynb
│
│-- src/ # Código modular (procesamiento, training, forecast...)
│-- models/ # Modelos entrenados (.pkl)
│-- app_streamlit/ # Demo en Streamlit
│-- docs/ # Presentación, memoria y documentación
│-- README.md

1. Clonar el repositorio o descargarlo.  

2. Crear un entorno virtual:  
   ```bash
   python -m venv venv

3. Activar el entorno virtual:

4. pip install -r requirements.txt

5. Explorar los notebooks en la carpeta notebooks/.

6. El modelo final entrenado se encuentra en:
models/final_model.pkl

Puede usarse para predecir próximos días ejecutando:
python src/forecast.py

Resultados
El modelo final combina:
Random Forest con lags/medias móviles.
Features derivados de clustering (KMeans) y detección de anomalías (Isolation Forest).
Métricas en Test (marzo 2011):
MAE = 3,694
RMSE = 5,632
sMAPE = 44.7%
nMAE = 19.8%

Esto significa que el modelo se equivoca en promedio un 20% respecto a las ventas diarias reales.

Autor

Proyecto desarrollado por Gilberto Gregorio Figueira