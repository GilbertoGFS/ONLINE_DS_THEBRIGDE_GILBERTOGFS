# 📊 Memoria del Proyecto: Predicción de Ventas

## 1. Introducción
El objetivo de este proyecto es **predecir las ventas diarias** de un e-commerce utilizando datos históricos.  
La motivación principal es ayudar al negocio a **planificar inventario y recursos**, reduciendo la incertidumbre en la demanda.

## 2. Datos
- Fuente: Online Retail Dataset (UCI Machine Learning Repository).  
- Periodo utilizado: Q1 2011 (enero a marzo).  
- Filtrado para **United Kingdom**.  
- Preprocesamiento:
  - Conversión de fechas a serie temporal.
  - Agregación a nivel **ventas diarias**.
  - Generación de variables (lags, medias móviles, día de la semana, fin de semana, etc.).
  - Incorporación de **clustering (KMeans)** y **detección de anomalías (Isolation Forest)**.

## 3. Metodología
Se probaron diferentes enfoques:
- **Modelos clásicos de series temporales**: ETS (Holt-Winters), SARIMA.
- **Modelos de machine learning supervisado**: Ridge, Random Forest, XGBoost, LightGBM.
- **Modelos no supervisados**: KMeans (para detectar patrones de días) e Isolation Forest (para anomalías).
- **Ensemble**: combinación de SARIMA + Random Forest.

El modelo final elegido fue un **Random Forest** enriquecido con features de **cluster** y **anomalía**.

## 4. Resultados
Comparativa de modelos (en Test – marzo 2011):

| Modelo                     | MAE   | RMSE  |
|-----------------------------|-------|-------|
| ETS (Holt-Winters)          | 6071  | 8754  |
| SARIMA (2,0,1)x(1,1,1,7)    | 5346  | 8046  |
| Ridge (lags + MA)           | 5315  | 7683  |
| Random Forest               | 4785  | 8115  |
| XGBoost                     | 4993  | 7682  |
| LightGBM                    | 8423  | 9997  |
| Ensemble SARIMA + RF        | 4637  | 7879  |
| **RF + cluster/anomaly** ⭐ | **3694** | **5632** |

Métricas adicionales:
- **sMAPE** = 44.7%  
- **nMAE** = 19.8%

## 5. Modelo Final
El modelo final es un **Random Forest** con:
- Variables de series temporales (lags, medias móviles).
- Variables de calendario (día de la semana, fin de semana).
- Variables no supervisadas:
  - Cluster (KMeans → días normales, bajos, picos).
  - Anomaly (Isolation Forest → outliers detectados).

Este modelo fue el que obtuvo **mejor desempeño** en términos de MAE y RMSE.

## 6. Conclusiones
- La combinación de enfoques **supervisados y no supervisados** mejora la precisión.
- El modelo es interpretable: sabemos qué variables son más importantes (ej. cluster, weekday).
- Se desarrolló una **app en Streamlit** que permite:
  - Visualizar el histórico de ventas.
  - Comparar predicciones vs reales.
  - Realizar forecast hasta 90 días.

## 7. Futuras Mejoras
- Probar más datos (años completos).
- Incluir variables externas (festivos, promociones, clima).
- Potenciar su desarrollo con Deep Learning.

