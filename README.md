# secom
Quality Optimization in Chip Production: A Data-Driven Approach
# **Optimización de la Calidad en la Producción de Chips: Un Enfoque Basado en Datos**

Este repositorio contiene los scripts y resultados del análisis de datos llevado a cabo para predecir productos defectuosos y priorizar áreas de mejora en la producción de chips. Se implementaron tres modelos de machine learning: Random Forest básico, Random Forest con hiperparámetros y XGBoost, representando una evolución progresiva para un primer proyecto.

---

## **Contenido del Repositorio**

- **`data/`**: Archivos de datos, incluyendo el conjunto de datos original (`uci_secom.csv`) y el conjunto limpio (`cleaned_secom.csv`).
- **`scripts/`**: Scripts en R utilizados para el análisis y entrenamiento de modelos.
  - `secom_random_forest.R`: Implementación de Random Forest básico.
  - `secom_random_forest_hyperparameters.R`: Random Forest con optimización de hiperparámetros.
  - `secom_data_xboost.R`: Implementación de XGBoost.
- **`results/`**: Visualizaciones y métricas generadas por los modelos.
- **`README.md`**: Este archivo de documentación.

---

## **Descripción del Proyecto**

El objetivo de este proyecto es analizar datos de sensores provenientes de una fábrica de chips para:
1. Predecir si un producto será defectuoso (Pass/Fail).
2. Identificar sensores críticos en el proceso de producción.
3. Proponer mejoras en la calidad del proceso basado en los hallazgos.

---

### **Pasos del Análisis**

1. **Preguntar:** Definir las necesidades del cliente.
2. **Preparar:** Limpieza y normalización de datos.
3. **Procesar:** Entrenamiento de tres modelos predictivos.
4. **Analizar:** Evaluación del desempeño de los modelos y análisis de características importantes.
5. **Compartir:** Creación de visualizaciones y reportes.
6. **Actuar:** Propuesta de recomendaciones basadas en los resultados.

---

## **Modelos Implementados y Resultados**

### **1. Random Forest Básico**
- **Precisión (Accuracy):** 70%
- **Sensibilidad (Recall):** 70%
- **Especificidad:** 70%
- **Observación:** Un modelo inicial para establecer una línea base de predicción.

### **2. Random Forest con Hiperparámetros**
- **Precisión (Accuracy):** 68%
- **Sensibilidad (Recall):** 70%
- **Especificidad:** 65%
- **Mejoras:** Uso de validación cruzada y ajuste de hiperparámetros.

### **3. XGBoost**
- **Precisión (Accuracy):** 94%
- **Sensibilidad (Recall):** 100%
- **Especificidad:** 70%
- **Área Bajo la Curva (AUC):** 0.7519
- **Observación:** Modelo final que muestra el mejor desempeño para identificar productos defectuosos.

---

## **Requisitos**

Antes de ejecutar los scripts, asegúrate de tener instalados los siguientes paquetes en R:

```R
install.packages(c("tidyverse", "caret", "xgboost", "randomForest", "pROC", "DMwR"))
