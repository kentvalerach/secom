# secom

# **Optimización de la Calidad en la Producción de Chips: Un Enfoque Basado en Datos**

Este repositorio contiene los scripts y resultados del análisis de datos llevado a cabo para predecir productos defectuosos y priorizar áreas de mejora en la producción de chips. Se implementaron tres modelos de machine learning: Random Forest básico, Random Forest con hiperparámetros y XGBoost, representando una evolución progresiva para un primer proyecto.

---

## **Contenido del Repositorio**

- **`data/`**: Archivos de datos, incluyendo el conjunto de datos original (`uci_secom.csv`) y el conjunto limpio (`cleaned_secom.csv`).
- **`scripts/`**: Scripts en R utilizados para el análisis y entrenamiento de modelos.
  - `secom_random_forest.R`: Implementación de Random Forest básico.
  - `secom_random_forest_hyperparameters.R`: Random Forest con optimización de hiperparámetros.
  - `secom_data_xboost.R`: Implementación de XGBoost.
- **`results/`**: `Informes de la elaboración de scripts paso a paso con métricas generadas por los modelos, conclusiones, visualizaciones, ajustes y optimizacion,` `SOW,` `Hoja de ruta analisis de datos.`
 
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
- **Precisión (Accuracy):** 97%
- **Sensibilidad (Recall):** 62%
- **Especificidad:** 76%
- **Área Bajo la Curva (AUC):** 0.7711
- **Observación:** Modelo final que muestra el mejor desempeño para identificar productos defectuosos.


---
### **Resumen del Proyecto**

Este proyecto analiza datos de producción de una fábrica de chips para identificar patrones en los defectos y proponer estrategias de mejora basadas en datos. Se aborda el problema como una clasificacion binaria ya que la variable objetivo es categórica. Se desarrollaron tres modelos predictivos para clasificar productos defectuosos: un modelo básico de Random Forest, un modelo optimizado con hiperparámetros y un modelo XGBoost. Cada modelo fue evaluado en términos de precisión, sensibilidad y AUC para determinar su efectividad en la tarea. 

### **Informes y Documentos**

- [Informe Modelo Básico Random Forest](https://github.com/kentvalerach/secom/blob/main/Informe_ramdom_forest_basico.pdf)
- [Informe Modelo Random Forest con Hiperparámetros](https://github.com/kentvalerach/secom/blob/main/Informe_ramdom_forest_Hiperparametros.pdf)
- [Informe Modelo XGBoost](https://github.com/kentvalerach/secom/blob/main/Informe_modelo_xgboost.pdf)
- [Statement of Work (SOW)](https://github.com/kentvalerach/secom/blob/main/Proyecto_Analisis_Datos_SOW.pdf)

El objetivo final es proporcionar recomendaciones accionables para mejorar la calidad y reducir costos en la producción de chips.

---
## **Requisitos**

Antes de ejecutar los scripts, asegúrate de tener instalados los siguientes paquetes en R:

```R
install.packages(c("tidyverse", "caret", "xgboost", "randomForest", "pROC", "DMwR2" , "dplyr"))



