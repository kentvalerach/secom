# secom

# **Optimización de la Calidad en la Producción de Chips: Un Enfoque Basado en Datos**

Este repositorio contiene los scripts y resultados del análisis de datos llevado a cabo para predecir productos defectuosos y priorizar áreas de mejora en la producción de micro-chips. En este particular la data representa 590 procesos con un estado determinado que arroja un resultado Pass/Fail. Se implementaron tres modelos de Machine Learning: Random Forest básico, Random Forest con hiperparámetros y XGboost, representando una evolución progresiva para un primer proyecto. La data utilizada para este analisis es proporcionada por el sitio https://www.kaggle.com/datasets/paresh2047/uci-semcom

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

- [Informe Modelo Básico Random Forest](https://github.com/kentvalerach/secom/blob/main/results/Informe_random_forest_basico.pdf )
- [Informe Modelo Random Forest con Hiperparámetros](https://github.com/kentvalerach/secom/blob/main/results/Informe_random_forest_Hiperparametros.pdf)
- [Informe Modelo XGBoost](https://github.com/kentvalerach/secom/blob/main/results/Informe_modelo_XGboost_Machine_Learning.pdf)
- [Statement of Work (SOW)](https://github.com/kentvalerach/secom/blob/main/results/Proyecto_Analisis_Datos_SOW.pdf)
- [Hoja de ruta para el analisis de datos](https://github.com/kentvalerach/secom/blob/main/results/Hoja_de_ruta.pdf)
- [Visualizaciones](https://github.com/kentvalerach/secom/blob/main/results/Top_10_caracteristicas_mas_importantes.pdf)

El objetivo final es proporcionar recomendaciones accionables para mejorar la calidad y reducir costos en la producción de chips, mejorando los atascos en la cola de produccion. 

Nota: Usted podra visualizar los resultados de estos tres algoritmos de Machine Learning simplemente con acceder a su cuenta github y en la pestana superior derecha de color verde; en code, pulse mas para un codespace. Se abrira un terminal o consola (tardara unos minutos en cargar los paquetes necesarios) luego llame el comando: Rscript + la ruta, ejemplo: Rscript scripts/Modelo_XGBoost.R desde su consola de codesspaces y listo.

Para ejecutarlo en su propio entorno R descarge los paquetes que a continuacion se describen. Luego descarge los dos archivos de datos: para XGboost  "cleaned_secom.csv"   y "uci_secom.csv" para los modelos de Random Forest. Recuerde que debe enrutar en el script del modelo la data necesaria para el analisis. O enrutamiento para guardar la data luego del proceso de limpieza. (solo en modelos  Random Forest) Ejemplo en scripts Random Forest: write.csv(secom_data, "C:/ruta_para_guardar_tus_datos/cleaned_secom.csv", row.names = FALSE)
Ejemplo en script XGboost: file_path <- "ruta_de_tu_descarga/cleaned_secom.csv" o en scripts Random_Forest;    secom_data <- read.csv("tu_ruta_de_descarga/uci_secom.csv") 

---
## **Requisitos**

Antes de ejecutar los scripts, asegúrate de tener instalados los siguientes paquetes en R:

```R
install.packages(c("tidyverse", "caret", "xgboost", "randomForest", "pROC", "DMwR2" , "dplyr"))



