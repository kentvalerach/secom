# secom

# **Quality Optimization in Chip Production: A Data-Driven Approach**

This repository contains the scripts and results of the data analysis carried out to predict defective products and prioritize areas of improvement in micro-chip production. In this particular case the data represents 590 processes with a given state that yields a Pass/Fail result. Three Machine Learning models were implemented: basic Random Forest, Random Forest with hyperparameters and XGboost, representing a progressive evolution for a first project. The data used for this analysis is provided by the site https://www.kaggle.com/datasets/paresh2047/uci-semcom.

# **Optimización de la Calidad en la Producción de Chips: Un Enfoque Basado en Datos**

Este repositorio contiene los scripts y resultados del análisis de datos llevado a cabo para predecir productos defectuosos y priorizar áreas de mejora en la producción de micro-chips. En este particular la data representa 590 procesos con un estado determinado que arroja un resultado Pass/Fail. Se implementaron tres modelos de Machine Learning: Random Forest básico, Random Forest con hiperparámetros y XGboost, representando una evolución progresiva para un primer proyecto. La data utilizada para este analisis es proporcionada por el sitio https://www.kaggle.com/datasets/paresh2047/uci-semcom

---

## **Repository content**
## **Contenido del Repositorio**

- **`data/`**: Archivos de datos, incluyendo datos originales (`uci_secom.csv`) y datos limpios (`cleaned_secom.csv`)/Data files, including original data (`uci_secom.csv`) and clean data (`cleaned_secom.csv`).
- **`scripts/`**: R scripts used for model analysis and training / Scripts en R utilizados para el análisis y entrenamiento de modelos.
  - `secom_random_forest.R`: Implementation of basic Random Forest / Implementación de Random Forest básico.
  - `secom_random_forest_hyperparameters.R`: Random Forest with hyperparameter optimization / Random Forest con optimización de hiperparámetros.
  - `secom_data_xboost.R`: XGBoost model implementation / Implementación de modelo XGBoost.
- **`results/`**:  Step-by-step scripting reports with model-generated metrics, conclusions, visualizations, adjustments and optimization, `SOW,`  `Data analysis roadmap` / `Informes de la elaboración de scripts paso a paso con métricas generadas por los modelos, conclusiones, visualizaciones, ajustes y optimizacion,` `SOW,` `Hoja de ruta analisis de datos`.
 
- **`README.md`**: This documentation archive / Este archivo de documentación.

---

## **Project Description**

The objective of this project is to analyze sensor data from a chip factory to:
1. predict whether a product will be defective (Pass/Fail).
2. Identify critical sensors in the production process.
3. Propose improvements in the quality of the process based on the findings.

## **Descripción del Proyecto**

El objetivo de este proyecto es analizar datos de sensores provenientes de una fábrica de chips para:
1. Predecir si un producto será defectuoso (Pass/Fail).
2. Identificar sensores críticos en el proceso de producción.
3. Proponer mejoras en la calidad del proceso basado en los hallazgos.

---

### **Steps of Analysis** 

1. **Questioning:** Define the customer's needs.
2. **Prepare:** Data cleansing and normalization.
3. **Process:** Training of three predictive models.
4. **Analyze:** Evaluation of model performance and analysis of important features.
5. **Sharing:** Creation of visualizations and reports.
6. **Acting:** Proposal of recommendations based on the results.
### **Pasos del Análisis**

1. **Preguntar:** Definir las necesidades del cliente.
2. **Preparar:** Limpieza y normalización de datos.
3. **Procesar:** Entrenamiento de tres modelos predictivos.
4. **Analizar:** Evaluación del desempeño de los modelos y análisis de características importantes.
5. **Compartir:** Creación de visualizaciones y reportes.
6. **Actuar:** Propuesta de recomendaciones basadas en los resultados.

---

## **Implemented Models and Results / Modelos Implementados y Resultados**

### **1. Random Forest Básic**
- **Precisión (Accuracy):** 70%
- **Sensibilidad (Recall):** 70%
- **Especificidad (Specificity):** 70%
- **Observatión:** An initial model to establish a predictive baseline / Un modelo inicial para establecer una línea base de predicción.

### **2. Random Forest with Hyperparameters**
- **Precisión (Accuracy):** 68%
- **Sensibilidad (Recall):** 70%
- **Especificidad (Specificity):** 65%
- **Observation:** Use of cross-validation and hyperparameter tuning / Uso de validación cruzada y ajuste de hiperparámetros.

### **3. XGBoost**
- **Precisión (Accuracy):** 97%
- **Sensibilidad (Recall):** 62%
- **Especificidad (Specificity):** 76%
- **Área Bajo la Curva (AUC):** 0.7711
- **Observatión:** Final model showing best performance in identifying defective products / Modelo final que muestra el mejor desempeño para identificar productos defectuosos.


---

###  **Project Summary**

This project analyzes production data from a chip fab to identify patterns in defects and propose data-driven improvement strategies. The problem is approached as a binary classification problem since the target variable is categorical. Three predictive models were developed to classify defective products: a basic Random Forest model, an optimized model with hyperparameters, and an XGBoost model. Each model was evaluated in terms of accuracy, sensitivity and AUC to determine its effectiveness on the task. 

### **Resumen del Proyecto**

Este proyecto analiza datos de producción de una fábrica de chips para identificar patrones en los defectos y proponer estrategias de mejora basadas en datos. Se aborda el problema como una clasificacion binaria ya que la variable objetivo es categórica. Se desarrollaron tres modelos predictivos para clasificar productos defectuosos: un modelo básico de Random Forest, un modelo optimizado con hiperparámetros y un modelo XGBoost. Cada modelo fue evaluado en términos de precisión, sensibilidad y AUC para determinar su efectividad en la tarea. 

### **Reports and Documents/Informes y Documentos**

- [Informe Modelo Básico Random Forest](https://github.com/kentvalerach/secom/blob/main/results/Informe_random_forest_basico.pdf )
- [Informe Modelo Random Forest con Hiperparámetros](https://github.com/kentvalerach/secom/blob/main/results/Informe_random_forest_Hiperparametros.pdf)
- [Informe Modelo XGBoost](https://github.com/kentvalerach/secom/blob/main/results/Informe_modelo_XGboost_Machine_Learning.pdf)
- [Statement of Work (SOW)](https://github.com/kentvalerach/secom/blob/main/results/Proyecto_Analisis_Datos_SOW.pdf)
- [Hoja de ruta para el analisis de datos](https://github.com/kentvalerach/secom/blob/main/results/Hoja_de_ruta.pdf)
- [Visualizaciones](https://github.com/kentvalerach/secom/blob/main/results/Top_10_caracteristicas_mas_importantes.pdf)

The ultimate goal is to provide actionable recommendations to improve quality and reduce costs in chip production by improving production queue bottlenecks. 

Note: You will be able to visualize the results of these three Machine Learning algorithms by simply logging into your github account and in the top right green tab; under code, click more for a codespace. A terminal or console will open (it will take a few minutes to load the necessary packages) then call the command: Rscript + the path, example: Rscript scripts/Model_XGBoost.R from your codesspaces console and you are done.

To run it in your own R environment download the packages described below. Then download the two data files: for XGboost “cleaned_secom.csv” and “uci_secom.csv” for the Random Forest models. Remember to route in the model script the data needed for the analysis. Example in Random Forest scripts: write.csv(secom_data, “C:/route_to_save_your_data/cleaned_secom.csv”, row.names = FALSE)
Example in XGboost script: file_path <- “path_of_your_download/cleaned_secom.csv” or in Random_Forest scripts; secom_data <- read.csv(“your_download_path/uci_secom.csv”). 

---

El objetivo final es proporcionar recomendaciones accionables para mejorar la calidad y reducir costos en la producción de chips, mejorando los atascos en la cola de produccion. 

Nota: Usted podra visualizar los resultados de estos tres algoritmos de Machine Learning simplemente con acceder a su cuenta github y en la pestana superior derecha de color verde; en code, pulse mas para un codespace. Se abrira un terminal o consola (tardara unos minutos en cargar los paquetes necesarios) luego llame el comando: Rscript + la ruta, ejemplo: Rscript scripts/Modelo_XGBoost.R desde su consola de codesspaces y listo.

Para ejecutarlo en su propio entorno R descarge los paquetes que a continuacion se describen. Luego descarge los dos archivos de datos: para XGboost  "cleaned_secom.csv"   y "uci_secom.csv" para los modelos de Random Forest. Recuerde que debe enrutar en el script del modelo la data necesaria para el analisis. O enrutamiento para guardar la data luego del proceso de limpieza. (solo en modelos  Random Forest) Ejemplo en scripts Random Forest: write.csv(secom_data, "C:/ruta_para_guardar_tus_datos/cleaned_secom.csv", row.names = FALSE)
Ejemplo en script XGboost: file_path <- "ruta_de_tu_descarga/cleaned_secom.csv" o en scripts Random_Forest;    secom_data <- read.csv("tu_ruta_de_descarga/uci_secom.csv")

---
## **Requirements / Requisitos**

Before running the scripts, make sure you have the following packages installed in R:

Antes de ejecutar los scripts, asegúrate de tener instalados los siguientes paquetes en R:

---
```R
install.packages(c("tidyverse", "caret", "xgboost", "randomForest", "pROC", "DMwR2" , "dplyr"))





