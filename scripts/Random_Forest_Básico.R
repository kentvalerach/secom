# ============
# LOAD DATASET
# ============

# Dataset Klagge
secom_data <- read.csv("data/uci_secom.csv")

# Verificar que el dataset se cargó correctamente
dim(secom_data)      # Dimensiones del dataset
colnames(secom_data) # Nombres de columnas
head(secom_data)     # Primeras filas del dataset

# ===========
# CLEAN DATA
#============

# Contar valores NA por columna
na_counts <- colSums(is.na(secom_data))

# Calcular el porcentaje de valores NA por columna
na_percentage <- colMeans(is.na(secom_data)) * 100

# Visualizar los resultados
na_counts[na_counts > 0]  # Columnas con valores faltantes
na_percentage[na_percentage > 0]  # Porcentaje de valores faltantes

# Eliminar columnas con más del 50% de NA
secom_data <- secom_data[, colMeans(is.na(secom_data)) < 0.5]

# Verificar dimensiones después de eliminar columnas
dim(secom_data)

# Imputar valores faltantes con la mediana
library(dplyr)
secom_data <- secom_data %>%
  dplyr::mutate(dplyr::across(everything(), ~ ifelse(is.na(.), median(., na.rm = TRUE), .)))

# Verificar que no queden valores faltantes
colSums(is.na(secom_data))

# =============================
# NORMALIZACION Y ESCALAR DATOS
# =============================

# Escalar datos numéricos (excluyendo la primera y última columna)
scaled_data <- scale(secom_data[, -c(1, ncol(secom_data))])

# Verificar dimensiones del conjunto escalado
dim(scaled_data)
# ===================================
# ELIMINAR COLUMNAS CON VARIANZA CERO
# ===================================

col_var <- apply(scaled_data, 2, stats::var)
scaled_data <- scaled_data[, col_var > 0]

# Filtrar columnas con varianza mayor a 0
scaled_data <- scaled_data[, col_var > 0]

# Verificar dimensiones después de eliminar columnas de varianza cero
dim(scaled_data)

# =============================
# PREPARAR LA VARIABLE OBJETIVA
# =============================

# Convertir `Pass/Fail` a factor
secom_data$Pass.Fail <- factor(secom_data$Pass.Fail, levels = c(-1,1), labels = c("Fail", "Pass"))


# Confirmar los niveles
table(secom_data$Pass.Fail)

# =======================
# GUARDAR DATASET LIMPIO
# =======================

write.csv(secom_data, "data/cleaned_secom.csv", row.names = FALSE)


# ==============
# ANALYSIS (EDA)
# ==============

# Tabla de frecuencias
table(secom_data$Pass.Fail)

# Gráfico de barras
barplot(
  table(secom_data$Pass.Fail), 
  col = c("red", "green"), 
  main = "Distributión de Pass/Fail", 
  xlab = "Stand", 
  ylab = "Frecuency"
)

# ===========
# CORRELACION
# ===========

# Seleccionar las columnas numéricas
numeric_columns <- secom_data[, sapply(secom_data, is.numeric)]

# Identificar columnas con desviación estándar cero
zero_sd_columns <- sapply(numeric_columns, function(col) sd(col, na.rm = TRUE) == 0)

# Eliminar esas columnas
numeric_columns <- numeric_columns[, !zero_sd_columns]

# Confirmar que no quedan columnas con desviación estándar cero
if (any(sapply(numeric_columns, function(col) sd(col, na.rm = TRUE) == 0))) {
  print("Todavía hay columnas con desviación estándar cero.")
} else {
  print("Todas las columnas con desviación estándar cero han sido eliminadas.")
}

# Recalcular la matriz de correlación
cor_matrix <- cor(numeric_columns)

# Identificar correlación con la variable objetivo
if ("Pass.Fail" %in% colnames(cor_matrix)) {
  cor_target <- cor_matrix["Pass.Fail", ]
  significant_vars <- names(cor_target[abs(cor_target) > 0.5]) # Umbral de 0.5
  print(significant_vars)
} else {
  print("No se encontró correlación directa con Pass.Fail")
}

# =============================
# REDUCCION DE LA DIMENSION PCA
# =============================
# Escalo, las columnas numéricas excluyendo la columna Pass/Fail
scaled_data <- scale(secom_data[, -c(1, ncol(secom_data))])

# Identificar columnas con varianza cero
zero_var_columns <- apply(scaled_data, 2, var) == 0

# Mostrar las columnas constantes
cat("Columnas constantes o nulas:", names(zero_var_columns[zero_var_columns]), "\n")

# Eliminar columnas constantes o nulas
scaled_data <- scaled_data[, !zero_var_columns]

# Confirmar dimensiones tras la limpieza
cat("Dimensiones de scaled_data después de eliminar columnas constantes:", dim(scaled_data), "\n")

# Escalar datos excluyendo columnas no numéricas
numeric_columns <- secom_data[, sapply(secom_data, is.numeric)]

# Eliminar columnas con varianza cero
scaled_data <- numeric_columns[, apply(numeric_columns, 2, var) > 0]

# Escalar los datos
scaled_data <- scale(scaled_data)

# Realizar PCA
pca_result <- prcomp(scaled_data, center = TRUE, scale. = TRUE)

# Resumen de PCA
summary(pca_result)

# Extraer la varianza explicada
explained_variance <- pca_result$sdev^2 / sum(pca_result$sdev^2)

# Graficar la varianza explicada con puntos y líneas
plot(
  explained_variance,
  type = "b",  # Tipo de gráfico: línea con puntos
  pch = 16,    # Forma del punto
  col = "blue",# Color de la línea y los puntos
  main = "Varianza Explicada por Componente", 
  xlab = "Componentes Principales", 
  ylab = "Varianza Explicada"
)

# ====================
# EQUILIBRIO DE CLASES
# ====================

# revisO el balance de clases
table(secom_data$Pass.Fail)

# Realizo un sobremuestreo de la clase minoritaria
library(dplyr)

# Separar por clase
failures <- secom_data[secom_data$Pass.Fail == "Fail", ]
successes <- secom_data[secom_data$Pass.Fail == "Pass", ]

# Sobremuestrear la clase minoritaria
set.seed(123)
balanced_failures <- failures[sample(nrow(failures), nrow(successes), replace = TRUE), ]
balanced_data <- rbind(successes, balanced_failures)

# Verificar nuevo balance
table(balanced_data$Pass.Fail)

# ====================
# DIVISION DEL DATASET
# ====================

library(caret)

set.seed(123)

# Dividimos los datos en conjuntos de entrenamiento y prueba (80/20)
train_index <- createDataPartition(balanced_data$Pass.Fail, p = 0.8, list = FALSE)

train_data <- balanced_data[train_index, ]
test_data <- balanced_data[-train_index, ]

# Confirmar tamaños
dim(train_data)
dim(test_data)

# =================
# MODELO PREDICTIVO
# =================

# Elijo modelo Random Forest para predecir el estado Pass/Fail
library(randomForest)

# Entrenar modelo Random Forest
rf_model <- randomForest(Pass.Fail ~ ., data = train_data, importance = TRUE)

# Evaluar modelo
rf_predictions <- predict(rf_model, test_data)
conf_matrix <- table(test_data$Pass.Fail, rf_predictions)
print(conf_matrix)

# ======================
# METRICAS DE EVALUACION
# ======================

# Calcular métricas a partir de la matriz de confusión
true_positive <- conf_matrix["Pass", "Pass"]
true_negative <- conf_matrix["Fail", "Fail"]
false_positive <- conf_matrix["Fail", "Pass"]
false_negative <- conf_matrix["Pass", "Fail"]

# Métricas
accuracy <- (true_positive + true_negative) / sum(conf_matrix)
precision <- true_positive / (true_positive + false_positive)
recall <- true_positive / (true_positive + false_negative)  # Sensibilidad
specificity <- true_negative / (true_negative + false_positive)

# Imprimir métricas
cat("Precisión (Accuracy):", round(accuracy, 2), "\n")
cat("Precisión (Precision):", round(precision, 2), "\n")
cat("Sensibilidad (Recall):", round(recall, 2), "\n")
cat("Especificidad (Specificity):", round(specificity, 2), "\n")

# ==========
# CURVA(ROC)
# ==========

library(pROC)
rf_prob <- predict(rf_model, test_data, type = "prob")[, 2]
roc_curve <- pROC::roc(test_data$Pass.Fail, rf_prob, levels = c("Fail", "Pass"))
plot(roc_curve, main = "Curva ROC", col = "blue")
auc_value <- pROC::auc(roc_curve)
cat("Área bajo la curva (AUC):", auc_value, "\n")

# =======================================
# VISUALIZAR IMPORTANCIA DE LAS VARIABLES
# =======================================

# Usar rf_model para mostrar importancia
var_imp <- importance(rf_model)
plot(var_imp, main = "Importancia de Variables")
