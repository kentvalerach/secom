# Script: xgboost_model.R
# Objetivo: Usar XGBoost para predecir el resultado del proceso de fabricación (Pass/Fail)
# Conjunto de datos: cleaned_secom.csv (limpio y filtrado)

# ============
# LOAD DATASET
# ============

# 1. Configuración inicial
install.packages("xgboost")
install.packages("caret")
install.packages("pROC")

library(xgboost)
library(caret)
library(pROC)

# Cargar el conjunto de datos limpio
data <- read.csv("data/cleaned_secom.csv")


# Verificar dimensiones y primeras filas
cat("Dimensiones del conjunto de datos:", dim(data), "\n")
head(data)

# 2. Verificación de la variable objetivo
# Asegurarse de que 'Pass/Fail' esté en formato categórico
data$Pass.Fail <- as.factor(data$Pass.Fail)

# 3. División de los datos
set.seed(123)  # Para reproducibilidad
train_index <- createDataPartition(data$Pass.Fail, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Documentar dimensiones
cat("Dimensiones de datos de entrenamiento:", dim(train_data), "\n")
cat("Dimensiones de datos de prueba:", dim(test_data), "\n")

# Crear matrices de entrenamiento y prueba
train_matrix <- xgb.DMatrix(
  data = as.matrix(train_data[, -c(1, ncol(train_data))]),  # Excluir la columna "Time" y la variable objetivo
  label = as.numeric(train_data$Pass.Fail) - 1              # Convertir "Pass/Fail" a valores numéricos (0 y 1)
)

test_matrix <- xgb.DMatrix(
  data = as.matrix(test_data[, -c(1, ncol(test_data))]),
  label = as.numeric(test_data$Pass.Fail) - 1
)

# Establesco los parámetros básicos para entrenar el modelo inicial
params <- list(
  objective = "binary:logistic",  # Clasificación binaria
  eval_metric = "auc",           # Área bajo la curva como métrica de evaluación
  eta = 0.1,                     # Tasa de aprendizaje
  max_depth = 6,                 # Profundidad máxima de los árboles
  subsample = 0.8,               # Submuestreo de filas
  colsample_bytree = 0.8         # Submuestreo de columnas
)

# Entreno un modelo básico para evaluar el rendimiento inicial
xgb_model <- xgb.train(
  params = params,
  data = train_matrix,
  nrounds = 100,                      # Número de iteraciones
  watchlist = list(train = train_matrix, test = test_matrix), 
  verbose = 1                         # Mostrar progreso del entrenamiento
)


# =========================
# EVALUACION MODELO INICIAL
# =========================

# Predicciones para el conjunto de prueba
predictions <- predict(xgb_model, test_matrix)

# Convertir probabilidades a etiquetas
predicted_labels <- ifelse(predictions > 0.5, "Pass", "Fail")

# Confusion Matrix
confusion_matrix <- table(test_data$Pass.Fail, predicted_labels)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)

cat("Confusion Matrix:\n")
print(confusion_matrix)
cat("\nAccuracy:", round(accuracy, 2), "\n")

# =======================
# CALCULAR LA CURVA (ROC)
# =======================

# Calcular la curva ROC
roc_curve <- roc(as.numeric(test_data$Pass.Fail) - 1, predictions)
plot(roc_curve, main = "Curva ROC", col = "blue")
cat("Área bajo la curva (AUC):", auc(roc_curve), "\n")

# =======================================
# EXTRAER CARACTERISTICAS MAS IMPORTANTES
# =======================================

# Importancia de características
importance <- xgb.importance(feature_names = colnames(train_data[, -c(1, ncol(train_data))]), model = xgb_model)
print(importance)

# Gráfico de importancia
xgb.plot.importance(importance_matrix = importance, top_n = 10, main = "Top 10 Características Importantes")

# ==============
# GUARDAR MODELO
# ==============

# Guardo el modelo para futuros analisis
if (!dir.exists("data/SECOM_Analysis/models")) {
  dir.create("data/SECOM_Analysis/models", recursive = TRUE)
}
