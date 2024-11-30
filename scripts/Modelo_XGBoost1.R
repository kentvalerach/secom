# ========================
# Modelo XGBoost Mejorado
# ========================

# Librerías necesarias
library(xgboost)
library(caret)
library(pROC)
library(dplyr)

# =====================
# Cargar Dataset
# =====================
data <- read.csv("data/cleaned_secom.csv")

# Verificar y preparar la variable objetivo
data$Pass.Fail <- as.factor(data$Pass.Fail)

# Dividir datos en entrenamiento y prueba
set.seed(123)
train_index <- createDataPartition(data$Pass.Fail, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Crear matrices para XGBoost
train_matrix <- xgb.DMatrix(
  data = as.matrix(train_data[, -ncol(train_data)]),
  label = as.numeric(train_data$Pass.Fail) - 1
)

test_matrix <- xgb.DMatrix(
  data = as.matrix(test_data[, -ncol(test_data)]),
  label = as.numeric(test_data$Pass.Fail) - 1
)

# =====================
# Configurar Parámetros
# =====================
params <- list(
  objective = "binary:logistic",  # Clasificación binaria
  eval_metric = "auc",           # Usar AUC para evaluar el modelo
  eta = 0.1,                     # Tasa de aprendizaje
  max_depth = 6,                 # Profundidad máxima del árbol
  subsample = 0.8,               # Submuestreo de filas
  colsample_bytree = 0.8         # Submuestreo de columnas
)

# =====================
# Entrenar el Modelo
# =====================
xgb_model <- xgb.train(
  params = params,
  data = train_matrix,
  nrounds = 100,
  watchlist = list(train = train_matrix, test = test_matrix),
  verbose = 1
)

# ===========================
# Evaluar el Modelo
# ===========================
# Predicciones con probabilidades
predictions <- predict(xgb_model, test_matrix)

# Ajustar el umbral
umbral <- 0.3  # Reducir el umbral para capturar más instancias positivas
predicted_labels <- ifelse(predictions > umbral, "Pass", "Fail")
predicted_labels <- factor(predicted_labels, levels = c("Fail", "Pass"))

# Crear matriz de confusión
confusion_matrix <- table(test_data$Pass.Fail, predicted_labels)
cat("Matriz de Confusión:\n")
print(confusion_matrix)

# ========================
# Calcular Sensibilidad
# ========================
if (all(c("Pass", "Fail") %in% colnames(confusion_matrix))) {
  tp <- confusion_matrix["Pass", "Pass"]
  fn <- confusion_matrix["Pass", "Fail"]
  sensitivity <- tp / (tp + fn)
  cat("Sensibilidad (Recall):", round(sensitivity, 2), "\n")
} else {
  cat("El modelo no predijo ambas clases. Revisa el umbral o el balance de datos.\n")
}

# ====================
# Métricas Adicionales
# ====================
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
cat("Precisión (Accuracy):", round(accuracy, 2), "\n")

# ========================
# Calcular Curva ROC y AUC
# ========================
roc_curve <- roc(as.numeric(test_data$Pass.Fail) - 1, predictions)
cat("Área bajo la curva (AUC):", auc(roc_curve), "\n")
plot(roc_curve, main = "Curva ROC", col = "blue")

# =============================
# Visualizar Importancia de las Variables
# =============================
importance <- xgb.importance(feature_names = colnames(train_data[, -ncol(train_data)]), model = xgb_model)
cat("Importancia de Características:\n")
print(importance)
xgb.plot.importance(importance, top_n = 10, main = "Top 10 Características Importantes")

# ==================
# Guardar el Modelo
# ==================
if (!dir.exists("models")) dir.create("models")
xgb.save(xgb_model, "models/xgb_model.bin")

cat("Modelo guardado exitosamente en 'models/xgb_model.bin'.\n")
