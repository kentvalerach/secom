library(xgboost)
library(data.table)
library(caret)
library(pROC)
library(ggplot2)

# Cargar datos
file_path <- "data/cleaned_secom.csv"
data <- fread(file_path)

# Separar X e y
X <- data[, !"Pass.Fail"]
y <- ifelse(data$Pass.Fail == "Pass", 1, 0)

# Convertir X a numérico
X <- as.data.frame(X)
X <- X[, sapply(X, is.numeric)]

# Dividir datos
set.seed(42)
train_index <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_index, ]
X_test <- X[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index]

# Submuestreo manual (alternativa a ROSE)
minority_class <- data.frame(X_train, Class = y_train) %>% filter(Class == 1)
majority_class <- data.frame(X_train, Class = y_train) %>% filter(Class == 0) %>% sample_n(nrow(minority_class))
balanced_data <- rbind(minority_class, majority_class)

# Crear DMatrix
X_train <- as.matrix(balanced_data[, -ncol(balanced_data)])
y_train <- balanced_data$Class
dtrain <- xgb.DMatrix(data = X_train, label = y_train)
dtest <- xgb.DMatrix(data = as.matrix(X_test), label = y_test)

# Parámetros ajustados
params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  eta = 0.1,
  max_depth = 4,
  min_child_weight = 10
)

# Entrenar modelo
set.seed(42)
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,
  watchlist = list(train = dtrain, test = dtest),
  early_stopping_rounds = 10,
  print_every_n = 10
)

# Predicciones
y_pred <- predict(xgb_model, dtest)
y_pred_class <- ifelse(y_pred > 0.5, 1, 0)

# Métricas
conf_matrix <- confusionMatrix(
  factor(y_pred_class, levels = c(0, 1)),
  factor(y_test, levels = c(0, 1))
)
sensitivity <- conf_matrix$byClass["Sensitivity"]
specificity <- conf_matrix$byClass["Specificity"]
precision <- conf_matrix$byClass["Precision"]

cat("Sensibilidad (Recall):", sensitivity, "\n")
cat("Especificidad:", specificity, "\n")
cat("Precisión:", precision, "\n")

# AUC
roc_obj <- roc(y_test, y_pred)
auc_value <- auc(roc_obj)
cat("Área bajo la curva (AUC):", auc_value, "\n")

# Obtener las 10 características más importantes
importance <- xgb.importance(feature_names = colnames(X_train), model = xgb_model)
top_10_features <- importance[1:10, ]

# Crear el gráfico
ggplot(top_10_features, aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  coord_flip() +
  labs(
    title = "Top 10 características más importantes",
    x = "Características",
    y = "Importancia (Gain)"
  ) +
  theme_minimal()
