import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
from sklearn.preprocessing import LabelEncoder
# import numpy as np

df_data = pd.read_csv("dados.csv", sep=";")

features = ["Tipo", "Mês", "Dia da Semana", "Hora", "Pago"]
targets = ["Número comentários", "like", "Compartilhamento"]

X = df_data[features].copy()  # Fazer uma cópia das features
y = df_data[targets].values

# Codificar a feature "Tipo" com LabelEncoder
label_encoder = LabelEncoder()
X["Tipo_encoded"] = label_encoder.fit_transform(X["Tipo"])

# Remover a feature original "Tipo"
X.drop("Tipo", axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 score: {r2}")

# Salvando os modelos
joblib.dump(model, "./linear_regression_le.pkl")
joblib.dump(label_encoder, "./label_encoder.pkl")
print("Models saved successfully.")
