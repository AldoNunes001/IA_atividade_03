import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from sklearn.preprocessing import LabelEncoder

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

# Treinar o modelo com todos os dados
model = LinearRegression()
model.fit(X, y)

# Fazer previsões com os mesmos dados usados para treino
y_pred = model.predict(X)

mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 score: {r2}")

# Salvando o modelo e o encoder
joblib.dump(model, "./linear_regression.pkl")
joblib.dump(label_encoder, "./label_encoder.pkl")
print("Models saved successfully.")
