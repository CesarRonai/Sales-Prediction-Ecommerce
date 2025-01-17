import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Carregar e processar a série temporal (substitua pelo seu dataset real)
df_series = pd.read_csv("olist_orders_dataset.csv", parse_dates=["order_purchase_timestamp"])
df_series.set_index("order_purchase_timestamp", inplace=True)

# Criar a série temporal agregando pedidos por dia
df_series = df_series.resample('D').size().to_frame(name='num_pedidos')

# Converter para agregação mensal
df_series_monthly = df_series.resample('M').sum()

# Visualizar a série temporal mensal
plt.figure(figsize=(12, 6))
plt.plot(df_series_monthly, label='Número de pedidos por mês', marker='o')
plt.title('Número de Pedidos ao Longo do Tempo (Mensal)')
plt.xlabel('Data')
plt.ylabel('Pedidos')
plt.legend()
plt.show()

# Definir parâmetros do ARIMA
p, d, q = 5, 1, 0

# Treinar ARIMA
model_arima = ARIMA(df_series_monthly, order=(p, d, q))
model_fit_arima = model_arima.fit()

# Previsão ARIMA para 12 meses futuros
forecast_arima = model_fit_arima.forecast(steps=12)

# Normalização para LSTM
scaler = MinMaxScaler()
scaled_orders = scaler.fit_transform(df_series_monthly.values.reshape(-1, 1))

# Criar sequências para LSTM
def create_sequences(data, seq_length=12):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 12
X, y = create_sequences(scaled_orders, seq_length)

# Divisão treino e teste
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Criar modelo LSTM
model_lstm = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(seq_length, 1)),
    LSTM(50, activation='relu'),
    Dense(1)
])

model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Previsão LSTM
y_pred_lstm = model_lstm.predict(X_test)

# Transformar previsões de volta para escala original
y_pred_inv = scaler.inverse_transform(y_pred_lstm)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Comparação ARIMA vs LSTM
plt.figure(figsize=(12, 6))
plt.plot(df_series_monthly[-len(y_test_inv):], label='Real', marker='o')
plt.plot(pd.date_range(df_series_monthly.index[-len(y_test_inv)], periods=len(y_test_inv), freq='M'), y_pred_inv, label='LSTM', linestyle='dashed')
plt.plot(pd.date_range(df_series_monthly.index[-len(forecast_arima):], periods=len(forecast_arima), freq='M'), forecast_arima, label='ARIMA', linestyle='dotted')
plt.title('Comparação ARIMA vs LSTM')
plt.xlabel('Data')
plt.ylabel('Pedidos')
plt.legend()
plt.show()

# Avaliação dos Modelos
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae_arima = mean_absolute_error(df_series_monthly[-12:], forecast_arima)
rmse_arima = np.sqrt(mean_squared_error(df_series_monthly[-12:], forecast_arima))

mae_lstm = mean_absolute_error(y_test_inv, y_pred_inv)
rmse_lstm = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))

print(f"ARIMA - MAE: {mae_arima:.2f}, RMSE: {rmse_arima:.2f}")
print(f"LSTM - MAE: {mae_lstm:.2f}, RMSE: {rmse_lstm:.2f}")
