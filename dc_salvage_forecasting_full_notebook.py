
# DC Salvage Forecasting - ML Comparison Notebook

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, mean_absolute_error
from prophet import Prophet
import statsmodels.api as sm
from xgboost import XGBRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv('dc_salvage_data.csv')  # Replace with actual file
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# Aggregate weekly
weekly_df = df.groupby('date')['salvage_qty'].sum().reset_index()
weekly_df.columns = ['ds', 'y']

# --------------------- Prophet ---------------------
prophet_model = Prophet()
prophet_model.fit(weekly_df)
future = prophet_model.make_future_dataframe(periods=12, freq='W')
forecast_prophet = prophet_model.predict(future)
y_true = weekly_df['y'][-12:]
y_pred = forecast_prophet['yhat'][-12:]
rmse_prophet = np.sqrt(mean_squared_error(y_true, y_pred))

# --------------------- SARIMA ---------------------
sarima_model = sm.tsa.SARIMAX(weekly_df['y'], order=(1,1,1), seasonal_order=(1,1,1,52))
sarima_result = sarima_model.fit(disp=False)
forecast_sarima = sarima_result.forecast(12)
rmse_sarima = np.sqrt(mean_squared_error(y_true, forecast_sarima))

# --------------------- XGBoost ---------------------
weekly_df['weekofyear'] = weekly_df['ds'].dt.isocalendar().week
weekly_df['year'] = weekly_df['ds'].dt.year

X = weekly_df[['weekofyear', 'year']]
y = weekly_df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

xgb_model = XGBRegressor()
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))

# --------------------- LSTM ---------------------
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(weekly_df[['y']])

X_lstm, y_lstm = [], []
for i in range(10, len(scaled_data)):
    X_lstm.append(scaled_data[i-10:i, 0])
    y_lstm.append(scaled_data[i, 0])
X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))

model_lstm = Sequential()
model_lstm.add(LSTM(50, return_sequences=False, input_shape=(X_lstm.shape[1], 1)))
model_lstm.add(Dense(1))
model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.fit(X_lstm, y_lstm, epochs=10, batch_size=1, verbose=0)

y_pred_lstm = model_lstm.predict(X_lstm[-12:])
y_pred_lstm = scaler.inverse_transform(y_pred_lstm)
y_true_lstm = scaler.inverse_transform(scaled_data[-12:])
rmse_lstm = np.sqrt(mean_squared_error(y_true_lstm, y_pred_lstm))

# --------------------- Comparison ---------------------
print("Model Comparison (RMSE):")
print(f"Prophet: {rmse_prophet:.2f}")
print(f"SARIMA: {rmse_sarima:.2f}")
print(f"XGBoost: {rmse_xgb:.2f}")
print(f"LSTM: {rmse_lstm:.2f}")

# Notes:
# - Use Prophet for quick interpretable forecasts
# - Use SARIMA for small clean datasets with clear seasonality
# - Use XGBoost for complex feature-rich data
# - Use LSTM when long-term patterns and deep learning infra is available
