
import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from xgboost import XGBRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import plotly.express as px

st.title("📦 Enhanced DC Salvage Forecasting Dashboard")

# Load data
df = pd.read_csv('dc_salvage_data_with_category.csv')
df['date'] = pd.to_datetime(df['date'])

# Sidebar filters
dc_selected = st.sidebar.selectbox("Select Distribution Center", options=df['dc_id'].unique())
category_selected = st.sidebar.selectbox("Select Product Category", options=df['product_category'].unique())
date_range = st.sidebar.date_input("Select Date Range", [df['date'].min(), df['date'].max()])

# Filter data
filtered_df = df[
    (df['dc_id'] == dc_selected) &
    (df['product_category'] == category_selected) &
    (df['date'] >= pd.to_datetime(date_range[0])) &
    (df['date'] <= pd.to_datetime(date_range[1]))
]

# Aggregate weekly
weekly_df = filtered_df.groupby('date')['salvage_qty'].sum().reset_index()
weekly_df.columns = ['ds', 'y']

# Plot raw salvage trend
st.subheader(f"Salvage Quantity Trend for {dc_selected} - {category_selected}")
fig1 = px.line(weekly_df, x='ds', y='y', labels={'ds': 'Date', 'y': 'Salvage Quantity'})
st.plotly_chart(fig1)

# Model selection
model_choice = st.selectbox("Choose Forecasting Model", ['Prophet', 'XGBoost', 'LSTM'])

if model_choice == 'Prophet':
    model = Prophet()
    model.fit(weekly_df)
    future = model.make_future_dataframe(periods=12, freq='W')
    forecast = model.predict(future)
    forecasted = forecast[['ds', 'yhat']].tail(12)
    y_true = weekly_df['y'][-12:] if len(weekly_df) >= 24 else weekly_df['y']
    y_pred = forecast['yhat'][-12:] if len(weekly_df) >= 24 else forecast['yhat']
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
elif model_choice == 'XGBoost':
    if len(weekly_df) < 24:
        st.warning("Not enough data for XGBoost.")
    else:
        weekly_df['weekofyear'] = weekly_df['ds'].dt.isocalendar().week
        weekly_df['year'] = weekly_df['ds'].dt.year
        X = weekly_df[['weekofyear', 'year']]
        y = weekly_df['y']
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
        model = XGBRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        forecasted = pd.DataFrame({'ds': weekly_df['ds'].iloc[-len(y_pred):], 'yhat': y_pred})
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
elif model_choice == 'LSTM':
    if len(weekly_df) < 24:
        st.warning("Not enough data for LSTM.")
    else:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(weekly_df[['y']])
        X_lstm, y_lstm = [], []
        for i in range(10, len(scaled_data)):
            X_lstm.append(scaled_data[i-10:i, 0])
            y_lstm.append(scaled_data[i, 0])
        X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
        X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))
        model = Sequential()
        model.add(LSTM(50, return_sequences=False, input_shape=(X_lstm.shape[1], 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_lstm, y_lstm, epochs=10, batch_size=1, verbose=0)
        y_pred = model.predict(X_lstm[-12:])
        y_pred = scaler.inverse_transform(y_pred)
        forecasted = pd.DataFrame({'ds': weekly_df['ds'].iloc[-12:], 'yhat': y_pred.flatten()})
        y_true = weekly_df['y'].iloc[-12:]
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

# Display forecast and RMSE
if model_choice in ['Prophet', 'XGBoost', 'LSTM'] and 'forecasted' in locals():
    st.subheader(f"📈 Forecast Results - {model_choice}")
    fig2 = px.line(forecasted, x='ds', y='yhat', labels={'ds': 'Date', 'yhat': 'Forecast'})
    st.plotly_chart(fig2)
    st.metric(label="RMSE", value=f"{rmse:.2f}")
    st.dataframe(forecasted.rename(columns={'ds': 'Date', 'yhat': 'Forecast'}))
