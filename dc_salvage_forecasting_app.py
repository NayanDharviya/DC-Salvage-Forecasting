
import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet

# Load data
df = pd.read_csv('dc_salvage_data.csv')
df['date'] = pd.to_datetime(df['date'])

st.title("📦 DC Salvage Forecasting Dashboard - Target Corp")

# Sidebar filters
dc_selected = st.sidebar.selectbox("Select Distribution Center", options=df['dc_id'].unique())

# Filter data
filtered_df = df[df['dc_id'] == dc_selected]
weekly_df = filtered_df.groupby('date')['salvage_qty'].sum().reset_index()
weekly_df.columns = ['ds', 'y']

# Plot raw salvage trend
st.subheader(f"Salvage Quantity Trend for {dc_selected}")
fig1 = px.line(weekly_df, x='ds', y='y', labels={'ds': 'Date', 'y': 'Salvage Quantity'})
st.plotly_chart(fig1)

# Forecast
st.subheader("📈 Forecasting with Prophet")
model = Prophet()
model.fit(weekly_df)
future = model.make_future_dataframe(periods=12, freq='W')
forecast = model.predict(future)

fig2 = px.line(forecast, x='ds', y='yhat', labels={'ds': 'Date', 'yhat': 'Forecasted Salvage'})
fig2.add_scatter(x=weekly_df['ds'], y=weekly_df['y'], mode='markers', name='Actual')
st.plotly_chart(fig2)

# Show forecast table
st.subheader("Forecast Table")
st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12).rename(columns={
    'ds': 'Date',
    'yhat': 'Forecast',
    'yhat_lower': 'Lower Bound',
    'yhat_upper': 'Upper Bound'
}))
