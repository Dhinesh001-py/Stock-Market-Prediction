import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import requests
from datetime import datetime, timedelta
Fetch S&P 500 stock symbols
def get_sp500_tickers():
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
tables = pd.read_html(url)
return tables[0]["Symbol"].tolist()
Fetch exchange rates
@st.cache_data(ttl=600)
def fetch_exchange_rates(base_currency="USD"):
try:
    response = requests.get(f"https://api.exchangerate.host/latest?base={base_currency}")
    return response.json().get("rates", {})
except:
    return {}
Currency converter
@st.cache_data(ttl=600)
def convert_currency(amount, from_currency, to_currency):
rates = fetch_exchange_rates(from_currency)
return amount * rates.get(to_currency, 1)
Generate future trading dates (skipping weekends)
def get_future_trading_dates(n_days):
dates = []
current_date = datetime.now()
while len(dates) < n_days:
    current_date += timedelta(days=1)
    if current_date.weekday() < 5:  # Monday–Friday are 0–4
        dates.append(current_date)
return [d.strftime("%Y-%m-%d") for d in dates]
Currency symbols
currency_symbols = {"INR": "₹"}
st.sidebar.title("Stock Market Dashboard")
stocks = get_sp500_tickers()
selected_stock = st.sidebar.selectbox("Select a Stock", stocks)
currency = st.sidebar.selectbox("Currency", list(currency_symbols.keys()))
currency_symbol = currency_symbols.get(currency, "")
base_currency = "USD"
def get_live_price(ticker):
return yf.Ticker(ticker).history(period='1d')['Close'].values[-1]
base_price = get_live_price(selected_stock)
live_price = convert_currency(base_price, base_currency, currency)
st.sidebar.markdown(f"Live Price: <span style='font-weight:bold; font-size:18px;'>{currency_symbol}{live_price:.2f}</span>", unsafe_allow_html=True)
date_range_options = {"1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y","2y": "2y"}
date_range = st.sidebar.selectbox("Date Range", list(date_range_options.keys()))
prediction_days = st.sidebar.slider("Prediction Days", 15, 200, 30)
tabs = st.tabs(["Stock Predictor", "Stock Summary"])
with tabs[0]:
st.title("Stock Market Predictor")
data = yf.Ticker(selected_stock).history(period=date_range_options[date_range])
df = data[['Close']]
if len(df) < 90:
    st.warning("Not enough data. Switching to 6M...")
    data = yf.Ticker(selected_stock).history(period="6mo")
    df = data[['Close']]
st.subheader("Live Stock Graph")
fig_live = go.Figure()
fig_live.add_trace(go.Scatter(x=data.index, y=[convert_currency(v, base_currency, currency) for v in data['Close']], mode='lines', name='Closing Price', line=dict(width=4, color='blue')))
fig_live.update_layout(title=f"Live Prices for {selected_stock}", xaxis_title="Date", yaxis_title=f"Price ({currency_symbol})", template="plotly_white", height=500)
st.plotly_chart(fig_live)
if st.button("Predict"):
    st.write("Training model...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    x_train, y_train = [], []
    for i in range(60, len(scaled_data)):
        x_train.append(scaled_data[i-60:i, 0])
        y_train.append(scaled_data[i, 0])
    if len(x_train) == 0:
        st.error("Not enough data to train.")
    else:
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)),
            Dropout(0.3),
            LSTM(128, return_sequences=False),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dense(1)

        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, epochs=25, batch_size=32, verbose=0)
        input_seq = scaled_data[-60:]
        predicted_prices = []
        current_input = input_seq.reshape(1, 60, 1)
        for _ in range(prediction_days):
            pred = model.predict(current_input, verbose=0)[0][0]
            predicted_prices.append(pred)
            current_input = np.append(current_input[:, 1:, :], [[[pred]]], axis=1)
        predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))
        predicted_prices = [convert_currency(p[0], base_currency, currency) for p in predicted_prices]
        # Generate valid future trading dates
        future_dates = get_future_trading_dates(prediction_days)
        st.subheader("Predicted Prices")
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=future_dates, y=predicted_prices, mode='lines', name='Predicted Price', line=dict(width=4, color='red')))
        fig_pred.update_layout(title="Predicted Prices", xaxis_title="Date", yaxis_title=f"Price ({currency_symbol})", template="plotly_white", height=500)
        st.plotly_chart(fig_pred)
        # Show table of predicted prices with dates
        st.write("### Predicted Prices (Next Trading Days)")
        pred_df = pd.DataFrame({
            "Date": future_dates,
            f"Predicted Price ({currency})": [f"{currency_symbol}{p:.2f}" for p in predicted_prices]
        })
        st.dataframe(pred_df, use_container_width=True)
with tabs[1]:
st.title("Stock Summary")
stock_info = yf.Ticker(selected_stock).info
with st.container():
    col1, col2, col3 = st.columns(3)
    col1.metric("Open", f"{currency_symbol}{convert_currency(stock_info.get('open', 0), base_currency, currency):.2f}")
    col2.metric("Close", f"{currency_symbol}{convert_currency(stock_info.get('previousClose', 0), base_currency, currency):.2f}")
    col3.metric("High", f"{currency_symbol}{convert_currency(stock_info.get('dayHigh', 0), base_currency, currency):.2f}")
    col1.metric("Low", f"{currency_symbol}{convert_currency(stock_info.get('dayLow', 0), base_currency, currency):.2f}")
    col2.metric("Market Cap", f"{currency_symbol}{convert_currency(stock_info.get('marketCap', 0), base_currency, currency):,.2f}")
    col3.metric("P/E Ratio", stock_info.get('trailingPE', 'N/A'))
