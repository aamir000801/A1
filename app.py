import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np

# ============================================
# 1. USER INPUTS (Customize for Any Asset)
# ============================================
st.sidebar.title("Settings")
ticker = st.sidebar.text_input("Enter Ticker (e.g., BTC-USD)", "BTC-USD")
interval = st.sidebar.selectbox("Interval", ["1h", "1d", "5m"])
period = st.sidebar.selectbox("Period", ["7d", "30d", "1y"])

# ============================================
# 2. DATA LOADING & FORMATTING
# ============================================
@st.cache_data
def load_data(ticker, interval, period):
    data = yf.download(ticker, interval=interval, period=period)
    data.columns = [col.lower() for col in data.columns]  # Ensure lowercase
    data.dropna(inplace=True)
    return data

data = load_data(ticker, interval, period)

# ============================================
# 3. AI/ML FEATURES (Add More Below!)
# ============================================
# ----- Trend Prediction (Linear Regression) -----
def predict_trend(data, feature='close', forecast_days=5):
    x = np.arange(len(data)).reshape(-1, 1)
    y = data[feature].values
    model = LinearRegression().fit(x, y)
    future_x = np.arange(len(data), len(data)+forecast_days).reshape(-1, 1)
    future_y = model.predict(future_x)
    return future_y

forecast = predict_trend(data)

# ----- Support/Resistance Detection -----
def find_support_resistance(data, window=5):
    levels = []
    for i in range(window, len(data)-window):
        high = data['high'][i]
        low = data['low'][i]
        # Resistance
        if all(high > data['high'][i-window:i]) and all(high > data['high'][i+1:i+window+1]):
            levels.append(('resistance', high))
        # Support
        if all(low < data['low'][i-window:i]) and all(low < data['low'][i+1:i+window+1]):
            levels.append(('support', low))
    return levels

levels = find_support_resistance(data)

# ============================================
# 4. TRADING SIGNALS (Expandable)
# ============================================
# Moving Average Crossover
data['ma_fast'] = data['close'].rolling(window=5).mean()
data['ma_slow'] = data['close'].rolling(window=20).mean()
data['signal'] = np.where(data['ma_fast'] > data['ma_slow'], 1, -1)

# Generate BUY/SELL markers
signals = data[data['signal'].diff() != 0].index
buy_signals = data[(data['signal'] == 1) & (data['signal'].diff() != 0)]
sell_signals = data[(data['signal'] == -1) & (data['signal'].diff() != 0)]

# ============================================
# 5. INTERACTIVE PLOT (Plotly)
# ============================================
fig = go.Figure()
# Candlesticks
fig.add_trace(go.Candlestick(
    x=data.index, open=data['open'], high=data['high'],
    low=data['low'], close=data['close'], name="Price"
))
# Trend Forecast
fig.add_trace(go.Scatter(
    x=pd.date_range(data.index[-1], periods=5, freq=data.index.freq),
    y=forecast, line=dict(color='purple', dash='dot'), name="AI Forecast"
))
# Support/Resistance
for typ, price in levels:
    fig.add_hline(y=price, line_color="green" if typ == "support" else "red", opacity=0.3)
# Buy/Sell Signals
fig.add_trace(go.Scatter(
    x=buy_signals.index, y=buy_signals['close'],
    mode='markers', marker=dict(color='green', size=10), name='BUY'
))
fig.add_trace(go.Scatter(
    x=sell_signals.index, y=sell_signals['close'],
    mode='markers', marker=dict(color='red', size=10), name='SELL'
))

fig.update_layout(title=f"{ticker} Chart with AI Signals")
st.plotly_chart(fig, use_container_width=True)

# ============================================
# 6. ADDITIONAL ANALYSIS (Customize!)
# ============================================
st.subheader("Technical Summary")
col1, col2 = st.columns(2)
with col1:
    st.metric("Latest Close", f"${data['close'][-1]:.2f}")
    st.write(f"**Support Levels:** {[level[1] for level in levels if level[0]=='support']}")
with col2:
    st.metric("Trend Forecast (5d)", f"${forecast[-1]:.2f}")
    st.write(f"**Resistance Levels:** {[level[1] for level in levels if level[0]=='resistance']}")

# Add more metrics (RSI, Volume, etc.) here!
