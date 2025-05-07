import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np

# ============================================
# 1. APP CONFIGURATION
# ============================================
st.set_page_config(page_title="AI Trading Assistant", layout="wide")
st.title("AI Trading Assistant - Multi-Asset Analysis")

# ============================================
# 2. USER INPUTS (Sidebar)
# ============================================
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Enter Ticker Symbol", "BTC-USD")
interval = st.sidebar.selectbox("Time Interval", ["1h", "1d", "5m", "15m", "30m"])
period = st.sidebar.selectbox("Period", ["7d", "30d", "60d", "1y"])

# ============================================
# 3. DATA LOADING WITH ERROR HANDLING
# ============================================
@st.cache_data
def load_data(ticker, interval, period):
    try:
        data = yf.download(ticker, interval=interval, period=period)
        if data.empty:
            st.error("⚠️ No data downloaded! Check ticker symbol and parameters.")
            return pd.DataFrame()
            
        # Clean and format data
        data.columns = [col.lower() for col in data.columns]
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        return data[-500:]  # Limit to last 500 periods for stability
        
    except Exception as e:
        st.error(f"🚨 Data download failed: {str(e)}")
        return pd.DataFrame()

data = load_data(ticker, interval, period)
if data.empty:
    st.stop()

# ============================================
# 4. AI/ML FEATURES (Robust Implementation)
# ============================================
def predict_trend(data, feature='close', forecast_days=5):
    """Safe trend prediction with error handling"""
    if len(data) < 10:  # Minimum data check
        st.warning("⚠️ Insufficient data for trend prediction")
        return np.array([])
    
    try:
        x = np.arange(len(data)).reshape(-1, 1)
        y = data[feature].values
        model = LinearRegression().fit(x, y)
        future_x = np.arange(len(data), len(data)+forecast_days).reshape(-1, 1)
        return model.predict(future_x)
    except Exception as e:
        st.warning(f"⚠️ Trend prediction skipped: {str(e)}")
        return np.array([])

forecast = predict_trend(data)

def find_support_resistance(data, window=5):
    """Safe support/resistance detection"""
    levels = []
    if len(data) < 2*window:
        return levels
        
    try:
        for i in range(window, len(data)-window):
            high = data['high'].iloc[i]
            low = data['low'].iloc[i]
            
            # Check resistance
            if (high > data['high'].iloc[i-window:i]).all() and \
               (high > data['high'].iloc[i+1:i+window+1]).all():
                levels.append(('resistance', high))
                
            # Check support
            if (low < data['low'].iloc[i-window:i]).all() and \
               (low < data['low'].iloc[i+1:i+window+1]).all():
                levels.append(('support', low))
        return levels
    except:
        return levels

levels = find_support_resistance(data)

# ============================================
# 5. TRADING SIGNALS (Vectorized Implementation)
# ============================================
data['ma_fast'] = data['close'].rolling(window=5, min_periods=1).mean()
data['ma_slow'] = data['close'].rolling(window=20, min_periods=1).mean()
data['signal'] = np.where(data['ma_fast'] > data['ma_slow'], 1, -1)

# Generate signals safely
signals = data[data['signal'].diff() != 0]
buy_signals = signals[signals['signal'] == 1]
sell_signals = signals[signals['signal'] == -1]

# ============================================
# 6. INTERACTIVE VISUALIZATION
# ============================================
fig = go.Figure()

# Candlestick Chart
fig.add_trace(go.Candlestick(
    x=data.index,
    open=data['open'],
    high=data['high'],
    low=data['low'],
    close=data['close'],
    name='Price'
))

# Trend Forecast
if len(forecast) > 0:
    forecast_dates = pd.date_range(
        start=data.index[-1],
        periods=len(forecast)+1,
        freq=data.index.freq
    )[1:]
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast,
        line=dict(color='purple', dash='dot'),
        name='AI Forecast'
    ))

# Support/Resistance Levels
for typ, price in levels:
    fig.add_hline(
        y=price,
        line_color="green" if typ == "support" else "red",
        opacity=0.3,
        annotation_text=f"{typ.capitalize()} Level"
    )

# Buy/Sell Signals
if not buy_signals.empty:
    fig.add_trace(go.Scatter(
        x=buy_signals.index,
        y=buy_signals['close'],
        mode='markers+text',
        marker=dict(color='green', size=12),
        text='BUY',
        textposition='top center',
        name='BUY Signals'
    ))

if not sell_signals.empty:
    fig.add_trace(go.Scatter(
        x=sell_signals.index,
        y=sell_signals['close'],
        mode='markers+text',
        marker=dict(color='red', size=12),
        text='SELL', 
        textposition='top center',
        name='SELL Signals'
    ))

fig.update_layout(
    title=f"{ticker} Analysis",
    xaxis_rangeslider_visible=False,
    height=800
)

st.plotly_chart(fig, use_container_width=True)

# ============================================
# 7. ANALYSIS SUMMARY
# ============================================
st.subheader("Market Analysis Summary")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Current Price", f"${data['close'].iloc[-1]:.2f}")
    st.metric("24h Change", f"{data['close'].pct_change().iloc[-1]*100:.2f}%")

with col2:
    st.metric("Support Levels", ", ".join([f"${x[1]:.2f}" for x in levels if x[0]=='support'][-3:]) or "None")
    st.metric("Resistance Levels", ", ".join([f"${x[1]:.2f}" for x in levels if x[0]=='resistance'][-3:]) or "None")

with col3:
    st.metric("Trend Forecast", 
             f"${forecast[-1]:.2f}" if len(forecast) > 0 else "N/A",
             delta=f"{(forecast[-1]/data['close'].iloc[-1]-1)*100:.2f}%" if len(forecast) > 0 else None)

st.write("---")
st.write("**Disclaimer:** This is an educational tool, not financial advice. Always do your own research.")
