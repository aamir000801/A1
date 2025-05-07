import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np

# ============================================
# 1. APP CONFIGURATION
# ============================================
st.set_page_config(page_title="AI Trading Pro", layout="wide")
st.title("📈 AI Trading Assistant - Multi-Asset Analysis")

# ============================================
# 2. USER INPUTS (Sidebar with Validation)
# ============================================
st.sidebar.header("⚙️ Settings")
ticker = st.sidebar.text_input("Ticker Symbol (e.g., BTC-USD, AAPL)", "BTC-USD")
interval = st.sidebar.selectbox("Time Frame", ["1h", "1d", "5m", "15m", "30m", "1m"])
period = st.sidebar.selectbox("History Period", ["7d", "30d", "60d", "1y", "2y"])

# ============================================
# 3. ROBUST DATA LOADING SYSTEM
# ============================================
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data(ticker, interval, period):
    try:
        # Validate interval-period combination
        if interval in ["1m", "5m", "15m", "30m"] and any(y in period for y in ["y", "2y"]):
            st.error(f"🚫 Minute data not available for {period} period. Max 60 days for {interval} intervals.")
            return pd.DataFrame()

        data = yf.download(
            tickers=ticker,
            interval=interval,
            period=period,
            progress=False,
            auto_adjust=True
        )
        
        if data.empty:
            st.error(f"""
            🔍 No data found for **{ticker}**!
            Common solutions:
            1. Check ticker format (Examples: BTC-USD, AAPL, EURUSD=X)
            2. Try shorter period for minute intervals
            3. Verify on [Yahoo Finance](https://finance.yahoo.com/)
            """)
            return pd.DataFrame()

        # Clean and format data
        data.columns = [col.lower() for col in data.columns]
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.dropna()
        return data.iloc[-500:]  # Limit to last 500 periods

    except Exception as e:
        st.error(f"""
        🚨 Critical Error: {str(e)}
        Possible fixes:
        - Check internet connection
        - Try different ticker/parameters
        - Wait 1 minute (API rate limit)
        """)
        return pd.DataFrame()

data = load_data(ticker, interval, period)
if data.empty:
    st.stop()

# ============================================
# 4. FAIL-SAFE ANALYTICS ENGINE
# ============================================
def predict_trend(data, feature='close', forecast_days=5):
    """Safe trend prediction with validation"""
    if len(data) < 10:
        st.warning("📉 Insufficient data for trend prediction")
        return np.array([])
    
    try:
        x = np.arange(len(data)).reshape(-1, 1)
        y = data[feature].values
        model = LinearRegression().fit(x, y)
        future_x = np.arange(len(data), len(data)+forecast_days).reshape(-1, 1)
        return model.predict(future_x)
    except Exception as e:
        st.warning(f"⚠️ Trend prediction failed: {str(e)}")
        return np.array([])

forecast = predict_trend(data)

def find_support_resistance(data, window=5):
    """Robust level detection"""
    levels = []
    if len(data) < 2 * window:
        return levels
    
    try:
        for i in range(window, len(data)-window):
            high = data['high'].iloc[i]
            low = data['low'].iloc[i]
            
            # Resistance check
            resist_condition = (
                (high > data['high'].iloc[i-window:i]).all() and 
                (high > data['high'].iloc[i+1:i+window+1]).all()
            )
            
            # Support check
            support_condition = (
                (low < data['low'].iloc[i-window:i]).all() and 
                (low < data['low'].iloc[i+1:i+window+1]).all()
            )
            
            if resist_condition:
                levels.append(('resistance', high))
            if support_condition:
                levels.append(('support', low))
        return levels[-10:]  # Return last 10 levels
    except:
        return levels

levels = find_support_resistance(data)

# ============================================
# 5. SMART TRADING SIGNALS
# ============================================
data['ma_fast'] = data['close'].rolling(window=5, min_periods=1).mean()
data['ma_slow'] = data['close'].rolling(window=20, min_periods=1).mean()
data['signal'] = np.where(data['ma_fast'] > data['ma_slow'], 1, -1)

# Generate signals with edge detection
signals = data[data['signal'].diff() != 0]
buy_signals = signals[signals['signal'] == 1]
sell_signals = signals[signals['signal'] == -1]

# ============================================
# 6. PROFESSIONAL CHARTING
# ============================================
fig = go.Figure()

# Candlestick Base
fig.add_trace(go.Candlestick(
    x=data.index,
    open=data['open'],
    high=data['high'],
    low=data['low'],
    close=data['close'],
    name='Price Action'
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
        line=dict(color='#636EFA', dash='dot'),
        name='AI Forecast'
    ))

# Support/Resistance Levels
for typ, price in levels:
    fig.add_hline(
        y=price,
        line_color="#00CC96" if typ == "support" else "#EF553B",
        opacity=0.5,
        annotation_text=f"{typ.title()} Zone",
        annotation_position="right"
    )

# Trading Signals
if not buy_signals.empty:
    fig.add_trace(go.Scatter(
        x=buy_signals.index,
        y=buy_signals['close'],
        mode='markers+text',
        marker=dict(color='#00CC96', size=12, symbol='triangle-up'),
        text='BUY',
        textposition='top center',
        name='Buy Signals'
    ))

if not sell_signals.empty:
    fig.add_trace(go.Scatter(
        x=sell_signals.index,
        y=sell_signals['close'],
        mode='markers+text',
        marker=dict(color='#EF553B', size=12, symbol='triangle-down'),
        text='SELL',
        textposition='bottom center',
        name='Sell Signals'
    ))

fig.update_layout(
    title=f"{ticker} Advanced Analysis",
    xaxis_rangeslider_visible=False,
    height=800,
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)

# ============================================
# 7. MARKET INTELLIGENCE DASHBOARD
# ============================================
st.subheader("📊 Market Intelligence")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Current Price", f"${data['close'].iloc[-1]:.2f}")
    st.metric("24h Volatility", f"{data['high'].iloc[-1]/data['low'].iloc[-1]-1:.2%}")

with col2:
    st.metric("Key Support", 
             f"${min([x[1] for x in levels if x[0]=='support'], default='N/A')}" if levels else "N/A")
    st.metric("Key Resistance", 
             f"${max([x[1] for x in levels if x[0]=='resistance'], default='N/A')}" if levels else "N/A")

with col3:
    forecast_status = (
        f"${forecast[-1]:.2f} ▲" if len(forecast) > 0 and forecast[-1] > data['close'].iloc[-1]
        else f"${forecast[-1]:.2f} ▼" if len(forecast) > 0 
        else "N/A"
    )
    st.metric("5-Period Forecast", forecast_status)

st.write("---")
st.write("⚠️ **Disclaimer:** This tool provides educational insights only. Past performance ≠ future results.")

# ============================================
# 8. REQUIRED DEPENDENCIES (requirements.txt)
# ============================================
"""
requirements.txt content:
yfinance==0.2.37
streamlit==1.32.0
pandas==2.2.1
plotly==5.18.0
scikit-learn==1.4.1.post1
numpy==1.26.4
"""
