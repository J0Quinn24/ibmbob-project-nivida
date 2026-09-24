# -*- coding: utf-8 -*-
# =============================================================================
# NVIDIA STOCK PRICE PREDICTION
# Single-file project — Backend (Data, Features, LSTM Model) +
#                        Frontend (Streamlit Dashboard)
# File   : Student_NvidiaStockPrediction.py
# Run    : streamlit run Student_NvidiaStockPrediction.py
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NVIDIA Stock Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #76b900;
        text-align: center;
        padding: 1rem 0 0.3rem 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #888;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #76b900;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #76b900;
        border-bottom: 1px solid #333;
        padding-bottom: 4px;
        margin: 1rem 0 0.6rem 0;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e1e2e;
        border-radius: 6px 6px 0 0;
        padding: 6px 18px;
        color: #ccc;
    }
    .stTabs [aria-selected="true"] {
        background-color: #76b900 !important;
        color: #000 !important;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH   = "Nvidia_stock_data.csv"
MODEL_PATH  = "nvidia_lstm_model.keras"
SEQ_LEN     = 60        # look-back window (days)
FEATURES    = ["Close", "High", "Low", "Open", "Volume",
               "MA_7", "MA_21", "MA_50",
               "EMA_12", "EMA_26",
               "RSI", "MACD", "Signal_Line",
               "BB_Upper", "BB_Lower", "BB_Mid",
               "Volatility", "Price_Change", "Volume_Change"]

# ─────────────────────────────────────────────────────────────────────────────
# BACKEND — DATA LOADING & FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_engineer(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # --- Moving Averages ---
    df["MA_7"]  = df["Close"].rolling(7).mean()
    df["MA_21"] = df["Close"].rolling(21).mean()
    df["MA_50"] = df["Close"].rolling(50).mean()

    # --- Exponential Moving Averages ---
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()

    # --- MACD ---
    df["MACD"]        = df["EMA_12"] - df["EMA_26"]
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # --- RSI ---
    delta    = df["Close"].diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs       = avg_gain / (avg_loss + 1e-10)
    df["RSI"] = 100 - (100 / (1 + rs))

    # --- Bollinger Bands ---
    df["BB_Mid"]   = df["Close"].rolling(20).mean()
    bb_std         = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["BB_Mid"] + 2 * bb_std
    df["BB_Lower"] = df["BB_Mid"] - 2 * bb_std

    # --- Volatility & Returns ---
    df["Volatility"]    = df["Close"].rolling(20).std()
    df["Price_Change"]  = df["Close"].pct_change()
    df["Volume_Change"] = df["Volume"].pct_change()

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# BACKEND — SEQUENCE BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def build_sequences(scaled: np.ndarray, seq_len: int):
    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i - seq_len:i])
        y.append(scaled[i, 0])          # index 0 = Close (after scaling)
    return np.array(X), np.array(y)


# ─────────────────────────────────────────────────────────────────────────────
# BACKEND — MODEL BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def build_lstm_model(input_shape: tuple) -> Sequential:
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
        Dropout(0.25),
        LSTM(64, return_sequences=True),
        Dropout(0.25),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="huber")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# BACKEND — TRAIN / LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────
def train_model(df: pd.DataFrame, epochs: int = 40, batch_size: int = 64):
    feat_data = df[FEATURES].values
    scaler    = MinMaxScaler(feature_range=(0, 1))
    scaled    = scaler.fit_transform(feat_data)

    split     = int(len(scaled) * 0.80)
    train_sc  = scaled[:split]
    test_sc   = scaled[split:]

    X_train, y_train = build_sequences(train_sc, SEQ_LEN)
    X_test,  y_test  = build_sequences(test_sc,  SEQ_LEN)

    model = build_lstm_model((SEQ_LEN, len(FEATURES)))

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
    ]

    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )

    model.save(MODEL_PATH)

    # ── Inverse-transform predictions ──
    # We need full-feature dummy array to invert only the Close column
    def inverse_close(y_scaled):
        dummy = np.zeros((len(y_scaled), len(FEATURES)))
        dummy[:, 0] = y_scaled
        return scaler.inverse_transform(dummy)[:, 0]

    train_pred = inverse_close(model.predict(X_train, verbose=0).flatten())
    test_pred  = inverse_close(model.predict(X_test,  verbose=0).flatten())
    train_true = inverse_close(y_train)
    test_true  = inverse_close(y_test)

    metrics = {
        "train_mae":  mean_absolute_error(train_true, train_pred),
        "test_mae":   mean_absolute_error(test_true,  test_pred),
        "train_rmse": np.sqrt(mean_squared_error(train_true, train_pred)),
        "test_rmse":  np.sqrt(mean_squared_error(test_true,  test_pred)),
        "train_r2":   r2_score(train_true, train_pred),
        "test_r2":    r2_score(test_true,  test_pred),
        "test_mape":  np.mean(np.abs((test_true - test_pred) / (test_true + 1e-10))) * 100,
    }

    # Align with original dataframe index
    train_idx = df.index[SEQ_LEN : split]
    test_idx  = df.index[split + SEQ_LEN :]

    return model, scaler, history, metrics, train_idx, test_idx, train_pred, test_pred, train_true, test_true


# ─────────────────────────────────────────────────────────────────────────────
# BACKEND — FUTURE FORECAST
# ─────────────────────────────────────────────────────────────────────────────
def forecast_future(model, scaler, df: pd.DataFrame, n_days: int = 30):
    feat_data   = df[FEATURES].values
    scaled_full = scaler.transform(feat_data)
    last_seq    = scaled_full[-SEQ_LEN:].copy()

    preds_scaled = []
    seq = last_seq.copy()

    for _ in range(n_days):
        x_in  = seq.reshape(1, SEQ_LEN, len(FEATURES))
        p     = model.predict(x_in, verbose=0)[0, 0]
        preds_scaled.append(p)
        # roll window: shift by 1 and set Close of new step
        new_row       = seq[-1].copy()
        new_row[0]    = p
        seq           = np.vstack([seq[1:], new_row])

    dummy = np.zeros((n_days, len(FEATURES)))
    dummy[:, 0] = preds_scaled
    future_close = scaler.inverse_transform(dummy)[:, 0]

    last_date    = df["Date"].iloc[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=n_days)
    return pd.DataFrame({"Date": future_dates, "Predicted_Close": future_close})


# ─────────────────────────────────────────────────────────────────────────────
# FRONTEND — HELPER PLOTS
# ─────────────────────────────────────────────────────────────────────────────
NVIDIA_GREEN = "#76b900"
NVIDIA_BLUE  = "#00a3e0"
PLOT_BG      = "#0e1117"
PAPER_BG     = "#0e1117"
GRID_COLOR   = "#2a2a3a"
FONT_COLOR   = "#e0e0e0"

LAYOUT_DEFAULTS = dict(
    plot_bgcolor=PLOT_BG,
    paper_bgcolor=PAPER_BG,
    font=dict(color=FONT_COLOR, size=12),
    xaxis=dict(gridcolor=GRID_COLOR, showgrid=True),
    yaxis=dict(gridcolor=GRID_COLOR, showgrid=True),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID_COLOR, borderwidth=1),
    margin=dict(l=50, r=30, t=50, b=50),
)


def fig_candle(df: pd.DataFrame, n: int = 180) -> go.Figure:
    sub = df.tail(n)
    fig = go.Figure(go.Candlestick(
        x=sub["Date"], open=sub["Open"], high=sub["High"],
        low=sub["Low"],  close=sub["Close"],
        increasing_line_color=NVIDIA_GREEN,
        decreasing_line_color="#e05252",
        name="OHLC",
    ))
    fig.add_trace(go.Scatter(
        x=sub["Date"], y=sub["MA_21"],
        line=dict(color=NVIDIA_BLUE, width=1.2), name="MA 21"))
    fig.add_trace(go.Scatter(
        x=sub["Date"], y=sub["MA_50"],
        line=dict(color="#f0a500", width=1.2), name="MA 50"))
    fig.update_layout(title=f"Candlestick — Last {n} Trading Days",
                      xaxis_rangeslider_visible=False, **LAYOUT_DEFAULTS)
    return fig


def fig_volume(df: pd.DataFrame, n: int = 180) -> go.Figure:
    sub   = df.tail(n)
    color = [NVIDIA_GREEN if c >= o else "#e05252"
             for c, o in zip(sub["Close"], sub["Open"])]
    fig = go.Figure(go.Bar(x=sub["Date"], y=sub["Volume"],
                           marker_color=color, name="Volume"))
    fig.update_layout(title="Trading Volume", **LAYOUT_DEFAULTS)
    return fig


def fig_indicators(df: pd.DataFrame, n: int = 365) -> go.Figure:
    sub = df.tail(n)
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=("Close + Bollinger Bands", "RSI (14)", "MACD"),
        vertical_spacing=0.06,
    )
    # BB
    fig.add_trace(go.Scatter(x=sub["Date"], y=sub["BB_Upper"],
                             line=dict(color="rgba(118,185,0,0.3)", width=1),
                             name="BB Upper", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=sub["Date"], y=sub["BB_Lower"],
                             line=dict(color="rgba(118,185,0,0.3)", width=1),
                             fill="tonexty", fillcolor="rgba(118,185,0,0.06)",
                             name="BB Band", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=sub["Date"], y=sub["Close"],
                             line=dict(color=NVIDIA_GREEN, width=1.5),
                             name="Close"), row=1, col=1)
    # RSI
    fig.add_trace(go.Scatter(x=sub["Date"], y=sub["RSI"],
                             line=dict(color=NVIDIA_BLUE, width=1.2),
                             name="RSI"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="#e05252",
                  annotation_text="Overbought", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color=NVIDIA_GREEN,
                  annotation_text="Oversold",   row=2, col=1)
    # MACD
    fig.add_trace(go.Scatter(x=sub["Date"], y=sub["MACD"],
                             line=dict(color=NVIDIA_GREEN, width=1.2),
                             name="MACD"), row=3, col=1)
    fig.add_trace(go.Scatter(x=sub["Date"], y=sub["Signal_Line"],
                             line=dict(color="#f0a500", width=1.2),
                             name="Signal"), row=3, col=1)
    macd_hist = sub["MACD"] - sub["Signal_Line"]
    hist_color = [NVIDIA_GREEN if v >= 0 else "#e05252" for v in macd_hist]
    fig.add_trace(go.Bar(x=sub["Date"], y=macd_hist,
                         marker_color=hist_color, name="Histogram",
                         showlegend=False), row=3, col=1)
    fig.update_layout(height=680, **LAYOUT_DEFAULTS)
    return fig


def fig_train_test(df, train_idx, test_idx, train_pred, test_pred, train_true, test_true) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"].iloc[train_idx], y=train_true,
        line=dict(color="#aaa", width=1), name="Train Actual"))
    fig.add_trace(go.Scatter(
        x=df["Date"].iloc[train_idx], y=train_pred,
        line=dict(color=NVIDIA_BLUE, width=1.2, dash="dot"), name="Train Predicted"))
    fig.add_trace(go.Scatter(
        x=df["Date"].iloc[test_idx], y=test_true,
        line=dict(color="#e0e0e0", width=1.5), name="Test Actual"))
    fig.add_trace(go.Scatter(
        x=df["Date"].iloc[test_idx], y=test_pred,
        line=dict(color=NVIDIA_GREEN, width=2), name="Test Predicted"))
    fig.update_layout(title="LSTM — Train vs Test Predictions",
                      yaxis_title="Price (USD)", **LAYOUT_DEFAULTS)
    return fig


def fig_forecast(df, future_df) -> go.Figure:
    hist = df.tail(120)
    fig  = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["Date"], y=hist["Close"],
        line=dict(color=NVIDIA_GREEN, width=2), name="Historical Close"))
    fig.add_trace(go.Scatter(
        x=future_df["Date"], y=future_df["Predicted_Close"],
        line=dict(color=NVIDIA_BLUE, width=2, dash="dash"),
        mode="lines+markers",
        marker=dict(size=5, color=NVIDIA_BLUE),
        name="Forecast"))
    fig.add_vrect(
        x0=future_df["Date"].iloc[0], x1=future_df["Date"].iloc[-1],
        fillcolor="rgba(0,163,224,0.07)", layer="below", line_width=0)
    fig.update_layout(title="Future Price Forecast", yaxis_title="Price (USD)",
                      **LAYOUT_DEFAULTS)
    return fig


def fig_loss(history) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=history.history["loss"], line=dict(color=NVIDIA_GREEN, width=2),
        name="Train Loss"))
    fig.add_trace(go.Scatter(
        y=history.history["val_loss"], line=dict(color=NVIDIA_BLUE, width=2),
        name="Val Loss"))
    fig.update_layout(title="Training Loss Curve",
                      xaxis_title="Epoch", yaxis_title="Huber Loss",
                      **LAYOUT_DEFAULTS)
    return fig


def fig_residuals(test_true, test_pred) -> go.Figure:
    residuals = test_true - test_pred
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Residuals Over Time", "Residual Distribution"))
    fig.add_trace(go.Scatter(
        y=residuals, mode="lines",
        line=dict(color=NVIDIA_BLUE, width=1), name="Residual"), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="#e05252", row=1, col=1)
    fig.add_trace(go.Histogram(
        x=residuals, nbinsx=50, marker_color=NVIDIA_GREEN,
        name="Distribution"), row=1, col=2)
    fig.update_layout(height=380, **LAYOUT_DEFAULTS)
    return fig


def fig_correlation(df: pd.DataFrame) -> go.Figure:
    cols = ["Close", "Open", "High", "Low", "Volume",
            "MA_7", "MA_21", "RSI", "MACD", "Volatility"]
    corr = df[cols].corr().round(2)
    fig  = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale="RdYlGn", zmid=0, text=corr.values,
        texttemplate="%{text}", textfont=dict(size=10),
        showscale=True,
    ))
    fig.update_layout(title="Feature Correlation Heatmap",
                      height=520, **LAYOUT_DEFAULTS)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────────
if "trained" not in st.session_state:
    st.session_state.trained    = False
if "model"   not in st.session_state:
    st.session_state.model      = None
if "scaler"  not in st.session_state:
    st.session_state.scaler     = None
if "history" not in st.session_state:
    st.session_state.history    = None
if "metrics" not in st.session_state:
    st.session_state.metrics    = {}
if "results" not in st.session_state:
    st.session_state.results    = {}

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">📈 NVIDIA Stock Price Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Bidirectional LSTM · Technical Indicators · Interactive Dashboard</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — CONTROLS
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/6/6d/Nvidia_image_logo.svg",
             width=160)
    st.markdown("---")
    st.markdown("### ⚙️ Model Configuration")
    epochs     = st.slider("Training Epochs",     min_value=10,  max_value=100, value=40, step=5)
    batch_size = st.select_slider("Batch Size",   options=[32, 64, 128], value=64)
    n_forecast = st.slider("Forecast Days",       min_value=7,   max_value=90,  value=30)
    candle_win = st.slider("Candlestick Window",  min_value=60,  max_value=500, value=180)
    ind_window = st.slider("Indicator Window",    min_value=90,  max_value=730, value=365)
    st.markdown("---")

    train_btn = st.button("🚀 Train / Re-Train Model", use_container_width=True, type="primary")
    st.markdown("---")
    st.caption("Data: NVIDIA Corp. (NVDA) | 1999–2026")
    st.caption("Model: Bidirectional LSTM (3 layers)")
    st.caption("Features: 19 technical indicators")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading & engineering features…"):
    df = load_and_engineer(DATA_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# TRAIN MODEL (on button click)
# ─────────────────────────────────────────────────────────────────────────────
if train_btn:
    with st.spinner("Training Bidirectional LSTM… This may take a few minutes."):
        (model, scaler, history, metrics,
         train_idx, test_idx,
         train_pred, test_pred,
         train_true, test_true) = train_model(df, epochs=epochs, batch_size=batch_size)

        st.session_state.trained  = True
        st.session_state.model    = model
        st.session_state.scaler   = scaler
        st.session_state.history  = history
        st.session_state.metrics  = metrics
        st.session_state.results  = dict(
            train_idx=train_idx, test_idx=test_idx,
            train_pred=train_pred, test_pred=test_pred,
            train_true=train_true, test_true=test_true,
        )
    st.success("✅ Model trained and saved!")

# ─────────────────────────────────────────────────────────────────────────────
# TOP KPI ROW
# ─────────────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)
latest  = df.iloc[-1]
prev    = df.iloc[-2]
delta_p = latest["Close"] - prev["Close"]
delta_r = delta_p / prev["Close"] * 100

k1.metric("Latest Close",  f"${latest['Close']:.2f}",   f"{delta_r:+.2f}%")
k2.metric("Open",          f"${latest['Open']:.2f}")
k3.metric("High",          f"${latest['High']:.2f}")
k4.metric("Low",           f"${latest['Low']:.2f}")
k5.metric("Volume",        f"{int(latest['Volume']):,}")
k6.metric("RSI (14)",      f"{latest['RSI']:.1f}")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Market Overview",
    "🔬 Technical Analysis",
    "🤖 Model Training",
    "🔮 Forecast",
    "📉 Model Diagnostics",
    "🗃️ Data Explorer",
])

# ── TAB 1 : Market Overview ───────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-title">Price History</div>', unsafe_allow_html=True)
    # Full close price line
    fig_full = go.Figure()
    fig_full.add_trace(go.Scatter(
        x=df["Date"], y=df["Close"],
        fill="tozeroy", fillcolor="rgba(118,185,0,0.10)",
        line=dict(color=NVIDIA_GREEN, width=1.5), name="Close"))
    fig_full.update_layout(title="NVIDIA Close Price (Full History)",
                           yaxis_title="Price (USD)", **LAYOUT_DEFAULTS)
    st.plotly_chart(fig_full, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">Returns Distribution</div>', unsafe_allow_html=True)
        fig_ret = px.histogram(df, x="Price_Change", nbins=100,
                               title="Daily Return Distribution",
                               color_discrete_sequence=[NVIDIA_GREEN])
        fig_ret.update_layout(**LAYOUT_DEFAULTS)
        st.plotly_chart(fig_ret, use_container_width=True)
    with c2:
        st.markdown('<div class="section-title">Annual Volatility</div>', unsafe_allow_html=True)
        df_yr  = df.copy()
        df_yr["Year"] = df_yr["Date"].dt.year
        ann_vol = df_yr.groupby("Year")["Price_Change"].std() * np.sqrt(252) * 100
        fig_v   = px.bar(ann_vol, title="Annualised Volatility (%)",
                         color_discrete_sequence=[NVIDIA_BLUE])
        fig_v.update_layout(**LAYOUT_DEFAULTS)
        st.plotly_chart(fig_v, use_container_width=True)

# ── TAB 2 : Technical Analysis ────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">Candlestick Chart</div>', unsafe_allow_html=True)
    st.plotly_chart(fig_candle(df, candle_win), use_container_width=True)
    st.plotly_chart(fig_volume(df, candle_win), use_container_width=True)

    st.markdown('<div class="section-title">Indicators Panel</div>', unsafe_allow_html=True)
    st.plotly_chart(fig_indicators(df, ind_window), use_container_width=True)

    st.markdown('<div class="section-title">Feature Correlation</div>', unsafe_allow_html=True)
    st.plotly_chart(fig_correlation(df), use_container_width=True)

# ── TAB 3 : Model Training ────────────────────────────────────────────────────
with tab3:
    if not st.session_state.trained:
        st.info("👈 Click **Train / Re-Train Model** in the sidebar to start training.")
    else:
        m = st.session_state.metrics
        st.markdown('<div class="section-title">Model Performance Metrics</div>', unsafe_allow_html=True)
        mc1, mc2, mc3, mc4, mc5, mc6, mc7 = st.columns(7)
        mc1.metric("Train MAE",  f"${m['train_mae']:.2f}")
        mc2.metric("Test MAE",   f"${m['test_mae']:.2f}")
        mc3.metric("Train RMSE", f"${m['train_rmse']:.2f}")
        mc4.metric("Test RMSE",  f"${m['test_rmse']:.2f}")
        mc5.metric("Train R²",   f"{m['train_r2']:.4f}")
        mc6.metric("Test R²",    f"{m['test_r2']:.4f}")
        mc7.metric("Test MAPE",  f"{m['test_mape']:.2f}%")

        st.plotly_chart(fig_loss(st.session_state.history), use_container_width=True)

        r = st.session_state.results
        st.plotly_chart(
            fig_train_test(df,
                           r["train_idx"], r["test_idx"],
                           r["train_pred"], r["test_pred"],
                           r["train_true"], r["test_true"]),
            use_container_width=True,
        )

# ── TAB 4 : Forecast ─────────────────────────────────────────────────────────
with tab4:
    if not st.session_state.trained:
        st.info("👈 Train the model first to generate forecasts.")
    else:
        with st.spinner("Generating forecast…"):
            future_df = forecast_future(
                st.session_state.model,
                st.session_state.scaler,
                df, n_days=n_forecast,
            )
        st.plotly_chart(fig_forecast(df, future_df), use_container_width=True)

        st.markdown('<div class="section-title">Forecast Table</div>', unsafe_allow_html=True)
        future_df["Date"] = future_df["Date"].dt.strftime("%Y-%m-%d")
        future_df["Predicted_Close"] = future_df["Predicted_Close"].round(2)
        st.dataframe(future_df.style.format({"Predicted_Close": "${:.2f}"}),
                     use_container_width=True, height=300)

        # Download button
        csv_bytes = future_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Forecast CSV", csv_bytes,
                           file_name="nvidia_forecast.csv", mime="text/csv")

# ── TAB 5 : Model Diagnostics ────────────────────────────────────────────────
with tab5:
    if not st.session_state.trained:
        st.info("👈 Train the model first.")
    else:
        r = st.session_state.results
        st.markdown('<div class="section-title">Residual Analysis</div>', unsafe_allow_html=True)
        st.plotly_chart(
            fig_residuals(r["test_true"], r["test_pred"]),
            use_container_width=True,
        )

        st.markdown('<div class="section-title">Actual vs Predicted Scatter</div>', unsafe_allow_html=True)
        fig_sc = go.Figure()
        mn = min(r["test_true"].min(), r["test_pred"].min())
        mx = max(r["test_true"].max(), r["test_pred"].max())
        fig_sc.add_trace(go.Scatter(
            x=r["test_true"], y=r["test_pred"],
            mode="markers",
            marker=dict(color=NVIDIA_GREEN, size=4, opacity=0.6),
            name="Predictions",
        ))
        fig_sc.add_trace(go.Scatter(
            x=[mn, mx], y=[mn, mx],
            line=dict(color="#e05252", dash="dash"), name="Perfect Fit"))
        fig_sc.update_layout(
            title="Actual vs Predicted (Test Set)",
            xaxis_title="Actual Price", yaxis_title="Predicted Price",
            **LAYOUT_DEFAULTS,
        )
        st.plotly_chart(fig_sc, use_container_width=True)

# ── TAB 6 : Data Explorer ────────────────────────────────────────────────────
with tab6:
    st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Features",      len(FEATURES))
    c3.metric("Date Range",    f"{df['Date'].dt.year.min()} – {df['Date'].dt.year.max()}")
    c4.metric("All-Time High", f"${df['High'].max():.2f}")

    st.markdown('<div class="section-title">Summary Statistics</div>', unsafe_allow_html=True)
    st.dataframe(df[["Close", "Open", "High", "Low", "Volume",
                      "RSI", "MACD", "Volatility"]].describe().round(4),
                 use_container_width=True)

    st.markdown('<div class="section-title">Raw Data (Latest 100 Rows)</div>', unsafe_allow_html=True)
    disp = df[["Date", "Open", "High", "Low", "Close", "Volume",
               "MA_7", "MA_21", "RSI", "MACD"]].tail(100).copy()
    disp["Date"] = disp["Date"].dt.strftime("%Y-%m-%d")
    st.dataframe(disp.style.format({
        "Open": "${:.2f}", "High": "${:.2f}", "Low": "${:.2f}", "Close": "${:.2f}",
        "Volume": "{:,.0f}", "MA_7": "${:.2f}", "MA_21": "${:.2f}",
        "RSI": "{:.1f}", "MACD": "{:.4f}",
    }), use_container_width=True, height=380)

    # Allow full data download
    csv_full = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Engineered Dataset", csv_full,
                       file_name="nvidia_engineered.csv", mime="text/csv")
