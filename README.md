# 📈 NVIDIA Stock Price Prediction

> A deep-learning stock forecasting system built with **Bidirectional LSTM**, **19 technical indicators**, and an interactive **Streamlit** dashboard — all in a single Python file.

---

## 🗂 Project Description

This project predicts the future closing price of **NVIDIA Corporation (NASDAQ: NVDA)** using a multi-layer Bidirectional LSTM neural network trained on ~27 years of historical daily OHLCV data (January 1999 – January 2026).

The entire system — data ingestion, feature engineering, model training, inference, and frontend visualisation — lives in **one file**: `Student_NvidiaStockPrediction.py`.

### Key capabilities

| Feature | Details |
|---|---|
| **Dataset** | 6,796 daily records · 5 raw columns (Date, Open, High, Low, Close, Volume) |
| **Feature Engineering** | 14 derived indicators → 19 total input dimensions |
| **Model** | Bidirectional LSTM (3 LSTM blocks + Dropout + Dense output) |
| **Look-back window** | 60 trading days |
| **Train / Test split** | 80% / 20% (chronological, no shuffle) |
| **Evaluation** | MAE · RMSE · R² · MAPE |
| **Forecast horizon** | Configurable 7–90 business days |
| **Frontend** | Streamlit app · 6 tabs · Plotly charts · live sidebar controls |

---

## 📁 File Structure

```
.
├── Student_NvidiaStockPrediction.py   ← Single code file (backend + frontend)
├── Nvidia_stock_data.csv              ← Dataset
├── requirements.txt                   ← Python dependencies
├── Student_NvidiaStockReport.docx     ← Full project report (Word)
└── README.md                          ← This file
```

> **Note:** `nvidia_lstm_model.keras` is auto-created when you click **Train / Re-Train Model** for the first time.

---

## 📊 Dataset

**Source:** Yahoo Finance — NVIDIA Corp. (NVDA) Historical Data  
**URL:** https://finance.yahoo.com/quote/NVDA/history  
**Local file:** `Nvidia_stock_data.csv`

| Field | Description |
|---|---|
| `Date` | Trading date |
| `Open` | Opening price (USD) |
| `High` | Intraday high (USD) |
| `Low` | Intraday low (USD) |
| `Close` | Adjusted closing price (USD) · **Prediction target** |
| `Volume` | Shares traded |

---

## 🧠 Technical Indicators Engineered

| Indicator | Category |
|---|---|
| MA_7, MA_21, MA_50 | Trend |
| EMA_12, EMA_26 | Trend |
| MACD, Signal_Line | Momentum |
| RSI (14-day) | Momentum |
| BB_Upper, BB_Lower, BB_Mid | Volatility |
| Volatility (20-day std) | Volatility |
| Price_Change (daily %) | Return |
| Volume_Change (daily %) | Volume |

---

## 🏗 Model Architecture

```
Input  →  (60, 19)  — 60-day window × 19 features

Bidirectional LSTM (128 units, return_sequences=True)
Dropout (0.25)
LSTM (64 units, return_sequences=True)
Dropout (0.25)
LSTM (32 units)
Dropout (0.20)
Dense (32, ReLU)
Dense (1)            ← predicted next-day Close price
```

- **Loss:** Huber (robust to outlier price spikes)  
- **Optimiser:** Adam (lr=1e-3)  
- **Callbacks:** EarlyStopping (patience=8) · ReduceLROnPlateau (factor=0.5, patience=4)

---

## 🛠 Technologies Used

| Library | Purpose |
|---|---|
| `pandas` | Data loading & feature engineering |
| `numpy` | Numerical operations |
| `scikit-learn` | MinMaxScaler · evaluation metrics |
| `tensorflow / keras` | Bidirectional LSTM model |
| `streamlit` | Interactive web dashboard (frontend) |
| `plotly` | Interactive charts (candlestick, heatmap, etc.) |
| `matplotlib` | Utility plotting (Agg backend) |

---

## ⚙️ Setup & Run Instructions

### 1. Prerequisites

- Python **3.9 – 3.11** (recommended)
- `pip` package manager

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> For Apple Silicon (M1/M2) replace `tensorflow` in `requirements.txt` with `tensorflow-macos` and `tensorflow-metal`.

### 3. Ensure dataset is present

Place `Nvidia_stock_data.csv` in the **same directory** as `Student_NvidiaStockPrediction.py`.

### 4. Launch the dashboard

```bash
streamlit run Student_NvidiaStockPrediction.py
```

The app opens automatically at **http://localhost:8501**

### 5. Train the model

1. Use the **sidebar** to configure epochs, batch size, and forecast horizon.
2. Click **🚀 Train / Re-Train Model**.
3. Wait for the training spinner to complete (~1–5 min depending on hardware).
4. Navigate to **Model Training**, **Forecast**, and **Model Diagnostics** tabs.

---

## 📐 Dashboard Tabs

| Tab | Contents |
|---|---|
| 📊 Market Overview | Full price history, returns distribution, annual volatility |
| 🔬 Technical Analysis | Candlestick + MAs, volume, RSI, MACD, Bollinger Bands, correlation heatmap |
| 🤖 Model Training | KPI metrics, loss curve, train vs test prediction overlay |
| 🔮 Forecast | Future price chart (7–90 days), downloadable forecast CSV |
| 📉 Model Diagnostics | Residual analysis, actual vs predicted scatter |
| 🗃️ Data Explorer | Summary statistics, raw data table, download engineered dataset |

---

## 📏 Performance (indicative)

| Metric | Train | Test |
|---|---|---|
| MAE | ~$1.50 | ~$2.80 |
| RMSE | ~$2.10 | ~$4.20 |
| R² | ~0.994 | ~0.978 |
| MAPE | — | ~2.1% |

> Exact values vary with epochs and random weight initialisation.

---

## ⚠️ Disclaimer

This project is for **academic and educational purposes only**.  
Stock price predictions produced by this model are **not financial advice** and should not be used to make investment decisions.

---

## 👤 Author

**JOUANA ** — Data Analytics / Machine Learning Project  
Dataset courtesy of Yahoo Finance.
