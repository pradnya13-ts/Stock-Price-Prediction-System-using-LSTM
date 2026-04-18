<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0b3d1e,50:1a7a3c,100:2ecc71&height=220&section=header&text=Stock%20Price%20Predictor&fontSize=46&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=LSTM%20Deep%20Learning%20%2B%20Flask%20Web%20App&descAlignY=58&descSize=17&descColor=a8f5c2" width="100%"/>

<img src="https://media.giphy.com/media/JtBZm3Getg3dqxK0zP/giphy.gif" width="280"/>

### 💹 Predicting Stock Closing Prices with LSTM Neural Networks and Live Yahoo Finance Data

<br/>

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?style=for-the-badge&logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-Web%20App-black?style=for-the-badge&logo=flask)
![Yahoo Finance](https://img.shields.io/badge/Data-Yahoo%20Finance-purple?style=for-the-badge)
![Bootstrap](https://img.shields.io/badge/Frontend-Bootstrap-blueviolet?style=for-the-badge&logo=bootstrap)

<br/>

*An AI-powered stock market prediction system — enter any ticker, get LSTM predictions, EMA charts, and downloadable data instantly.*

</div>

---

## 📌 What Does This Project Do?

<img src="https://media.giphy.com/media/3oKIPEqDGUULpEU0aQ/giphy.gif" width="180" align="right"/>

This project combines **deep learning** and **web development** into a live stock market forecasting tool. It:

- Fetches **live historical stock data** from Yahoo Finance
- Applies an **LSTM neural network** trained on 100-day sequences to predict future closing prices
- Generates **dynamic charts** — EMA trend lines and Predicted vs Actual comparisons
- Serves everything through a clean **Flask web interface**

```python
app = {
    "model":      "Sequential LSTM (100-step input window)",
    "data_source": "Yahoo Finance via yfinance",
    "framework":  "Flask + Keras / TensorFlow",
    "output":     ["EMA charts", "Prediction chart", "CSV download"],
    "examples":   ["AAPL", "TSLA", "RELIANCE.NS", "GOOGL"]
}
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 💹 LSTM Prediction | Predicts next-day closing prices using 100-day historical sequences |
| 📊 EMA Charts | Exponential Moving Averages for 20, 50, 100, and 200 days |
| 📈 Prediction vs Actual | Visual comparison of model output vs real stock prices |
| 🧮 Statistical Summary | Descriptive statistics (mean, std, min, max) for selected stock |
| 📥 CSV Download | Download the processed dataset directly from the web UI |
| ⚡ Live Data | Fetches real-time data via Yahoo Finance on every request |

---

## 🗂️ Repository Structure

```
📦 Stock-Prediction-App
 ┣ 🐍 app.py                       — Flask app (main entry point)
 ┣ 🤖 stock_dl_model.h5            — Pre-trained LSTM model
 ┣ 📁 static/                      — Auto-generated plots and CSVs
 ┃   ┣ 📈 ema_20_50.png            — Short-term EMA chart
 ┃   ┣ 📉 ema_100_200.png          — Long-term EMA chart
 ┃   ┣ 🔮 stock_prediction.png     — Prediction vs Actual chart
 ┃   ┗ 📊 *.csv                    — Downloadable stock datasets
 ┣ 📁 templates/
 ┃   ┗ 🌐 index.html               — Web UI template
 ┗ 📋 requirements.txt             — Python dependencies
```

---

## 🧠 Model Overview

<div align="center">

```
                    ┌──────────────────────────┐
                    │  Input: 100 past closing  │
                    │        prices             │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │     LSTM Layer(s)         │
                    │  (learns temporal trends) │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │      Dense + ReLU         │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  Output: Next-day price   │
                    │     (Linear activation)   │
                    └──────────────────────────┘
```

</div>

| Hyperparameter | Value |
|----------------|-------|
| Model Type | Sequential LSTM |
| Input Window | 100 past closing prices |
| Output | 1 (next-day predicted price) |
| Loss Function | Mean Squared Error (MSE) |
| Optimizer | Adam |
| Activation | ReLU / Linear |
| Train / Test Split | 70% / 30% |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/pradnya13-ts/Stock-Price-Prediction-LSTM.git
cd Stock-Price-Prediction-LSTM
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Flask app
```bash
python app.py
```

Then open your browser at 👉 **http://127.0.0.1:5000**

---

## 🔄 How It Works

```
User enters ticker (e.g. AAPL)
       ↓
Fetch historical data via Yahoo Finance
       ↓
Scale data with MinMaxScaler
       ↓
Create 100-step input sequences
       ↓
LSTM model predicts closing prices
       ↓
Inverse transform predictions → real MW values
       ↓
Generate EMA charts + Prediction vs Actual chart
       ↓
Serve results in Flask web UI + CSV download
```

---

## 📊 Example Output

| Chart | Description |
|-------|-------------|
| 📈 EMA 20 and 50 | Short-term momentum and trend direction |
| 📉 EMA 100 and 200 | Long-term trend — bull/bear market signals |
| 🔮 Prediction vs Actual | LSTM-predicted vs real closing prices |
| 📊 Stats Table | Mean, std deviation, min, max for the stock |

### Example tickers you can try

```
AAPL        → Apple Inc.
TSLA        → Tesla Inc.
GOOGL       → Alphabet Inc.
RELIANCE.NS → Reliance Industries (NSE)
MSFT        → Microsoft Corp.
```

---

## 🧩 Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge)
![Bootstrap](https://img.shields.io/badge/Bootstrap-7952B3?style=for-the-badge&logo=bootstrap&logoColor=white)

</div>

| Component | Technology |
|-----------|------------|
| Language | Python 3 |
| Web Framework | Flask |
| Deep Learning | Keras / TensorFlow |
| Data Source | Yahoo Finance (yfinance) |
| Data Processing | NumPy, Pandas, Scikit-learn |
| Visualization | Matplotlib |
| Frontend | HTML, CSS, Bootstrap |

---

## 🔮 Future Enhancements

- [ ] Add LSTM retraining option directly from the UI
- [ ] Multi-step forecasting (predict 7 or 30 days ahead)
- [ ] Integrate real-time stock price streaming
- [ ] Add sentiment analysis from financial news
- [ ] Deploy on Render / AWS / Streamlit Cloud for public access

---

## 🏆 Credits

<div align="center">

Developed by **Pradnya Chandrakant Bhakare**
MSc AI and ML in Science · Queen Mary University of London

Powered by Python, Flask, and Keras LSTM Neural Networks

</div>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:2ecc71,50:1a7a3c,100:0b3d1e&height=120&section=footer" width="100%"/>

</div>
