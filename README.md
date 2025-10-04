📈 Stock Price Prediction System using LSTM (Flask + Deep Learning)

This project is an AI-powered stock market prediction system built using LSTM (Long Short-Term Memory) neural networks.
It predicts future stock prices based on historical data and visualizes performance trends with moving averages and prediction charts.
The app is built with Flask, features a web interface, and integrates Yahoo Finance for live stock data.

✨ Features

💹 Deep Learning Model (LSTM): Predicts stock closing prices using historical data.

📊 Dynamic Charts: Visualizes trends with Exponential Moving Averages (20, 50, 100, 200 days).

📈 Prediction vs Actual Graph: Compares model performance visually.

🧮 Statistical Summary: Generates key descriptive statistics for the stock data.

📥 Dataset Download: Allows users to download the processed dataset as CSV.

⚡ Web Interface: Simple and interactive Flask web app for prediction and visualization.

🧠 Tech Stack
Component	Technology
Programming Language	Python 3
Web Framework	Flask
Deep Learning	Keras / TensorFlow
Data Source	Yahoo Finance (via yfinance)
Data Processing	NumPy, Pandas, Scikit-learn
Visualization	Matplotlib
Frontend	HTML, CSS (Bootstrap templates)
🧩 Project Structure
📦 Stock-Prediction-App
├── app.py                        # Flask app (main entry point)
├── stock_dl_model.h5             # Pre-trained LSTM model
├── static/                       # Folder for saved plots and downloadable CSVs
│   ├── ema_20_50.png
│   ├── ema_100_200.png
│   ├── stock_prediction.png
│   └── *.csv
├── templates/                    # HTML templates
│   └── index.html
└── requirements.txt              # Python dependencies

🚀 How It Works

User enters a stock ticker symbol (e.g., AAPL, TSLA, RELIANCE.NS, etc.).

The app fetches historical stock data from Yahoo Finance.

It processes and scales the data, then uses the LSTM model to predict future closing prices.

It generates:

EMA trend charts (20–50, 100–200 days)

Prediction vs Original trend chart

The user can view descriptive statistics and download the dataset.

📊 Example Output
Chart	Description
📈 EMA 20 & 50 Chart	Short-term moving average trend
📉 EMA 100 & 200 Chart	Long-term moving average trend
🔮 Prediction vs Actual	LSTM-predicted vs real stock prices
⚙️ Setup Instructions
1. Clone the Repository
git clone https://github.com/yourusername/Stock-Price-Prediction-LSTM.git
cd Stock-Price-Prediction-LSTM

2. Create Virtual Environment (Optional)
python -m venv venv
venv\Scripts\activate  # On Windows
# OR
source venv/bin/activate  # On macOS/Linux

3. Install Dependencies
pip install -r requirements.txt

4. Run the Flask App
python app.py


The app will open in your browser at 👉 http://127.0.0.1:5000

🧠 Model Overview (LSTM)

Model Type: Sequential LSTM

Input: 100 past closing prices

Output: Next-day predicted price

Loss Function: Mean Squared Error (MSE)

Optimizer: Adam

Activation Function: ReLU / Linear

Trained on 70% of the dataset, tested on 30%.

🧩 Example Usage
Input	Output
Stock Symbol: AAPL	Generates charts and dataset for Apple stock
Stock Symbol: RELIANCE.NS	Predicts and visualizes Reliance Industries stock prices
📁 Example File Outputs

static/ema_20_50.png → Short-term EMA chart

static/ema_100_200.png → Long-term EMA chart

static/stock_prediction.png → Prediction vs Actual trend

static/<stock>_dataset.csv → Downloadable dataset

🔮 Future Enhancements

🧾 Add LSTM retraining option from UI.

📆 Predict multiple days ahead (multi-step forecasting).

📈 Integrate real-time stock streaming.

🌐 Deploy on Render / AWS / Streamlit for online access.

🏆 Credits

Developed by Pradnya Chandrakant Bhakare
Powered by Python, Flask, and Keras LSTM Neural Networks

