# nocturne_bloom
# 🧠 Nvidia Stock Price Prediction with CNN & Dash

This project uses a 1D Convolutional Neural Network (CNN) to predict **daily closing prices** of Nvidia Corporation (NVDA) stock using historical OHLCV data. It includes a full pipeline for data preprocessing, model training, backtesting, and deployment through an interactive **Dash dashboard**.

## 📈 Project Goals

- Predict next-day closing prices of NVDA using historical patterns
- Evaluate predictive performance using MSE, RMSE, and directional accuracy
- Simulate a simple trading strategy with **rolling backtesting**
- Provide a **user-friendly dashboard** for visualizing predictions and portfolio returns

---
## 🔍 Methodology

1. **Data Preprocessing**  
   - Load historical OHLCV data from Yahoo Finance  
   - Normalize features using `MinMaxScaler`  
   - Create lookback windows (e.g., past 60 days to predict the next day)

2. **Model Architecture** (`src/model_cnn.py`)  
   - Input shape: `(samples, timesteps, features)`  
   - Layers: Conv1D → ReLU → MaxPooling → Flatten → Dense  
   - Output: 1 neuron (regression output for next-day closing price)  

3. **Backtesting** (`src/backtest.py`)  
   - Walk-forward evaluation on test data  
   - Simulated trading strategy (buy if predicted ↑)  
   - Metrics: cumulative return, directional accuracy, MSE, RMSE  

4. **Deployment** (`app/`)  
   - Dash dashboard displays predicted vs actual prices  
   - Portfolio simulation over time  
   - Dropdowns, sliders, and graphs for interactivity  

---
## 🚀 Getting Started
    ``` bash
    git clone https://github.com/rayan-joshi/nvda-stock-predictor.git
    cd nvda-stock-predictor
    pip install -r requirements.txt
    python app/app.py
    ```
