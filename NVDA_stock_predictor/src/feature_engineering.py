import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from config import RAW_DATA_PATH, PROCESSED_DATA_PATH, SCALER_PATH

def feature_engineering(df):
    print("Starting feature engineering...")

    for col in ['Close','Open', 'High', 'Low', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    # Calculate RSI
    rsi = RSIIndicator(close=df['Close'], window=14)
    df['RSI'] = rsi.rsi()

    # Calculate MACD
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()

    # Calculate Bollinger Bands
    bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()

    # Drop rows with NaN values
    df.dropna(inplace=True)

    # Scale features
    scaler = StandardScaler()
    indicator_cols = ['RSI', 'MACD', 'MACD_Signal', 'BB_High', 'BB_Low']
    df[indicator_cols] = scaler.fit_transform(df[indicator_cols])
    
    # Save the scaler for future use
    Path(SCALER_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH,)

    # Save processed data
    Path(PROCESSED_DATA_PATH).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=True)
    
    print(f"Feature engineering completed. Processed data saved to {PROCESSED_DATA_PATH}")

    return df

if __name__ == "__main__":
    df = pd.read_csv(RAW_DATA_PATH, header=0, skiprows=[1,2], index_col=0, parse_dates=True)
    feature_engineering(df)