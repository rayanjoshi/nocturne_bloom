import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from sklearn.preprocessing import MinMaxScaler
import joblib
from pathlib import Path
import hydra
from omegaconf import DictConfig

def feature_engineering(dataFrame, cfg: DictConfig):
    print("Starting feature engineering...")

    for col in ['Close','Open', 'High', 'Low', 'Volume']:
        if col in dataFrame.columns:
            dataFrame[col] = pd.to_numeric(dataFrame[col], errors='coerce')
    dataFrame.dropna(inplace=True) # Ensure no NaN values before calculations
    # Calculate RSI
    rsi = RSIIndicator(close=dataFrame['Close'], window=14)
    dataFrame['RSI'] = rsi.rsi()

    # Calculate MACD
    macd = MACD(close=dataFrame['Close'])
    dataFrame['MACD'] = macd.macd()
    dataFrame['MACD_Signal'] = macd.macd_signal()

    # Calculate Bollinger Bands
    bb = BollingerBands(close=dataFrame['Close'], window=20, window_dev=2)
    dataFrame['BB_High'] = bb.bollinger_hband()
    dataFrame['BB_Low'] = bb.bollinger_lband()

    # Calculate SMA (Simple Moving Average)
    dataFrame['SMA_20'] = dataFrame['Close'].rolling(window=20).mean()

    # Calculate EMA (Exponential Moving Average)
    dataFrame['EMA_12'] = dataFrame['Close'].ewm(span=12, adjust=False).mean()

    # Calculate price change and the percentage change of neighbouring closing price
    dataFrame['Close_Change'] = dataFrame['Close'].diff()
    dataFrame['Close_Change_Percentage'] = dataFrame['Close'].pct_change()

    # Drop rows with NaN values
    dataFrame.dropna(inplace=True)

    # Scale features
    scaler = MinMaxScaler()
    # Include all relevant columns for scaling
    indicator_cols = cfg.features.indicator_cols
    
    # Scale the features and assign them back individually
    scaled_data = scaler.fit_transform(dataFrame[indicator_cols])
    for i, col in enumerate(indicator_cols):
        dataFrame[col] = scaled_data[:, i]
    
    # Convert relative paths to absolute paths within the repository
    script_dir = Path(__file__).parent  # /path/to/repo/NVDA_stock_predictor/src
    repo_root = script_dir.parent  # /path/to/repo/NVDA_stock_predictor
    
    scaler_path = repo_root / cfg.features.SCALER_PATH.lstrip('../')
    processed_data_path = repo_root / cfg.features.processed_data_path.lstrip('../')
    
    # Save the scaler for future use
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)

    # Save processed data
    processed_data_path.parent.mkdir(parents=True, exist_ok=True)
    dataFrame.to_csv(processed_data_path, index=True)

    print(f"Feature engineering completed. Processed data saved to {processed_data_path.absolute()}")
    print("--------- Generated Features ---------")
    print(f"Total features created: {len(dataFrame.columns)}")
    print(f"Features scaled: {len(indicator_cols)}")
    print(f"Dataset shape: {dataFrame.shape}")
    print(f"Scaled features: {indicator_cols}")
    print(f"All features: {list(dataFrame.columns)}")
    print("--------- Feature Engineering Completed ---------")

    return dataFrame

@hydra.main(version_base=None, config_path="../configs", config_name="feature_engineering")
def main(cfg: DictConfig):
    try:
        # Convert relative path to absolute path within the repository
        script_dir = Path(__file__).parent  # /path/to/repo/NVDA_stock_predictor/src
        repo_root = script_dir.parent  # /path/to/repo/NVDA_stock_predictor
        raw_data_path = repo_root / cfg.data_loader.raw_data_path.lstrip('../')
        
        print(f"Reading raw data from: {raw_data_path.absolute()}")
        dataFrame = pd.read_csv(raw_data_path, header=0, skiprows=[1,2], index_col=0, parse_dates=True)
        feature_engineering(dataFrame, cfg)
    except Exception as e:
        print(f"An error occurred: {e}")
        raise
if __name__ == "__main__":
    main()