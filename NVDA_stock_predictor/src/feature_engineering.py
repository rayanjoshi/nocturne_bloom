import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import hydra
from omegaconf import DictConfig

def feature_engineering(df, cfg: DictConfig):
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
    df.to_csv(processed_data_path, index=True)

    print(f"Feature engineering completed. Processed data saved to {processed_data_path.absolute()}")

    return df

@hydra.main(version_base=None, config_path="../configs", config_name="feature_engineering")
def main(cfg: DictConfig):
    try:
        # Convert relative path to absolute path within the repository
        script_dir = Path(__file__).parent  # /path/to/repo/NVDA_stock_predictor/src
        repo_root = script_dir.parent  # /path/to/repo/NVDA_stock_predictor
        raw_data_path = repo_root / cfg.data_loader.raw_data_path.lstrip('../')
        
        print(f"Reading raw data from: {raw_data_path.absolute()}")
        df = pd.read_csv(raw_data_path, header=0, skiprows=[1,2], index_col=0, parse_dates=True)
        feature_engineering(df, cfg)
        print("Feature engineering completed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise
if __name__ == "__main__":
    main()