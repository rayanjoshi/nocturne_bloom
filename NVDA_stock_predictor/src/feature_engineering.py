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
    # -------- Momentum/ Trend indicators -------- #
    rsi = RSIIndicator(close=dataFrame['Close'], window=14)
    dataFrame['RSI'] = rsi.rsi()
    macd = MACD(close=dataFrame['Close'])
    dataFrame['MACD'] = macd.macd()
    dataFrame['MACDSignal'] = macd.macd_signal()
    
    dataFrame['SMA_20'] = dataFrame['Close'].rolling(window=20).mean()
    dataFrame['EMA_12'] = dataFrame['Close'].ewm(span=12, adjust=False).mean()
    dataFrame['EMA_26'] = dataFrame['Close'].ewm(span=26, adjust=False).mean()
    # First 11 values will be NaN, so we can calculate ROC from the 12th value onwards
    dataFrame['rateOfChange'] = (dataFrame['Close'] - dataFrame['Close'].shift(12)) / dataFrame['Close'].shift(12) * 100
    dataFrame['Momentum'] = dataFrame['Close'].diff(10)

    # --------- Volatility indicators --------- #
    bb = BollingerBands(close=dataFrame['Close'], window=20, window_dev=2)
    dataFrame['BBHigh'] = bb.bollinger_hband()
    dataFrame['BBLow'] = bb.bollinger_lband()
    
    dataFrame['priceChange'] = dataFrame['Close'].diff()
    dataFrame['closeChangePercentage'] = dataFrame['Close'].pct_change()
    
    range = dataFrame['High'] - dataFrame['Low']
    previousClosingHigh = (dataFrame['High'] - dataFrame['Close'].shift(1)).abs()
    previousClosingLow = (dataFrame['Low'] - dataFrame['Close'].shift(1)).abs()
    true_range = pd.concat([range, previousClosingHigh, previousClosingLow], axis=1).max(axis=1)
    dataFrame['averageTrueRange'] = true_range.rolling(window=14).mean()
    dataFrame['rollingStd'] = dataFrame['Close'].rolling(window=14).std()

    # --------- Volume indicators --------- #
    dataFrame['volumeChange'] = dataFrame['Volume'].diff()
    dataFrame['volumeRateOfChange'] = (dataFrame['Volume'] - dataFrame['Volume'].shift(12)) / dataFrame['Volume'].shift(12) * 100
    
    priceChange = dataFrame['Close'].diff()
    volume_direction = pd.Series(index=dataFrame.index, dtype=float)
    volume_direction[priceChange > 0] = dataFrame['Volume'][priceChange > 0]
    volume_direction[priceChange < 0] = -dataFrame['Volume'][priceChange < 0]
    volume_direction[priceChange == 0] = 0
    volume_direction.iloc[0] = 0  # First value is 0
    dataFrame['onBalanceVolume'] = volume_direction.cumsum()

    # -------- Candle indicators --------- #
    dataFrame['Body'] = dataFrame['Close'] - dataFrame['Open']
    dataFrame['upperWick'] = dataFrame['High'] - dataFrame[['Close', 'Open']].max(axis=1)
    dataFrame['lowerWick'] = dataFrame[['Close', 'Open']].min(axis=1) - dataFrame['Low']
    
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
    preprocessing_data_path = repo_root / cfg.features.preprocessing_data_path.lstrip('../')

    # Save the scaler for future use
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)

    # Save processed data
    preprocessing_data_path.parent.mkdir(parents=True, exist_ok=True)
    dataFrame.to_csv(preprocessing_data_path, index=True)

    print(f"Processed data saved to {preprocessing_data_path.absolute()}")
    print("--------- Feature Engineering Statistics ---------")
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