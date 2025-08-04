import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, ADXIndicator, MassIndex, AroonIndicator, CCIIndicator
from ta.volatility import BollingerBands, AverageTrueRange, UlcerIndex
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator, ForceIndexIndicator, NegativeVolumeIndexIndicator
from pathlib import Path
import hydra
from omegaconf import DictConfig

def feature_engineering(dataFrame, cfg: DictConfig, save_data_path):
    print("Starting feature engineering...")

    for col in ['Close','Open', 'High', 'Low', 'Volume', 'PB_Ratio', 'PE_Ratio']:
        if col in dataFrame.columns:
            dataFrame[col] = pd.to_numeric(dataFrame[col], errors='coerce')

    # Ensure no NaN values before calculations
    dataFrame.dropna(inplace=True)
    
    # -------- Momentum/ Trend Indicators -------- #
    rsi = RSIIndicator(close=dataFrame['Close'], window=14)
    dataFrame['RSI'] = rsi.rsi()
    macd = MACD(close=dataFrame['Close'])
    dataFrame['MACD'] = macd.macd()
    dataFrame['MACDSignal'] = macd.macd_signal()
    
    dataFrame['SMA_20'] = dataFrame['Close'].rolling(window=20).mean()
    dataFrame['SMA_50'] = dataFrame['Close'].rolling(window=50).mean()
    dataFrame['SMA_200'] = dataFrame['Close'].rolling(window=200).mean()
    dataFrame['EMA_12'] = dataFrame['Close'].ewm(span=12, adjust=False).mean()
    dataFrame['EMA_26'] = dataFrame['Close'].ewm(span=26, adjust=False).mean()
    dataFrame['rateOfChange'] = (dataFrame['Close'] - dataFrame['Close'].shift(12)) / dataFrame['Close'].shift(12) * 100
    dataFrame['momentum'] = dataFrame['Close'].diff(10)
    
    adx = ADXIndicator(high=dataFrame['High'], low=dataFrame['Low'], close=dataFrame['Close'], window=14)
    dataFrame['averageDirectionalIndex'] = adx.adx()
    
    mi = MassIndex(high=dataFrame['High'], low=dataFrame['Low'], window_fast=9, window_slow=25)
    dataFrame['massIndex'] = mi.mass_index()
    
    CCI = CCIIndicator(high=dataFrame['High'], low=dataFrame['Low'], close=dataFrame['Close'], window=20)
    dataFrame['commodityChannelIndex'] = CCI.cci()

    # --------- Volatility Indicators --------- #
    bb = BollingerBands(close=dataFrame['Close'], window=20, window_dev=2)
    dataFrame['BBHigh'] = bb.bollinger_hband()
    dataFrame['BBLow'] = bb.bollinger_lband()
    
    dataFrame['priceChange'] = dataFrame['Close'].diff()
    dataFrame['closeChangePercentage'] = dataFrame['Close'].pct_change()
    
    atr = AverageTrueRange(high=dataFrame['High'], low=dataFrame['Low'], close=dataFrame['Close'], window=14)
    dataFrame['averageTrueRange'] = atr.average_true_range()
    dataFrame['rollingStd'] = dataFrame['Close'].rolling(window=14).std()
    
    aroon= AroonIndicator(high=dataFrame['High'], low=dataFrame['Low'], window=14)
    dataFrame['aroonUp'] = aroon.aroon_up()
    dataFrame['aroonDown'] = aroon.aroon_down()
    
    ui = UlcerIndex(close=dataFrame['Close'], window=14)
    dataFrame['ulcerIndex'] = ui.ulcer_index()

    # --------- Volume Indicators --------- #
    dataFrame['volumeChange'] = dataFrame['Volume'].diff()
    dataFrame['volumeRateOfChange'] = (dataFrame['Volume'] - dataFrame['Volume'].shift(12)) / dataFrame['Volume'].shift(12) * 100
    
    obv = OnBalanceVolumeIndicator(close=dataFrame['Close'], volume=dataFrame['Volume'])
    dataFrame['onBalanceVolume'] = obv.on_balance_volume()
    
    cmf = ChaikinMoneyFlowIndicator(high=dataFrame['High'], low=dataFrame['Low'], close=dataFrame['Close'], volume=dataFrame['Volume'], window=20)
    dataFrame['chaikinMoneyFlow'] = cmf.chaikin_money_flow()
    
    fi = ForceIndexIndicator(close=dataFrame['Close'], volume=dataFrame['Volume'], window=20)
    dataFrame['forceIndex'] = fi.force_index()
    
    nvi = NegativeVolumeIndexIndicator(close=dataFrame['Close'], volume=dataFrame['Volume'])
    dataFrame['negativeVolumeIndex'] = nvi.negative_volume_index()

    # -------- Candle Indicators --------- #
    dataFrame['Body'] = dataFrame['Close'] - dataFrame['Open']
    dataFrame['upperWick'] = dataFrame['High'] - dataFrame[['Close', 'Open']].max(axis=1)
    dataFrame['lowerWick'] = dataFrame[['Close', 'Open']].min(axis=1) - dataFrame['Low']
    
    # --------- Technical Indicators --------- #
    dataFrame['sigma'] = dataFrame['Close'].rolling(window=20).std()
    dataFrame['beta'] = dataFrame['Close'].rolling(window=20).cov(dataFrame['Volume']) / dataFrame['Volume'].rolling(window=20).var()
        
    dataFrame['skewness'] = dataFrame['Close'].rolling(window=20).skew()

    # Drop rows with NaN values
    dataFrame.dropna(inplace=True)

    # No scaling here - will be done in data_module.py after temporal split
    print("Feature engineering complete - no scaling applied")
    print("Scaling will be performed in data_module.py after train/val split to avoid data leakage")

    # Convert relative paths to absolute paths within the repository
    script_dir = Path(__file__).parent  # /path/to/repo/NVDA_stock_predictor/src
    repo_root = script_dir.parent  # /path/to/repo/NVDA_stock_predictor

    # Save processed data
    save_data_path.parent.mkdir(parents=True, exist_ok=True)
    dataFrame.to_csv(save_data_path, index=True)

    print(f"Processed data saved to {save_data_path.absolute()}")
    print("--------- Feature Engineering Statistics ---------")
    print(f"Total features created: {len(dataFrame.columns)}")
    print(f"Dataset shape: {dataFrame.shape}")
    print(f"Features will be scaled in data_module.py after train/val split")
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
        save_data_path = repo_root / cfg.features.preprocessing_data_path.lstrip('../')
        
        print(f"Reading raw data from: {raw_data_path.absolute()}")
        dataFrame = pd.read_csv(raw_data_path, header=0, index_col=0, parse_dates=True)
        feature_engineering(dataFrame, cfg, save_data_path)
    except Exception as e:
        print(f"An error occurred: {e}")
        raise
if __name__ == "__main__":
    main()