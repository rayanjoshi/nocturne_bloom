import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, ADXIndicator, MassIndex, AroonIndicator, CCIIndicator
from ta.volatility import BollingerBands, AverageTrueRange, UlcerIndex
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator, ForceIndexIndicator, NegativeVolumeIndexIndicator
from pathlib import Path
import hydra
from omegaconf import DictConfig
import numpy as np
import sys

script_dir = Path(__file__).parent    # /path/to/repo/NVDA_stock_predictor/src
repo_root = script_dir.parent         # /path/to/repo/NVDA_stock_predictor
scripts_path = repo_root / "scripts"
sys.path.append(str(scripts_path))
from logging_config import get_logger, setup_logging, log_function_start, log_function_end

def feature_engineering(dataFrame, cfg: DictConfig, save_data_path):
    logger = log_function_start("feature_engineering",dataFrame=dataFrame, save_data_path=save_data_path)
    logger.info("Starting feature engineering...")
    
    for col in ['Close','Open', 'High', 'Low', 'Volume', 'PB_Ratio', 'PE_Ratio', 'SPY_Close', 'QQQ_Close', 'SOXX_Close', 'VIX_Proxy', 'Treasury_10Y']:
        if col in dataFrame.columns:
            dataFrame[col] = pd.to_numeric(dataFrame[col], errors='coerce')
    
    # Ensure no NaN values before calculations
    nan_count = dataFrame.isna().sum().sum()
    total_values = dataFrame.size
    if nan_count > 0:
        nan_pct = (nan_count / total_values) * 100
        logger.warning("Found %d NaN values in initial DataFrame (%.2f%% of all values)", nan_count, nan_pct)
    dataFrame.dropna(inplace=True)
    logger.debug("Rows dropped after NaN removal: %d", dataFrame.shape[0])
    
    # -------- Momentum/ Trend Indicators -------- #
    rsi = RSIIndicator(close=dataFrame['Close'], window=14)
    dataFrame['RSI'] = rsi.rsi()
    logger.debug("Created 'RSI' with mean: %.2f, std: %.2f", dataFrame['RSI'].mean(), dataFrame['RSI'].std())
    dataFrame['rsi_momentum'] = dataFrame['RSI'].diff(5)
    
    macd = MACD(close=dataFrame['Close'])
    dataFrame['MACD'] = macd.macd()
    dataFrame['MACDSignal'] = macd.macd_signal()
    logger.debug("Created MACD features: MACD range [%.2f, %.2f], MACDSignal range [%.2f, %.2f]",
                    dataFrame['MACD'].min(), dataFrame['MACD'].max(),
                    dataFrame['MACDSignal'].min(), dataFrame['MACDSignal'].max())
    
    dataFrame['SMA_20'] = dataFrame['Close'].rolling(window=20).mean()
    dataFrame['SMA_50'] = dataFrame['Close'].rolling(window=50).mean()
    dataFrame['SMA_200'] = dataFrame['Close'].rolling(window=200).mean()
    if len(dataFrame) < 200:
        logger.error("Insufficient data for SMA_200 calculation: %d rows available", len(dataFrame))
        raise ValueError("Insufficient data for 200-day SMA")
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
    
    dataFrame['price_acceleration'] = dataFrame['Close'].pct_change(5) - dataFrame['Close'].pct_change(10)
    
    dataFrame['volume_surge'] = dataFrame['Volume'] / dataFrame['Volume'].rolling(20).mean()
    dataFrame['volume_momentum'] = dataFrame['volume_surge'].diff(3)
    
    # Multi-timeframe momentum
    dataFrame['momentum_1d'] = dataFrame['Close'].pct_change(1)
    dataFrame['momentum_5d'] = dataFrame['Close'].pct_change(5)
    dataFrame['momentum_10d'] = dataFrame['Close'].pct_change(10)
    dataFrame['momentum_consistency'] = (
        (dataFrame['momentum_1d'] > 0).rolling(5).sum() / 5
    )
    
    # --------- Volatility Indicators --------- #
    bb = BollingerBands(close=dataFrame['Close'], window=20, window_dev=2)
    dataFrame['BBHigh'] = bb.bollinger_hband()
    dataFrame['BBLow'] = bb.bollinger_lband()
    logger.debug("Created Bollinger Bands: BBHigh mean %.2f, BBLow mean %.2f",
                    dataFrame['BBHigh'].mean(), dataFrame['BBLow'].mean())
    
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
    
    dataFrame['vix_normalized'] = (dataFrame['VIX_Proxy'] - dataFrame['VIX_Proxy'].rolling(126).mean()) / dataFrame['VIX_Proxy'].rolling(126).std()  # Reduced from 252
    dataFrame['vix_regime'] = (dataFrame['VIX_Proxy'] > dataFrame['VIX_Proxy'].rolling(126).quantile(0.75)).astype(int)

    
    # --------- Volume Indicators --------- #
    dataFrame['volumeChange'] = dataFrame['Volume'].diff()
    dataFrame['volumeRateOfChange'] = (dataFrame['Volume'] - dataFrame['Volume'].shift(12)) / dataFrame['Volume'].shift(12) * 100
    
    obv = OnBalanceVolumeIndicator(close=dataFrame['Close'], volume=dataFrame['Volume'])
    dataFrame['onBalanceVolume'] = obv.on_balance_volume()
    logger.debug("Created 'onBalanceVolume' with mean: %.2f, std: %.2f",
                    dataFrame['onBalanceVolume'].mean(), dataFrame['onBalanceVolume'].std())

    cmf = ChaikinMoneyFlowIndicator(high=dataFrame['High'], low=dataFrame['Low'], close=dataFrame['Close'], volume=dataFrame['Volume'], window=20)
    dataFrame['chaikinMoneyFlow'] = cmf.chaikin_money_flow()
    logger.debug("Created 'chaikinMoneyFlow' with mean: %.2f, std: %.2f",
                    dataFrame['chaikinMoneyFlow'].mean(), dataFrame['chaikinMoneyFlow'].std())

    fi = ForceIndexIndicator(close=dataFrame['Close'], volume=dataFrame['Volume'], window=20)
    dataFrame['forceIndex'] = fi.force_index()
    logger.debug("Created 'forceIndex' with mean: %.2f, std: %.2f",
                    dataFrame['forceIndex'].mean(), dataFrame['forceIndex'].std())

    nvi = NegativeVolumeIndexIndicator(close=dataFrame['Close'], volume=dataFrame['Volume'])
    dataFrame['negativeVolumeIndex'] = nvi.negative_volume_index()
    logger.debug("Created 'negativeVolumeIndex' with mean: %.2f, std: %.2f",
                    dataFrame['negativeVolumeIndex'].mean(), dataFrame['negativeVolumeIndex'].std())

    dataFrame['rolling_vol_5'] = dataFrame['Close'].pct_change().rolling(5).std()
    dataFrame['rolling_vol_21'] = dataFrame['Close'].pct_change().rolling(21).std()
    dataFrame['vol_percentile_21d'] = dataFrame['rolling_vol_21'].rolling(126).rank(pct=True)  # Reduced from 252 to 126 (6 months)
    dataFrame['vol_percentile_5d'] = dataFrame['rolling_vol_5'].rolling(63).rank(pct=True)
    dataFrame['regime_high_vol'] = (dataFrame['vol_percentile_21d'] > 0.8).astype(int)
    
    dataFrame['vol_divergence'] = dataFrame['rolling_vol_21'] - (dataFrame['VIX_Proxy'] / 100)  # Normalize VIX to comparable scale
    
    dataFrame['volume_vol'] = dataFrame['Volume'].rolling(20).std() / dataFrame['Volume'].rolling(20).mean()
    dataFrame['volume_price_trend'] = dataFrame['Volume'].rolling(5).corr(dataFrame['Close'])
    
    dataFrame['volume_skew'] = dataFrame['Volume'].rolling(20).skew()
    dataFrame['volume_acceleration'] = dataFrame['Volume'].pct_change(5)
    dataFrame['volume_breakout'] = (dataFrame['Volume'] > dataFrame['Volume'].rolling(50).quantile(0.9)).astype(int)
    
    dataFrame['volume_price_divergence'] = (
        dataFrame['volume_surge'] - dataFrame['Close'].pct_change().abs().rolling(5).mean()
    )
    
    # -------- Candle Indicators --------- #
    dataFrame['Body'] = dataFrame['Close'] - dataFrame['Open']
    dataFrame['upperWick'] = dataFrame['High'] - dataFrame[['Close', 'Open']].max(axis=1)
    dataFrame['lowerWick'] = dataFrame[['Close', 'Open']].min(axis=1) - dataFrame['Low']
    logger.debug("Created candle features: Body mean %.2f, upperWick mean %.2f, lowerWick mean %.2f",
                    dataFrame['Body'].mean(), dataFrame['upperWick'].mean(), dataFrame['lowerWick'].mean())

    # --------- Statistical Indicators --------- #
    dataFrame['sigma'] = dataFrame['Close'].rolling(window=20).std()
    dataFrame['beta'] = dataFrame['Close'].rolling(window=20).cov(dataFrame['Volume']) / dataFrame['Volume'].rolling(window=20).var()   
    dataFrame['skewness'] = dataFrame['Close'].rolling(window=20).skew()
    
    # --------- Gap Analysis --------- #
    dataFrame['overnight_gap'] = (dataFrame['Open'] - dataFrame['Close'].shift(1)) / dataFrame['Close'].shift(1)
    dataFrame['gap_magnitude_avg_5d'] = dataFrame['overnight_gap'].abs().rolling(5).mean()
    dataFrame['large_gap_frequency'] = (dataFrame['overnight_gap'].abs() > 0.03).rolling(20).sum()
    
    # --------- AI/Tech Sector Proxies --------- #
    dataFrame['tech_sector_rotation'] = dataFrame['Close'] / dataFrame['SPY_Close'] - 1
    dataFrame['nasdaq_relative_strength'] = dataFrame['Close'] / dataFrame['QQQ_Close'] - 1
    dataFrame['semiconductor_strength'] = dataFrame['Close'] / dataFrame['SOXX_Close'] - 1
    
    dataFrame['spy_qqq_spread'] = (dataFrame['QQQ_Close'] / dataFrame['SPY_Close']).pct_change()
    dataFrame['soxx_qqq_spread'] = (dataFrame['SOXX_Close'] / dataFrame['QQQ_Close']).pct_change()
    
    dataFrame['tech_leadership'] = (dataFrame['QQQ_Close'].pct_change() - dataFrame['SPY_Close'].pct_change()).rolling(10).mean()
    dataFrame['semiconductor_leadership'] = (dataFrame['SOXX_Close'].pct_change() - dataFrame['QQQ_Close'].pct_change()).rolling(10).mean()
    dataFrame['nvda_outperformance'] = (dataFrame['Close'].pct_change() - dataFrame['SOXX_Close'].pct_change()).rolling(5).mean()
    dataFrame['sector_momentum_divergence'] = (
        dataFrame['Close'].pct_change(10) - dataFrame['SOXX_Close'].pct_change(10)
    )
    
    dataFrame['mega_cap_rotation'] = (
        dataFrame['QQQ_Close'].pct_change(5) - dataFrame['SPY_Close'].pct_change(5)
    ).rolling(10).mean()
    
    dataFrame['momentum_persistence_3d'] = (
        (dataFrame['nvda_outperformance'] > 0).rolling(3).sum() / 3
    )
    
    dataFrame['ai_momentum_strength'] = (
        dataFrame['nvda_outperformance'] * 
        dataFrame['tech_leadership'] * 
        dataFrame['semiconductor_strength']
    )
    logger.debug("Created 'ai_momentum_strength' with range: [%.2f, %.2f]", 
                    dataFrame['ai_momentum_strength'].min(), 
                    dataFrame['ai_momentum_strength'].max()
                    )
    
    dataFrame['momentum_acceleration'] = (
        dataFrame['ai_momentum_strength'].diff(1) + 
        dataFrame['ai_momentum_strength'].diff(2)
    ) / 2
    logger.debug("Created 'momentum_acceleration' with mean: %.2f, std: %.2f",
                    dataFrame['momentum_acceleration'].mean(),
                    dataFrame['momentum_acceleration'].std()
                    )

    dataFrame['confirmed_momentum'] = (
        dataFrame['ai_momentum_strength'] * 
        (dataFrame['volume_surge'] > 1).astype(float)
    )
    logger.debug("Created 'confirmed_momentum' with mean: %.2f, std: %.2f",
                    dataFrame['confirmed_momentum'].mean(),
                    dataFrame['confirmed_momentum'].std()
                    )

    # --------- Cross-Asset Correlations --------- #
    dataFrame['vix_spread'] = dataFrame['VIX_Proxy'] - dataFrame['VIX_Proxy'].rolling(20).mean()
    dataFrame['spy_nvda_correlation'] = dataFrame['Close'].pct_change().rolling(20).corr(dataFrame['SPY_Close'].pct_change())
    dataFrame['qqq_nvda_correlation'] = dataFrame['Close'].pct_change().rolling(20).corr(dataFrame['QQQ_Close'].pct_change())
    dataFrame['treasury_equity_spread'] = dataFrame['Treasury_10Y'].diff() * -1  # Inverse relationship with equities
    
    
    # --------- Trend and Breakout Indicators --------- #
    dataFrame['trend_strength'] = (dataFrame['Close'] > dataFrame['SMA_200']).rolling(20).mean()
    dataFrame['bull_market_intensity'] = (
        (dataFrame['Close'] > dataFrame['SMA_50']).astype(int) +
        (dataFrame['SMA_50'] > dataFrame['SMA_200']).astype(int) +
        (dataFrame['Close'].pct_change(20) > 0.1).astype(int)
    )
    
    dataFrame['price_vs_bb_position'] = (dataFrame['Close'] - dataFrame['BBLow']) / (dataFrame['BBHigh'] - dataFrame['BBLow'])
    dataFrame['breakout_signal'] = (dataFrame['Close'] > dataFrame['BBHigh']).astype(int)
    
    # --------- Seasonal Indicators --------- #
    dataFrame['days_in_quarter'] = (dataFrame.index.dayofyear % 90)
    dataFrame['earnings_proximity'] = np.where(
        dataFrame['days_in_quarter'].isin([1, 2, 3, 88, 89, 0]), 1, 0
    )
    
    # Shift target column by -1 for next-day prediction BEFORE dropping NaN
    dataFrame['Target'] = dataFrame['Close'].shift(-1)
    nan_count = dataFrame.isna().sum().sum()
    total_values = dataFrame.size
    if nan_count > 0:
        nan_pct = (nan_count / total_values) * 100
        logger.warning("Found %d NaN values in final DataFrame (%.2f%% of all values)", nan_count, nan_pct)
    dataFrame.dropna(inplace=True)  # Drop NaN values after shifting target
    logger.debug("Rows dropped after NaN removal: %d", dataFrame.shape[0])
    
    # No scaling here - will be done in data_module.py after temporal split
    logger.info("Feature engineering complete - no scaling applied")
    logger.info("Scaling will be performed in data_module.py after train/val split to avoid data leakage")

    # Save processed data
    save_data_path.parent.mkdir(parents=True, exist_ok=True)
    dataFrame.to_csv(save_data_path, index=True)

    logger.info(f"Processed data saved to {save_data_path.absolute()}")
    logger.info("--------- Feature Engineering Statistics ---------")
    logger.info(f"Total features created: {len(dataFrame.columns)}")
    logger.info(f"Dataset shape: {dataFrame.shape}")
    logger.info(f"Features will be scaled in data_module.py after train/val split")
    logger.info(f"All features: {list(dataFrame.columns)}")
    logger.info("--------- Feature Engineering Completed ---------")
    log_function_end("feature_engineering", dataFrame=dataFrame, save_data_path=save_data_path)
    return dataFrame

@hydra.main(version_base=None, config_path="../configs", config_name="feature_engineering")
def main(cfg: DictConfig):
    try:
        setup_logging(log_level="INFO", console_output=True, file_output=True)
        logger = get_logger("main")
        # Convert relative path to absolute path within the repository
        script_dir = Path(__file__).parent  # /path/to/repo/NVDA_stock_predictor/src
        repo_root = script_dir.parent  # /path/to/repo/NVDA_stock_predictor
        raw_data_path = repo_root / cfg.data_loader.raw_data_path.lstrip('../')
        save_data_path = repo_root / cfg.features.preprocessing_data_path.lstrip('../')

        logger.info(f"Reading raw data from: {raw_data_path.absolute()}")
        dataFrame = pd.read_csv(raw_data_path, header=0, index_col=0, parse_dates=True)
        feature_engineering(dataFrame, cfg, save_data_path)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise
if __name__ == "__main__":
    main()