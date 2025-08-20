import pandas as pd
import pandas_ta as ta
from pathlib import Path
import hydra
from omegaconf import DictConfig
import numpy as np
from ray import logger
from scripts.logging_config import get_logger, setup_logging, log_function_start, log_function_end

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
        logger.warning(f"Found {nan_count} NaN values in initial DataFrame ({nan_pct:.2f}% of all values)")
    dataFrame.dropna(inplace=True)
    logger.debug(f"Rows dropped after NaN removal: {dataFrame.shape[0]}")
    
    # -------- Candle Indicators --------- #
    dataFrame['Body'] = dataFrame['Close'] - dataFrame['Open']
    dataFrame['upperWick'] = dataFrame['High'] - dataFrame[['Close', 'Open']].max(axis=1)
    dataFrame['lowerWick'] = dataFrame[['Close', 'Open']].min(axis=1) - dataFrame['Low']
    logger.debug(f"Created candle features: Body mean {dataFrame['Body'].mean():.2f}, upperWick mean {dataFrame['upperWick'].mean():.2f}, lowerWick mean {dataFrame['lowerWick'].mean():.2f}")
    dataFrame['Doji'] = (abs(dataFrame['Close'] - dataFrame['Open']) / dataFrame['Close'] < 0.001).astype(int)
    logger.debug(f"Created 'Doji' feature: {dataFrame['Doji'].sum()} Doji candles detected")
    
    dataFrame['Bullish_Engulfing'] = (
        (dataFrame['Close'].shift(1) < dataFrame['Open'].shift(1)) &  # Previous bearish candle
        (dataFrame['Close'] > dataFrame['Open']) &  # Current bullish candle
        (dataFrame['Close'] > dataFrame['Open'].shift(1)) &  # Engulfs previous open
        (dataFrame['Open'] < dataFrame['Close'].shift(1))  # Engulfs previous close
    ).astype(int)
    logger.debug(f"Created 'Bullish_Engulfing': {dataFrame['Bullish_Engulfing'].sum()} patterns detected")
    
    dataFrame['Bearish_Engulfing'] = (
        (dataFrame['Close'].shift(1) > dataFrame['Open'].shift(1)) &  # Previous bullish candle
        (dataFrame['Close'] < dataFrame['Open']) &  # Current bearish candle
        (dataFrame['Close'] < dataFrame['Open'].shift(1)) &  # Engulfs previous open
        (dataFrame['Open'] > dataFrame['Close'].shift(1))  # Engulfs previous close
    ).astype(int)
    logger.debug(f"Created 'Bearish_Engulfing': {dataFrame['Bearish_Engulfing'].sum()} patterns detected")
    
    dataFrame['Hammer'] = (
        (dataFrame['Body'].abs() < 0.3 * (dataFrame['High'] - dataFrame['Low'])) &  # Small body
        (dataFrame['lowerWick'] > 2 * dataFrame['upperWick']) &  # Long lower wick
        (dataFrame['lowerWick'] > 0.5 * dataFrame['Body'].abs())  # Lower wick dominates
    ).astype(int)
    logger.debug(f"Created 'Hammer': {dataFrame['Hammer'].sum()} patterns detected")
    
    dataFrame['Shooting_Star'] = (
        (dataFrame['Body'].abs() < 0.3 * (dataFrame['High'] - dataFrame['Low'])) &  # Small body
        (dataFrame['upperWick'] > 2 * dataFrame['lowerWick']) &  # Long upper wick
        (dataFrame['upperWick'] > 0.5 * dataFrame['Body'].abs())  # Upper wick dominates
    ).astype(int)
    logger.debug(f"Created 'Shooting_Star': {dataFrame['Shooting_Star'].sum()} patterns detected")
    
    dataFrame['Bullish_Trend'] = (dataFrame['Close'] > dataFrame['Open']).rolling(5).mean()
    logger.debug(f"Created 'Bullish_Trend' with mean: {dataFrame['Bullish_Trend'].mean():.2f}")
    dataFrame['Bearish_Trend'] = (dataFrame['Close'] < dataFrame['Open']).rolling(5).mean()
    logger.debug(f"Created 'Bearish_Trend' with mean: {dataFrame['Bearish_Trend'].mean():.2f}")
    
    dataFrame['Wick_to_Body_Ratio'] = (dataFrame['upperWick'] + dataFrame['lowerWick']) / (dataFrame['Body'].abs() + 1e-6)
    logger.debug(f"Created 'Wick_to_Body_Ratio' with mean: {dataFrame['Wick_to_Body_Ratio'].mean():.2f}, std: {dataFrame['Wick_to_Body_Ratio'].std():.2f}")
    
    dataFrame['Consecutive_Bullish'] = (dataFrame['Close'] > dataFrame['Open']).rolling(3).sum()
    dataFrame['Consecutive_Bearish'] = (dataFrame['Close'] < dataFrame['Open']).rolling(3).sum()
    logger.debug(f"Created 'Consecutive_Bullish' with mean: {dataFrame['Consecutive_Bullish'].mean():.2f}")
    logger.debug(f"Created 'Consecutive_Bearish' with mean: {dataFrame['Consecutive_Bearish'].mean():.2f}")

    # -------- Momentum/ Trend Indicators -------- #
    dataFrame['RSI'] = ta.rsi(dataFrame['Close'], length=14)
    logger.debug(f"Created 'RSI' with mean: {dataFrame['RSI'].mean():.2f}, std: {dataFrame['RSI'].std():.2f}")
    dataFrame['rsi_momentum'] = dataFrame['RSI'].diff(5)
    
    macd_data = ta.macd(dataFrame['Close'])
    dataFrame['MACD'] = macd_data['MACD_12_26_9']
    dataFrame['MACDSignal'] = macd_data['MACDs_12_26_9']
    logger.debug(f"Created MACD features: MACD range [{dataFrame['MACD'].min():.2f}, {dataFrame['MACD'].max():.2f}], MACDSignal range [{dataFrame['MACDSignal'].min():.2f}, {dataFrame['MACDSignal'].max():.2f}]")

    dataFrame['SMA_20'] = dataFrame['Close'].rolling(window=20).mean()
    dataFrame['SMA_50'] = dataFrame['Close'].rolling(window=50).mean()
    dataFrame['SMA_200'] = dataFrame['Close'].rolling(window=200).mean()
    if len(dataFrame) < 200:
        logger.error(f"Insufficient data for SMA_200 calculation: {len(dataFrame)} rows available")
        raise ValueError("Insufficient data for 200-day SMA")
    dataFrame['EMA_12'] = dataFrame['Close'].ewm(span=12, adjust=False).mean()
    dataFrame['EMA_26'] = dataFrame['Close'].ewm(span=26, adjust=False).mean()
    
    dataFrame['rateOfChange'] = (dataFrame['Close'] - dataFrame['Close'].shift(12)) / dataFrame['Close'].shift(12) * 100
    dataFrame['momentum'] = dataFrame['Close'].diff(10)
    
    dataFrame['averageDirectionalIndex'] = ta.adx(dataFrame['High'], dataFrame['Low'], dataFrame['Close'], length=14)['ADX_14']
    
    dataFrame['massIndex'] = ta.massi(dataFrame['High'], dataFrame['Low'], fast=9, slow=25)
    
    dataFrame['commodityChannelIndex'] = ta.cci(dataFrame['High'], dataFrame['Low'], dataFrame['Close'], length=20)
    
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
    
    dataFrame['Stochastic_K'] = ta.stoch(dataFrame['High'], dataFrame['Low'], dataFrame['Close'], k=14, d=3, smooth_k=3)['STOCHk_14_3_3']
    dataFrame['Stochastic_D'] = ta.stoch(dataFrame['High'], dataFrame['Low'], dataFrame['Close'], k=14, d=3, smooth_k=3)['STOCHd_14_3_3']
    logger.debug(f"Created 'Stochastic_K' with mean: {dataFrame['Stochastic_K'].mean():.2f}, std: {dataFrame['Stochastic_K'].std():.2f}")
    logger.debug(f"Created 'Stochastic_D' with mean: {dataFrame['Stochastic_D'].mean():.2f}, std: {dataFrame['Stochastic_D'].std():.2f}")
    
    dataFrame['Price_RSI_Divergence'] = dataFrame['momentum_5d'] - dataFrame['RSI'].diff(5)
    logger.debug(f"Created 'Price_RSI_Divergence' with mean: {dataFrame['Price_RSI_Divergence'].mean():.2f}, std: {dataFrame['Price_RSI_Divergence'].std():.2f}")
    
    dataFrame['Reversal_Score'] = (
        (dataFrame['RSI'] < 30).astype(int) - (dataFrame['RSI'] > 70).astype(int) +  # Oversold/overbought
        (dataFrame['MACD'] > dataFrame['MACDSignal']).astype(int) - (dataFrame['MACD'] < dataFrame['MACDSignal']).astype(int) +  # MACD crossover
        (dataFrame['Stochastic_K'] < 20).astype(int) - (dataFrame['Stochastic_K'] > 80).astype(int)  # Stochastic thresholds
    )
    logger.debug(f"Created 'Reversal_Score' with range: [{dataFrame['Reversal_Score'].min():.2f}, {dataFrame['Reversal_Score'].max():.2f}]")

    # --------- Volatility Indicators --------- #
    bb_data = ta.bbands(dataFrame['Close'], length=20, std=2)
    dataFrame['BBHigh'] = bb_data['BBU_20_2.0']
    dataFrame['BBLow'] = bb_data['BBL_20_2.0']
    logger.debug(f"Created Bollinger Bands: BBHigh mean {dataFrame['BBHigh'].mean():.2f}, BBLow mean {dataFrame['BBLow'].mean():.2f}")
    
    dataFrame['priceChange'] = dataFrame['Close'].diff()
    dataFrame['closeChangePercentage'] = dataFrame['Close'].pct_change()
    
    dataFrame['averageTrueRange'] = ta.atr(dataFrame['High'], dataFrame['Low'], dataFrame['Close'], length=14)
    dataFrame['ATR_Normalized_Change'] = dataFrame['closeChangePercentage'] / (dataFrame['averageTrueRange'] / dataFrame['Close'])
    logger.debug(f"Created 'ATR_Normalized_Change' with mean: {dataFrame['ATR_Normalized_Change'].mean():.2f}, std: {dataFrame['ATR_Normalized_Change'].std():.2f}")
    
    dataFrame['rollingStd'] = dataFrame['Close'].rolling(window=14).std()
    
    aroon_data = ta.aroon(dataFrame['High'], dataFrame['Low'], length=14)
    dataFrame['aroonUp'] = aroon_data['AROONU_14']
    dataFrame['aroonDown'] = aroon_data['AROOND_14']
    
    dataFrame['ulcerIndex'] = ta.ui(dataFrame['Close'], length=14)
    
    dataFrame['vix_normalized'] = (dataFrame['VIX_Proxy'] - dataFrame['VIX_Proxy'].rolling(126).mean()) / dataFrame['VIX_Proxy'].rolling(126).std()  # Reduced from 252
    dataFrame['vix_regime'] = (dataFrame['VIX_Proxy'] > dataFrame['VIX_Proxy'].rolling(126).quantile(0.75)).astype(int)
    
    dataFrame['Volatility_Breakout_Up'] = (dataFrame['Close'] > dataFrame['BBHigh'] + dataFrame['averageTrueRange']).astype(int)
    logger.debug(f"Created 'Volatility_Breakout_Up': {dataFrame['Volatility_Breakout_Up'].sum()} events")
    dataFrame['Volatility_Breakout_Down'] = (dataFrame['Close'] < dataFrame['BBLow'] - dataFrame['averageTrueRange']).astype(int)
    logger.debug(f"Created 'Volatility_Breakout_Down': {dataFrame['Volatility_Breakout_Down'].sum()} events")

    # --------- Volume Indicators --------- #
    dataFrame['volumeChange'] = dataFrame['Volume'].diff()
    dataFrame['volumeRateOfChange'] = (dataFrame['Volume'] - dataFrame['Volume'].shift(12)) / dataFrame['Volume'].shift(12) * 100
    
    dataFrame['onBalanceVolume'] = ta.obv(dataFrame['Close'], dataFrame['Volume'])
    logger.debug(f"Created 'onBalanceVolume' with mean: {dataFrame['onBalanceVolume'].mean():.2f}, std: {dataFrame['onBalanceVolume'].std():.2f}")

    dataFrame['chaikinMoneyFlow'] = ta.cmf(dataFrame['High'], dataFrame['Low'], dataFrame['Close'], dataFrame['Volume'], length=20)
    logger.debug(f"Created 'chaikinMoneyFlow' with mean: {dataFrame['chaikinMoneyFlow'].mean():.2f}, std: {dataFrame['chaikinMoneyFlow'].std():.2f}")

    # Force Index - using custom calculation as pandas-ta might not have exact equivalent
    dataFrame['forceIndex'] = (dataFrame['Close'] - dataFrame['Close'].shift(1)) * dataFrame['Volume']
    dataFrame['forceIndex'] = dataFrame['forceIndex'].rolling(window=20).mean()
    logger.debug(f"Created 'forceIndex' with mean: {dataFrame['forceIndex'].mean():.2f}, std: {dataFrame['forceIndex'].std():.2f}")

    dataFrame['Volume_Breakout_Up'] = (
        (dataFrame['Close'] > dataFrame['BBHigh']) & 
        (dataFrame['volume_surge'] > 1.5)
    ).astype(int)
    logger.debug(f"Created 'Volume_Breakout_Up': {dataFrame['Volume_Breakout_Up'].sum()} events")
    dataFrame['Volume_Breakout_Down'] = (
        (dataFrame['Close'] < dataFrame['BBLow']) & 
        (dataFrame['volume_surge'] > 1.5)
    ).astype(int)
    logger.debug(f"Created 'Volume_Breakout_Down': {dataFrame['Volume_Breakout_Down'].sum()} events")
    
    dataFrame['Volume_Price_Divergence'] = (
    dataFrame['volume_surge'] - dataFrame['momentum_5d'].abs()
    )
    logger.debug(f"Created 'Volume_Price_Divergence' with mean: {dataFrame['Volume_Price_Divergence'].mean():.2f}, std: {dataFrame['Volume_Price_Divergence'].std():.2f}")

    # Negative Volume Index - using custom calculation
    nvi = []
    nvi_value = 1000  # Starting value
    for i in range(len(dataFrame)):
        if i == 0:
            nvi.append(nvi_value)
        else:
            if dataFrame['Volume'].iloc[i] < dataFrame['Volume'].iloc[i-1]:
                nvi_value = nvi_value * (dataFrame['Close'].iloc[i] / dataFrame['Close'].iloc[i-1])
            nvi.append(nvi_value)
    dataFrame['negativeVolumeIndex'] = nvi
    logger.debug(f"Created 'negativeVolumeIndex' with mean: {dataFrame['negativeVolumeIndex'].mean():.2f}, std: {dataFrame['negativeVolumeIndex'].std():.2f}")

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
    logger.debug(f"Created 'ai_momentum_strength' with range: [{dataFrame['ai_momentum_strength'].min():.2f}, {dataFrame['ai_momentum_strength'].max():.2f}]")

    dataFrame['momentum_acceleration'] = (
        dataFrame['ai_momentum_strength'].diff(1) + 
        dataFrame['ai_momentum_strength'].diff(2)
    ) / 2
    logger.debug(f"Created 'momentum_acceleration' with mean: {dataFrame['momentum_acceleration'].mean():.2f}, std: {dataFrame['momentum_acceleration'].std():.2f}")

    dataFrame['confirmed_momentum'] = (
        dataFrame['ai_momentum_strength'] * 
        (dataFrame['volume_surge'] > 1).astype(float)
    )
    logger.debug(f"Created 'confirmed_momentum' with mean: {dataFrame['confirmed_momentum'].mean():.2f}, std: {dataFrame['confirmed_momentum'].std():.2f}")

    # --------- Cross-Asset Correlations --------- #
    dataFrame['vix_spread'] = dataFrame['VIX_Proxy'] - dataFrame['VIX_Proxy'].rolling(20).mean()
    dataFrame['spy_nvda_correlation'] = dataFrame['Close'].pct_change().rolling(20).corr(dataFrame['SPY_Close'].pct_change())
    dataFrame['qqq_nvda_correlation'] = dataFrame['Close'].pct_change().rolling(20).corr(dataFrame['QQQ_Close'].pct_change())
    dataFrame['treasury_equity_spread'] = dataFrame['Treasury_10Y'].diff() * -1  # Inverse relationship with equities
    
    dataFrame['Sector_Direction_Divergence'] = (
        dataFrame['momentum_5d'] - dataFrame['SOXX_Close'].pct_change(5)
    )
    logger.debug(f"Created 'Sector_Direction_Divergence' with mean: {dataFrame['Sector_Direction_Divergence'].mean():.2f}, std: {dataFrame['Sector_Direction_Divergence'].std():.2f}")
    
    dataFrame['VIX_Direction_Signal'] = (dataFrame['VIX_Proxy'] > dataFrame['VIX_Proxy'].rolling(20).quantile(0.9)).astype(int)
    logger.debug(f"Created 'VIX_Direction_Signal': {dataFrame['VIX_Direction_Signal'].sum()} high VIX events")
    
    # --------- Trend and Breakout Indicators --------- #
    dataFrame['trend_strength'] = (dataFrame['Close'] > dataFrame['SMA_200']).rolling(20).mean()
    dataFrame['bull_market_intensity'] = (
        (dataFrame['Close'] > dataFrame['SMA_50']).astype(int) +
        (dataFrame['SMA_50'] > dataFrame['SMA_200']).astype(int) +
        (dataFrame['Close'].pct_change(20) > 0.1).astype(int)
    )
    
    dataFrame['price_vs_bb_position'] = (dataFrame['Close'] - dataFrame['BBLow']) / (dataFrame['BBHigh'] - dataFrame['BBLow'])
    dataFrame['breakout_signal'] = (dataFrame['Close'] > dataFrame['BBHigh']).astype(int)
    
    dataFrame['Time_Since_Breakout'] = 0
    breakout_indices = dataFrame[(dataFrame['breakout_signal'] == 1) | 
                            (dataFrame['Volatility_Breakout_Up'] == 1) | 
                            (dataFrame['Volatility_Breakout_Down'] == 1)].index
    for i in range(len(dataFrame)):
        if i == 0:
            continue
    last_breakout = max([idx for idx in breakout_indices if idx <= dataFrame.index[i]], default=dataFrame.index[0])
    # Use .loc to avoid chained-assignment and to be compatible with pandas Copy-on-Write
    dataFrame.loc[dataFrame.index[i], 'Time_Since_Breakout'] = (dataFrame.index[i] - last_breakout).days
    logger.debug(f"Created 'Time_Since_Breakout' with mean: {dataFrame['Time_Since_Breakout'].mean():.2f}")

    # --------- Seasonal Indicators --------- #
    dataFrame['days_in_quarter'] = (dataFrame.index.dayofyear % 90)
    dataFrame['earnings_proximity'] = np.where(
        dataFrame['days_in_quarter'].isin([1, 2, 3, 88, 89, 0]), 1, 0
    )
    
    # Shift target column by -1 for next-day prediction BEFORE dropping NaN
    dataFrame['Price_Target'] = dataFrame['Close'].shift(-1)
    logger.debug("Shifted 'Price_Target' column for next-day prediction")
    dataFrame['direction_pct'] = ((dataFrame['Price_Target'] - dataFrame['Close']) / dataFrame['Close']) * 100
    logger.debug(f"Calculated 'direction_pct' with mean: {dataFrame['direction_pct'].mean():.2f}, std: {dataFrame['direction_pct'].std():.2f}")

    threshold = dataFrame['averageTrueRange'].rolling(20).mean() * 1.5
    dataFrame['Direction_Target'] = 1  # Default to sideways
    dataFrame.loc[dataFrame['direction_pct'] > threshold, 'Direction_Target'] = 2  # Strong up
    dataFrame.loc[(-threshold < dataFrame['direction_pct']) & (dataFrame['direction_pct'] < threshold), 'Direction_Target'] = 1  # Sideways
    dataFrame.loc[dataFrame['direction_pct'] < -threshold, 'Direction_Target'] = 0  # Strong down
    logger.debug("Created 'Direction_Target' before shifting")
    dataFrame['Direction_Target'] = dataFrame['Direction_Target'].shift(-1)
    logger.debug("Shifted 'Direction_Target' to align with current day's features")

    nan_count = dataFrame.isna().sum().sum()
    total_values = dataFrame.size
    if nan_count > 0:
        nan_pct = (nan_count / total_values) * 100
        logger.warning(f"Found {nan_count} NaN values in final DataFrame ({nan_pct:.2f}% of all values)")
    dataFrame.dropna(inplace=True)  # Drop NaN values after shifting target
    logger.debug(f"Rows dropped after NaN removal: {dataFrame.shape[0]}")
    dataFrame['Direction_Target'] = dataFrame['Direction_Target'].astype(int)
    logger.info("Converted 'Direction_Target' to integer type")

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
        script_dir = Path(__file__).parent  # /path/to/repo/src
        repo_root = script_dir.parent  # /path/to/repo/
        raw_data_path = Path(repo_root / cfg.data_loader.raw_data_path).resolve()
        save_data_path = Path(repo_root / cfg.features.preprocessing_data_path).resolve()

        logger.info(f"Reading raw data from: {raw_data_path.absolute()}")
        dataFrame = pd.read_csv(raw_data_path, header=0, index_col=0, parse_dates=True)
        feature_engineering(dataFrame, cfg, save_data_path)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise
if __name__ == "__main__":
    main()