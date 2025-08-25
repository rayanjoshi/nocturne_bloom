"""
Feature engineering pipeline for NVDA stock time-series data.

This module implements a comprehensive feature engineering process 
designed to extract technical, statistical, and macroeconomic features 
from raw financial time-series data. It prepares the dataset for 
machine learning applications such as price movement classification.

The pipeline includes:
- Candlestick pattern detection
- Trend and momentum indicators
- Volatility and volume analytics
- Sector-relative strength measures
- Cross-asset correlations and macroeconomic signals
- Seasonal and gap-related features
- Adaptive target label generation

Features are added to the raw input DataFrame and saved to disk for 
subsequent modeling steps. Logging is integrated for traceability 
throughout the process.
"""
from pathlib import Path
from typing import Optional
import pandas as pd
import pandas_ta as ta
import hydra
from omegaconf import DictConfig
import numpy as np
from scripts.logging_config import get_logger, setup_logging, log_function_start, log_function_end

def feature_engineering(df, save_data_path):
    """
    Perform feature engineering on financial time series data for NVDA stock.
    
    This function creates a comprehensive set of technical, statistical, and
    sentiment-based features using the input price and volume data. Features
    include candlestick patterns, trend indicators, volatility measures, volume
    metrics, sector strength proxies, and more. It also handles data cleaning,
    target label generation, and exports the processed df to a CSV file.
    
    Args:
        df (pd.df): The raw input df containing columns such as
            'Open', 'High', 'Low', 'Close', 'Volume', and various macro indicators.
        cfg (DictConfig): Hydra configuration object with data processing settings.
        save_data_path (Path or str): File path to save the processed df as CSV.
    
    Returns:
        pd.df: The processed df with engineered features and target labels.
    
    Raises:
        ValueError: If there is insufficient data for computing long-term indicators
            such as the 200-day SMA.
    
    Notes:
        - The target label 'Direction_Target' is a 3-class label indicating
            bullish (2), bearish (0), or sideways (1) movement, based on adaptive thresholds.
    """
    logger = log_function_start("feature_engineering",df=df, save_data_path=save_data_path)
    logger.info("Starting feature engineering...")

    columns = [
        'Close', 'Open', 'High', 'Low', 'Volume',
        'PB_Ratio', 'PE_Ratio', 'SPY_Close', 'QQQ_Close',
        'SOXX_Close', 'VIX_Proxy', 'Treasury_10Y',
    ]
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Ensure no NaN values before calculations
    nan_count = df.isna().sum().sum()
    total_values = df.size
    if nan_count > 0:
        nan_pct = (nan_count / total_values) * 100
        logger.warning(f"Found {nan_count} NaN values in initial df ({nan_pct:.2f}% of all values)")
    df.dropna(inplace=True)
    logger.debug(f"Rows dropped after NaN removal: {df.shape[0]}")

    # Create a dictionary to store all new features
    new_features = {}

    # -------- Candle Indicators --------- #
    new_features['Body'] = df['Close'] - df['Open']
    new_features['upperWick'] = df['High'] - df[['Close', 'Open']].max(axis=1)
    new_features['lowerWick'] = df[['Close', 'Open']].min(axis=1) - df['Low']
    b_mean = new_features['Body'].mean()
    u_mean = new_features['upperWick'].mean()
    l_mean = new_features['lowerWick'].mean()
    logger.debug(
        f"Created candle features: Body mean {b_mean:.2f}, "
        f"upperWick mean {u_mean:.2f}, "
        f"lowerWick mean {l_mean:.2f}"
    )
    new_features['Doji'] = (abs(df['Close'] - df['Open']) / df['Close'] < 0.001).astype(int)
    logger.debug(f"Created 'Doji' feature: {new_features['Doji'].sum()} Doji candles detected")

    new_features['Bullish_Engulfing'] = (
        (df['Close'].shift(1) < df['Open'].shift(1)) &  # Previous bearish candle
        (df['Close'] > df['Open']) &  # Current bullish candle
        (df['Close'] > df['Open'].shift(1)) &  # Engulfs previous open
        (df['Open'] < df['Close'].shift(1))  # Engulfs previous close
    ).astype(int)
    count_bullish_engulf = int(new_features['Bullish_Engulfing'].sum())
    logger.debug(f"Created 'Bullish_Engulfing': {count_bullish_engulf} patterns detected")

    new_features['Bearish_Engulfing'] = (
        (df['Close'].shift(1) > df['Open'].shift(1)) &  # Previous bullish candle
        (df['Close'] < df['Open']) &  # Current bearish candle
        (df['Close'] < df['Open'].shift(1)) &  # Engulfs previous open
        (df['Open'] > df['Close'].shift(1))  # Engulfs previous close
    ).astype(int)
    count_bearish_engulf = int(new_features['Bearish_Engulfing'].sum())
    logger.debug(f"Created 'Bearish_Engulfing': {count_bearish_engulf} patterns detected")

    new_features['Hammer'] = (
        (new_features['Body'].abs() < 0.3 * (df['High'] - df['Low'])) &  # Small body
        (new_features['lowerWick'] > 2 * new_features['upperWick']) &  # Long lower wick
        (new_features['lowerWick'] > 0.5 * new_features['Body'].abs())  # Lower wick dominates
    ).astype(int)
    logger.debug(f"Created 'Hammer': {new_features['Hammer'].sum()} patterns detected")

    new_features['Shooting_Star'] = (
        (new_features['Body'].abs() < 0.3 * (df['High'] - df['Low'])) &  # Small body
        (new_features['upperWick'] > 2 * new_features['lowerWick']) &  # Long upper wick
        (new_features['upperWick'] > 0.5 * new_features['Body'].abs())  # Upper wick dominates
    ).astype(int)
    logger.debug(
        f"Created 'Shooting_Star': "
        f"{new_features['Shooting_Star'].sum()} patterns detected"
    )

    new_features['Bullish_Trend'] = (df['Close'] > df['Open']).rolling(5).mean()
    logger.debug(f"Created 'Bullish_Trend' with mean: {new_features['Bullish_Trend'].mean():.2f}")
    new_features['Bearish_Trend'] = (df['Close'] < df['Open']).rolling(5).mean()
    logger.debug(f"Created 'Bearish_Trend' with mean: {new_features['Bearish_Trend'].mean():.2f}")

    new_features['Wick_to_Body_Ratio'] = (
        (new_features['upperWick'] + new_features['lowerWick'])
        / (new_features['Body'].abs() + 1e-6)
    )
    mean_wbr = new_features['Wick_to_Body_Ratio'].mean()
    std_wbr = new_features['Wick_to_Body_Ratio'].std()
    logger.debug(f"Created 'Wick_to_Body_Ratio' with mean: {mean_wbr:.2f}, std: {std_wbr:.2f}")

    new_features['Consecutive_Bullish'] = (df['Close'] > df['Open']).rolling(3).sum()
    new_features['Consecutive_Bearish'] = (df['Close'] < df['Open']).rolling(3).sum()
    cb_mean = new_features['Consecutive_Bullish'].mean()
    logger.debug(f"Created 'Consecutive_Bullish' with mean: {cb_mean:.2f}")
    mean_consecutive_bear = new_features['Consecutive_Bearish'].mean()
    logger.debug(
        f"Created 'Consecutive_Bearish' with mean: {mean_consecutive_bear:.2f}"
    )

    # -------- Momentum/ Trend Indicators -------- #
    new_features['RSI'] = ta.rsi(df['Close'], length=14)
    rsi_mean = new_features['RSI'].mean()
    rsi_std = new_features['RSI'].std()
    logger.debug(f"Created 'RSI' with mean: {rsi_mean:.2f}, std: {rsi_std:.2f}")
    new_features['rsi_momentum'] = new_features['RSI'].diff(5)

    macd_data = ta.macd(df['Close'])
    new_features['MACD'] = macd_data['MACD_12_26_9']
    new_features['MACDSignal'] = macd_data['MACDs_12_26_9']
    macd_min = new_features['MACD'].min()
    macd_max = new_features['MACD'].max()
    macd_sig_min = new_features['MACDSignal'].min()
    macd_sig_max = new_features['MACDSignal'].max()
    logger.debug(
        f"Created MACD features: MACD range [{macd_min:.2f}, {macd_max:.2f}], "
        f"MACDSignal range [{macd_sig_min:.2f}, {macd_sig_max:.2f}]"
    )

    new_features['SMA_20'] = df['Close'].rolling(window=20).mean()
    new_features['SMA_50'] = df['Close'].rolling(window=50).mean()
    new_features['SMA_200'] = df['Close'].rolling(window=200).mean()
    if len(df) < 200:
        logger.error(f"Insufficient data for SMA_200 calculation: {len(df)} rows available")
        raise ValueError("Insufficient data for 200-day SMA")
    new_features['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    new_features['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

    prev_close_12 = df['Close'].shift(12)
    new_features['rateOfChange'] = (
        (df['Close'] - prev_close_12)
        / (prev_close_12 + 1e-6)
    ) * 100
    new_features['momentum'] = df['Close'].diff(10)

    adx_df = ta.adx(df['High'], df['Low'], df['Close'], length=14)
    new_features['averageDirectionalIndex'] = adx_df['ADX_14']
    logger.debug(
        f"Created 'averageDirectionalIndex' "
        f"mean: {new_features['averageDirectionalIndex'].mean():.2f}, "
        f"std: {new_features['averageDirectionalIndex'].std():.2f}"
    )

    new_features['massIndex'] = ta.massi(df['High'], df['Low'], fast=9, slow=25)

    new_features['commodityChannelIndex'] = ta.cci(df['High'], df['Low'], df['Close'], length=20)

    new_features['price_acceleration'] = df['Close'].pct_change(5) - df['Close'].pct_change(10)

    new_features['volume_surge'] = df['Volume'] / df['Volume'].rolling(20).mean()
    new_features['volume_momentum'] = new_features['volume_surge'].diff(3)

    # Multi-timeframe momentum
    new_features['momentum_1d'] = df['Close'].pct_change(1)
    new_features['momentum_5d'] = df['Close'].pct_change(5)
    new_features['momentum_10d'] = df['Close'].pct_change(10)
    new_features['momentum_consistency'] = (
        (new_features['momentum_1d'] > 0).rolling(5).sum() / 5
    )

    stoch = ta.stoch(
        df['High'], df['Low'], df['Close'], k=14, d=3, smooth_k=3
    )
    new_features['Stochastic_K'] = stoch['STOCHk_14_3_3']
    new_features['Stochastic_D'] = stoch['STOCHd_14_3_3']

    k_mean = new_features['Stochastic_K'].mean()
    k_std = new_features['Stochastic_K'].std()
    d_mean = new_features['Stochastic_D'].mean()
    d_std = new_features['Stochastic_D'].std()

    logger.debug(
        f"Created 'Stochastic_K' mean {k_mean:.2f} std {k_std:.2f}"
    )
    logger.debug(
        f"Created 'Stochastic_D' mean {d_mean:.2f} std {d_std:.2f}"
    )

    new_features['Price_RSI_Divergence'] = new_features['momentum_5d'] - new_features['RSI'].diff(5)
    logger.debug(
        "Created 'Price_RSI_Divergence' "
        f"mean: {new_features['Price_RSI_Divergence'].mean():.2f}, "
        f"std: {new_features['Price_RSI_Divergence'].std():.2f}"
    )

    rsi_buy = (new_features['RSI'] < 30).astype(int)
    rsi_sell = (new_features['RSI'] > 70).astype(int)
    macd_pos = (new_features['MACD'] > new_features['MACDSignal']).astype(int)
    macd_neg = (new_features['MACD'] < new_features['MACDSignal']).astype(int)
    stoch_buy = (new_features['Stochastic_K'] < 20).astype(int)
    stoch_sell = (new_features['Stochastic_K'] > 80).astype(int)

    new_features['Reversal_Score'] = (
        rsi_buy - rsi_sell +
        macd_pos - macd_neg +
        stoch_buy - stoch_sell
    )
    min_rs = new_features['Reversal_Score'].min()
    max_rs = new_features['Reversal_Score'].max()
    logger.debug(
        f"Created 'Reversal_Score' with range: [{min_rs:.2f}, {max_rs:.2f}]"
    )

    # --------- Volatility Indicators --------- #
    bb_data = ta.bbands(df['Close'], length=20, std=2)
    new_features['BBHigh'] = bb_data['BBU_20_2.0']
    new_features['BBLow'] = bb_data['BBL_20_2.0']
    bbhigh_mean = new_features['BBHigh'].mean()
    bblow_mean = new_features['BBLow'].mean()
    logger.debug(
        f"Created Bollinger Bands: BBHigh mean {bbhigh_mean:.2f}, BBLow mean {bblow_mean:.2f}",
    )

    new_features['priceChange'] = df['Close'].diff()
    new_features['closeChangePercentage'] = df['Close'].pct_change()

    new_features['averageTrueRange'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    # Normalize ATR change by price to avoid a long single line
    atr_ratio = new_features['averageTrueRange'] / df['Close']
    new_features['ATR_Normalized_Change'] = new_features['closeChangePercentage'] / atr_ratio
    mean_atr = new_features['ATR_Normalized_Change'].mean()
    std_atr = new_features['ATR_Normalized_Change'].std()
    logger.debug(f"Created 'ATR_Normalized_Change' with mean: {mean_atr:.2f}, std: {std_atr:.2f}")

    new_features['rollingStd'] = df['Close'].rolling(window=14).std()

    aroon_data = ta.aroon(df['High'], df['Low'], length=14)
    new_features['aroonUp'] = aroon_data['AROONU_14']
    new_features['aroonDown'] = aroon_data['AROOND_14']

    new_features['ulcerIndex'] = ta.ui(df['Close'], length=14)

    vix_roll = df['VIX_Proxy'].rolling(126)
    vix_mean = vix_roll.mean()
    vix_std = vix_roll.std() + 1e-6  # avoid division-by-zero
    new_features['vix_normalized'] = (df['VIX_Proxy'] - vix_mean) / vix_std
    vix_q75 = vix_roll.quantile(0.75)
    new_features['vix_regime'] = (df['VIX_Proxy'] > vix_q75).astype(int)
    logger.debug(f"Created 'vix_regime': {int(new_features['vix_regime'].sum())} high VIX periods")

    vol_up_threshold = new_features['BBHigh'] + new_features['averageTrueRange']
    new_features['Volatility_Breakout_Up'] = (df['Close'] > vol_up_threshold).astype(int)
    vol_up_count = int(new_features['Volatility_Breakout_Up'].sum())
    logger.debug(f"Created 'Volatility_Breakout_Up': {vol_up_count} events")
    # Volatility breakout down: close below BBLow minus ATR
    vol_breakout_down = df['Close'] < (new_features['BBLow'] - new_features['averageTrueRange'])
    new_features['Volatility_Breakout_Down'] = vol_breakout_down.astype(int)
    count_vol_breakout_down = int(new_features['Volatility_Breakout_Down'].sum())
    logger.debug(
        f"Created 'Volatility_Breakout_Down': {count_vol_breakout_down} events",
    )

    # --------- Volume Indicators --------- #
    new_features['volumeChange'] = df['Volume'].diff()
    new_features['volumeRateOfChange'] = (
        (df['Volume'] - df['Volume'].shift(12))
        / df['Volume'].shift(12)
        * 100
    )

    new_features['onBalanceVolume'] = ta.obv(df['Close'], df['Volume'])
    mean_obv = new_features['onBalanceVolume'].mean()
    std_obv = new_features['onBalanceVolume'].std()
    logger.debug(
        f"Created 'onBalanceVolume' with mean: {mean_obv:.2f}, std: {std_obv:.2f}",
    )

    cmf = ta.cmf(
        df['High'],
        df['Low'],
        df['Close'],
        df['Volume'],
        length=20,
    )
    new_features['chaikinMoneyFlow'] = cmf
    cmf_mean = new_features['chaikinMoneyFlow'].mean()
    cmf_std = new_features['chaikinMoneyFlow'].std()
    logger.debug(
        f"Created 'chaikinMoneyFlow' with mean: {cmf_mean:.2f}, std: {cmf_std:.2f}",
    )

    # Force Index - using custom calculation as pandas-ta might not have exact equivalent
    new_features['forceIndex'] = (df['Close'] - df['Close'].shift(1)) * df['Volume']
    new_features['forceIndex'] = new_features['forceIndex'].rolling(window=20).mean()
    fi_mean = new_features['forceIndex'].mean()
    fi_std = new_features['forceIndex'].std()
    logger.debug(
        f"Created 'forceIndex' with mean: {fi_mean:.2f}, "
        f"std: {fi_std:.2f}"
    )

    new_features['Volume_Breakout_Up'] = (
        (df['Close'] > new_features['BBHigh']) &
        (new_features['volume_surge'] > 1.5)
    ).astype(int)
    logger.debug(f"Created 'Volume_Breakout_Up': {new_features['Volume_Breakout_Up'].sum()} events")
    new_features['Volume_Breakout_Down'] = (
        (df['Close'] < new_features['BBLow']) &
        (new_features['volume_surge'] > 1.5)
    ).astype(int)
    vol_breakout_down_count = int(new_features['Volume_Breakout_Down'].sum())
    logger.debug(f"Created 'Volume_Breakout_Down': {vol_breakout_down_count} events")

    new_features['Volume_Price_Divergence'] = (
    new_features['volume_surge'] - new_features['momentum_5d'].abs()
    )
    mean_vpd = new_features['Volume_Price_Divergence'].mean()
    std_vpd = new_features['Volume_Price_Divergence'].std()
    logger.debug(
        "Created 'Volume_Price_Divergence' "
        f"with mean: {mean_vpd:.2f}, std: {std_vpd:.2f}"
    )

    # Negative Volume Index - using custom calculation
    nvi = []
    nvi_value = 1000  # Starting value
    for i in range(len(df)):
        if i == 0:
            nvi.append(nvi_value)
        else:
            if df['Volume'].iloc[i] < df['Volume'].iloc[i-1]:
                nvi_value = nvi_value * (df['Close'].iloc[i] / df['Close'].iloc[i-1])
            nvi.append(nvi_value)
    # store as a pandas Series so we can call .mean() / .std() on it
    new_features['negativeVolumeIndex'] = pd.Series(nvi, index=df.index)
    mean_nvi = new_features['negativeVolumeIndex'].mean()
    std_nvi = new_features['negativeVolumeIndex'].std()
    logger.debug(
        "Created 'negativeVolumeIndex' with mean: %.2f, std: %.2f",
        mean_nvi, std_nvi
    )

    new_features['rolling_vol_5'] = df['Close'].pct_change().rolling(5).std()
    new_features['rolling_vol_21'] = df['Close'].pct_change().rolling(21).std()
    new_features['vol_percentile_21d'] = new_features['rolling_vol_21'].rolling(126).rank(pct=True)
    new_features['vol_percentile_5d'] = new_features['rolling_vol_5'].rolling(63).rank(pct=True)
    new_features['regime_high_vol'] = (new_features['vol_percentile_21d'] > 0.8).astype(int)

    vix_normalized = df['VIX_Proxy'] / 100.0  # normalize VIX to comparable scale
    new_features['vol_divergence'] = new_features['rolling_vol_21'] - vix_normalized
    logger.debug(
        f"Created 'vol_divergence' mean {new_features['vol_divergence'].mean():.4f}, "
        f"std {new_features['vol_divergence'].std():.4f}"
    )
    new_features['volume_vol'] = df['Volume'].rolling(20).std() / df['Volume'].rolling(20).mean()
    new_features['volume_price_trend'] = df['Volume'].rolling(5).corr(df['Close'])

    new_features['volume_skew'] = df['Volume'].rolling(20).skew()
    new_features['volume_acceleration'] = df['Volume'].pct_change(5)
    new_features['volume_breakout'] = (
        df['Volume'] > df['Volume'].rolling(50).quantile(0.9)
    ).astype(int)

    new_features['volume_price_divergence'] = (
        new_features['volume_surge'] - df['Close'].pct_change().abs().rolling(5).mean()
    )

    # --------- Statistical Indicators --------- #
    new_features['sigma'] = df['Close'].rolling(window=20).std()
    close_rolling = df['Close'].rolling(window=20)
    volume_rolling = df['Volume'].rolling(window=20)
    new_features['beta'] = close_rolling.cov(df['Volume']) / volume_rolling.var()
    new_features['skewness'] = df['Close'].rolling(window=20).skew()

    # --------- Gap Analysis --------- #
    new_features['overnight_gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    new_features['gap_magnitude_avg_5d'] = new_features['overnight_gap'].abs().rolling(5).mean()
    new_features['large_gap_frequency'] = (
        (new_features['overnight_gap'].abs() > 0.03)
        .rolling(20)
        .sum()
    )
    logger.debug(
        f"Created 'large_gap_frequency' with total events: "
        f"{int(new_features['large_gap_frequency'].sum())}"
    )

    # --------- AI/Tech Sector Proxies --------- #
    new_features['tech_sector_rotation'] = df['Close'] / df['SPY_Close'] - 1
    new_features['nasdaq_relative_strength'] = df['Close'] / df['QQQ_Close'] - 1
    new_features['semiconductor_strength'] = df['Close'] / df['SOXX_Close'] - 1

    new_features['spy_qqq_spread'] = (df['QQQ_Close'] / df['SPY_Close']).pct_change()
    new_features['soxx_qqq_spread'] = (df['SOXX_Close'] / df['QQQ_Close']).pct_change()

    qqq_pct = df['QQQ_Close'].pct_change()
    spy_pct = df['SPY_Close'].pct_change()
    soxx_pct = df['SOXX_Close'].pct_change()
    nvda_pct = df['Close'].pct_change()

    new_features['tech_leadership'] = (
        (qqq_pct - spy_pct)
        .rolling(10)
        .mean()
    )

    new_features['semiconductor_leadership'] = (
        (soxx_pct - qqq_pct)
        .rolling(10)
        .mean()
    )

    new_features['nvda_outperformance'] = (
        (nvda_pct - soxx_pct)
        .rolling(5)
        .mean()
    )
    new_features['sector_momentum_divergence'] = (
        df['Close'].pct_change(10) - df['SOXX_Close'].pct_change(10)
    )

    new_features['mega_cap_rotation'] = (
        df['QQQ_Close'].pct_change(5) - df['SPY_Close'].pct_change(5)
    ).rolling(10).mean()

    new_features['momentum_persistence_3d'] = (
        (new_features['nvda_outperformance'] > 0).rolling(3).sum() / 3
    )

    new_features['ai_momentum_strength'] = (
        new_features['nvda_outperformance'] *
        new_features['tech_leadership'] *
        new_features['semiconductor_strength']
    )
    ai_min = new_features['ai_momentum_strength'].min()
    ai_max = new_features['ai_momentum_strength'].max()
    logger.debug(
        "Created 'ai_momentum_strength' with range: "
        f"[{ai_min:.2f}, {ai_max:.2f}]"
    )

    new_features['momentum_acceleration'] = (
        new_features['ai_momentum_strength'].diff(1) +
        new_features['ai_momentum_strength'].diff(2)
        ) / 2
    mean_ma = new_features['momentum_acceleration'].mean()
    std_ma = new_features['momentum_acceleration'].std()
    logger.debug(
        f"Created 'momentum_acceleration' with mean: {mean_ma:.2f}, "
        f"std: {std_ma:.2f}"
    )

    new_features['confirmed_momentum'] = (
        new_features['ai_momentum_strength'] *
        (new_features['volume_surge'] > 1).astype(float)
        )
    cm_mean = new_features['confirmed_momentum'].mean()
    cm_std = new_features['confirmed_momentum'].std()
    logger.debug(
        "Created 'confirmed_momentum' with mean: %.2f, std: %.2f",
        cm_mean, cm_std,
    )

    # --------- Cross-Asset Correlations --------- #
    new_features['vix_spread'] = df['VIX_Proxy'] - df['VIX_Proxy'].rolling(20).mean()
    spy_pct_change = df['SPY_Close'].pct_change()
    nvda_pct_change = df['Close'].pct_change()
    new_features['spy_nvda_correlation'] = (
        nvda_pct_change.rolling(20)
        .corr(spy_pct_change)
    )

    qqq_pct_change = df['QQQ_Close'].pct_change()
    new_features['qqq_nvda_correlation'] = (
        nvda_pct_change.rolling(20)
        .corr(qqq_pct_change)
    )
    new_features['treasury_equity_spread'] = df['Treasury_10Y'].diff() * -1

    new_features['Sector_Direction_Divergence'] = (
        new_features['momentum_5d'] - df['SOXX_Close'].pct_change(5)
    )
    sector_mean = new_features['Sector_Direction_Divergence'].mean()
    sector_std = new_features['Sector_Direction_Divergence'].std()
    logger.debug(
        "Created 'Sector_Direction_Divergence' "
        f"with mean: {sector_mean:.2f}, std: {sector_std:.2f}"
    )

    vix_roll = df['VIX_Proxy'].rolling(20)
    vix_thresh = vix_roll.quantile(0.9)
    new_features['VIX_Direction_Signal'] = (
        df['VIX_Proxy'] > vix_thresh
    ).astype(int)
    logger.debug(
        "Created 'VIX_Direction_Signal': "
        f"{new_features['VIX_Direction_Signal'].sum()} high VIX events"
    )

    # --------- Trend and Breakout Indicators --------- #
    new_features['trend_strength'] = (df['Close'] > new_features['SMA_200']).rolling(20).mean()
    new_features['bull_market_intensity'] = (
        (df['Close'] > new_features['SMA_50']).astype(int) +
        (new_features['SMA_50'] > new_features['SMA_200']).astype(int) +
        (df['Close'].pct_change(20) > 0.1).astype(int)
    )

    # Price position within Bollinger Bands (0=BBLow, 1=BBHigh)
    _bb_range = new_features['BBHigh'] - new_features['BBLow'] + 1e-6
    new_features['price_vs_bb_position'] = (
        (df['Close'] - new_features['BBLow']) / _bb_range
    )
    logger.debug(
        f"Created 'price_vs_bb_position' mean {new_features['price_vs_bb_position'].mean():.4f}, "
        f"std {new_features['price_vs_bb_position'].std():.4f}"
    )
    new_features['breakout_signal'] = (df['Close'] > new_features['BBHigh']).astype(int)

    # Time since breakout calculation
    time_since_breakout = []
    last_breakout_idx = 0
    for i in range(len(df)):
        # Check for breakout at current index
        if (new_features['breakout_signal'].iloc[i] == 1 or
            new_features['Volatility_Breakout_Up'].iloc[i] == 1 or
            new_features['Volatility_Breakout_Down'].iloc[i] == 1):
            last_breakout_idx = i

        # Calculate days since last breakout
        days_since = i - last_breakout_idx
        time_since_breakout.append(days_since)

    new_features['Time_Since_Breakout'] = time_since_breakout
    logger.debug(f"Created 'Time_Since_Breakout' with mean: {np.mean(time_since_breakout):.2f}")

    # --------- Seasonal Indicators --------- #
    new_features['days_in_quarter'] = df.index.dayofyear % 90
    new_features['earnings_proximity'] = np.where(
        new_features['days_in_quarter'].isin([1, 2, 3, 88, 89, 0]), 1, 0
    )

    # Shifted next-day price
    new_features['Price_Target'] = df['Close'].shift(-1)
    logger.debug("Shifted 'Price_Target' column for next-day prediction")

    # Calculate percentage change
    price_target = new_features['Price_Target']
    close = df['Close']
    # compute next-day percent change (with small eps to avoid div-by-zero)
    new_features['direction_pct'] = ((price_target - close) / (close + 1e-6)) * 100
    logger.debug(
        f"Created 'direction_pct' mean: {new_features['direction_pct'].mean():.2f}, "
        f"std: {new_features['direction_pct'].std():.2f}"
    )
    mean_dir_pct = new_features['direction_pct'].mean()
    std_dir_pct = new_features['direction_pct'].std()
    logger.debug(
        f"Calculated 'direction_pct' with mean: {mean_dir_pct:.2f}, std: {std_dir_pct:.2f}"
    )

    # Fixed minimum threshold: 0.5%
    fixed_thresh = 0.5

    # Rolling ATR-based adaptive threshold (20-day window, scaled)
    atr_window = 20
    atr_multiplier = 1.5
    atr_thresh = new_features['averageTrueRange'].rolling(atr_window).mean() * atr_multiplier

    # Combine thresholds: use the max between fixed and ATR-based
    hybrid_thresh = np.maximum(fixed_thresh, atr_thresh)
    logger.debug(f"Hybrid threshold calculated with mean: {hybrid_thresh.mean():.2f}")

    # Step 4: Initialize Direction_Target as sideways (1)
    # Assign directional labels using numpy.where on the Series/array stored in the dict
    new_features['Direction_Target'] = np.where(
        new_features['direction_pct'] > hybrid_thresh, 2,
        np.where(new_features['direction_pct'] < -hybrid_thresh, 0, 1)
    )

    logger.debug("'Direction_Target' already shifted for next day prediction")

    # Convert new_features dict to DataFrame and concatenate with original df
    logger.info("Concatenating all new features to original DataFrame...")
    new_features_df = pd.DataFrame(new_features, index=df.index)

    # Use pd.concat to add all features at once - this avoids fragmentation warnings
    df = pd.concat([df, new_features_df], axis=1)
    logger.info(f"Successfully added {len(new_features)} new features to DataFrame")

    nan_count = df.isna().sum().sum()
    total_values = df.size
    if nan_count > 0:
        nan_pct = (nan_count / total_values) * 100
        logger.warning(f"Found {nan_count} NaN values in final df ({nan_pct:.2f}% of all values)")
    df.dropna(inplace=True)  # Drop NaN values after shifting target
    logger.debug(f"Rows dropped after NaN removal: {df.shape[0]}")
    df['Direction_Target'] = df['Direction_Target'].astype(int)
    logger.info("Converted 'Direction_Target' to integer type")

    # No scaling here - will be done in data_module.py after temporal split
    logger.info("Feature engineering complete - no scaling applied")
    logger.info(
        "Scaling will be performed in data_module.py after "
        "train/val split to avoid data leakage"
    )

    # Save processed data
    save_data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_data_path, index=True)

    logger.info(f"Processed data saved to {save_data_path.absolute()}")
    logger.info("--------- Feature Engineering Statistics ---------")
    logger.info(f"Total features created: {len(df.columns)}")
    logger.info(f"Dataset shape: {df.shape}")
    logger.info("Features will be scaled in data_module.py after train/val split")
    logger.info(f"All features: {list(df.columns)}")
    logger.info("--------- Feature Engineering Completed ---------")
    log_function_end("feature_engineering", df=df, save_data_path=save_data_path)
    return df

@hydra.main(version_base=None, config_path="../configs", config_name="feature_engineering")
def main(cfg: Optional[DictConfig] = None):
    """
    Entry point for the feature engineering pipeline.
    
    This function sets up logging, loads the raw data from the specified path,
    and triggers the feature engineering process. Paths are resolved relative
    to the repository root to ensure portability.
    
    Args:
        cfg (DictConfig): Hydra configuration object containing paths and parameters
            for data loading and preprocessing.
    
    Raises:
        Exception: Propagates any exception raised during the data loading or
        feature engineering process.
    """
    try:
        setup_logging(log_level="INFO", console_output=True, file_output=True)
        logger = get_logger("main")
        # Convert relative path to absolute path within the repository
        script_dir = Path(__file__).parent  # /path/to/repo/src
        repo_root = script_dir.parent  # /path/to/repo/
        raw_data_path = Path(repo_root / cfg.data_loader.raw_data_path).resolve()
        save_data_path = Path(repo_root / cfg.features.preprocessing_data_path).resolve()

        logger.info(f"Reading raw data from: {raw_data_path.absolute()}")
        df = pd.read_csv(raw_data_path, header=0, index_col=0, parse_dates=True)
        feature_engineering(df, save_data_path)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise
if __name__ == "__main__":
    main()
