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

    # -------- Candle Indicators --------- #
    df['Body'] = df['Close'] - df['Open']
    df['upperWick'] = df['High'] - df[['Close', 'Open']].max(axis=1)
    df['lowerWick'] = df[['Close', 'Open']].min(axis=1) - df['Low']
    b_mean = df['Body'].mean()
    u_mean = df['upperWick'].mean()
    l_mean = df['lowerWick'].mean()
    logger.debug(
        f"Created candle features: Body mean {b_mean:.2f}, "
        f"upperWick mean {u_mean:.2f}, "
        f"lowerWick mean {l_mean:.2f}"
    )
    df['Doji'] = (abs(df['Close'] - df['Open']) / df['Close'] < 0.001).astype(int)
    logger.debug(f"Created 'Doji' feature: {df['Doji'].sum()} Doji candles detected")

    df['Bullish_Engulfing'] = (
        (df['Close'].shift(1) < df['Open'].shift(1)) &  # Previous bearish candle
        (df['Close'] > df['Open']) &  # Current bullish candle
        (df['Close'] > df['Open'].shift(1)) &  # Engulfs previous open
        (df['Open'] < df['Close'].shift(1))  # Engulfs previous close
    ).astype(int)
    logger.debug(f"Created 'Bullish_Engulfing': {df['Bullish_Engulfing'].sum()} patterns detected")

    df['Bearish_Engulfing'] = (
        (df['Close'].shift(1) > df['Open'].shift(1)) &  # Previous bullish candle
        (df['Close'] < df['Open']) &  # Current bearish candle
        (df['Close'] < df['Open'].shift(1)) &  # Engulfs previous open
        (df['Open'] > df['Close'].shift(1))  # Engulfs previous close
    ).astype(int)
    logger.debug(f"Created 'Bearish_Engulfing': {df['Bearish_Engulfing'].sum()} patterns detected")

    df['Hammer'] = (
        (df['Body'].abs() < 0.3 * (df['High'] - df['Low'])) &  # Small body
        (df['lowerWick'] > 2 * df['upperWick']) &  # Long lower wick
        (df['lowerWick'] > 0.5 * df['Body'].abs())  # Lower wick dominates
    ).astype(int)
    logger.debug(f"Created 'Hammer': {df['Hammer'].sum()} patterns detected")

    df['Shooting_Star'] = (
        (df['Body'].abs() < 0.3 * (df['High'] - df['Low'])) &  # Small body
        (df['upperWick'] > 2 * df['lowerWick']) &  # Long upper wick
        (df['upperWick'] > 0.5 * df['Body'].abs())  # Upper wick dominates
    ).astype(int)
    logger.debug(f"Created 'Shooting_Star': {df['Shooting_Star'].sum()} patterns detected")

    df['Bullish_Trend'] = (df['Close'] > df['Open']).rolling(5).mean()
    logger.debug(f"Created 'Bullish_Trend' with mean: {df['Bullish_Trend'].mean():.2f}")
    df['Bearish_Trend'] = (df['Close'] < df['Open']).rolling(5).mean()
    logger.debug(f"Created 'Bearish_Trend' with mean: {df['Bearish_Trend'].mean():.2f}")

    df['Wick_to_Body_Ratio'] = (df['upperWick'] + df['lowerWick']) / (df['Body'].abs() + 1e-6)
    mean_wbr = df['Wick_to_Body_Ratio'].mean()
    std_wbr = df['Wick_to_Body_Ratio'].std()
    logger.debug(f"Created 'Wick_to_Body_Ratio' with mean: {mean_wbr:.2f}, std: {std_wbr:.2f}")

    df['Consecutive_Bullish'] = (df['Close'] > df['Open']).rolling(3).sum()
    df['Consecutive_Bearish'] = (df['Close'] < df['Open']).rolling(3).sum()
    logger.debug(f"Created 'Consecutive_Bullish' with mean: {df['Consecutive_Bullish'].mean():.2f}")
    logger.debug(f"Created 'Consecutive_Bearish' with mean: {df['Consecutive_Bearish'].mean():.2f}")

    # -------- Momentum/ Trend Indicators -------- #
    df['RSI'] = ta.rsi(df['Close'], length=14)
    logger.debug(f"Created 'RSI' with mean: {df['RSI'].mean():.2f}, std: {df['RSI'].std():.2f}")
    df['rsi_momentum'] = df['RSI'].diff(5)

    macd_data = ta.macd(df['Close'])
    df['MACD'] = macd_data['MACD_12_26_9']
    df['MACDSignal'] = macd_data['MACDs_12_26_9']
    macd_min = df['MACD'].min()
    macd_max = df['MACD'].max()
    macd_sig_min = df['MACDSignal'].min()
    macd_sig_max = df['MACDSignal'].max()
    logger.debug(
        f"Created MACD features: MACD range [{macd_min:.2f}, {macd_max:.2f}], "
        f"MACDSignal range [{macd_sig_min:.2f}, {macd_sig_max:.2f}]"
    )

    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    if len(df) < 200:
        logger.error(f"Insufficient data for SMA_200 calculation: {len(df)} rows available")
        raise ValueError("Insufficient data for 200-day SMA")
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

    df['rateOfChange'] = (df['Close'] - df['Close'].shift(12)) / df['Close'].shift(12) * 100
    df['momentum'] = df['Close'].diff(10)

    df['averageDirectionalIndex'] = ta.adx(df['High'], df['Low'], df['Close'], length=14)['ADX_14']

    df['massIndex'] = ta.massi(df['High'], df['Low'], fast=9, slow=25)

    df['commodityChannelIndex'] = ta.cci(df['High'], df['Low'], df['Close'], length=20)

    df['price_acceleration'] = df['Close'].pct_change(5) - df['Close'].pct_change(10)

    df['volume_surge'] = df['Volume'] / df['Volume'].rolling(20).mean()
    df['volume_momentum'] = df['volume_surge'].diff(3)

    # Multi-timeframe momentum
    df['momentum_1d'] = df['Close'].pct_change(1)
    df['momentum_5d'] = df['Close'].pct_change(5)
    df['momentum_10d'] = df['Close'].pct_change(10)
    df['momentum_consistency'] = (
        (df['momentum_1d'] > 0).rolling(5).sum() / 5
    )

    stoch = ta.stoch(
        df['High'], df['Low'], df['Close'], k=14, d=3, smooth_k=3
    )
    df['Stochastic_K'] = stoch['STOCHk_14_3_3']
    df['Stochastic_D'] = stoch['STOCHd_14_3_3']

    k_mean = df['Stochastic_K'].mean()
    k_std = df['Stochastic_K'].std()
    d_mean = df['Stochastic_D'].mean()
    d_std = df['Stochastic_D'].std()

    logger.debug(
        f"Created 'Stochastic_K' mean {k_mean:.2f} std {k_std:.2f}"
    )
    logger.debug(
        f"Created 'Stochastic_D' mean {d_mean:.2f} std {d_std:.2f}"
    )

    df['Price_RSI_Divergence'] = df['momentum_5d'] - df['RSI'].diff(5)
    logger.debug(
        "Created 'Price_RSI_Divergence' "
        f"mean: {df['Price_RSI_Divergence'].mean():.2f}, "
        f"std: {df['Price_RSI_Divergence'].std():.2f}"
    )

    df['Reversal_Score'] = (
        (df['RSI'] < 30).astype(int) - (df['RSI'] > 70).astype(int) +  # Oversold/overbought
        (df['MACD'] > df['MACDSignal']).astype(int) - (df['MACD'] < df['MACDSignal']).astype(int) +
        (df['Stochastic_K'] < 20).astype(int) - (df['Stochastic_K'] > 80).astype(int)
    )
    min_rs = df['Reversal_Score'].min()
    max_rs = df['Reversal_Score'].max()
    logger.debug(
        f"Created 'Reversal_Score' with range: [{min_rs:.2f}, {max_rs:.2f}]"
    )

    # --------- Volatility Indicators --------- #
    bb_data = ta.bbands(df['Close'], length=20, std=2)
    df['BBHigh'] = bb_data['BBU_20_2.0']
    df['BBLow'] = bb_data['BBL_20_2.0']
    bbhigh_mean = df['BBHigh'].mean()
    bblow_mean = df['BBLow'].mean()
    logger.debug(
        f"Created Bollinger Bands: BBHigh mean {bbhigh_mean:.2f}, BBLow mean {bblow_mean:.2f}",
    )

    df['priceChange'] = df['Close'].diff()
    df['closeChangePercentage'] = df['Close'].pct_change()

    df['averageTrueRange'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    # Normalize ATR change by price to avoid a long single line
    atr_ratio = df['averageTrueRange'] / df['Close']
    df['ATR_Normalized_Change'] = df['closeChangePercentage'] / atr_ratio
    mean_atr = df['ATR_Normalized_Change'].mean()
    std_atr = df['ATR_Normalized_Change'].std()
    logger.debug(f"Created 'ATR_Normalized_Change' with mean: {mean_atr:.2f}, std: {std_atr:.2f}")

    df['rollingStd'] = df['Close'].rolling(window=14).std()

    aroon_data = ta.aroon(df['High'], df['Low'], length=14)
    df['aroonUp'] = aroon_data['AROONU_14']
    df['aroonDown'] = aroon_data['AROOND_14']

    df['ulcerIndex'] = ta.ui(df['Close'], length=14)

    vix_roll = df['VIX_Proxy'].rolling(126)
    vix_mean = vix_roll.mean()
    vix_std = vix_roll.std() + 1e-6  # avoid division-by-zero
    df['vix_normalized'] = (df['VIX_Proxy'] - vix_mean) / vix_std
    df['vix_regime'] = (df['VIX_Proxy'] > df['VIX_Proxy'].rolling(126).quantile(0.75)).astype(int)

    df['Volatility_Breakout_Up'] = (df['Close'] > df['BBHigh'] + df['averageTrueRange']).astype(int)
    logger.debug(f"Created 'Volatility_Breakout_Up': {df['Volatility_Breakout_Up'].sum()} events")
    # Volatility breakout down: close below BBLow minus ATR
    vol_breakout_down = df['Close'] < (df['BBLow'] - df['averageTrueRange'])
    df['Volatility_Breakout_Down'] = vol_breakout_down.astype(int)
    count_vol_breakout_down = int(df['Volatility_Breakout_Down'].sum())
    logger.debug(
        f"Created 'Volatility_Breakout_Down': {count_vol_breakout_down} events",
    )

    # --------- Volume Indicators --------- #
    df['volumeChange'] = df['Volume'].diff()
    df['volumeRateOfChange'] = (
        (df['Volume'] - df['Volume'].shift(12))
        / df['Volume'].shift(12)
        * 100
    )

    df['onBalanceVolume'] = ta.obv(df['Close'], df['Volume'])
    mean_obv = df['onBalanceVolume'].mean()
    std_obv = df['onBalanceVolume'].std()
    logger.debug(
        f"Created 'onBalanceVolume' with mean: {mean_obv:.2f}, std: {std_obv:.2f}",
    )

    df['chaikinMoneyFlow'] = ta.cmf(df['High'], df['Low'], df['Close'], df['Volume'], length=20)
    cmf_mean = df['chaikinMoneyFlow'].mean()
    cmf_std = df['chaikinMoneyFlow'].std()
    logger.debug(
        f"Created 'chaikinMoneyFlow' with mean: {cmf_mean:.2f}, std: {cmf_std:.2f}",
    )

    # Force Index - using custom calculation as pandas-ta might not have exact equivalent
    df['forceIndex'] = (df['Close'] - df['Close'].shift(1)) * df['Volume']
    df['forceIndex'] = df['forceIndex'].rolling(window=20).mean()
    fi_mean = df['forceIndex'].mean()
    fi_std = df['forceIndex'].std()
    logger.debug(
        f"Created 'forceIndex' with mean: {fi_mean:.2f}, "
        f"std: {fi_std:.2f}"
    )

    df['Volume_Breakout_Up'] = (
        (df['Close'] > df['BBHigh']) &
        (df['volume_surge'] > 1.5)
    ).astype(int)
    logger.debug(f"Created 'Volume_Breakout_Up': {df['Volume_Breakout_Up'].sum()} events")
    df['Volume_Breakout_Down'] = (
        (df['Close'] < df['BBLow']) &
        (df['volume_surge'] > 1.5)
    ).astype(int)
    logger.debug(f"Created 'Volume_Breakout_Down': {df['Volume_Breakout_Down'].sum()} events")

    df['Volume_Price_Divergence'] = (
    df['volume_surge'] - df['momentum_5d'].abs()
    )
    mean_vpd = df['Volume_Price_Divergence'].mean()
    std_vpd = df['Volume_Price_Divergence'].std()
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
    df['negativeVolumeIndex'] = nvi
    mean_nvi = df['negativeVolumeIndex'].mean()
    std_nvi = df['negativeVolumeIndex'].std()
    logger.debug(
        "Created 'negativeVolumeIndex' with mean: %.2f, std: %.2f",
        mean_nvi, std_nvi
    )

    df['rolling_vol_5'] = df['Close'].pct_change().rolling(5).std()
    df['rolling_vol_21'] = df['Close'].pct_change().rolling(21).std()
    df['vol_percentile_21d'] = df['rolling_vol_21'].rolling(126).rank(pct=True)
    df['vol_percentile_5d'] = df['rolling_vol_5'].rolling(63).rank(pct=True)
    df['regime_high_vol'] = (df['vol_percentile_21d'] > 0.8).astype(int)

    df['vol_divergence'] = df['rolling_vol_21'] - (df['VIX_Proxy'] / 100) # Normalize VIX

    df['volume_vol'] = df['Volume'].rolling(20).std() / df['Volume'].rolling(20).mean()
    df['volume_price_trend'] = df['Volume'].rolling(5).corr(df['Close'])

    df['volume_skew'] = df['Volume'].rolling(20).skew()
    df['volume_acceleration'] = df['Volume'].pct_change(5)
    df['volume_breakout'] = (df['Volume'] > df['Volume'].rolling(50).quantile(0.9)).astype(int)

    df['volume_price_divergence'] = (
        df['volume_surge'] - df['Close'].pct_change().abs().rolling(5).mean()
    )

    # --------- Statistical Indicators --------- #
    df['sigma'] = df['Close'].rolling(window=20).std()
    close_rolling = df['Close'].rolling(window=20)
    volume_rolling = df['Volume'].rolling(window=20)
    df['beta'] = close_rolling.cov(df['Volume']) / volume_rolling.var()
    df['skewness'] = df['Close'].rolling(window=20).skew()

    # --------- Gap Analysis --------- #
    df['overnight_gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    df['gap_magnitude_avg_5d'] = df['overnight_gap'].abs().rolling(5).mean()
    df['large_gap_frequency'] = (df['overnight_gap'].abs() > 0.03).rolling(20).sum()

    # --------- AI/Tech Sector Proxies --------- #
    df['tech_sector_rotation'] = df['Close'] / df['SPY_Close'] - 1
    df['nasdaq_relative_strength'] = df['Close'] / df['QQQ_Close'] - 1
    df['semiconductor_strength'] = df['Close'] / df['SOXX_Close'] - 1

    df['spy_qqq_spread'] = (df['QQQ_Close'] / df['SPY_Close']).pct_change()
    df['soxx_qqq_spread'] = (df['SOXX_Close'] / df['QQQ_Close']).pct_change()

    qqq_pct = df['QQQ_Close'].pct_change()
    spy_pct = df['SPY_Close'].pct_change()
    soxx_pct = df['SOXX_Close'].pct_change()
    nvda_pct = df['Close'].pct_change()

    df['tech_leadership'] = (
        (qqq_pct - spy_pct)
        .rolling(10)
        .mean()
    )

    df['semiconductor_leadership'] = (
        (soxx_pct - qqq_pct)
        .rolling(10)
        .mean()
    )

    df['nvda_outperformance'] = (
        (nvda_pct - soxx_pct)
        .rolling(5)
        .mean()
    )
    df['sector_momentum_divergence'] = (
        df['Close'].pct_change(10) - df['SOXX_Close'].pct_change(10)
    )

    df['mega_cap_rotation'] = (
        df['QQQ_Close'].pct_change(5) - df['SPY_Close'].pct_change(5)
    ).rolling(10).mean()

    df['momentum_persistence_3d'] = (
        (df['nvda_outperformance'] > 0).rolling(3).sum() / 3
    )

    df['ai_momentum_strength'] = (
        df['nvda_outperformance'] *
        df['tech_leadership'] *
        df['semiconductor_strength']
    )
    ai_min = df['ai_momentum_strength'].min()
    ai_max = df['ai_momentum_strength'].max()
    logger.debug(
        "Created 'ai_momentum_strength' with range: "
        f"[{ai_min:.2f}, {ai_max:.2f}]"
    )

    df['momentum_acceleration'] = (
        df['ai_momentum_strength'].diff(1) +
        df['ai_momentum_strength'].diff(2)
        ) / 2
    mean_ma = df['momentum_acceleration'].mean()
    std_ma = df['momentum_acceleration'].std()
    logger.debug(
        f"Created 'momentum_acceleration' with mean: {mean_ma:.2f}, "
        f"std: {std_ma:.2f}"
    )

    df['confirmed_momentum'] = (
        df['ai_momentum_strength'] *
        (df['volume_surge'] > 1).astype(float)
        )
    cm_mean = df['confirmed_momentum'].mean()
    cm_std = df['confirmed_momentum'].std()
    logger.debug(
        "Created 'confirmed_momentum' with mean: %.2f, std: %.2f",
        cm_mean, cm_std,
    )

    # --------- Cross-Asset Correlations --------- #
    df['vix_spread'] = df['VIX_Proxy'] - df['VIX_Proxy'].rolling(20).mean()
    spy_pct_change = df['SPY_Close'].pct_change()
    nvda_pct_change = df['Close'].pct_change()
    df['spy_nvda_correlation'] = (
        nvda_pct_change.rolling(20)
        .corr(spy_pct_change)
    )

    qqq_pct_change = df['QQQ_Close'].pct_change()
    df['qqq_nvda_correlation'] = (
        nvda_pct_change.rolling(20)
        .corr(qqq_pct_change)
    )
    df['treasury_equity_spread'] = df['Treasury_10Y'].diff() * -1

    df['Sector_Direction_Divergence'] = (
        df['momentum_5d'] - df['SOXX_Close'].pct_change(5)
    )
    sector_mean = df['Sector_Direction_Divergence'].mean()
    sector_std = df['Sector_Direction_Divergence'].std()
    logger.debug(
        "Created 'Sector_Direction_Divergence' "
        f"with mean: {sector_mean:.2f}, std: {sector_std:.2f}"
    )

    vix_roll = df['VIX_Proxy'].rolling(20)
    vix_thresh = vix_roll.quantile(0.9)
    df['VIX_Direction_Signal'] = (
        df['VIX_Proxy'] > vix_thresh
    ).astype(int)
    logger.debug(
        "Created 'VIX_Direction_Signal': "
        f"{df['VIX_Direction_Signal'].sum()} high VIX events"
    )

    # --------- Trend and Breakout Indicators --------- #
    df['trend_strength'] = (df['Close'] > df['SMA_200']).rolling(20).mean()
    df['bull_market_intensity'] = (
        (df['Close'] > df['SMA_50']).astype(int) +
        (df['SMA_50'] > df['SMA_200']).astype(int) +
        (df['Close'].pct_change(20) > 0.1).astype(int)
    )

    df['price_vs_bb_position'] = (df['Close'] - df['BBLow']) / (df['BBHigh'] - df['BBLow'])
    df['breakout_signal'] = (df['Close'] > df['BBHigh']).astype(int)

    df['Time_Since_Breakout'] = 0
    breakout_indices = df[(df['breakout_signal'] == 1) |
                            (df['Volatility_Breakout_Up'] == 1) |
                            (df['Volatility_Breakout_Down'] == 1)].index
    for i in range(len(df)):
        if i == 0:
            continue
    eligible_breakouts = [idx for idx in breakout_indices if idx <= df.index[i]]
    if eligible_breakouts:
        last_breakout = max(eligible_breakouts)
    else:
        last_breakout = df.index[0]
    # Use .loc to avoid chained-assignment and to be compatible with pandas Copy-on-Write
    df.loc[df.index[i], 'Time_Since_Breakout'] = (df.index[i] - last_breakout).days
    logger.debug(f"Created 'Time_Since_Breakout' with mean: {df['Time_Since_Breakout'].mean():.2f}")

    # --------- Seasonal Indicators --------- #
    df['days_in_quarter'] = df.index.dayofyear % 90
    df['earnings_proximity'] = np.where(
        df['days_in_quarter'].isin([1, 2, 3, 88, 89, 0]), 1, 0
    )

    # Shifted next-day price
    df['Price_Target'] = df['Close'].shift(-1)
    logger.debug("Shifted 'Price_Target' column for next-day prediction")

    # Calculate percentage change
    df['direction_pct'] = ((df['Price_Target'] - df['Close']) / df['Close']) * 100
    mean_dir_pct = df['direction_pct'].mean()
    std_dir_pct = df['direction_pct'].std()
    logger.debug(
        f"Calculated 'direction_pct' with mean: {mean_dir_pct:.2f}, std: {std_dir_pct:.2f}"
    )

    # Fixed minimum threshold: 0.5%
    fixed_thresh = 0.5

    # Rolling ATR-based adaptive threshold (20-day window, scaled)
    atr_window = 20
    atr_multiplier = 1.5
    atr_thresh = df['averageTrueRange'].rolling(atr_window).mean() * atr_multiplier

    # Combine thresholds: use the max between fixed and ATR-based
    hybrid_thresh = np.maximum(fixed_thresh, atr_thresh)
    logger.debug(f"Hybrid threshold calculated with mean: {hybrid_thresh.mean():.2f}")

    # Step 4: Initialize Direction_Target as sideways (1)
    df['Direction_Target'] = 1

    # Step 5: Assign directional labels
    df.loc[df['direction_pct'] > hybrid_thresh, 'Direction_Target'] = 2  # Up
    df.loc[df['direction_pct'] < -hybrid_thresh, 'Direction_Target'] = 0  # Down

    logger.debug("'Direction_Target' already shifted for next day prediction")

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
