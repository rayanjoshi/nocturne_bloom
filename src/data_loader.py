import wrds
import pandas as pd
from pathlib import Path
import hydra
from omegaconf import DictConfig
import sys
from scripts.logging_config import get_logger, setup_logging, log_function_start, log_function_end

def load_data(cfg: DictConfig, ticker, permno, gvkey, start_date, end_date, save_name):
    logger = log_function_start("load_data", 
                            ticker=ticker, permno=permno, gvkey=gvkey,
                            start_date=start_date, end_date=end_date, save_name=save_name)
    
    logger.info("Loading data from WRDS...")
    db = wrds.Connection(wrds_username=cfg.data_loader.WRDS_USERNAME)
    logger.info("Connected to WRDS successfully.")
    # print(db.list_libraries())  # prints accessible data sets
    sql_path = Path(__file__).parent / cfg.data_loader.sql_save_path
    logger.info(f"Reading SQL query from {sql_path}.")
    with open(sql_path, 'r') as file:
        WRDS_query = file.read()
        
    sql_query = WRDS_query.format(
        TICKER=ticker,
        PERMNO=permno,
        GVKEY=gvkey,
        START_DATE=start_date,
        END_DATE=end_date
    )
    logger.info(f"Executing SQL query...")
    
    dataFrame = db.raw_sql(sql_query)
    if dataFrame.empty:
        error_msg = f"No data found for {ticker} in the specified date range."
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    
    # Process the data
    dataFrame['date'] = pd.to_datetime(dataFrame['date'])
    dataFrame.set_index('date', inplace=True)
    
    # Forward fill fundamental components since they are quarterly data
    dataFrame['book_value_per_share'] = dataFrame['book_value_per_share'].ffill()
    dataFrame['earnings_per_share'] = dataFrame['earnings_per_share'].ffill()
    
    # Forward fill market data in case of missing values
    dataFrame['spy_close'] = dataFrame['spy_close'].ffill()
    dataFrame['qqq_close'] = dataFrame['qqq_close'].ffill()
    dataFrame['qqq_return'] = dataFrame['qqq_return'].ffill()
    dataFrame['soxx_close'] = dataFrame['soxx_close'].ffill()
    dataFrame['vix_proxy'] = dataFrame['vix_proxy'].ffill()
    dataFrame['treasury_10y'] = dataFrame['treasury_10y'].ffill()
    
    # Rename columns to match expected format
    dataFrame.columns = ['permno', 'High', 'Low', 'Open', 'Close', 'Volume', 
                        'SPY_Close', 'QQQ_Close', 'QQQ_Return', 'SOXX_Close', 'VIX_Proxy', 'Treasury_10Y', 'PB_Ratio', 'PE_Ratio']
    
    # Drop permno column as it's not needed for modeling
    dataFrame = dataFrame.drop('permno', axis=1)
    
    # ===== INTELLIGENT IMPUTATION FOR 100% DATA COVERAGE =====
    logger.info("Applying intelligent imputation for missing market data...")
    
    # 1. SOXX imputation: Use QQQ as tech proxy when SOXX is missing
    missing_soxx = dataFrame['SOXX_Close'].isna()
    if missing_soxx.any():
        # Calculate SOXX/QQQ ratio from available data to maintain relative scaling
        available_both = dataFrame.dropna(subset=['SOXX_Close', 'QQQ_Close'])
        if not available_both.empty:
            soxx_qqq_ratio = (available_both['SOXX_Close'] / available_both['QQQ_Close']).median()
            dataFrame.loc[missing_soxx, 'SOXX_Close'] = dataFrame.loc[missing_soxx, 'QQQ_Close'] * soxx_qqq_ratio
            logger.info(f"Imputed {missing_soxx.sum()} SOXX values using QQQ proxy (ratio: {soxx_qqq_ratio:.2f})")
    
    # 2. VIX Proxy imputation: Calculate rolling volatility from QQQ returns when VIX proxy is missing
    missing_vix = dataFrame['VIX_Proxy'].isna()
    if missing_vix.any():
        # Calculate 30-day rolling volatility from QQQ returns (annualized)
        dataFrame['QQQ_Vol_30d'] = dataFrame['QQQ_Return'].rolling(window=30, min_periods=10).std() * (252**0.5) * 100
        
        # Scale to VIX-like levels (VIX is typically 10-80, our vol might be different scale)
        available_both_vix = dataFrame.dropna(subset=['VIX_Proxy', 'QQQ_Vol_30d'])
        if not available_both_vix.empty:
            vix_scale_factor = (available_both_vix['VIX_Proxy'] / available_both_vix['QQQ_Vol_30d']).median()
            dataFrame.loc[missing_vix, 'VIX_Proxy'] = dataFrame.loc[missing_vix, 'QQQ_Vol_30d'] * vix_scale_factor
            logger.info(f"Imputed {missing_vix.sum()} VIX values using QQQ volatility (scale factor: {vix_scale_factor:.2f})")
        else:
            # Fallback: use median VIX value
            median_vix = dataFrame['VIX_Proxy'].median()
            dataFrame.loc[missing_vix, 'VIX_Proxy'] = median_vix
            logger.info(f"Imputed {missing_vix.sum()} VIX values using median fallback ({median_vix:.2f})")
        
        # Clean up temporary column
        dataFrame = dataFrame.drop('QQQ_Vol_30d', axis=1)
    
    # 3. Final check: ensure no missing values remain in key columns
    final_missing = dataFrame[['SPY_Close', 'QQQ_Close', 'SOXX_Close', 'VIX_Proxy', 'Treasury_10Y']].isna().sum()
    if final_missing.any():
        logger.warning("Remaining missing values after imputation:")
        logger.warning(final_missing[final_missing > 0])
        
        # Final forward/backward fill for any remaining gaps
        for col in ['SPY_Close', 'QQQ_Close', 'SOXX_Close', 'VIX_Proxy', 'Treasury_10Y']:
            if dataFrame[col].isna().any():
                    dataFrame[col] = dataFrame[col].ffill().bfill()
                    logger.info(f"Applied forward/backward fill to {col}")
    
    logger.info("=== Final Data Coverage Check ===")
    coverage_check = dataFrame.isna().sum()
    total_missing = coverage_check.sum()
    if total_missing == 0:
        logger.info("PERFECT: 100% data coverage achieved!")
    else:
        logger.warning(f"This amount {total_missing} of missing values remain.")
        logger.warning(coverage_check[coverage_check > 0])
    
    # Drop QQQ_Return as it was only needed for imputation
    dataFrame = dataFrame.drop('QQQ_Return', axis=1)
    
    # Convert relative path to absolute path within the repository
    script_dir = Path(__file__).parent  # /path/to/repo/src
    repo_root = script_dir.parent  # /path/to/repo/
    output_path = Path(repo_root / save_name.lstrip("./")).resolve()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataFrame.to_csv(output_path, index=True)
    
    log_function_end("load_data", success=True,
                        output_file=str(output_path),
                        final_shape=dataFrame.shape,
                        date_range=f"{dataFrame.index.min()} to {dataFrame.index.max()}")
    
    return dataFrame

@hydra.main(version_base=None, config_path="../configs", config_name="data_loader")
def main(cfg: DictConfig):
    try:
        setup_logging(log_level="INFO", console_output=True, file_output=True)
        logger = get_logger("main")
        logger.info("🚀 Starting NVDA Stock Predictor Data Loader")
        logger.info(f"Configuration loaded from: data_loader config")
        
        # Log configuration parameters (excluding sensitive ones)
        logger.info("Configuration Parameters:")
        for key, value in cfg.data_loader.items():
            if not any(sensitive in key.upper() for sensitive in ['PASSWORD', 'USERNAME', 'TOKEN']):
                logger.info(f"  {key}: {value}")
        load_data(cfg, cfg.data_loader.TICKER, cfg.data_loader.PERMNO, cfg.data_loader.GVKEY, cfg.data_loader.START_DATE, cfg.data_loader.END_DATE, cfg.data_loader.raw_data_path)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        logger.error("Full traceback:", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
