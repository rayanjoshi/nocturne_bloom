import wrds
import pandas as pd
from pathlib import Path
import hydra
from omegaconf import DictConfig

def load_data(cfg: DictConfig, ticker, permno, gvkey, start_date, end_date, save_name):
    db = wrds.Connection(wrds_username=cfg.data_loader.WRDS_USERNAME)

    # print(db.list_libraries())  # prints accessible data sets

    sql_path = Path(__file__).parent / cfg.data_loader.sql_save_path.lstrip('../')
    with open(sql_path, 'r') as file:
        WRDS_query = file.read()
        
    sql_query = WRDS_query.format(
        TICKER=ticker,
        PERMNO=permno,
        GVKEY=gvkey,
        START_DATE=start_date,
        END_DATE=end_date
    )

    dataFrame = db.raw_sql(sql_query)
    if dataFrame.empty:
        raise ValueError(f"No data found for {ticker} in the specified date range.")

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
    print("Applying intelligent imputation for missing market data...")
    
    # 1. SOXX imputation: Use QQQ as tech proxy when SOXX is missing
    missing_soxx = dataFrame['SOXX_Close'].isna()
    if missing_soxx.any():
        # Calculate SOXX/QQQ ratio from available data to maintain relative scaling
        available_both = dataFrame.dropna(subset=['SOXX_Close', 'QQQ_Close'])
        if not available_both.empty:
            soxx_qqq_ratio = (available_both['SOXX_Close'] / available_both['QQQ_Close']).median()
            dataFrame.loc[missing_soxx, 'SOXX_Close'] = dataFrame.loc[missing_soxx, 'QQQ_Close'] * soxx_qqq_ratio
            print(f"Imputed {missing_soxx.sum()} SOXX values using QQQ proxy (ratio: {soxx_qqq_ratio:.2f})")
    
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
            print(f"Imputed {missing_vix.sum()} VIX values using QQQ volatility (scale factor: {vix_scale_factor:.2f})")
        else:
            # Fallback: use median VIX value
            median_vix = dataFrame['VIX_Proxy'].median()
            dataFrame.loc[missing_vix, 'VIX_Proxy'] = median_vix
            print(f"Imputed {missing_vix.sum()} VIX values using median fallback ({median_vix:.2f})")
        
        # Clean up temporary column
        dataFrame = dataFrame.drop('QQQ_Vol_30d', axis=1)
    
    # 3. Final check: ensure no missing values remain in key columns
    final_missing = dataFrame[['SPY_Close', 'QQQ_Close', 'SOXX_Close', 'VIX_Proxy', 'Treasury_10Y']].isna().sum()
    if final_missing.any():
        print("Remaining missing values after imputation:")
        print(final_missing[final_missing > 0])
        
        # Final forward/backward fill for any remaining gaps
        for col in ['SPY_Close', 'QQQ_Close', 'SOXX_Close', 'VIX_Proxy', 'Treasury_10Y']:
            if dataFrame[col].isna().any():
                dataFrame[col] = dataFrame[col].fillna(method='ffill').fillna(method='bfill')
                print(f"Applied forward/backward fill to {col}")
    
    print("=== Final Data Coverage Check ===")
    coverage_check = dataFrame.isna().sum()
    total_missing = coverage_check.sum()
    if total_missing == 0:
        print("✅ PERFECT: 100% data coverage achieved!")
    else:
        print(f"❌ WARNING: {total_missing} missing values remain")
        print(coverage_check[coverage_check > 0])
    
    # Drop QQQ_Return as it was only needed for imputation
    dataFrame = dataFrame.drop('QQQ_Return', axis=1)

    # Convert relative path to absolute path within the repository
    script_dir = Path(__file__).parent  # /path/to/repo/NVDA_stock_predictor/src
    repo_root = script_dir.parent  # /path/to/repo/NVDA_stock_predictor
    output_path = repo_root / save_name.lstrip('../')  # Remove leading ../

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataFrame.to_csv(output_path, index=True)
    
    print(f"Data saved to {output_path.absolute()}")
    print("--------- Data Loading Statistics ---------")
    print(f"Total features created: {len(dataFrame.columns)}")
    print(f"Dataset shape: {dataFrame.shape}")
    print(f"All features: {list(dataFrame.columns)}")
    print("--------- Data Loading Completed ---------")

    return dataFrame

@hydra.main(version_base=None, config_path="../configs", config_name="data_loader")
def main(cfg: DictConfig):
    try:
        print("The following parameters will be used to load data:")
        print(f"TICKER: {cfg.data_loader.TICKER}")
        print(f"PERMNO: {cfg.data_loader.PERMNO}")
        print(f"GVKEY: {cfg.data_loader.GVKEY}")
        print(f"START_DATE: {cfg.data_loader.START_DATE}")
        print(f"END_DATE: {cfg.data_loader.END_DATE}")
        print(f"SAVE_NAME: {cfg.data_loader.raw_data_path}")
        load_data(cfg, cfg.data_loader.TICKER, cfg.data_loader.PERMNO, cfg.data_loader.GVKEY, cfg.data_loader.START_DATE, cfg.data_loader.END_DATE, cfg.data_loader.raw_data_path)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
