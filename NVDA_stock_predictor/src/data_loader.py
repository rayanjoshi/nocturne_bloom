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
    
    # Rename columns to match expected format
    dataFrame.columns = ['permno', 'High', 'Low', 'Open', 'Close', 'Volume', 
                        'PB_Ratio', 'PE_Ratio']
    
    # Drop permno column as it's not needed for modeling
    dataFrame = dataFrame.drop('permno', axis=1)

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
