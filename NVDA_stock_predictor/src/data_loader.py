import wrds
import pandas as pd
from pathlib import Path
import hydra
from omegaconf import DictConfig

def load_data(cfg: DictConfig):
    print(f"Loading data for {cfg.data_loader.TICKER} from {cfg.data_loader.START_DATE} to {cfg.data_loader.END_DATE}")
    db = wrds.Connection(wrds_username='rj524')

    # print(db.list_libraries())  # prints accessible data sets

    sql_path = Path(__file__).parent / cfg.data_loader.sql_save_path.lstrip('../')
    with open(sql_path, 'r') as file:
        WRDS_query = file.read()
        
    sql_query = WRDS_query.format(
        TICKER=cfg.data_loader.TICKER,
        PERMNO=cfg.data_loader.PERMNO,
        GVKEY=cfg.data_loader.GVKEY,
        START_DATE=cfg.data_loader.START_DATE,
        END_DATE=cfg.data_loader.END_DATE
    )

    dataFrame = db.raw_sql(sql_query)
    if dataFrame.empty:
        raise ValueError(f"No data found for {cfg.data_loader.TICKER} in the specified date range.")

    # Process the data
    dataFrame['date'] = pd.to_datetime(dataFrame['date'])
    dataFrame.set_index('date', inplace=True)
    
    # Forward fill fundamental components since they are quarterly data
    dataFrame['book_value_per_share'] = dataFrame['book_value_per_share'].ffill()
    dataFrame['earnings_per_share'] = dataFrame['earnings_per_share'].ffill()
    
    # Rename columns to match expected format
    dataFrame.columns = ['permno', 'High', 'Low', 'Open', 'Close', 'Volume', 
                        'Book_Value_Per_Share', 'Earnings_Per_Share']
    
    # Drop permno column as it's not needed for modeling
    dataFrame = dataFrame.drop('permno', axis=1)

    # Convert relative path to absolute path within the repository
    script_dir = Path(__file__).parent  # /path/to/repo/NVDA_stock_predictor/src
    repo_root = script_dir.parent  # /path/to/repo/NVDA_stock_predictor
    output_path = repo_root / cfg.data_loader.raw_data_path.lstrip('../')  # Remove leading ../
    
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
        load_data(cfg)
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()
