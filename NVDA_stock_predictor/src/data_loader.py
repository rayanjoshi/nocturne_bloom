import wrds
import pandas as pd
from pathlib import Path
import hydra
from omegaconf import DictConfig

def load_data(cfg: DictConfig):
    print(f"Loading data for {cfg.data_loader.TICKER} from {cfg.data_loader.START_DATE} to {cfg.data_loader.END_DATE}")
    db = wrds.Connection(wrds_username='rj524')

    # print(db.list_libraries())  # prints accessible data sets

    sql_query = f"""
    WITH stock_data AS (
        SELECT 
            date,
            permno,
            ABS(prc) / cfacpr as close,
            CASE WHEN openprc IS NOT NULL THEN ABS(openprc) / cfacpr ELSE ABS(prc) / cfacpr END as open,
            CASE WHEN askhi IS NOT NULL THEN askhi / cfacpr ELSE ABS(prc) / cfacpr END as high,
            CASE WHEN bidlo IS NOT NULL THEN bidlo / cfacpr ELSE ABS(prc) / cfacpr END as low,
            vol * cfacshr as volume
        FROM crsp.dsf
        WHERE permno = {cfg.data_loader.PERMNO}
        AND date >= '{cfg.data_loader.START_DATE}' AND date <= '{cfg.data_loader.END_DATE}'
        AND prc IS NOT NULL
    ),
    fundamental_data AS (
        SELECT 
            l.lpermno as permno,
            f.datadate,
            CASE WHEN f.ceqq > 0 AND f.cshoq > 0 THEN f.ceqq / f.cshoq ELSE NULL END as book_value_per_share,
            CASE WHEN f.niq IS NOT NULL AND f.cshoq > 0 THEN (f.niq * 4) / f.cshoq ELSE NULL END as earnings_per_share
        FROM comp.fundq f
        JOIN crsp_q_ccm.ccm_lookup l ON f.gvkey = l.gvkey
        WHERE l.lpermno = {cfg.data_loader.PERMNO}
        AND f.datadate >= '{cfg.data_loader.START_DATE}' AND f.datadate <= '{cfg.data_loader.END_DATE}'
        AND f.cshoq IS NOT NULL
    )
    SELECT 
        s.date,
        s.permno,
        s.high,
        s.low, 
        s.open,
        s.close,
        s.volume,
        f.book_value_per_share,
        f.earnings_per_share
    FROM stock_data s
    LEFT JOIN fundamental_data f ON s.permno = f.permno 
        AND f.datadate = (
            SELECT MAX(f2.datadate) 
            FROM fundamental_data f2 
            WHERE f2.permno = s.permno 
            AND f2.datadate <= s.date
        )
    ORDER BY s.date;
    """
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
