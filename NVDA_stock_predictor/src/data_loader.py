import yfinance as yf
import pandas as pd
from config import TICKER, START_DATE, END_DATE, RAW_DATA_PATH

def load_data():
    print(f"Loading data for {TICKER} from {START_DATE} to {END_DATE}")
    df = yf.download(TICKER, start=START_DATE, end=END_DATE)
    if df.empty:
        raise ValueError(f"No data found for {TICKER} in the specified date range.")

    df.to_csv(RAW_DATA_PATH, index=True)
    print(f"Data saved to {RAW_DATA_PATH}")
    return df

if __name__ == "__main__":
    try:
        data = load_data()
        print("Data loading completed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise