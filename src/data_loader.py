"""
Data Loader Module for NVDA Stock Predictor Pipeline.

This module connects to the WRDS (Wharton Research Data Services) database and retrieves
historical market and fundamental data for a specified security over a defined time period.
The data is cleaned, transformed, and saved as a CSV file for downstream machine learning
or analytics workflows.

The primary function, `load_data`, handles:
    - Authentication and connection to WRDS
    - Execution of a parameterized SQL query
    - Preprocessing including date parsing and intelligent imputation
    - Saving cleaned data to disk with full coverage of key variables

The script is designed to be run standalone with Hydra configuration support and includes
rich logging for traceability and debugging.

Dependencies:
    - wrds
    - python-dotenv
    - pandas
    - hydra-core
    - omegaconf
    - scripts.logging_config (custom logging utilities)
"""

import sys
import os
from typing import Optional
from pathlib import Path
from dotenv import dotenv_values
import wrds
import pandas as pd
import hydra
from omegaconf import DictConfig
from scripts.logging_config import (
    get_logger,
    setup_logging,
    log_function_start,
    log_function_end,
)


def load_data(cfg: DictConfig, ticker, permno, gvkey, start_date, end_date, save_name):
    """
    Load and preprocess financial data from WRDS for a given security and time period.

    This function connects to the WRDS database using provided credentials, retrieves
    data using a parameterized SQL query, applies data cleaning and intelligent imputation
    strategies to ensure complete coverage, and saves the resulting df to a CSV file.

    Args:
        cfg (DictConfig): Configuration object with WRDS credentials and SQL query path.
        ticker (str): Ticker symbol of the target company.
        permno (str or int): CRSP PERMNO identifier for the security.
        gvkey (str or int): Compustat GVKEY identifier for the company.
        start_date (str): Start date for data retrieval in 'YYYY-MM-DD' format.
        end_date (str): End date for data retrieval in 'YYYY-MM-DD' format.
        save_name (str): Output file path for saving the processed data as CSV.

    Returns:
        pd.df: A cleaned and imputed df indexed by date, containing
        market and fundamental data.
    """

    logger = log_function_start(
        "load_data",
        ticker=ticker,
        permno=permno,
        gvkey=gvkey,
        start_date=start_date,
        end_date=end_date,
        save_name=save_name,
    )

    logger.info("Loading data from WRDS...")
    try:
        script_dir = Path(__file__).parent
        repo_root = script_dir.parent
        dotenv_path = repo_root / ".env"
        if dotenv_path.exists():
            try:
                values = dotenv_values(dotenv_path)
                for k in ("WRDS_USERNAME", "WRDS_PASSWORD"):
                    if values.get(k) and not os.getenv(k):
                        os.environ[k] = values.get(k)
            except Exception:
                # Fallback to load_dotenv if dotenv_values isn't available
                try:
                    from dotenv import load_dotenv

                    load_dotenv(dotenv_path=str(dotenv_path))
                except Exception:
                    pass
    except Exception:
        # Non-fatal; continue and let WRDS client fail later if needed
        pass

    def _ensure_pgpass(
        host: str = "wrds-pgdata.wharton.upenn.edu",
        port: str = "9737",
        dbname: str = "wrds",
    ) -> bool:
        user = os.getenv("WRDS_USERNAME")
        pw = os.getenv("WRDS_PASSWORD")
        if not user or not pw:
            return False
        pgpass = Path.home() / ".pgpass"
        entry = f"{host}:{port}:{dbname}:{user}:{pw}\n"
        try:
            if pgpass.exists():
                existing = pgpass.read_text()
                if entry.strip() in existing:
                    return True
            # Append the entry and set strict permissions
            with open(pgpass, "a", encoding="utf-8") as f:
                f.write(entry)
            pgpass.chmod(0o600)
            return True
        except Exception:
            return False

    # Try to ensure ~/.pgpass exists before connecting so WRDS client won't prompt.
    pgpass_ready = _ensure_pgpass()

    db = wrds.Connection(
        wrds_username=os.getenv("WRDS_USERNAME"),
        wrds_password=os.getenv("WRDS_PASSWORD"),
    )
    # If we couldn't create ~/.pgpass beforehand, fall back to the WRDS helper
    # which may prompt interactively. If pgpass_ready is True, skip calling
    # create_pgpass_file() to avoid unnecessary prompts.
    if not pgpass_ready:
        try:
            db.create_pgpass_file()
        except Exception:
            # If create_pgpass_file fails (e.g., non-interactive CI), continue;
            # the connection may still work if credentials were supplied.
            logger.debug("db.create_pgpass_file() failed or was skipped")
    logger.info("Connected to WRDS successfully.")
    # print(db.list_libraries())  # prints accessible data sets
    sql_path = Path(__file__).parent / cfg.data_loader.sql_save_path
    logger.info(f"Reading SQL query from {sql_path}.")
    with open(sql_path, "r", encoding="utf-8") as file:
        wrds_query = file.read()

    sql_query = wrds_query.format(
        TICKER=ticker,
        PERMNO=permno,
        GVKEY=gvkey,
        START_DATE=start_date,
        END_DATE=end_date,
    )
    logger.info("Executing SQL query...")

    # Attempt to execute the query using the WRDS helper. If the underlying
    # connection is a SQLAlchemy Engine/Connection (common with modern wrds clients)
    # pandas may attempt to call .cursor() on it which will fail. Provide a
    # graceful fallback: try db.raw_sql first, then fall back to pd.read_sql_query
    # using SQLAlchemy engine or a raw DB-API connection via engine.raw_connection().
    try:
        df = db.raw_sql(sql_query)
    except AttributeError as raw_err:
        logger.warning(
            "db.raw_sql raised AttributeError; attempting SQLAlchemy fallback: %s",
            raw_err,
        )
        engine = None
        # prefer explicit engine attribute if available
        for candidate in ("engine", "connection", "conn"):
            if hasattr(db, candidate):
                engine = getattr(db, candidate)
                logger.info(
                    "Using db.%s of type %s for fallback", candidate, type(engine)
                )
                break

        if engine is None:
            raise RuntimeError(
                "Unable to obtain a SQLAlchemy engine/connection from the WRDS client. "
                "Ensure a DB-API driver (e.g. psycopg2-binary) is installed or inspect the `db` object."
            ) from raw_err

        try:
            # Let pandas handle SQLAlchemy Engine/Connection directly if possible
            df = pd.read_sql_query(sql_query, con=engine)
        except Exception:
            # Fall back to obtaining a raw DB-API connection from the engine
            if hasattr(engine, "raw_connection"):
                raw_conn = engine.raw_connection()
                try:
                    df = pd.read_sql_query(sql_query, con=raw_conn)
                finally:
                    try:
                        raw_conn.close()
                    except Exception:
                        pass
            else:
                # Re-raise with context if no workable fallback exists
                raise

    if df.empty:
        error_msg = f"No data found for {ticker} in the specified date range."
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Process the data
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    # Forward fill fundamental components since they are quarterly data
    df["book_value_per_share"] = df["book_value_per_share"].ffill()
    df["earnings_per_share"] = df["earnings_per_share"].ffill()

    # Forward fill market data in case of missing values
    df["spy_close"] = df["spy_close"].ffill()
    df["qqq_close"] = df["qqq_close"].ffill()
    df["qqq_return"] = df["qqq_return"].ffill()
    df["soxx_close"] = df["soxx_close"].ffill()
    df["vix_proxy"] = df["vix_proxy"].ffill()
    df["treasury_10y"] = df["treasury_10y"].ffill()

    # Rename columns to match expected format
    df.columns = [
        "permno",
        "High",
        "Low",
        "Open",
        "Close",
        "Volume",
        "SPY_Close",
        "QQQ_Close",
        "QQQ_Return",
        "SOXX_Close",
        "VIX_Proxy",
        "Treasury_10Y",
        "PB_Ratio",
        "PE_Ratio",
    ]

    # Drop permno column as it's not needed for modeling
    df = df.drop("permno", axis=1)

    # ===== INTELLIGENT IMPUTATION FOR 100% DATA COVERAGE =====
    logger.info("Applying intelligent imputation for missing market data...")

    # SOXX imputation: Use QQQ as tech proxy when SOXX is missing
    missing_soxx = df["SOXX_Close"].isna()
    if missing_soxx.any():
        # Calculate SOXX/QQQ ratio from available data to maintain relative scaling
        available_both = df.dropna(subset=["SOXX_Close", "QQQ_Close"])
        if not available_both.empty:
            soxx_qqq_ratio = (
                available_both["SOXX_Close"] / available_both["QQQ_Close"]
            ).median()
            df.loc[missing_soxx, "SOXX_Close"] = (
                df.loc[missing_soxx, "QQQ_Close"] * soxx_qqq_ratio
            )
            logger.info(
                f"Imputed {missing_soxx.sum()} SOXX values using QQQ proxy "
                f"(ratio: {soxx_qqq_ratio:.2f})"
            )

    # VIX Proxy imputation: Calculate rolling volatility from QQQ returns when VIX proxy is missing
    missing_vix = df["VIX_Proxy"].isna()
    if missing_vix.any():
        # Calculate 30-day rolling volatility from QQQ returns
        rolling_std = df["QQQ_Return"].rolling(window=30, min_periods=10).std()
        df["QQQ_Vol_30d"] = rolling_std * (252**0.5) * 100

        # Scale to VIX-like levels (VIX is typically 10-80, our vol might be different scale)
        available_both_vix = df.dropna(subset=["VIX_Proxy", "QQQ_Vol_30d"])
        if not available_both_vix.empty:
            vix_scale_factor = (
                available_both_vix["VIX_Proxy"] / available_both_vix["QQQ_Vol_30d"]
            ).median()
            scaled_vix = df.loc[missing_vix, "QQQ_Vol_30d"] * vix_scale_factor
            df.loc[missing_vix, "VIX_Proxy"] = scaled_vix
            logger.info(
                f"Imputed {missing_vix.sum()} "
                f"VIX values using QQQ volatility (scale factor: {vix_scale_factor:.2f})"
            )
        else:
            # Fallback: use median VIX value
            median_vix = df["VIX_Proxy"].median()
            df.loc[missing_vix, "VIX_Proxy"] = median_vix
            logger.info(
                f"Imputed {missing_vix.sum()} "
                f"VIX values using median fallback ({median_vix:.2f})"
            )

        # Clean up temporary column
        df = df.drop("QQQ_Vol_30d", axis=1)

    # Final check: ensure no missing values remain in key columns
    final_missing = (
        df[
            [
                "SPY_Close",
                "QQQ_Close",
                "SOXX_Close",
                "VIX_Proxy",
                "Treasury_10Y",
            ]
        ]
        .isna()
        .sum()
    )
    if final_missing.any():
        logger.warning("Remaining missing values after imputation:")
        logger.warning(final_missing[final_missing > 0])

        # Final forward/backward fill for any remaining gaps
        for col in [
            "SPY_Close",
            "QQQ_Close",
            "SOXX_Close",
            "VIX_Proxy",
            "Treasury_10Y",
        ]:
            if df[col].isna().any():
                df[col] = df[col].ffill().bfill()
                logger.info(f"Applied forward/backward fill to {col}")

    logger.info("=== Final Data Coverage Check ===")
    coverage_check = df.isna().sum()
    total_missing = coverage_check.sum()
    if total_missing == 0:
        logger.info("PERFECT: 100% data coverage achieved!")
    else:
        logger.warning(f"This amount {total_missing} of missing values remain.")
        logger.warning(coverage_check[coverage_check > 0])

    # Drop QQQ_Return as it was only needed for imputation
    df = df.drop("QQQ_Return", axis=1)

    # Convert relative path to absolute path within the repository
    script_dir = Path(__file__).parent  # /path/to/repo/src
    repo_root = script_dir.parent  # /path/to/repo/
    output_path = Path(repo_root / save_name.lstrip("./")).resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=True)

    log_function_end(
        "load_data",
        success=True,
        output_file=str(output_path),
        final_shape=df.shape,
        date_range=f"{df.index.min()} to {df.index.max()}",
    )

    return df


@hydra.main(version_base=None, config_path="../configs", config_name="data_loader")
def main(cfg: Optional[DictConfig] = None):
    """
    Entry point for the NVDA Stock Predictor data loading pipeline.

    Sets up logging, logs relevant configuration details, and initiates the
    data loading and preprocessing process via `load_data`.

    Args:
        cfg (DictConfig): Configuration object containing all required
        parameters for data loading, including identifiers, date range, and
        output paths.
    """
    setup_logging(log_level="INFO", console_output=True, file_output=True)
    logger = get_logger("main")
    try:
        logger.info("Starting NVDA Stock Predictor Data Loader")
        logger.info("Configuration loaded from: data_loader config")
        logger.info("Configuration Parameters:")
        for key, value in cfg.data_loader.items():
            if not any(
                sensitive in key.upper()
                for sensitive in ["PASSWORD", "USERNAME", "TOKEN"]
            ):
                logger.info(f"  {key}: {value}")
    except AttributeError as e:
        logger.error(f"Configuration error: {e}", exc_info=True)
        sys.exit(1)
    try:
        load_data(
            cfg,
            cfg.data_loader.TICKER,
            cfg.data_loader.PERMNO,
            cfg.data_loader.GVKEY,
            cfg.data_loader.START_DATE,
            cfg.data_loader.END_DATE,
            cfg.data_loader.raw_data_path,
        )
    except (FileNotFoundError, ConnectionError) as e:
        logger.error(f"Data loading failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
