"""
Module for processing financial data, generating predictions, and running trading simulations.

This module provides classes and functions to load financial data, engineer features,
create data modules for model input, generate predictions using an ensemble model,
and run trading simulations using Backtrader.
"""
from pathlib import Path
from typing import Optional
import pandas as pd
import hydra
from omegaconf import DictConfig
import numpy as np
import joblib
import torch
import backtrader as bt
import quantstats as qs


from src.data_loader import load_data
from src.feature_engineering import feature_engineering
from src.model_ensemble import EnsembleModule
from scripts.logging_config import get_logger, setup_logging

class DataProcessor:
    """
    Handles data loading, feature engineering, and data preparation for modeling.

    Args:
        cfg (DictConfig): Configuration object containing data processing parameters.
    """
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.logger = get_logger("DataProcessor")

    def load_data(self):
        """
        Loads financial data based on configuration parameters.
        """
        ticker = self.cfg.data_loader.TICKER
        start = self.cfg.data_loader.START_DATE
        end = self.cfg.data_loader.END_DATE
        self.logger.info(f"Loading data for {ticker} from {start} to {end}")
        load_data(
            self.cfg,
            self.cfg.data_loader.TICKER,
            self.cfg.data_loader.PERMNO,
            self.cfg.data_loader.GVKEY,
            self.cfg.data_loader.START_DATE,
            self.cfg.data_loader.END_DATE,
            self.cfg.data_loader.raw_data_path
        )

    def engineer_features(self):
        """
        Engineers features from raw financial data.

        Returns:
            pd.DataFrame: Processed DataFrame with engineered features.
        """
        self.logger.info(f"Engineering features for {self.cfg.data_loader.TICKER}")
        script_dir = Path(__file__).parent  # /path/to/repo/scripts
        repo_root = script_dir.parent  # /path/to/repo/
        raw_data_path = (repo_root / Path(self.cfg.data_loader.raw_data_path)).resolve()
        save_data_path = (repo_root / Path(self.cfg.features.preprocessing_data_path)).resolve()
        df = pd.read_csv(
            raw_data_path,
            header=0,
            index_col=0,
            parse_dates=True
        )
        return feature_engineering(df, save_data_path)

    def data_module(self, df):
        """
        Converts features and targets to scaled tensors for model input.

        Args:
            df (pd.DataFrame): DataFrame containing features and target columns.

        Returns:
            tuple: Scaled feature tensor (scaledX) and target tensor (scaledY).
        """
        self.logger.info(f"Converting features to tensors for {self.cfg.data_loader.TICKER}")
        window_size = self.cfg.data_module.window_size
        target_cols = self.cfg.data_module.target_cols

        features = df.drop(columns=target_cols)
        target = df["Price_Target"]

        x, y = [], []
        for i in range(window_size, len(df)):
            x.append(features.iloc[i-window_size:i].values)  # past window_size days features
            y.append(target.iloc[i])                         # target is the value at day i

        x = np.array(x)
        y = np.array(y)

        self.logger.info(f"Created {len(x)} windows with shape {x.shape}")
        self.logger.info(f"Target range before scaling: [{y.min():.6f}, {y.max():.6f}]")

        self.logger.info("Scaling features and targets...")
        script_dir = Path(__file__).parent  # /path/to/repo/scripts
        repo_root = script_dir.parent  # /path/to/repo/
        get_feature_scaler = (repo_root / Path(self.cfg.data_module.feature_scaler_path)).resolve()
        get_target_scaler = (repo_root / Path(self.cfg.data_module.target_scaler_path)).resolve()
        feature_scaler = joblib.load(get_feature_scaler)
        target_scaler = joblib.load(get_target_scaler)

        x_scaled = feature_scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
        y_scaled = target_scaler.transform(y.reshape(-1, 1)).flatten()

        feat_min = x_scaled.min()
        feat_max = x_scaled.max()
        tgt_min = y_scaled.min()
        tgt_max = y_scaled.max()
        self.logger.info(f"Feature range after scaling: [{feat_min:.6f}, {feat_max:.6f}]")
        self.logger.info(f"Target range after scaling: [{tgt_min:.6f}, {tgt_max:.6f}]")

        x_scaled = torch.tensor(x_scaled, dtype=torch.float32)
        y_scaled = torch.tensor(y_scaled, dtype=torch.float32)

        torch.save(x_scaled, (repo_root / Path(self.cfg.data_module.x_scaled_save_path)).resolve())
        torch.save(y_scaled, (repo_root / Path(self.cfg.data_module.y_scaled_save_path)).resolve())
        return x_scaled, y_scaled


class MakePredictions:
    """
    Generates and saves predictions using an ensemble model.

    Args:
        cfg (DictConfig): Configuration object containing model and data parameters.
    """
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.logger = get_logger("MakePredictions")

    def load_model(self):
        """
        Loads the ensemble model and generates predictions.

        Returns:
            list: List of predictions from the ensemble model.
        """
        script_dir = Path(__file__).parent  # /path/to/repo/src
        repo_root = script_dir.parent  # /path/to/repo/
        x_backtest = (repo_root / Path(self.cfg.data_module.x_scaled_save_path)).resolve()
        y_backtest = (repo_root / Path(self.cfg.data_module.y_scaled_save_path)).resolve()
        x = torch.load(x_backtest)
        y = torch.load(y_backtest)
        self.logger.info(f"Loaded {len(x)} samples with shape {x.shape} and target shape {y.shape}")
        model = EnsembleModule(self.cfg)

        cnn_path = Path(repo_root / self.cfg.model.cnnPath).resolve()
        cnn_state_dict = torch.load(cnn_path)
        model.cnn.load_state_dict(cnn_state_dict)

        ridge_path = Path(repo_root / self.cfg.model.ridgePath).resolve()
        ridge_state_dict = torch.load(ridge_path)
        model.ridge.load_state_dict(ridge_state_dict)
        model.ridge.is_fitted = True

        if self.cfg.model.use_meta_learning:
            meta_price_path = Path(repo_root / self.cfg.model.meta_price_path).resolve()
            meta_price_state_dict = torch.load(meta_price_path)
            model.meta_price.load_state_dict(meta_price_state_dict)

            meta_direction_path = Path(repo_root / self.cfg.model.meta_direction_path).resolve()
            meta_direction_state_dict = torch.load(meta_direction_path)
            model.meta_dir.load_state_dict(meta_direction_state_dict)

        elastic_net_path = Path(repo_root / self.cfg.model.elasticNetPath).resolve()
        elastic_net_state_dict = torch.load(elastic_net_path)
        model.elasticNet.load_state_dict(elastic_net_state_dict)
        model.elasticNet.is_fitted = True

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        x = x.to(device)

        predictions = []
        for xi in x:
            x_i = xi.unsqueeze(0)  # add batch dim
            with torch.no_grad():
                cnn_pred = model.cnn(x_i)
            elastic_net_pred = model.elasticNet(x_i)
            pred = model.cnnWeight * cnn_pred + model.elasticNetWeight * elastic_net_pred
            if pred.dim() > 0:
                pred = pred.squeeze(-1)
            predictions.append(pred.item())

        self.logger.info(f"Generated {len(predictions)} predictions using CNN+ElasticNet")
        return predictions

    def save_predictions(self, predictions):
        """
        Saves predictions with additional metrics to a CSV file.

        Args:
            predictions (list): List of model predictions.
        """
        script_dir = Path(__file__).parent
        repo_root = script_dir.parent
        scaler_path = (repo_root / Path(self.cfg.data_module.target_scaler_path)).resolve()
        target_scaler = joblib.load(scaler_path)
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = target_scaler.inverse_transform(predictions).flatten()
        df_predictions = pd.DataFrame(predictions, columns=['Predicted'])

        evaluate_data_path = (repo_root / Path(self.cfg.features.preprocessing_data_path)).resolve()
        # load preprocessed data and align to predictions
        df_eval = pd.read_csv(evaluate_data_path, header=0, index_col=0, parse_dates=True)
        dates = df_eval.index[self.cfg.data_module.window_size:]
        df_predictions.insert(0, 'Time', dates)
        self.logger.debug(f"Inserted Time column with {len(dates)} entries.")

        close_values = df_eval['Close'].values[self.cfg.data_module.window_size:]
        self.logger.debug(f"Loaded Close values, shape: {close_values.shape}")

        df_predictions['Predicted'] = df_predictions['Predicted'].round(3)
        close_values = np.round(close_values, 3)
        df_predictions.insert(2, 'Close', close_values)
        self.logger.debug("Inserted Close column and rounded Predicted/Close values.")

        error = np.abs(df_predictions['Predicted'] - df_predictions['Close'])
        df_predictions.insert(3, 'Error', error.round(3))
        self.logger.debug("Inserted Error column.")

        # Calculate direction: 1 if next value > current, -1 if next < current, 0 if equal
        pred_diff = np.diff(df_predictions['Predicted'])
        close_diff = np.diff(df_predictions['Close'])
        direction = np.where((pred_diff > 0) == (close_diff > 0), 'yes', 'no')
        # Pad with NaN for first row to align length
        direction = np.insert(direction, 0, np.nan)
        df_predictions.insert(4, 'Direction_Match', direction)
        self.logger.debug("Inserted Direction_Match column.")

        save_path = (repo_root / Path(self.cfg.backtest.predictions_save_path)).resolve()
        self.logger.debug(f"Predictions will be saved to {save_path}")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df_predictions.to_csv(save_path, index=False)
        self.logger.info(f"Saved predictions with time column to {save_path}")


class TradingSimulation:
    """
    Runs a trading simulation using Backtrader based on model predictions.

    Args:
        cfg (DictConfig): Configuration object containing simulation parameters.
        predictions (list): List of model predictions.
    """
    def __init__(self, cfg: DictConfig, predictions):
        self.cfg = cfg
        self.predictions = predictions
        self.logger = get_logger("TradingSimulation")
        script_dir = Path(__file__).parent
        repo_root = script_dir.parent
        data_path = (repo_root / Path(self.cfg.backtest.predictions_save_path)).resolve()
        df = pd.read_csv(data_path)
        self.predictions = df['Predicted'].values
        self.close_prices = df['Close'].values
        self.dates = df['Time'].values if 'Time' in df.columns else None

    def run(self):
        """
        Executes the trading simulation and calculates performance metrics.
        """
        # Some Backtrader versions do not accept 'cheat_on_open' in the Cerebro constructor.
        # Create Cerebro without the keyword and set the attribute afterwards if supported.
        cerebro = bt.Cerebro()
        try:
            cerebro.cheat_on_open = True
        except AttributeError:
            # Attribute not present on this Backtrader version; proceed without setting it.
            pass
        except (TypeError, ValueError):
            # If assignment fails due to unexpected type/value, proceed without setting it.
            pass
        cerebro.addstrategy(
            StrategySimulation,
            predictions=self.predictions,
            close_prices=self.close_prices,
            dates=self.dates,
        )

        # For Backtrader, create OHLCV df
        # Use Close from CSV, fill OHLC with Close, Volume=0
        df = pd.DataFrame({
            'Open': self.close_prices,
            'High': self.close_prices,
            'Low': self.close_prices,
            'Close': self.close_prices,
            'Volume': np.zeros(len(self.close_prices))
        })
        if self.dates is not None:
            df.index = pd.to_datetime(self.dates)
        # Provide the DataFrame to the PandasData feed; default column names are used
        data = bt.feeds.PandasData(dataname=df)  # pylint: disable=unexpected-keyword-arg

        cerebro.adddata(data)

        cerebro.broker.setcash(1000000.00)
        cerebro.broker.setcommission(commission=0.001)

        self.logger.info(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")
        # Track portfolio value after each bar
        portfolio_values = []
        class ValueTracker(bt.Analyzer):
            """
            Custom analyzer to track portfolio values.
            """
            def __init__(self):
                super().__init__()
                self.strategy = None
                self.values = []
            def next(self):
                self.values.append(self.strategy.broker.getvalue())
        cerebro.addanalyzer(ValueTracker, _name='valtracker')
        results = cerebro.run()
        self.logger.info(f"Ending Portfolio Value: {cerebro.broker.getvalue():.2f}")

        # Get portfolio values from analyzer
        valtracker = results[0].analyzers.valtracker
        portfolio_values = valtracker.values
        if len(portfolio_values) > 1:
            # Calculate daily returns
            returns = pd.Series(portfolio_values).pct_change().dropna()
            sharpe = qs.stats.sharpe(returns)
            self.logger.info(f"QuantStats Sharpe Ratio: {sharpe:.4f}")

            # ...removed matplotlib and seaborn plotting code...
        else:
            self.logger.warning("Not enough data to calculate Sharpe ratio.")

class StrategySimulation(bt.Strategy):
    """
    Backtrader strategy for trading based on model predictions.

    Args:
        predictions (list): List of predicted prices.
        close_prices (list): List of actual close prices.
        dates (list, optional): List of corresponding dates. Defaults to None.

    Attributes:
        params (dict): Strategy parameters (threshold, size).
    """
    params = dict(threshold=0.01, size=100)

    def __init__(self, predictions, close_prices, dates=None):
        super().__init__()
        self.predictions = predictions
        self.close_prices = close_prices
        self.dates = dates
        self.order = None
        # Initialize attributes that will be assigned later to avoid defining them outside __init__
        self.buyprice = None
        self.buycomm = None
        self.bar_executed = None
        self.data.close = self.datas[0].close
        self.logger = get_logger("StrategySimulation")

    def log(self, txt, dt=None):
        """
        Logs a message with an optional timestamp.

        Args:
            txt (str): Message to log.
            dt (datetime, optional): Timestamp for the log. Defaults to None.
        """
        idx = len(self)
        if self.dates is not None and idx < len(self.dates):
            dt = pd.to_datetime(self.dates[idx]).date()
        else:
            dt = dt or self.datas[0].datetime.date(0)
        self.logger.info(f"{dt.isoformat()} {txt}")

    def next(self):
        """
        Executes the trading logic for each bar.
        """
        idx = len(self)
        if idx >= len(self.predictions):
            return  # Avoid index error if predictions are shorter than data
        current_close = self.close_prices[idx]
        predicted_close = self.predictions[idx]
        if np.isnan(predicted_close) or np.isnan(current_close):
            return
        predicted_return = (predicted_close - current_close) / current_close
        self.log(f"Close, {current_close:.2f} | Predicted, {predicted_close:.2f}")

        # Close any open position
        if self.position:
            if (self.position.size > 0 and predicted_return < self.p.threshold) or \
                (self.position.size < 0 and predicted_return > -self.p.threshold):
                self.log(f'CLOSE POSITION, {current_close:.2f}')
                self.order = self.close()

        # Long signal
        elif predicted_return > self.p.threshold:
            self.log(f'BUY CREATE, {current_close:.2f}')
            self.order = self.buy(size=self.p.size)

        # Short signal
        elif predicted_return < -self.p.threshold:
            self.log(f'SELL CREATE, {current_close:.2f}')
            self.order = self.sell(size=self.p.size)

    def notify_order(self, order):
        """
        Handles order notifications from the broker.

        Args:
            order (bt.Order): The order object with status updates.
        """
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                msg = (
                    f"BUY EXECUTED, Price: {order.executed.price:.2f}, "
                    f"Cost: {order.executed.value:.2f}, "
                    f"Comm {order.executed.comm:.2f}"
                )
                self.log(msg)

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                msg = (
                    f"SELL EXECUTED, Price: {order.executed.price:.2f}, "
                    f"Cost: {order.executed.value:.2f}, "
                    f"Comm {order.executed.comm:.2f}"
                )
                self.log(msg)

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # Write down: no pending order
        self.order = None

    def notify_trade(self, trade):
        """
        Handles trade notifications from the broker.

        Args:
            trade (bt.Trade): The trade object with profit/loss details.
        """
        if not trade.isclosed:
            return
        self.log(f'OPERATION PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}')

@hydra.main(version_base=None, config_path="../configs", config_name="backtest")
def main(cfg: Optional[DictConfig] = None):
    """
    Main function to orchestrate data processing, prediction, and trading simulation.

    Args:
        cfg (DictConfig, optional): Configuration object. Defaults to None.

    Raises:
        FileNotFoundError: If data or model files are not found.
        pd.errors.EmptyDataError: If data files are empty.
        KeyError: If required configuration keys are missing.
        ValueError: If data or model parameters are invalid.
        RuntimeError: If runtime issues occur during execution.
        OSError: If file operations fail.
    """
    try:
        setup_logging(log_level="DEBUG", console_output=True, file_output=True)
        logger = get_logger("main")
        data_processor = DataProcessor(cfg)
        data_processor.load_data()
        df = data_processor.engineer_features()
        data_processor.data_module(df)
        make_predictions = MakePredictions(cfg)

        predictions = make_predictions.load_model()
        make_predictions.save_predictions(predictions)
        backtest = TradingSimulation(cfg, predictions)
        backtest.run()
        logger.info("Backtest completed successfully.")
    except (FileNotFoundError,
            pd.errors.EmptyDataError,
            KeyError,
            ValueError,
            RuntimeError,
            OSError) as e:
        logger.error(f"An error occurred ({type(e).__name__}): {e}")
        raise

if __name__ == "__main__":
    main()
