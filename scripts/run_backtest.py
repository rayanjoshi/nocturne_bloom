"""
Module for processing financial data, generating predictions, and running trading simulations.

This module provides classes and functions to load financial data, engineer features,
create data modules for model input, generate predictions using an ensemble model,
and run trading simulations using Backtrader.
"""
from pathlib import Path
from typing import Optional
import json
import pandas as pd
import hydra
from omegaconf import DictConfig
import numpy as np
import joblib
import torch
import backtrader as bt
import quantstats as qs
from numpy.lib.stride_tricks import sliding_window_view

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
        Converts features and targets to scaled tensors for model input using pre-existing scalers.

        Args:
            df (pd.DataFrame): DataFrame containing features and target columns.

        Returns:
            tuple: Scaled feature tensor (x_scaled), price target tensor (price_y_scaled),
                   and direction target tensor (direction_y).
        """
        self.logger.info(f"Converting features to tensors for {self.cfg.data_loader.TICKER}")
        window_size = self.cfg.data_module.window_size
        price_target_col = self.cfg.data_module.price_target_col
        direction_target_col = self.cfg.data_module.direction_target_col
        output_seq_len = self.cfg.data_module.output_seq_len

        # Define target columns
        target_cols = [price_target_col, direction_target_col]
        features = df.drop(columns=[col for col in df.columns if col in target_cols])
        price_target = df[price_target_col]
        direction_target = df[direction_target_col]

        # Convert to numpy arrays
        features_array = features.values
        price_target_array = price_target.values
        direction_target_array = direction_target.values

        # Create sliding window for features
        x = sliding_window_view(
            features_array,
            window_shape=(window_size, features_array.shape[1])
        )[:-output_seq_len].reshape(-1, window_size, features_array.shape[1])
        price_y = sliding_window_view(
            price_target_array,
            window_shape=output_seq_len
        )[window_size:].reshape(-1, output_seq_len)
        direction_y = direction_target_array[window_size:window_size + len(price_y)]

        x = np.array(x)
        price_y = np.array(price_y)
        direction_y = np.array(direction_y)

        self.logger.info(f"Created {len(x)} windows with shape {x.shape}")
        pmin = price_y.min()
        pmax = price_y.max()
        self.logger.info(
            f"Price target range before scaling: [{pmin:.6f}, {pmax:.6f}]"
        )
        dmin = direction_y.min()
        dmax = direction_y.max()
        self.logger.info(
            f"Direction target range before scaling: [{dmin:.6f}, {dmax:.6f}]"
        )

        # Load pre-existing scalers
        self.logger.info("Loading pre-existing scalers for features and targets...")
        script_dir = Path(__file__).parent
        repo_root = script_dir.parent
        feature_scaler_path = (repo_root / Path(self.cfg.data_module.x_scaled_save_path)).resolve()
        price_target_rel = Path(self.cfg.data_module.price_y_scaled_save_path)
        price_target_scaler_path = (repo_root / price_target_rel).resolve()

        try:
            feature_scaler = joblib.load(feature_scaler_path)
            price_target_scaler = joblib.load(price_target_scaler_path)
            self.logger.info(f"Loaded feature scaler from: {feature_scaler_path}")
            self.logger.info(f"Loaded price target scaler from: {price_target_scaler_path}")
        except FileNotFoundError as e:
            self.logger.error(f"Scaler file not found: {e}")
            raise

        # Scale features and targets using loaded scalers
        x_reshaped = x.reshape(-1, x.shape[-1])
        x_scaled = feature_scaler.transform(x_reshaped).reshape(x.shape)
        # reshape price targets to 2D for scaler then restore original shape
        price_y_2d = price_y.reshape(-1, 1)
        price_y_scaled = price_target_scaler.transform(price_y_2d).reshape(price_y.shape)

        # Log scaled ranges
        feat_min = x_scaled.min()
        feat_max = x_scaled.max()
        price_min = price_y_scaled.min()
        price_max = price_y_scaled.max()
        self.logger.info(f"Feature range after scaling: [{feat_min:.6f}, {feat_max:.6f}]")
        self.logger.info(f"Price target range after scaling: [{price_min:.6f}, {price_max:.6f}]")
        self.logger.info(f"Direction targets not scaled: {np.bincount(direction_y.astype(int))}")

        # Convert to tensors
        x_scaled = torch.tensor(x_scaled, dtype=torch.float32)
        price_y_scaled = torch.tensor(price_y_scaled, dtype=torch.float32)
        direction_y = torch.tensor(direction_y, dtype=torch.long)

        # Save tensors
        x_save_path = (repo_root / Path(self.cfg.data_module.eval_x_scaled_save_path)).resolve()
        price_y_rel = Path(self.cfg.data_module.eval_price_y_scaled_save_path)
        price_y_save_path = (repo_root / price_y_rel).resolve()

        direction_y_rel = Path(self.cfg.data_module.eval_direction_y_scaled_save_path)
        direction_y_save_path = (repo_root / direction_y_rel).resolve()

        x_save_path.parent.mkdir(parents=True, exist_ok=True)
        price_y_save_path.parent.mkdir(parents=True, exist_ok=True)
        direction_y_save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(x_scaled, x_save_path)
        torch.save(price_y_scaled, price_y_save_path)
        torch.save(direction_y, direction_y_save_path)
        self.logger.info(f"Saved x_scaled to: {x_save_path}")
        self.logger.info(f"Saved price_y_scaled to: {price_y_save_path}")
        self.logger.info(f"Saved direction_y to: {direction_y_save_path}")

        return x_scaled, price_y_scaled, direction_y


class MakePredictions:
    """
    Generates and saves predictions using an ensemble model.

    Args:
        cfg (DictConfig): Configuration object containing model and data parameters.
    """
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.logger = get_logger("MakePredictions")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")

    def load_model(self):
        """
        Loads the ensemble model and generates predictions.

        Returns:
            list: List of price predictions from the ensemble model.
        """
        # Resolve paths relative to repo root
        script_dir = Path(__file__).parent
        repo_root = script_dir.parent
        x_backtest_path = (repo_root / Path(self.cfg.data_module.eval_x_scaled_save_path)).resolve()
        price_y_rel = Path(self.cfg.data_module.eval_price_y_scaled_save_path)
        price_y_path = (repo_root / price_y_rel).resolve()
        y_backtest_path = (repo_root / price_y_path).resolve()

        # Load scaled input data
        x = torch.load(x_backtest_path, map_location=self.device)
        y = torch.load(y_backtest_path, map_location=self.device)
        self.logger.info(f"Loaded {len(x)} samples with shape {x.shape} and target shape {y.shape}")

        # Initialize ensemble model
        model = EnsembleModule(self.cfg).to(self.device)
        model.eval()

        # Load model state dictionaries
        try:
            cnn_path = (repo_root / Path(self.cfg.model.cnn_path)).resolve()
            cnn_state_dict = torch.load(cnn_path, map_location=self.device)
            model.cnn.load_state_dict(cnn_state_dict)
            self.logger.info(f"Loaded CNN state from {cnn_path}")

            ridge_path = (repo_root / Path(self.cfg.model.ridge_path)).resolve()
            ridge_state_dict = torch.load(ridge_path, map_location=self.device)
            model.ridge.load_state_dict(ridge_state_dict)
            model.ridge.is_fitted = True
            model.ridge_fitted = True
            self.logger.info(f"Loaded Ridge state from {ridge_path}")

            lstm_path = (repo_root / Path(self.cfg.model.lstm_path)).resolve()
            lstm_state_dict = torch.load(lstm_path, map_location=self.device)
            model.lstm.load_state_dict(lstm_state_dict)
            self.logger.info(f"Loaded LSTM state from {lstm_path}")

            if self.cfg.model.use_meta_learning:
                meta_price_path = (repo_root / Path(self.cfg.model.meta_price_path)).resolve()
                meta_price_state_dict = torch.load(meta_price_path, map_location=self.device)
                model.meta_price.load_state_dict(meta_price_state_dict)
                self.logger.info(f"Loaded MetaPrice state from {meta_price_path}")

        except FileNotFoundError as e:
            self.logger.error(f"Model file not found: {e}")
            raise
        except RuntimeError as e:
            self.logger.error(f"Error loading model state: {e}")
            raise

        # Generate predictions
        predictions = []
        with torch.no_grad():
            for xi in x:
                xi = xi.unsqueeze(0).to(self.device)  # Add batch dimension
                price_pred, _, _, _ = model(xi)  # pylint: disable=not-callable
                if price_pred.dim() > 1:
                    price_pred = price_pred.squeeze(-1)  # Ensure 1D output
                predictions.append(price_pred.item())

        self.logger.info(f"Generated {len(predictions)} predictions using EnsembleModule")
        return predictions

    def save_predictions(self, predictions):
        """
        Saves predictions with additional metrics to a CSV file.

        Args:
            predictions (list): List of model predictions (scaled).
        """
        script_dir = Path(__file__).parent
        repo_root = script_dir.parent
        scaler_path = (repo_root / Path(self.cfg.data_module.price_y_scaled_save_path)).resolve()
        target_scaler = joblib.load(scaler_path)
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = target_scaler.inverse_transform(predictions).flatten()
        df_predictions = pd.DataFrame(predictions, columns=['Predicted'])

        evaluate_data_path = (repo_root / Path(self.cfg.features.preprocessing_data_path)).resolve()
        # Load preprocessed data and align to predictions
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

        # Calculate direction: 'yes' if predicted and actual directions match, 'no' otherwise
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

    def save_trading_metrics(self, portfolio_values):
        """
        Save trading performance metrics to a JSON file.

        Calculate key trading metrics from portfolio values, including Sharpe ratio,
        annual return, maximum drawdown, win rate, and total return, and save them
        to a JSON file in the data/metrics directory.

        Args:
            portfolio_values (list): List of portfolio values over time from the trading simulation.

        Returns:
            None
        """

        script_dir = Path(__file__).parent
        repo_root = script_dir.parent
        save_path = (repo_root / "data/predictions/trading_metrics.json").resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if len(portfolio_values) > 1:
            returns = pd.Series(portfolio_values).pct_change().dropna()

            metrics = {
                'sharpe_ratio': float(qs.stats.sharpe(returns)),
                'sortino_ratio': float(qs.stats.sortino(returns)),
                'annual_return': float(qs.stats.cagr(returns)),
                'max_drawdown': float(qs.stats.max_drawdown(returns)),
                'win_rate': float(qs.stats.win_rate(returns)),
                'total_return': float((portfolio_values[-1] / portfolio_values[0]) - 1),
            }

            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2)

            self.logger.info(f"Saved trading metrics to: {save_path}")

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
            cfg=self.cfg,
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

        cerebro.broker.setcash(self.cfg.backtest.starting_cash)
        cerebro.broker.setcommission(commission=self.cfg.backtest.commission)

        starting_value = cerebro.broker.getvalue()

        self.logger.info(f"Starting Portfolio Value: {starting_value:.2f}")
        # Track portfolio value after each bar
        portfolio_values = []

        class ValueTracker(bt.Analyzer):
            """
            Custom analyzer to track portfolio values.
            """
            def __init__(self):
                super().__init__()
                self.values = []

            def next(self):
                strat = getattr(self, 'strategy', None) or getattr(self, '_owner', None)
                if strat is None:
                    return
                broker = getattr(strat, 'broker', None)
                if broker is None:
                    return
                self.values.append(broker.getvalue())
        cerebro.addanalyzer(ValueTracker, _name='valtracker')
        results = cerebro.run()
        self.logger.info(f"Starting Portfolio Value: {starting_value:.2f}")
        self.logger.info(f"Ending Portfolio Value: {cerebro.broker.getvalue():.2f}")

        # Get portfolio values from analyzer
        valtracker = results[0].analyzers.valtracker
        portfolio_values = valtracker.values
        if len(portfolio_values) > 1:
            # Calculate daily returns
            returns = pd.Series(portfolio_values).pct_change().dropna()
            sharpe = qs.stats.sharpe(returns)
            self.logger.info(f"QuantStats Sharpe Ratio: {sharpe:.4f}")
            sortino = qs.stats.sortino(returns)
            self.logger.info(f"QuantStats Sortino Ratio: {sortino:.4f}")
            max_drawdown = qs.stats.max_drawdown(returns)
            self.logger.info(f"QuantStats Max Drawdown: {max_drawdown * 100:.2f}%")
            annual_return = qs.stats.cagr(returns)
            self.logger.info(f"QuantStats Annual Return: {annual_return * 100:.2f}%")
            win_rate = qs.stats.win_rate(returns)
            self.logger.info(f"QuantStats Win Rate: {win_rate * 100:.2f}%")

            self.save_trading_metrics(portfolio_values)

        else:
            self.logger.warning("Not enough data to calculate metrics.")

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
    params = dict(
        vol_window=19,     # lookback for volatility
        k=1.4,             # threshold multiplier
        base_size=500,     # base position size
        stop_loss=0.03,    # stop loss
        take_profit=0.1,  # take profit
        min_hold=1,
        max_hold=28,        # max bars to hold
        max_size=5000,      # maximum position size
        trend_window=30,      # lookback for trend
        atr_window=14,      # ATR window for volatility-adjusted holding period
    )

    def __init__(self, cfg: DictConfig, predictions, close_prices, dates=None):
        super().__init__()
        self.cfg = cfg
        self.predictions = predictions
        self.close_prices = close_prices
        self.dates = dates
        self.buyprice = None
        self.buycomm = None
        self.order = None
        self.bar_executed = None
        self.hold_counter = 0
        self.recent_trades = []
        self.data.close = self.datas[0].close
        self.returns = pd.Series(close_prices).pct_change()
        self.logger = get_logger("StrategySimulation")
        self.atr = bt.indicators.ATR(self.datas[0], period=self.p.atr_window)

    def log(self, txt, dt=None):
        """
        Log a message with a specified or derived date.

        Parameters
        ----------
        txt : str
            The message to be logged.
        dt : datetime.date, optional
            The date to associate with the log message. If None, derives the date
            from self.dates or the first data's datetime.
        """
        idx = len(self)
        if self.dates is not None and idx < len(self.dates):
            dt = pd.to_datetime(self.dates[idx]).date()
        else:
            dt = dt or self.datas[0].datetime.date(0)
        self.logger.info(f"{dt.isoformat()} {txt}")

    def next(self):
        idx = len(self)
        if idx >= len(self.predictions) or idx < self.p.vol_window:
            return

        current_value = self.broker.getvalue()
        starting_value = self.cfg.backtest.starting_cash
        current_drawdown = (starting_value - current_value) / starting_value

        if current_drawdown > 0.15:  # 15% drawdown limit
            self.log(f"Heat control: Drawdown {current_drawdown:.2%}, skipping new entries")
            return

        current_close = self.close_prices[idx]
        predicted_close = self.predictions[idx]
        self.log(f" ActualClose ={current_close} | Predicted Close ={predicted_close}")
        if np.isnan(predicted_close) or np.isnan(current_close):
            return

        predicted_return = (predicted_close - current_close) / current_close
        sigma = self.returns.iloc[idx - self.p.vol_window:idx].std()

        sma_50 = np.mean(self.close_prices[idx-self.p.trend_window:idx])
        is_uptrend = current_close > sma_50
        atr_value = self.atr[0]
        atr_normalized = atr_value / current_close if current_close != 0 else 0

        dynamic_max_hold = self.p.min_hold + int(10 * min(atr_normalized / 0.02, 1.0))

        dynamic_take_profit = self.p.take_profit * (1 + atr_normalized / 0.1)

        # Exit logic
        if self.position:
            self.hold_counter += 1
            pnl = (current_close - self.position.price) / self.position.price
            exit_condition = False
            if self.position.size > 0:  # Long position
                exit_condition = (
                    pnl <= -self.p.stop_loss or
                    pnl >= dynamic_take_profit or
                    self.hold_counter >= (dynamic_max_hold if is_uptrend else self.p.min_hold)
                )
            elif self.position.size < 0:  # Short position
                exit_condition = (
                    pnl >= self.p.stop_loss or
                    pnl <= -dynamic_take_profit or
                    self.hold_counter >= (dynamic_max_hold if not is_uptrend else self.p.min_hold)
                )
            if exit_condition:
                self.log(f"CLOSE, {current_close:.2f}, Profit={pnl:.2%}, "
                            f"Hold={self.hold_counter} bars")
                self.close()
                self.hold_counter = 0
            return

        # Entry logic
        if abs(predicted_return) > self.p.k * sigma:
            recent_returns = self.returns.iloc[idx-3:idx].mean()
            # Long trades: uptrend + positive prediction + not strongly negative momentum
            if predicted_return > 0 and is_uptrend and recent_returns > -0.01:
                confidence = predicted_return / (self.p.k * sigma)
                recent_window = self.recent_trades[-10:]
                if recent_window:
                    recent_win_rate = sum(recent_window) / len(recent_window)
                else:
                    recent_win_rate = 0.5

                if recent_win_rate > 0.6:
                    size_multiplier = 2
                elif recent_win_rate < 0.3:
                    size_multiplier = 0.5
                else:
                    size_multiplier = 1.0

                size = int(min(confidence * self.p.base_size * size_multiplier, self.p.max_size))
                self.buy(size=max(1, size))

            # Short trades: downtrend + negative prediction + not strongly positive momentum
            elif predicted_return < 0 and not is_uptrend and recent_returns < 0.01:
                confidence = abs(predicted_return) / (self.p.k * sigma)
                recent_window = self.recent_trades[-10:]
                if recent_window:
                    recent_win_rate = sum(recent_window) / len(recent_window)
                else:
                    recent_win_rate = 0.5

                if recent_win_rate > 0.6:
                    size_multiplier = 2
                elif recent_win_rate < 0.3:
                    size_multiplier = 0.5
                else:
                    size_multiplier = 1.0

                size = int(min(confidence * self.p.base_size * size_multiplier, self.p.max_size))
                self.sell(size=max(1, size))

    def notify_order(self, order):
        """
        Handles order notifications from the broker.

        Args:
            order (bt.Order): The order object with status updates.
        """
        if order.status in [order.Submitted, order.Accepted]:
            return

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
