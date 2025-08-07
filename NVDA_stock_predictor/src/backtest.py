import pandas as pd
from pathlib import Path
import hydra
from omegaconf import DictConfig
import numpy as np
import joblib
import torch
import backtrader as bt

from data_loader import load_data
from feature_engineering import feature_engineering
from model_Ensemble import EnsembleModule


class DataProcessor:
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

    def load_data(self):
        print(f"Loading data for {self.cfg.data_loader.TICKER} from {self.cfg.data_loader.START_DATE} to {self.cfg.data_loader.END_DATE}")
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
        print(f"Engineering features for {self.cfg.data_loader.TICKER}")
        script_dir = Path(__file__).parent  # /path/to/repo/NVDA_stock_predictor/src
        repo_root = script_dir.parent  # /path/to/repo/NVDA_stock_predictor
        raw_data_path = repo_root / self.cfg.data_loader.raw_data_path.lstrip('../')
        save_data_path = repo_root / self.cfg.features.preprocessing_data_path.lstrip('../')
        dataFrame = pd.read_csv(
            raw_data_path, 
            header=0, 
            index_col=0, 
            parse_dates=True
        )
        return feature_engineering(dataFrame, self.cfg, save_data_path)
    
    def data_module(self, dataFrame):
        print(f"Converting features to tensors for {self.cfg.data_loader.TICKER}")
        window_size = self.cfg.data_module.window_size
        target_col = self.cfg.data_module.target_col
        
        features = dataFrame.drop(columns=[target_col])
        target = dataFrame[target_col]
        
        x, y = [], []
        for i in range(window_size, len(dataFrame)):
            x.append(features.iloc[i-window_size:i].values)  # past window_size days features
            y.append(target.iloc[i])                         # target is the value at day i
        
        x = np.array(x)
        y = np.array(y)
        
        print(f"Created {len(x)} windows with shape {x.shape}")
        print(f"Target range before scaling: [{y.min():.6f}, {y.max():.6f}]")
        
        print("Scaling features and targets...")
        script_dir = Path(__file__).parent  # /path/to/repo/NVDA_stock_predictor/src
        repo_root = script_dir.parent  # /path/to/repo/NVDA_stock_predictor
        getFeatureScaler = repo_root / self.cfg.data_module.feature_scaler_path.lstrip('../')
        getTargetScaler = repo_root / self.cfg.data_module.target_scaler_path.lstrip('../')
        feature_scaler = joblib.load(getFeatureScaler)
        target_scaler = joblib.load(getTargetScaler)
        
        scaledX = feature_scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
        scaledY = target_scaler.transform(y.reshape(-1, 1)).flatten()
        
        print(f"Feature range after scaling: [{scaledX.min():.6f}, {scaledX.max():.6f}]")
        print(f"Target range after scaling: [{scaledY.min():.6f}, {scaledY.max():.6f}]")
        
        scaledX = torch.tensor(scaledX, dtype=torch.float32)
        scaledY = torch.tensor(scaledY, dtype=torch.float32)
        
        torch.save(scaledX, repo_root / self.cfg.data_module.x_scaled_save_path.lstrip('../'))
        torch.save(scaledY, repo_root / self.cfg.data_module.y_scaled_save_path.lstrip('../'))
        return scaledX, scaledY


class MakePredictions:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def load_model(self):
        script_dir = Path(__file__).parent  # /path/to/repo/NVDA_stock_predictor/src
        repo_root = script_dir.parent  # /path/to/repo/NVDA_stock_predict
        backtestX = repo_root / self.cfg.data_module.x_scaled_save_path.lstrip('../')
        backtestY = repo_root / self.cfg.data_module.y_scaled_save_path.lstrip('../')
        x = torch.load(backtestX)
        y = torch.load(backtestY)
        print(f"Loaded {len(x)} samples with shape {x.shape} and target shape {y.shape}")
        model = EnsembleModule(self.cfg)

        cnnPath = repo_root / self.cfg.model.cnnPath.lstrip('../')
        cnn_state_dict = torch.load(cnnPath)
        model.cnn.load_state_dict(cnn_state_dict)
        
        ridgePath = repo_root / self.cfg.model.ridgePath.lstrip('../')
        ridge_state_dict = torch.load(ridgePath)
        weightShape = ridge_state_dict['weight'].shape
        biasShape = ridge_state_dict['bias'].shape
        model.ridge.weight.data = torch.zeros(weightShape, dtype=torch.float32)
        model.ridge.bias.data = torch.zeros(biasShape, dtype=torch.float32)
        model.ridge.load_state_dict(ridge_state_dict)
        model.ridge.is_fitted = True
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        x = x.to(device)
        
        predictions = []
        for i in range(len(x)):
            x_i = x[i].unsqueeze(0)  # add batch dim
            with torch.no_grad():
                cnn_pred = model.cnn(x_i)
                ridge_pred = model.ridge(x_i)
                pred = model.cnnWeight * cnn_pred + model.ridgeWeight * ridge_pred
                if pred.dim() > 0:
                    pred = pred.squeeze(-1)
                predictions.append(pred.item())
        
        print(f"Generated {len(predictions)} predictions using CNN+Ridge")
        return predictions
    
    def savePredictions(self, predictions):
        script_dir = Path(__file__).parent
        repo_root = script_dir.parent
        scalerPath = repo_root / self.cfg.data_module.target_scaler_path.lstrip('../')
        target_scaler = joblib.load(scalerPath)
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = target_scaler.inverse_transform(predictions).flatten()
        dataFramePredictions = pd.DataFrame(predictions, columns=['Predicted'])
        
        evaluate_data_path = repo_root / self.cfg.features.preprocessing_data_path.lstrip('../')
        # load preprocessed data and align to predictions
        df_eval = pd.read_csv(evaluate_data_path, header=0, index_col=0, parse_dates=True)
        dates = df_eval.index[self.cfg.data_module.window_size:]
        dataFramePredictions.insert(0, 'Time', dates)
        close_values = df_eval['Close'].values[self.cfg.data_module.window_size:]
        dataFramePredictions['Predicted'] = dataFramePredictions['Predicted'].round(3)
        close_values = np.round(close_values, 3)
        dataFramePredictions.insert(2, 'Close', close_values)
        error = np.abs(dataFramePredictions['Predicted'] - dataFramePredictions['Close'])
        dataFramePredictions.insert(3, 'Error', error.round(3))
        # Calculate direction: 1 if next value > current, -1 if next < current, 0 if equal
        pred_diff = np.diff(dataFramePredictions['Predicted'])
        close_diff = np.diff(dataFramePredictions['Close'])
        direction = np.where((pred_diff > 0) == (close_diff > 0), 'yes', 'no')
        # Pad with NaN for first row to align length
        direction = np.insert(direction, 0, np.nan)
        dataFramePredictions.insert(4, 'Direction_Match', direction)
        save_path = repo_root / self.cfg.backtest.predictions_save_path.lstrip('../')
        save_path.parent.mkdir(parents=True, exist_ok=True)
        dataFramePredictions.to_csv(save_path, index=False)
        print(f"Saved predictions with time column to {save_path}")


class TradingSimulation:
    def __init__(self, cfg: DictConfig, predictions):
        self.cfg = cfg
        script_dir = Path(__file__).parent
        repo_root = script_dir.parent
        data_path = repo_root / self.cfg.backtest.predictions_save_path.lstrip('../')
        df = pd.read_csv(data_path)
        self.predictions = df['Predicted'].values
        self.close_prices = df['Close'].values
        self.dates = df['Time'].values if 'Time' in df.columns else None

    def run(self):
        cerebro = bt.Cerebro()
        cerebro.addstrategy(StrategySimulation, predictions=self.predictions, close_prices=self.close_prices, dates=self.dates)

        # For Backtrader, create OHLCV DataFrame
        # Use Close from CSV, fill OHLC with Close, Volume=0
        df = pd.DataFrame({
            'Open': self.close_prices,
            'High': self.close_prices,
            'Low': self.close_prices,
            'Close': self.close_prices,
            'Volume': 0
        })
        if self.dates is not None:
            df.index = pd.to_datetime(self.dates)
        data = bt.feeds.PandasData(
            dataname=df,
            datetime=None,
            open='Open',
            high='High',
            low='Low',
            close='Close',
            volume='Volume',
            openinterest=None
        )
        cerebro.adddata(data)

        cerebro.broker.setcash(1000000.00)
        cerebro.broker.setcommission(commission=0.001)

        print(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")
        cerebro.run()
        print(f"Ending Portfolio Value: {cerebro.broker.getvalue():.2f}")

class StrategySimulation(bt.Strategy):
    params = dict(threshold=0.01, size=100)

    def __init__(self, predictions, close_prices, dates=None):
        self.predictions = predictions
        self.close_prices = close_prices
        self.dates = dates
        self.order = None
        self.data.close = self.datas[0].close
    
    def log(self, txt, dt=None):
        idx = len(self)
        if self.dates is not None and idx < len(self.dates):
            dt = pd.to_datetime(self.dates[idx]).date()
        else:
            dt = dt or self.datas[0].datetime.date(0)
        print(f"{dt.isoformat()} {txt}")
    
    def next(self):
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
                self.log('CLOSE POSITION, %.2f' % current_close)
                self.order = self.close()

        # Long signal
        elif predicted_return > self.p.threshold:
            self.log('BUY CREATE, %.2f' % current_close)
            self.order = self.buy(size=self.p.size)

        # Short signal
        elif predicted_return < -self.p.threshold:
            self.log('SELL CREATE, %.2f' % current_close)
            self.order = self.sell(size=self.p.size)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price, order.executed.value, order.executed.comm)
                    )

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log(
                    'SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price, order.executed.value, order.executed.comm)
                    )

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # Write down: no pending order
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' % (trade.pnl, trade.pnlcomm))

@hydra.main(version_base=None, config_path="../configs", config_name="backtest")
def main(cfg: DictConfig):
    try:
        data_processor = DataProcessor(cfg)
        data_processor.load_data()
        dataFrame = data_processor.engineer_features()
        data_processor.data_module(dataFrame)
        make_predictions = MakePredictions(cfg)

        predictions = make_predictions.load_model()
        make_predictions.savePredictions(predictions)
        backtest = TradingSimulation(cfg, predictions)
        backtest.run()
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
