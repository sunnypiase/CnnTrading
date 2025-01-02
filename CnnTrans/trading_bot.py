import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import random
import pandas as pd
import datetime
import pytz
import yaml
import threading
import time
import logging
from binance import Client, ThreadedWebsocketManager
from readAndSortCsv import read_and_sort_csv  # Ensure this module is available
from TradingStatisticsCalculator import TradingStatisticsCalculator  # Ensure this module is available
from HybridCNN import HybridCNN  # Ensure this module is available
from ta import trend, momentum, volatility, volume

# Set device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_logging(log_file='trading_bot.log'):
    """
    Sets up logging to both console and a log file.
    """
    # Create a custom logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(log_file)
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)
    
    # Create formatters and add them to handlers
    c_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)
    
    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    
    return logger

def load_config(config_path='config.yaml'):
    """
    Loads configuration from a YAML file.
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def get_dynamic_time_range(minutes_back=183):
    """
    Calculates the start and end time based on the current UTC time and a specified window.
    """
    utc_now = datetime.datetime.utcnow().replace(tzinfo=pytz.UTC)
    end_time = utc_now
    start_time = end_time - datetime.timedelta(minutes=minutes_back)
    return start_time, end_time

def fetch_historical_klines(client, symbol, interval, start_time, end_time, logger):
    """
    Fetches historical kline data from Binance.
    """
    try:
        klines = client.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_str=start_time.strftime("%d %b %Y %H:%M:%S"),
            end_str=end_time.strftime("%d %b %Y %H:%M:%S")
        )
        logger.info(f"Fetched {len(klines)} klines from {start_time} to {end_time}")
        return klines
    except Exception as e:
        logger.error(f"An error occurred while fetching klines: {e}")
        return []

def convert_klines_to_dataframe(klines, logger):
    """
    Converts raw klines data into a pandas DataFrame with appropriate data types.
    """
    columns = [
        'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close Time', 'Quote Asset Volume', 'Number of Trades',
        'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
    ]
    
    df = pd.DataFrame(klines, columns=columns)
    
    # Convert columns to appropriate data types
    columns_to_convert = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'Quote Asset Volume', 'Number of Trades',
        'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume'
    ]
    
    for col in columns_to_convert:
        if col == 'Number of Trades':
            df[col] = df[col].astype(int)
        else:
            df[col] = df[col].astype(float)
    
    logger.info("Converted klines to DataFrame and set data types")
    
    return df

def format_dataframe(df, logger):
    """
    Formats the DataFrame by renaming columns, converting timestamps, and setting the index.
    """
    df['date'] = pd.to_datetime(df['Open Time'], unit='ms')
    df_formatted = df[['date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df_formatted.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }, inplace=True)
    df_formatted.set_index('date', inplace=True)
    logger.info("Formatted DataFrame with required columns and set 'date' as index")
    return df_formatted

def compute_features(df_input, logger):
    # df_input.to_csv("df_input before drpo nan.csv")

    df = df_input.iloc[:]
    df = df.iloc[::-1]
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()

    df['RSI_14'] = momentum.RSIIndicator(close=df['close'], window=14).rsi()

    macd = trend.MACD(close=df['close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff'] = macd.macd_diff()

    bollinger = volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['Bollinger_High'] = bollinger.bollinger_hband()
    df['Bollinger_Low'] = bollinger.bollinger_lband()
    df['Bollinger_Middle'] = bollinger.bollinger_mavg()

    df['ATR_14'] = volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()

    df['OBV'] = volume.OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()

    stochastic = momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3)
    df['Stochastic_%K'] = stochastic.stoch()
    df['Stochastic_%D'] = stochastic.stoch_signal()

    ichimoku = trend.IchimokuIndicator(high=df['high'], low=df['low'], window1=9, window2=26, window3=52)
    df['Ichimoku_A'] = ichimoku.ichimoku_a()
    df['Ichimoku_B'] = ichimoku.ichimoku_b()
    df['Ichimoku_Base_Line'] = ichimoku.ichimoku_base_line()
    df['Ichimoku_Conversion_Line'] = ichimoku.ichimoku_conversion_line()
    # df.to_csv("before drpo nan.csv")
    logger.info(f"DataFrame shape before dropping NaNs: {df.shape}")

    df.dropna(inplace=True)
    logger.info(f"DataFrame shape after dropping NaNs: {df.shape}")
    # df.to_csv("after before drpo nan.csv")

    logger.info("Computed technical indicators and features")
    
    return df.iloc[::-1]
    

class Predictor:
    """
    Handles data scaling and model prediction.
    """
    def __init__(self, model, logger):
        self.model = model
        self.model.eval()
        self.logger = logger
    
    def predict(self, df_input):
        X_single_scaled = MinMaxScaler().fit_transform(df_input)  # shape (183,5)
        if X_single_scaled.shape[0] < 150:
            # Handle cases where not enough data is present
            X_single_scaled = np.pad(
                X_single_scaled,
                ((150 - X_single_scaled.shape[0], 0), (0, 0)),
                'constant',
                constant_values=0
            )
            self.logger.warning(f"Padded input from {X_single_scaled.shape[0]} to 150")
        else:
            X_single_scaled = X_single_scaled[-150:]  # Ensure sequence length is 150
        X_single_scaled = np.expand_dims(X_single_scaled, axis=0)  # => (1,150,5)
        X_single_transposed = np.transpose(X_single_scaled, (0, 2, 1))  # => (1,5,150)
        X_single_tensor = torch.from_numpy(X_single_transposed).float().to(device)
        with torch.no_grad():
            output = self.model(X_single_tensor)   # => shape (1,3)
            _, predicted = torch.max(output, 1)
            predicted_label = predicted.item()  # 0,1,2

        label_map = {0:"short",1:"flat",2:"long"}
        return label_map[predicted_label]

class InputProvider:
    """
    Manages WebSocket connections to Binance and handles incoming candlestick data.
    """
    def __init__(self, initial_df, window_size=183, config_path='config.yaml', logger=None):
        self.window = window_size
        self.df = initial_df.copy()
        # self.df.to_csv("initial_df.csv")
        self.config_path = config_path
        self.lock = threading.Lock()
        self.new_candle = None
        self.logger = logger or logging.getLogger(__name__)
        self.ws_manager = ThreadedWebsocketManager(api_key=api_key, api_secret=api_secret)
        self.ws_manager.start()
        self.ws_manager.start_kline_socket(
            callback=self.handle_socket_message,
            symbol=symbol,
            interval=interval
        )
        self.running = True
        self.logger.info("WebSocket manager started and kline socket opened")
    
    def handle_socket_message(self, msg):
        try:
            if msg['e'] != 'kline':
                return
            k = msg['k']
            if k['x']:  # If the candle is closed
                candle = {
                    'date': pd.to_datetime(k['t'], unit='ms'),
                    'open': float(k['o']),
                    'high': float(k['h']),
                    'low': float(k['l']),
                    'close': float(k['c']),
                    'volume': float(k['v'])
                }
                with self.lock:
                # Create DataFrame with 'date' as index
                    new_candle_df = pd.DataFrame([candle])
                    new_candle_df.set_index('date', inplace=True)
                    
                    # Concatenate without ignoring the index
                    self.df = pd.concat([self.df, new_candle_df])
                    
                    # Ensure the DataFrame has only the last 'window' candles
                    self.df = self.df.iloc[-self.window:].copy()
                    
                    self.new_candle = candle

                self.logger.info(f"New candle added: {candle}")

        except Exception as e:
            self.logger.error(f"Error processing socket message: {e}")
    
    def can_get_next_input(self):
        # Reload config to check if we should continue
        config = load_config(self.config_path)
        if not config.get('run_prediction', False):
            self.running = False
            self.logger.info("run_prediction set to False in config, stopping input provider")
            return False
        with self.lock:
            if self.new_candle is not None:
                return True
            else:
                return False
    
    def get_next_input(self):
        with self.lock:
            if self.new_candle is None:
                return None
            # Compute features on the updated DataFrame
            features_df = compute_features(self.df, self.logger)
            self.new_candle = None
            return features_df.to_numpy()
    
    def stop(self):
        try:
            self.ws_manager.stop()
            self.running = False
            self.logger.info("WebSocket manager stopped")
        except Exception as e:
            self.logger.error(f"Error stopping WebSocket manager: {e}")

def main():
    # Setup logging
    logger = setup_logging()
    logger.info("Starting trading bot")
    
    # Load configuration
    config = load_config()
    
    # Binance API credentials
    global api_key, api_secret
    api_key = config.get('api_key')
    api_secret = config.get('api_secret')
    
    if not api_key or not api_secret:
        logger.error("API key and secret must be provided in config.yaml")
        return
    
    client = Client(api_key, api_secret)
    
    # Define the symbol and time interval
    global symbol, interval
    symbol = 'BTCUSDT'
    interval = Client.KLINE_INTERVAL_1MINUTE
    
    # Get dynamic time range
    start_time, end_time = get_dynamic_time_range()
    
    # Fetch historical klines
    klines = fetch_historical_klines(client, symbol, interval, start_time, end_time, logger)
    
    if not klines:
        logger.error("No klines fetched, exiting.")
        return
    
    # Convert klines to DataFrame
    df_M = convert_klines_to_dataframe(klines, logger)
    
    # Format DataFrame
    df_formatted = format_dataframe(df_M, logger)
    
    # Ensure we have the last 183 candles
    df_formatted = df_formatted.tail(183).copy()
    logger.info(f"DataFrame trimmed to last {len(df_formatted)} candles")
    
    # Initialize Trading Statistics Calculator
    algo_calc = TradingStatisticsCalculator(
        initial_capital=5000.0,
        position_size_dollars=1000.0,
        close_idx=3,
        high_idx=1,
        low_idx=2,
        commission_rate=0.0005,
        tp_percent=0.0034,
        sl_percent=0.0033
    )
    
    # Initialize and load the model
    model = HybridCNN(num_features=22, seq_len=150, num_classes=3)
    model_path = config.get('model_path', 'HybridCNN_best_on_valid_state_dict.pth')  # Update the path as needed
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return
    
    model.to(device)
    
    # Initialize Predictor
    predictor = Predictor(model, logger)
    
    # Initialize InputProvider
    input_provider = InputProvider(initial_df=df_formatted, config_path='config.yaml', logger=logger)
    
    # Define prediction loop
    def run_prediction():
        logger.info("Starting prediction loop")
        try:
            while input_provider.running:
                if input_provider.can_get_next_input():
                    df_input = input_provider.get_next_input()
                    if df_input is not None:
                        predicted_label = predictor.predict(df_input)
                        latest_candle = input_provider.df.iloc[-1]
                        algo_calc.process_candle(latest_candle, predicted_label)
                        logger.info(f"Processed candle with prediction: {predicted_label}")
                else:
                    time.sleep(1)  # Wait before checking again
        except Exception as e:
            logger.error(f"An error occurred in the prediction loop: {e}")
        finally:
            input_provider.stop()
            logger.info("Prediction loop stopped")
    
    # Start prediction loop in separate thread
    prediction_thread = threading.Thread(target=run_prediction, daemon=True)
    prediction_thread.start()
    
    # Keep the main thread alive to allow WebSocket processing
    try:
        while prediction_thread.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, stopping prediction loop")
        input_provider.stop()
        prediction_thread.join()
    
    # Retrieve and log statistics
    algo_stats = algo_calc.get_statistics()
    stats_df = algo_stats.to_dataframe().T
    logger.info("Trading statistics:")
    logger.info(f"\n{stats_df}")
    
    # Optionally, save statistics to a CSV file
    stats_df.to_csv("trading_statistics.csv")
    logger.info("Trading statistics saved to 'trading_statistics.csv'")

if __name__ == "__main__":
    main()
