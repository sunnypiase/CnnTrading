import numpy as np
from dataclasses import dataclass

@dataclass
class TradingStatisticDTO:
    initial_capital: float
    final_capital: float
    total_profit: float
    average_profit: float
    return_on_investment: float
    num_trades: int
    long_trades: int
    short_trades: int
    flat_trades: int
    position_size_per_trade: float  # Position size in dollars

def compute_trading_statistics(
    X,
    y,
    labels,
    initial_capital,
    position_size_dollars=1000.0,  # Fixed position size per trade in dollars
    close_idx=3,
    commission_rate=0.0005
):
    """
    Computes trading statistics based on initial capital, fixed dollar position size, and trade profits.

    Args:
        X (np.ndarray): Input prices, shape (num_samples, sequence_length, num_features).
        y (np.ndarray): Output prices, shape (num_samples, sequence_length, num_features).
        labels (list or np.ndarray): Trade labels for each sample ("long", "short", "flat").
        initial_capital (float): The starting capital for trading.
        position_size_dollars (float, optional): Fixed dollar amount per trade. Defaults to 1000.0.
        close_idx (int, optional): Index of the closing price in the feature array. Defaults to 3.
        commission_rate (float, optional): Commission rate per trade (as a decimal). Defaults to 0.0005.

    Returns:
        TradingStatisticDTO: An object containing various trading statistics.
    """
    # Parameter Validation
    if initial_capital <= 0:
        raise ValueError("Initial capital must be greater than zero.")
    if position_size_dollars <= 0:
        raise ValueError("Position size must be greater than zero.")
    if not isinstance(labels, (list, np.ndarray)):
        raise TypeError("Labels must be a list or numpy array.")
    if X.shape[0] != len(labels):
        raise ValueError("Number of samples in X must match the number of labels.")

    num_samples = X.shape[0]
    total_profit = 0.0
    long_trades = 0
    short_trades = 0
    flat_trades = 0

    for i in range(num_samples):
        price_in_end = X[i, -1, close_idx]
        price_out_end = y[i, -1, close_idx]

        # Calculate number of units based on position size in dollars
        units = position_size_dollars / price_in_end

        # Commission on both entry & exit (based on position size in dollars)
        trade_commission = commission_rate * position_size_dollars * 2  # Entry and exit

        if labels[i] == "long":
            # Profit: (price_out - price_in) * units - commission
            trade_profit = (price_out_end - price_in_end) * units - trade_commission
            long_trades += 1
        elif labels[i] == "short":
            # Profit: (price_in - price_out) * units - commission
            trade_profit = (price_in_end - price_out_end) * units - trade_commission
            short_trades += 1
        else:
            trade_profit = 0.0
            flat_trades += 1

        total_profit += trade_profit

    average_profit = total_profit / num_samples if num_samples > 0 else 0.0
    final_capital = initial_capital + total_profit
    return_on_investment = (total_profit / initial_capital) * 100 if initial_capital != 0 else 0.0

    return TradingStatisticDTO(
        initial_capital=initial_capital,
        final_capital=final_capital,
        total_profit=total_profit,
        average_profit=average_profit,
        return_on_investment=return_on_investment,
        num_trades=num_samples,
        long_trades=long_trades,
        short_trades=short_trades,
        flat_trades=flat_trades,
        position_size_per_trade=position_size_dollars
    )

def compute_profit(X, y, labels, close_idx=3, commission_rate=0.0005):
    """
    For each sequence i:
      - price_in_end = X[i, -1, close_idx]
      - price_out_end = y[i, -1, close_idx]

      - If label == "long":
          trade_profit = (price_out_end - price_in_end) - commission
        If label == "short":
          trade_profit = (price_in_end - price_out_end) - commission
        If label == "flat":
          trade_profit = 0

      Commission is assumed to be paid on both entry & exit.
      That is: commission = commission_rate * price_in_end + commission_rate * price_out_end

    Returns the average profit across all sequences.
    """
    total_profit = 0.0
    num_samples = X.shape[0]

    for i in range(num_samples):
        price_in_end = X[i, -1, close_idx]
        price_out_end = y[i, -1, close_idx]

        # Commission on both entry & exit
        trade_commission = commission_rate * price_in_end + commission_rate * price_out_end
        
        if labels[i] == "long":
            # Profit is final - initial - commission
            trade_profit = (price_out_end - price_in_end) - trade_commission
        elif labels[i] == "short":
            # Profit is initial - final - commission
            trade_profit = (price_in_end - price_out_end) - trade_commission
        else:
            trade_profit = 0.0
        
        total_profit += trade_profit
    
    return total_profit / num_samples if num_samples > 0 else 0.0

def compute_sharpe_ratio(X, y, labels, close_idx=3, risk_free=0.00254, commission_rate=0.0005):
    """
    Computes a simple Sharpe ratio. Interprets each labeled trade as:
      return = (PnL net of commissions) / initial_price

    Where:
      - If "long": 
          raw_pnl = price_out_end - price_in_end
      - If "short": 
          raw_pnl = price_in_end - price_out_end
      - If "flat": 
          raw_pnl = 0
      - Commission is subtracted from raw_pnl
        commission = commission_rate * price_in_end + commission_rate * price_out_end

    risk_free: annual risk-free rate (e.g., 0.02 for 2%). Here it's treated as a simple offset 
               on the average return for Sharpe calculations, though strictly speaking you might 
               annualize returns differently depending on your timeframe.

    Returns the computed Sharpe ratio.
    """
    returns = []
    num_samples = X.shape[0]

    for i in range(num_samples):
        price_in_end = X[i, -1, close_idx]
        price_out_end = y[i, -1, close_idx]
        
        if labels[i] in ("long", "short") and price_in_end != 0:
            # Commission
            trade_commission = commission_rate * price_in_end + commission_rate * price_out_end

            if labels[i] == "long":
                # PnL net of commission
                pnl = (price_out_end - price_in_end) - trade_commission
            else:  # "short"
                pnl = (price_in_end - price_out_end) - trade_commission

            trade_return = pnl / price_in_end
        else:
            # "flat" or price_in_end == 0
            trade_return = 0.0
        
        returns.append(trade_return)

    if len(returns) == 0:
        return 0.0
    
    arr = np.array(returns)
    mean_return = np.mean(arr)
    std_return = np.std(arr, ddof=1)
    
    if std_return == 0:
        return 0.0  # no variability => Sharpe is undefined or infinite

    # Basic Sharpe = (mean_return - risk_free) / stdev
    sharpe = (mean_return - risk_free) / std_return
    return sharpe
