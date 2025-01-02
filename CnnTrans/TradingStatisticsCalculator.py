import numpy as np
from TradingStatisticDTO import TradingStatisticDTO

class TradingStatisticsCalculator:
    def __init__(
        self,
        initial_capital: float,
        position_size_dollars: float = 1000.0,  # Fixed position size per trade in dollars
        close_idx: int = 3,
        high_idx: int = 1,  # Index for High price
        low_idx: int = 2,   # Index for Low price
        commission_rate: float = 0.0005,
        tp_percent: float = 0.01,  # 1% Take Profit
        sl_percent: float = 0.005  # 0.5% Stop Loss
    ):
        """
        Initializes the TradingStatisticsCalculator with the given parameters.

        Args:
            initial_capital (float): The starting capital for trading.
            position_size_dollars (float, optional): Fixed dollar amount per trade. Defaults to 1000.0.
            close_idx (int, optional): Index of the closing price in the feature array. Defaults to 3.
            high_idx (int, optional): Index of the high price in the feature array. Defaults to 1.
            low_idx (int, optional): Index of the low price in the feature array. Defaults to 2.
            commission_rate (float, optional): Commission rate per trade (as a decimal). Defaults to 0.0005.
            tp_percent (float, optional): Take Profit percentage (e.g., 0.01 for 1%). Defaults to 0.01.
            sl_percent (float, optional): Stop Loss percentage (e.g., 0.005 for 0.5%). Defaults to 0.005.
        """
        # Parameter Validation
        if initial_capital <= 0:
            raise ValueError("Initial capital must be greater than zero.")
        if position_size_dollars <= 0:
            raise ValueError("Position size must be greater than zero.")
        
        self.position_size_dollars = position_size_dollars
        self.initial_capital = initial_capital
        self.close_idx = close_idx
        self.high_idx = high_idx
        self.low_idx = low_idx
        self.commission_rate = commission_rate
        self.tp_percent = tp_percent
        self.sl_percent = sl_percent
        self.total_profit = 0.0
        self.long_trades = 0
        self.short_trades = 0
        self.flat_trades = 0
        self.num_trades = 0

        # Additional Metrics Variables
        self.win_trades = 0
        self.loss_trades = 0
        self.total_win = 0.0
        self.total_loss = 0.0
        self.trade_profits = []  # List to store individual trade profits for profit factor
        self.current_capital = initial_capital
        self.max_capital = initial_capital
        self.max_drawdown = 0.0
        self.consecutive_wins = 0
        self.max_consecutive_wins = 0
        self.consecutive_losses = 0
        self.max_consecutive_losses = 0
        self.max_win = float('-inf')  # Initialize to negative infinity
        self.max_loss = float('inf')  # Initialize to infinity

        # To track consecutive wins/losses
        self.last_trade_result = None  # 'win' or 'loss'

        self.current_position = 'flat'  # Possible values: 'flat', 'long', 'short'
        self.entry_price = 0.0
        self.tp_level = 0.0
        self.sl_level = 0.0

        # Initialize capital history
        self.capital_history = [initial_capital]

    def process_candle(self, candle: np.ndarray, label: str):
        """
        Processes a single candle with the given label.

        Args:
            candle (np.ndarray): The candle data containing prices.
            label (str): The trade label for the current time step ("long", "short", "flat").
        """
        
        close_price = candle[self.close_idx]
        high_price = candle[self.high_idx]
        low_price = candle[self.low_idx]

        if self.current_position == 'flat':
            # Only consider opening a new position if flat
            if label == "long":
                # Open a long position at Close price
                self.current_position = "long"
                self.entry_price = close_price
                self.long_trades += 1
                self.num_trades += 1
                # Calculate TP and SL levels based on entry price
                self.tp_level = self.entry_price * (1 + self.tp_percent)
                self.sl_level = self.entry_price * (1 - self.sl_percent)
                # Commission on entry
                commission = self.commission_rate * self.position_size_dollars
                self.total_profit -= commission
                self.current_capital -= commission
            elif label == "short":
                # Open a short position at Close price
                self.current_position = "short"
                self.entry_price = close_price
                self.short_trades += 1
                self.num_trades += 1
                # Calculate TP and SL levels based on entry price
                self.tp_level = self.entry_price * (1 - self.tp_percent)
                self.sl_level = self.entry_price * (1 + self.sl_percent)
                # Commission on entry
                commission = self.commission_rate * self.position_size_dollars
                self.total_profit -= commission
                self.current_capital -= commission
            else:
                self.flat_trades += 1
            # Record capital after deciding on position
            self.capital_history.append(self.current_capital)
        else:
            # Currently in a position; check for dynamic adjustments and TP/SL hits
            trade_closed = False
            profit = 0.0

            # **Dynamic Stop-Loss Adjustment on Reversal Labels**
            if (label == "long" and self.current_position == "short") or (label == "short" and self.current_position == "long"):
                if label == "long" and self.current_position == "short":
                    # Potential reversal from short to long
                    # Calculate new SL for the short position based on current price
                    new_sl_level = close_price * (1 + self.sl_percent)
                    # For short positions, SL is above the entry price
                    # Only update SL if new SL is closer to current price than existing SL
                    if new_sl_level < self.sl_level:
                        self.sl_level = new_sl_level  # Update to closer SL
                elif label == "short" and self.current_position == "long":
                    # Potential reversal from long to short
                    # Calculate new SL for the long position based on current price
                    new_sl_level = close_price * (1 - self.sl_percent)
                    # For long positions, SL is below the entry price
                    # Only update SL if new SL is closer to current price than existing SL
                    if new_sl_level > self.sl_level:
                        self.sl_level = new_sl_level  # Update to closer SL

            # **Dynamic Take-Profit Adjustment on Same Labels**
            if label == self.current_position:
                if self.current_position == "long":
                    # Calculate new TP based on current price
                    new_tp_level = close_price * (1 + self.tp_percent)
                    # Update TP only if new TP is further from current price than existing TP
                    if new_tp_level > self.tp_level:
                        self.tp_level = new_tp_level
                elif self.current_position == "short":
                    # Calculate new TP based on current price
                    new_tp_level = close_price * (1 - self.tp_percent)
                    # Update TP only if new TP is further from current price than existing TP
                    if new_tp_level < self.tp_level:
                        self.tp_level = new_tp_level

            # **Check for TP or SL Hits**
            if self.current_position == "long":
                # For long positions, TP is higher than entry, SL is lower
                # Check TP first using High price
                if high_price >= self.tp_level:
                    # Take Profit at TP level
                    exit_price = self.tp_level
                    profit = (exit_price - self.entry_price) * (self.position_size_dollars / self.entry_price) - (self.commission_rate * self.position_size_dollars)
                    trade_closed = True
                # If TP not hit, check SL using Low price
                elif low_price <= self.sl_level:
                    # Stop Loss at SL level
                    exit_price = self.sl_level
                    profit = (exit_price - self.entry_price) * (self.position_size_dollars / self.entry_price) - (self.commission_rate * self.position_size_dollars)
                    trade_closed = True
            elif self.current_position == "short":
                # For short positions, TP is lower than entry, SL is higher
                # Check TP first using Low price
                if low_price <= self.tp_level:
                    # Take Profit at TP level
                    exit_price = self.tp_level
                    profit = (self.entry_price - exit_price) * (self.position_size_dollars / self.entry_price) - (self.commission_rate * self.position_size_dollars)
                    trade_closed = True
                # If TP not hit, check SL using High price
                elif high_price >= self.sl_level:
                    # Stop Loss at SL level
                    exit_price = self.sl_level
                    profit = (self.entry_price - exit_price) * (self.position_size_dollars / self.entry_price) - (self.commission_rate * self.position_size_dollars)
                    trade_closed = True

            if trade_closed:
                self.total_profit += profit
                self.current_capital += profit
                self.trade_profits.append(profit)

                # Update max win and max loss
                if profit > self.max_win:
                    self.max_win = profit
                if profit < self.max_loss:
                    self.max_loss = profit

                # Update drawdown
                if self.current_capital > self.max_capital:
                    self.max_capital = self.current_capital
                drawdown = self.max_capital - self.current_capital
                if drawdown > self.max_drawdown:
                    self.max_drawdown = drawdown

                # Update win/loss statistics
                if profit > 0:
                    self.win_trades += 1
                    self.total_win += profit
                    if self.last_trade_result == 'win':
                        self.consecutive_wins += 1
                    else:
                        self.consecutive_wins = 1
                        self.consecutive_losses = 0
                    self.last_trade_result = 'win'
                    if self.consecutive_wins > self.max_consecutive_wins:
                        self.max_consecutive_wins = self.consecutive_wins
                elif profit < 0:
                    self.loss_trades += 1
                    self.total_loss += abs(profit)
                    if self.last_trade_result == 'loss':
                        self.consecutive_losses += 1
                    else:
                        self.consecutive_losses = 1
                        self.consecutive_wins = 0
                    self.last_trade_result = 'loss'
                    if self.consecutive_losses > self.max_consecutive_losses:
                        self.max_consecutive_losses = self.consecutive_losses
                else:
                    # Break-even trade; do not count as win or loss
                    self.consecutive_wins = 0
                    self.consecutive_losses = 0
                    self.last_trade_result = None

                # Reset position
                self.current_position = "flat"

                # Record capital after closing the trade
                self.capital_history.append(self.current_capital)
            else:
                # Position remains open; record current capital without changes
                self.capital_history.append(self.current_capital)

    def get_statistics(self) -> TradingStatisticDTO:
        """
        Computes and returns the trading statistics.

        Returns:
            TradingStatisticDTO: An object containing various trading statistics.
        """
        # Calculate average profit
        average_profit = self.total_profit / self.num_trades if self.num_trades > 0 else 0.0

        # Calculate ROI
        return_on_investment = (self.total_profit / self.initial_capital) * 100 if self.initial_capital != 0 else 0.0

        # Calculate Win/Loss Rates
        win_rate = (self.win_trades / self.num_trades) * 100 if self.num_trades > 0 else 0.0
        loss_rate = (self.loss_trades / self.num_trades) * 100 if self.num_trades > 0 else 0.0

        # Calculate Average Win and Average Loss
        average_win = (self.total_win / self.win_trades) if self.win_trades > 0 else 0.0
        average_loss = (self.total_loss / self.loss_trades) if self.loss_trades > 0 else 0.0

        # Calculate Profit Factor
        profit_factor = (self.total_win / self.total_loss) if self.total_loss > 0 else float('inf')

        # Handle cases where max_win and max_loss were never updated
        final_max_win = self.max_win if self.max_win != float('-inf') else 0.0
        final_max_loss = self.max_loss if self.max_loss != float('inf') else 0.0

        # Prepare the TradingStatisticDTO with all metrics
        trading_stats = TradingStatisticDTO(
            initial_capital=self.initial_capital,
            final_capital=self.current_capital,
            total_profit=self.total_profit,
            average_profit=average_profit,
            return_on_investment=return_on_investment,
            num_trades=self.num_trades,
            long_trades=self.long_trades,
            short_trades=self.short_trades,
            flat_trades=self.flat_trades,
            position_size_per_trade=self.position_size_dollars,
            win_trades=self.win_trades,
            loss_trades=self.loss_trades,
            win_rate=win_rate,
            loss_rate=loss_rate,
            average_win=average_win,
            average_loss=average_loss,
            profit_factor=profit_factor,
            max_drawdown=self.max_drawdown,
            max_consecutive_wins=self.max_consecutive_wins,
            max_consecutive_losses=self.max_consecutive_losses,
            max_win=final_max_win,
            max_loss=final_max_loss,
            capital_history=self.capital_history.copy(),
        )

        return trading_stats
