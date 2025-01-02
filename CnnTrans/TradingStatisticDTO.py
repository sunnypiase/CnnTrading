import numpy as np
import pandas as pd
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from scipy.stats import pearsonr  # Import Pearson correlation function

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
    
    # Additional Metrics
    win_trades: int
    loss_trades: int
    win_rate: float
    loss_rate: float
    average_win: float
    average_loss: float
    profit_factor: float
    max_drawdown: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    max_win: float        # New Metric: Maximum single trade profit
    max_loss: float       # New Metric: Maximum single trade loss
    
    # New Attribute: Capital History
    capital_history: list = field(default_factory=list, repr=False)
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the trading statistics into a Pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing all trading statistics.
        """
        data = {
            'Initial Capital': [self.initial_capital],
            'Final Capital': [self.final_capital],
            'Total Profit': [self.total_profit],
            'Average Profit': [self.average_profit],
            'Return on Investment (ROI)': [self.return_on_investment],
            'Number of Trades': [self.num_trades],
            'Long Trades': [self.long_trades],
            'Short Trades': [self.short_trades],
            'Flat Trades': [self.flat_trades],
            'Position Size per Trade': [self.position_size_per_trade],
            'Winning Trades': [self.win_trades],
            'Losing Trades': [self.loss_trades],
            'Win Rate (%)': [self.win_rate],
            'Loss Rate (%)': [self.loss_rate],
            'Average Win': [self.average_win],
            'Average Loss': [self.average_loss],
            'Profit Factor': [self.profit_factor],
            'Maximum Drawdown': [self.max_drawdown],
            'Max Consecutive Wins': [self.max_consecutive_wins],
            'Max Consecutive Losses': [self.max_consecutive_losses],
            'Max Win': [self.max_win],
            'Max Loss': [self.max_loss],
        }
        df = pd.DataFrame(data)
        return df

    def compare(self, other: 'TradingStatisticDTO', this_name: str, other_name: str) -> pd.DataFrame:
        """
        Compares this TradingStatisticDTO with another instance and returns a DataFrame of differences.

        Args:
            other (TradingStatisticDTO): The other trading statistics to compare against.
            this_name (str): Name/Identifier for the current instance.
            other_name (str): Name/Identifier for the other instance.

        Returns:
            pd.DataFrame: DataFrame containing comparison of metrics between the two instances.
        """
        metrics = {
            'Final Capital': ('final_capital', 'higher is better'),
            'Total Profit': ('total_profit', 'higher is better'),
            'Average Profit': ('average_profit', 'higher is better'),
            'Return on Investment (ROI)': ('return_on_investment', 'higher is better'),
            'Number of Trades': ('num_trades', 'lower is better'),
            'Long Trades': ('long_trades', 'no specific preference'),
            'Short Trades': ('short_trades', 'no specific preference'),
            'Win Trades': ('win_trades', 'higher is better'),
            'Loss Trades': ('loss_trades', 'lower is better'),
            'Win Rate (%)': ('win_rate', 'higher is better'),
            'Loss Rate (%)': ('loss_rate', 'lower is better'),
            'Average Win': ('average_win', 'higher is better'),
            'Average Loss': ('average_loss', 'lower is better'),
            'Profit Factor': ('profit_factor', 'higher is better'),
            'Maximum Drawdown': ('max_drawdown', 'lower is better'),
            'Max Consecutive Wins': ('max_consecutive_wins', 'higher is better'),
            'Max Consecutive Losses': ('max_consecutive_losses', 'lower is better'),
            'Max Win': ('max_win', 'higher is better'),
            'Max Loss': ('max_loss', 'lower is better'),
        }

        comparison_results = []

        for metric_name, (attribute, preference) in metrics.items():
            self_value = getattr(self, attribute)
            other_value = getattr(other, attribute)
            difference = self_value - other_value

            if preference == 'higher is better':
                if difference > 0:
                    status = 'Improved'
                elif difference < 0:
                    status = 'Declined'
                else:
                    status = 'Same'
            elif preference == 'lower is better':
                if difference < 0:
                    status = 'Improved'
                elif difference > 0:
                    status = 'Declined'
                else:
                    status = 'Same'
            else:
                # For metrics where no specific preference is set
                if difference > 0:
                    status = 'Increased'
                elif difference < 0:
                    status = 'Decreased'
                else:
                    status = 'Same'

            # Format the difference
            if isinstance(self_value, float):
                if 'Rate' in metric_name or 'Drawdown' in metric_name:
                    # Format rates and drawdowns with appropriate units
                    if 'Rate' in metric_name:
                        diff_formatted = f"{difference:+.2f}%"
                    elif 'Drawdown' in metric_name:
                        diff_formatted = f"${difference:+,.2f}"
                    else:
                        diff_formatted = f"${difference:+,.2f}"
                else:
                    diff_formatted = f"${difference:+,.2f}"
            else:
                diff_formatted = f"{difference:+}"

            # Prepare display values
            if isinstance(other_value, float):
                if 'Rate' in metric_name:
                    other_display = f"{other_value:.2f}%"
                elif 'Drawdown' in metric_name:
                    other_display = f"${other_value:,.2f}"
                else:
                    other_display = f"${other_value:,.2f}"
            else:
                other_display = f"{other_value}"
            
            if isinstance(self_value, float):
                if 'Rate' in metric_name:
                    self_display = f"{self_value:.2f}%"
                elif 'Drawdown' in metric_name:
                    self_display = f"${self_value:,.2f}"
                else:
                    self_display = f"${self_value:,.2f}"
            else:
                self_display = f"{self_value}"

            comparison_results.append({
                'Metric': metric_name,
                other_name: other_display,
                this_name: self_display,
                'Difference': diff_formatted,
                'Status': status
            })

        comparison_df = pd.DataFrame(comparison_results)
        return comparison_df

    def plot_capital_history(self, title: str = "Capital Over Time"):
        """
        Plots the capital history over time.

        Args:
            title (str, optional): The title of the plot. Defaults to "Capital Over Time".
        """
        if not self.capital_history:
            print("No capital history to plot.")
            return
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.capital_history, label='Capital', color='blue')
        plt.title(title)
        plt.xlabel('Trade Number')
        plt.ylabel('Capital ($)')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def correlation_with_linear_trend(self) -> float:
        """
        Calculates the Pearson correlation coefficient between the actual capital history
        and an ideal linear trend from initial to final capital.

        Returns:
            float: Pearson correlation coefficient. Returns np.nan if calculation is not possible.
        """
        if not self.capital_history:
            print("Capital history is empty.")
            return np.nan
        if len(self.capital_history) < 2:
            print("Not enough data points to calculate correlation.")
            return np.nan
        
        n = len(self.capital_history)
        linear_trend = np.linspace(self.initial_capital, self.final_capital, n)
        
        # Calculate Pearson correlation
        corr_coeff, _ = pearsonr(self.capital_history, linear_trend)
        return corr_coeff
