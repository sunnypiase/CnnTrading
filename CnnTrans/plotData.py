import pandas as pd
import mplfinance as mpf
import numpy as np

def plot_input_output_combined(
    df_original: pd.DataFrame, 
    start_idx: int, 
    input_window: int = 500, 
    output_window: int = 60,
    title: str = "Input + Output Window on One Chart"
):
    """
    Plots both input and output windows on a single candlestick chart.
    The output window candles will be overlaid in a different color.

    df_original is in descending order, but we flip it here to ascending for
    intuitive left->right time flow in the chart.
    """
    # Flip to ascending so index=0 is the oldest row (left side of the chart)
    df_asc = df_original.iloc[::-1]
    
    # Extract the input and output from the ascending data
    input_sequence = df_asc.iloc[start_idx : start_idx + input_window]
    output_sequence = df_asc.iloc[start_idx + input_window : start_idx + input_window + output_window]
    
    # Concatenate so the x-axis (dates) is continuous
    combined_df = pd.concat([input_sequence, output_sequence])
    
    # Create a market colors style for the input sequence
    input_market_colors = mpf.make_marketcolors(
        up='green', down='red', edge='inherit', wick='inherit', volume='inherit'
    )
    input_style = mpf.make_mpf_style(marketcolors=input_market_colors)
    
    # Plot the combined data with the input sequence style
    fig, axes = mpf.plot(
        combined_df,
        type='candle',
        style=input_style,
        volume=True,
        mav=(20, 50),
        returnfig=True,
        title=title,
        figsize=(12, 8),
        show_nontrading=True
    )
    ax_main = axes[0]  # main price axis
    
    # Create a market colors style for the output sequence
    output_market_colors = mpf.make_marketcolors(
        up='blue', down='blue', edge='inherit', wick='inherit', volume='inherit'
    )
    output_style = mpf.make_mpf_style(marketcolors=output_market_colors)
    
    # Overlay the output sequence with its own style
    mpf.plot(
        output_sequence,
        type='candle',
        ax=ax_main,
        style=output_style,
        volume=False,
        mav=(20, 50),
        show_nontrading=True
    )
    
    # Show the final chart
    fig.show()


def plot_input_output_combined_with_label(
    input_sequence: np.ndarray,
    output_sequence: np.ndarray,
    label: str,
    title: str = "Input + Output Window on One Chart"
):
    """
    Plots both input and output windows on a single candlestick chart.
    The output window candles will be overlaid in a color depending on 'label':
      - long -> green
      - short -> red
      - flat -> blue

    Args:
        input_sequence (np.ndarray): OHLCV data for the input window (shape: [N, 5])
        output_sequence (np.ndarray): OHLCV data for the output window (shape: [M, 5])
        label (str): The trading signal for the output window ('long', 'short', 'flat').
        title (str): The title of the chart.
    """

    # Ensure input and output arrays have 5 columns (OHLCV)
    if input_sequence.shape[1] != 5 or output_sequence.shape[1] != 5:
        raise ValueError("Input and output sequences must have exactly 5 columns (OHLCV).")

    # Generate a continuous datetime index for plotting
    input_dates = pd.date_range(start="2023-01-01", periods=len(input_sequence), freq="T")
    output_dates = pd.date_range(start=input_dates[-1] + pd.Timedelta(minutes=1), periods=len(output_sequence), freq="T")

    # Convert to DataFrames with appropriate columns
    input_df = pd.DataFrame(input_sequence, columns=["Open", "High", "Low", "Close", "Volume"], index=input_dates)
    output_df = pd.DataFrame(output_sequence, columns=["Open", "High", "Low", "Close", "Volume"], index=output_dates)

    # Concatenate input and output DataFrames
    combined_df = pd.concat([input_df, output_df])

    # Create a market colors style for the input sequence (standard green/red)
    input_market_colors = mpf.make_marketcolors(
        up='green', down='red', edge='inherit', wick='inherit', volume='inherit'
    )
    input_style = mpf.make_mpf_style(marketcolors=input_market_colors)

    # Plot the combined data with the input sequence style
    fig, axes = mpf.plot(
        combined_df,
        type='candle',
        style=input_style,
        volume=True,
        mav=(20, 50),
        returnfig=True,
        title=title,
        figsize=(12, 8),
        show_nontrading=True
    )
    ax_main = axes[0]  # main price axis

    # Determine the output candlestick color based on label
    if label == 'long':
        color_up = 'green'
        color_down = 'green'
    elif label == 'short':
        color_up = 'red'
        color_down = 'red'
    else:  # 'flat'
        color_up = 'blue'
        color_down = 'blue'

    # Create a market colors style for the output sequence
    output_market_colors = mpf.make_marketcolors(
        up=color_up,
        down=color_down,
        edge='inherit',
        wick='inherit',
        volume='inherit'
    )
    output_style = mpf.make_mpf_style(marketcolors=output_market_colors)

    # Overlay the output sequence with its own style
    mpf.plot(
        output_df,
        type='candle',
        ax=ax_main,
        style=output_style,
        volume=False,
        mav=(20, 50),
        show_nontrading=True
    )

    # Show the final chart
    fig.show()
