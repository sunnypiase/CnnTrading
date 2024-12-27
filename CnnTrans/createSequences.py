import numpy as np

def create_sequences(df, input_window=500, output_window=60):
    """
    Creates input and output sequences from the DataFrame.
    Even though df is in descending order, we flip it (df.iloc[::-1])
    so that index=0 in 'data' is the oldest candle.
    That ensures slicing X[i], y[i] proceeds from older to newer in chronological order.
    """
    # Flip to ascending inside this function for correct sequence ordering
    df_asc = df.iloc[::-1]
    data = df_asc.values  # Convert to NumPy array (oldest -> newest)

    num_features = data.shape[1]
    total_length = input_window + output_window
    num_samples = len(data) - total_length + 1

    if num_samples <= 0:
        raise ValueError("Not enough data to create even one sequence.")

    X = np.zeros((num_samples, input_window, num_features))
    y = np.zeros((num_samples, output_window, num_features))

    for i in range(num_samples):
        X[i] = data[i : i + input_window]
        y[i] = data[i + input_window : i + total_length]

    return X, y
