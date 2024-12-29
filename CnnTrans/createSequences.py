import numpy as np

def create_sequences(df, input_window=500, output_window=60, step=60):
    """
    Creates input and output sequences from the DataFrame with a specified step size.

    Args:
        df (pd.DataFrame): The input DataFrame containing time-series data.
        input_window (int): Number of timesteps in the input sequence.
        output_window (int): Number of timesteps in the output sequence.
        step (int): Step size between consecutive sequences to reduce overlap.

    Returns:
        X (np.ndarray): Input sequences of shape (num_samples, input_window, num_features).
        y (np.ndarray): Output sequences of shape (num_samples, output_window, num_features).
    """
    # Flip to ascending inside this function for correct sequence ordering
    df_asc = df.iloc[::-1]
    data = df_asc.values  # Convert to NumPy array (oldest -> newest)

    num_features = data.shape[1]
    total_length = input_window + output_window
    num_samples = (len(data) - total_length) // step + 1

    if num_samples <= 0:
        raise ValueError("Not enough data to create even one sequence with the given step size.")

    X = np.zeros((num_samples, input_window, num_features))
    y = np.zeros((num_samples, output_window, num_features))

    for i in range(num_samples):
        start = i * step
        end = start + input_window
        X[i] = data[start:end]
        y[i] = data[end:end + output_window]

    return X, y
