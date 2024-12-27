import numpy as np
from sklearn.preprocessing import MinMaxScaler

def scale_X_0_1(X):
    """
    Scales 3D array X (num_samples, input_window, num_features) to [0,1].
    
    Returns:
      X_scaled (same shape as X)
      scaler (the fitted MinMaxScaler object)
    """
    num_samples, input_window, num_features = X.shape

    # 1) Flatten to 2D for scaling
    X_reshaped = X.reshape(-1, num_features)  
    # shape: (num_samples * input_window, num_features)

    # 2) Fit and transform
    scaler = MinMaxScaler(feature_range=(0,1))
    X_reshaped_scaled = scaler.fit_transform(X_reshaped)

    # 3) Reshape back to 3D
    X_scaled = X_reshaped_scaled.reshape(num_samples, input_window, num_features)

    return X_scaled, scaler
def encode_labels(labels):
    """
    Convert string labels ('long', 'short', 'flat') to numeric codes.
    For example:
       'long' -> 2
       'flat' -> 1
       'short'-> 0
    
    Returns: np.array of shape (num_samples,) with integer codes.
    """
    # Your desired mapping (choose any scheme you like)
    label_map = {
        "long":  2,
        "flat":  1,
        "short": 0
    }
    encoded = [label_map[label] for label in labels]
    return np.array(encoded, dtype=int)

def compute_sample_profit(x_sample, y_sample, label, close_idx=3, commission_rate=0.0005):
    """
    Compute profit for a single sample (X[i], Y[i], label[i]).
    This is a 'per-sample' version of your compute_profit logic.
    """
    price_in_end = x_sample[-1, close_idx]
    price_out_end = y_sample[-1, close_idx]

    # Commission on both entry & exit (simplified 0.05% approach)
    trade_commission = commission_rate * price_in_end + commission_rate * price_out_end

    if label == "long":
        trade_profit = (price_out_end - price_in_end) - trade_commission
    elif label == "short":
        trade_profit = (price_in_end - price_out_end) - trade_commission
    else:  # "flat"
        trade_profit = 0.0

    return trade_profit

def get_profitable_indices(X, Y, labels, close_idx=3, commission_rate=0.0005):
    """
    Zips (X, Y, labels) and returns a list of indices i where the
    computed profit > 0.
    """
    profitable_idxs = []
    for i, (x_sample, y_sample, lbl) in enumerate(zip(X, Y, labels)):
        trade_profit = compute_sample_profit(
            x_sample, 
            y_sample, 
            lbl, 
            close_idx=close_idx, 
            commission_rate=commission_rate
        )
        if trade_profit > 0:
            profitable_idxs.append(i)
    return profitable_idxs
def get_flat_indices(labels, amount):
    """
    Zips (X, Y, labels) and returns a list of indices i where the
    computed profit > 0.
    """
    k = 0
    flat_idxs = []
    for i, v in enumerate(labels):
        if k == amount : break
        if v == "flat":
            flat_idxs.append(i)
            k+=1
    return flat_idxs