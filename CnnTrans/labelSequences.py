import numpy as np
from scipy.stats import ranksums

def precompute_label_info(
    X: np.ndarray,
    y: np.ndarray,
    close_idx=3,
    high_idx=1,
    low_idx=2,
    volume_idx=4
):
    """
    Precompute the statistics needed for labeling, but do NOT apply the
    alpha/beta/gamma/threshold logic yet.

    Returns:
      direction_pct_arr:  shape (num_samples,)
      range_diff_arr:     shape (num_samples,)
      p_val_arr:          shape (num_samples,) from Wilcoxon rank-sum
      median_in_arr:      shape (num_samples,) median volume for input window
      median_out_arr:     shape (num_samples,) median volume for output window
    """
    num_samples = X.shape[0]

    direction_pct_arr = np.zeros(num_samples, dtype=float)
    range_diff_arr = np.zeros(num_samples, dtype=float)
    p_val_arr = np.zeros(num_samples, dtype=float)
    median_in_arr = np.zeros(num_samples, dtype=float)
    median_out_arr = np.zeros(num_samples, dtype=float)
    
    for i in range(num_samples):
        X_window = X[i]  # (input_window, num_features)
        y_window = y[i]  # (output_window, num_features)

        price_in_end = X_window[-1, close_idx]
        price_out_end = y_window[-1, close_idx]
        
        # ----- Price Range -----
        in_high = np.max(X_window[:, high_idx])
        in_low  = np.min(X_window[:, low_idx])
        out_high = np.max(y_window[:, high_idx])
        out_low  = np.min(y_window[:, low_idx])
        
        if price_in_end == 0:
            range_in_pct  = 0
            range_out_pct = 0
        else:
            range_in_pct  = ((in_high - in_low)   / price_in_end) * 100
            range_out_pct = ((out_high - out_low) / price_in_end) * 100
        range_diff = range_out_pct - range_in_pct
        range_diff_arr[i] = range_diff

        # ----- Price Direction % -----
        if price_in_end == 0:
            direction_pct = 0
        else:
            direction_pct = ((price_out_end - price_in_end) / price_in_end) * 100
        direction_pct_arr[i] = direction_pct

        # ----- Volume Wilcoxon Test -----
        vol_in = X_window[:, volume_idx]
        vol_out = y_window[:, volume_idx]
        
        # Note: If you want "less", "greater", or "two-sided" depends on your use case
        stat, p_value = ranksums(vol_in, vol_out, alternative="less")
        p_val_arr[i] = p_value

        # We store median volumes for quick usage
        median_in = np.median(vol_in)
        median_out = np.median(vol_out)
        median_in_arr[i] = median_in
        median_out_arr[i] = median_out
    
    return (direction_pct_arr, range_diff_arr, p_val_arr, 
            median_in_arr, median_out_arr)

def get_labels_from_precomputed(
    direction_pct_arr,
    range_diff_arr,
    p_val_arr,
    median_in_arr,
    median_out_arr,
    alpha=0.7,
    beta=0.3,
    gamma=0.5,
    threshold=1.0
):
    """
    Given the precomputed arrays, apply alpha, beta, gamma, threshold
    to assign "long", "short", or "flat" for each sample.

    Returns: labels, shape (num_samples,)
    """
    num_samples = len(direction_pct_arr)
    labels = np.empty(num_samples, dtype=object)

    for i in range(num_samples):
        direction_pct = direction_pct_arr[i]
        range_diff = range_diff_arr[i]
        p_val = p_val_arr[i]
        med_in = median_in_arr[i]
        med_out = median_out_arr[i]

        # Volume Score
        if p_val < 0.05:
            volume_score = 1
        else:
            volume_score = 0.0

        # Weighted Score
        score = alpha * direction_pct + beta * range_diff
        # Final Label
        if direction_pct * volume_score > 0 and score > threshold:
            labels[i] = "long"
        elif direction_pct * volume_score < 0 and score < -threshold:
            labels[i] = "short"
        else:
            labels[i] = "flat"

    return labels
