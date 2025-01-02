# Standard Library Imports
import random
import os

# Third-Party Imports
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from ta import momentum, trend, volume, volatility
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim

# Local Application Imports
from createSequences import create_sequences
from prepareScaledData import  encode_labels, get_flat_indices, get_profitable_indices
from readAndSortCsv import read_and_sort_csv

MODEL_PATH = 'simple1dcnn_state_dict.pth'
required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
file_path = r"C:\GitCnn\CnnTrading\CnnTrans\merged_output.csv"
input_window = 1000
output_window = 60
np.set_printoptions(formatter={'float_kind': lambda x: f'{x:.2f}'})
random.seed(42)
np.random.seed(42)

if os.path.exists("X_train_scaled.npy") and os.path.exists("X_val_scaled.npy") and os.path.exists("X_test_scaled.npy")\
    and os.path.exists("y_train_encoded.npy") and os.path.exists("y_val_encoded.npy") and os.path.exists("y_test_encoded.npy"):
        print(f"Loading scaled data from files...")
        X_train_scaled = np.load("X_train_scaled.npy")
        X_val_scaled = np.load("X_val_scaled.npy")
        X_test_scaled = np.load("X_test_scaled.npy")
        y_train_encoded = np.load("y_train_encoded.npy")
        y_val_encoded   = np.load("y_val_encoded.npy")
        y_test_encoded  = np.load("y_test_encoded.npy")
else:
    # 1) Read & sort in descending order
    df = read_and_sort_csv(file_path, required_columns)

    # Assuming 'df' is your DataFrame with 'open', 'high', 'low', 'close', 'volume'

    # Moving Averages
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()

    # RSI
    df['RSI_14'] = momentum.RSIIndicator(close=df['close'], window=14).rsi()

    # MACD
    macd = trend.MACD(close=df['close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff'] = macd.macd_diff()

    # Bollinger Bands
    bollinger = volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['Bollinger_High'] = bollinger.bollinger_hband()
    df['Bollinger_Low'] = bollinger.bollinger_lband()
    df['Bollinger_Middle'] = bollinger.bollinger_mavg()

    # ATR
    df['ATR_14'] = volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()

    # OBV
    df['OBV'] = volume.OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()

    # Stochastic Oscillator
    stochastic = momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3)
    df['Stochastic_%K'] = stochastic.stoch()
    df['Stochastic_%D'] = stochastic.stoch_signal()

    # Ichimoku Cloud
    ichimoku = trend.IchimokuIndicator(high=df['high'], low=df['low'], window1=9, window2=26, window3=52)
    df['Ichimoku_A'] = ichimoku.ichimoku_a()
    df['Ichimoku_B'] = ichimoku.ichimoku_b()
    df['Ichimoku_Base_Line'] = ichimoku.ichimoku_base_line()
    df['Ichimoku_Conversion_Line'] = ichimoku.ichimoku_conversion_line()

    # Handle missing values
    df.dropna(inplace=True)  # or df.fillna(method='ffill', inplace=True)

    # Display the updated DataFrame
    X, y = create_sequences(df, input_window=input_window, output_window=output_window, step=300)

    def compute_labels_simple(
        X: np.ndarray, 
        y: np.ndarray, 
        close_idx: int = 3, 
        threshold: float = 0.03
    ) -> np.ndarray:
        """
        Computes labels based on the percentage difference between the last
        closing price of the input and output windows.

        Args:
            X: Input window data, shape (num_samples, input_window, num_features).
            y: Output window data, shape (num_samples, output_window, num_features).
            close_idx: Index of the closing price in the feature set.
            threshold: Percentage threshold to classify "long" or "short".

        Returns:
            labels: An array of labels ("long", "short", "flat"), shape (num_samples,).
        """
        num_samples = X.shape[0]
        labels = np.empty(num_samples, dtype=object)

        for i in range(num_samples):
            price_in_end = X[i, -1, close_idx]
            price_out_end = y[i, -1, close_idx]

            if price_in_end == 0:
                labels[i] = "flat"  # Avoid division by zero
                continue

            pct_diff = ((price_out_end - price_in_end) / price_in_end) * 100

            if pct_diff > threshold:
                labels[i] = "long"
            elif pct_diff < -threshold:
                labels[i] = "short"
            else:
                labels[i] = "flat"

        return labels
    labels = compute_labels_simple(X, y, threshold = 0.15)
    train_percent = 0.8    # 80% for training
    val_percent = 0.1      # 10% for validation
    test_percent = 0.1     # 10% for testing

    # Ensure that the percentages sum to 1
    assert train_percent + val_percent + test_percent == 1.0, "Percentages must sum to 1."

    # Calculate the number of samples
    total_samples = len(X)
    train_end = int(train_percent * total_samples)
    val_end = train_end + int(val_percent * total_samples)

    # Split the data
    X_train_slice = X[:train_end]
    y_train_slice = y[:train_end]
    labels_train_slice = labels[:train_end]

    X_val_slice = X[train_end:val_end]
    y_val_slice = y[train_end:val_end]
    labels_val_slice = labels[train_end:val_end]

    X_test_slice = X[val_end:]
    y_test_slice = y[val_end:]
    labels_test_slice = labels[val_end:]

    print("After Splitting:")
    print(f"X_train_slice shape: {X_train_slice.shape}")
    print(f"y_train_slice shape: {y_train_slice.shape}")
    print(f"labels_train_slice shape: {labels_train_slice.shape}\n")

    print(f"X_val_slice shape: {X_val_slice.shape}")
    print(f"y_val_slice shape: {y_val_slice.shape}")
    print(f"labels_val_slice shape: {labels_val_slice.shape}\n")

    print(f"X_test_slice shape: {X_test_slice.shape}")
    print(f"y_test_slice shape: {y_test_slice.shape}")
    print(f"labels_test_slice shape: {labels_test_slice.shape}\n")

    # Save the slices to disk
    # np.save('X_train_slice.npy', X_train_slice)
    # np.save('y_train_slice.npy', y_train_slice)
    # np.save('labels_train_slice.npy', labels_train_slice)

    # np.save('X_val_slice.npy', X_val_slice)
    # np.save('y_val_slice.npy', y_val_slice)
    # np.save('labels_val_slice.npy', labels_val_slice)

    # np.save('X_test_slice.npy', X_test_slice)
    # np.save('y_test_slice.npy', y_test_slice)
    # np.save('labels_test_slice.npy', labels_test_slice)

    # print("Data slices have been saved successfully.")
    del X
    del y
    del labels

    print(len(X_train_slice))
    print(len(X_val_slice))
    print(len(X_test_slice))

    def uniqueLabels(labelsToUnique):
        unique, counts = np.unique(labelsToUnique, return_counts=True)

        print(dict(zip(unique, counts)))

        print(len(np.where(labelsToUnique == 'long')[0]))
        print(len(np.where(labelsToUnique == 'short')[0]))
        print(len(np.where(labelsToUnique == 'flat')[0]))
        return counts.min()

    min_train = uniqueLabels(labels_train_slice)
    min_labels = uniqueLabels(labels_val_slice)
    min_test = uniqueLabels(labels_test_slice)

    def get_indexs_for_slice(x_input, y_input, labels_input, min_input):
        oaoao_long = []
        oaoao_short = []
        idxs = get_profitable_indices(x_input, y_input, labels_input)
        for i, v in enumerate(idxs):
            if labels_input[v] == "short":
                oaoao_short.append((v))
            elif labels_input[v] == "long":
                oaoao_long.append((v))
        minLen = min(len(oaoao_long), len(oaoao_short), min_input)
        print("minlem", minLen)
        flat_idxs = get_flat_indices(labels_input, minLen)
        oaoao_flat = random.sample(flat_idxs, minLen)
        print(len(oaoao_long))
        print(len(oaoao_short))
        print(len(oaoao_flat))
        return oaoao_long[:minLen] + oaoao_short[:minLen] + oaoao_flat

    train_idxs = get_indexs_for_slice(X_train_slice, y_train_slice, labels_train_slice, min_train)
    val_idxs = get_indexs_for_slice(X_val_slice, y_val_slice, labels_val_slice, min_labels)
    test_idxs = get_indexs_for_slice(X_test_slice, y_test_slice, labels_test_slice, min_test)

    # np.random.shuffle(range(len(X_train_slice)))
    # np.random.shuffle(val_idxs)
    # np.random.shuffle(test_idxs)

    # 5) Convert each to np.array
    X_train, y_train, out_train = np.array(X_train_slice[train_idxs]), np.array(labels_train_slice[train_idxs]), np.array(y_train_slice[train_idxs])
    X_val,   y_val, out_val    = np.array(X_val_slice[val_idxs]),   np.array(labels_val_slice[val_idxs]), np.array(y_val_slice[val_idxs])
    X_test,  y_test, out_test  = np.array(X_test_slice[test_idxs]),  np.array(labels_test_slice[test_idxs]), np.array(y_test_slice[test_idxs])

    del X_train_slice
    del X_val_slice
    del X_test_slice
    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)


    def scale_X_0_1_per_sequence(X: np.ndarray) -> np.ndarray:
        """
        Scales each feature (across the time steps) within each sequence of the 3D array X independently to [0, 1].
        A new MinMaxScaler instance is applied for each feature in each sequence.

        Args:
            X: 3D numpy array of shape (num_samples, input_window, num_features).

        Returns:
            X_scaled: Scaled version of X, where each feature is independently scaled within each sequence.
        """
        num_samples, input_window, num_features = X.shape

        # Initialize an array to store scaled data
        X_scaled = np.zeros_like(X)

        for sample_idx in range(num_samples):
            for feature_idx in range(num_features):
                # Extract the time series for a single feature in a single sequence
                feature_sequence = X[sample_idx, :, feature_idx].reshape(-1, 1)
                
                # Create a MinMaxScaler for this specific sequence
                scaler = MinMaxScaler(feature_range=(0, 1))
                
                # Fit and transform the feature sequence
                scaled_sequence = scaler.fit_transform(feature_sequence)
                
                # Assign the scaled sequence back to the corresponding location
                X_scaled[sample_idx, :, feature_idx] = scaled_sequence.flatten()

        return X_scaled



    def add_gaussian_noise(X, noise_factor=0.05):
        """
        Adds Gaussian noise to the input data.

        Parameters:
        - X: numpy array, shape (num_samples, input_window, num_features)
        - noise_factor: float, standard deviation of the Gaussian noise

        Returns:
        - X_noisy: numpy array with added Gaussian noise, same shape as X
        """
        # Generate Gaussian noise
        noise = np.random.normal(loc=0.0, scale=noise_factor, size=X.shape)
        
        # Add noise to the original data
        X_noisy = X + noise
        
        # Clip the values to ensure they remain within [0, 1]
        X_noisy = np.clip(X_noisy, 0.0, 1.0)
        
        return X_noisy


    # Suppose we have:
    # X_train, X_val, X_test as (num_samples, input_window, num_features)
    # y_train, y_val, y_test as string arrays of shape (num_samples,)

    # 1) Scale X's

    X_train_scaled = scale_X_0_1_per_sequence(X_train)
    # X_train_scaled = add_gaussian_noise(X_train_scaled, noise_factor=0.05)

    X_val_scaled = scale_X_0_1_per_sequence(X_val)
    X_test_scaled = scale_X_0_1_per_sequence(X_test)

    del X_train
    del X_val
    del X_test
    # 2) Encode labels
    y_train_encoded = encode_labels(y_train)
    y_val_encoded   = encode_labels(y_val)
    y_test_encoded  = encode_labels(y_test)

    print("X_train_scaled shape:", X_train_scaled.shape)
    print("y_train_encoded shape:", y_train_encoded.shape)
    print("Sample encoded labels:", np.unique(y_train_encoded))

    np.save("X_train_scaled.npy", X_train_scaled)
    np.save("X_val_scaled.npy", X_val_scaled)
    np.save("X_test_scaled.npy", X_test_scaled)

    np.save("y_train_encoded.npy", y_train_encoded)
    np.save("y_val_encoded.npy", y_val_encoded)
    np.save("y_test_encoded.npy", y_test_encoded)


import torch
import torch.nn as nn
import math

class CNNTransformer(nn.Module):
    """
    A Hybrid CNN-Transformer Model for Sequence Classification.

    This model combines multiple convolutional channels with different kernel sizes
    to extract local features from the input sequence. The extracted features are then
    processed by a Transformer encoder to capture global dependencies. Finally, the
    aggregated features are passed through a fully connected layer for classification.

    Args:
        num_features (int): Number of input features per time step. Default is 22.
        seq_len (int): Length of the input sequences. Default is 300.
        num_classes (int): Number of output classes. Default is 3.
        dropout_p (float): Dropout probability for CNN and Transformer dropout layers. Default is 0.3.
        cnn_out_channels (int): Number of output channels for each CNN branch. Default is 64.
        transformer_hidden_dim (int): Hidden dimension size for the Transformer feedforward network. Default is 128.
        transformer_num_heads (int): Number of attention heads in the Transformer. Default is 8.
        transformer_num_layers (int): Number of Transformer encoder layers. Default is 2.
        transformer_dropout (float): Dropout probability within the Transformer. Default is 0.1.
    """
    def __init__(self, num_features=22, seq_len=300, num_classes=3, dropout_p=0.3, 
                 cnn_out_channels=64, transformer_hidden_dim=128, transformer_num_heads=8, 
                 transformer_num_layers=2, transformer_dropout=0.1):
        super(CNNTransformer, self).__init__()
        
        # Define different kernel sizes for the CNN branches
        filter_sizes = [3, 5, 7, 11, 15]
        
        # Initialize CNN branches with varying kernel sizes
        self.cnn_channels = nn.ModuleList([
            self._create_cnn_branch(in_channels=num_features, 
                                    out_channels=cnn_out_channels, 
                                    kernel_size=ks, 
                                    dropout_p=dropout_p)
            for ks in filter_sizes
        ])
        
        # Total number of CNN output channels after concatenation
        self.total_cnn_out = cnn_out_channels * len(filter_sizes)
        
        # Positional Encoding for the Transformer
        self.positional_encoding = self._create_positional_encoding(d_model=self.total_cnn_out, 
                                                                     max_len=seq_len//2, 
                                                                     dropout_p=transformer_dropout)
        
        # Transformer Encoder Layer Configuration
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.total_cnn_out, 
                                                   nhead=transformer_num_heads, 
                                                   dim_feedforward=transformer_hidden_dim,
                                                   dropout=transformer_dropout,
                                                   activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 
                                                         num_layers=transformer_num_layers)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_p)
        
        # Fully Connected Layer for classification
        self.fc = nn.Linear(self.total_cnn_out, num_classes)
        
    def _create_cnn_branch(self, in_channels, out_channels, kernel_size, dropout_p):
        """
        Creates a single CNN branch with Conv1d, ReLU, BatchNorm, Dropout, and MaxPool layers.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            dropout_p (float): Dropout probability.

        Returns:
            nn.Sequential: A sequential container of CNN layers.
        """
        padding = (kernel_size - 1) // 2  # To maintain sequence length
        return nn.Sequential(
            nn.Conv1d(in_channels=in_channels, 
                      out_channels=out_channels, 
                      kernel_size=kernel_size, 
                      padding=padding),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(dropout_p),
            nn.MaxPool1d(kernel_size=2)  # Downsample by a factor of 2
        )
    
    def _create_positional_encoding(self, d_model, max_len, dropout_p):
        """
        Creates a positional encoding module to inject positional information into the model.

        Args:
            d_model (int): Embedding dimension.
            max_len (int): Maximum sequence length.
            dropout_p (float): Dropout probability for positional encoding.

        Returns:
            nn.Sequential: A sequential container with LayerNorm and Dropout.
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
        return nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout_p)
        )
    
    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, seq_len).

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, num_classes).
        """
        # Pass input through each CNN branch
        cnn_features = [branch(x) for branch in self.cnn_channels]  # List of tensors: (batch, out_channels, L)
        
        # Concatenate CNN features along the channel dimension
        cnn_concat = torch.cat(cnn_features, dim=1)  # Shape: (batch, total_cnn_out, L)
        
        # Permute to shape (batch, L, total_cnn_out) for Transformer input
        cnn_concat = cnn_concat.permute(0, 2, 1)  # Shape: (batch, L, total_cnn_out)
        
        # Add positional encoding
        # Ensure that the positional encoding length matches the sequence length
        if cnn_concat.size(1) > self.pe.pe.size(1):
            raise ValueError(f"Sequence length {cnn_concat.size(1)} exceeds maximum length {self.pe.pe.size(1)}.")
        cnn_encoded = cnn_concat + self.pe.pe[:, :cnn_concat.size(1), :]
        
        # Prepare for Transformer: shape needs to be (L, batch, total_cnn_out)
        transformer_input = cnn_encoded.permute(1, 0, 2)  # Shape: (L, batch, total_cnn_out)
        
        # Pass through Transformer Encoder
        transformer_output = self.transformer_encoder(transformer_input)  # Shape: (L, batch, total_cnn_out)
        
        # Permute back to (batch, L, total_cnn_out)
        transformer_output = transformer_output.permute(1, 0, 2)  # Shape: (batch, L, total_cnn_out)
        
        # Aggregate features: Global Average Pooling over the sequence length dimension
        aggregated_features = transformer_output.mean(dim=1)  # Shape: (batch, total_cnn_out)
        
        # Apply Dropout for regularization
        dropped_out = self.dropout(aggregated_features)  # Shape: (batch, total_cnn_out)
        
        # Final Fully Connected Layer
        logits = self.fc(dropped_out)  # Shape: (batch, num_classes)
        
        return logits



# Instantiate the modified model
num_features = 22
seq_len = input_window
num_classes = 3
dropout_p = 0.3

# Initialize the model
model = CNNTransformer(num_features=num_features, 
                        seq_len=seq_len, 
                        num_classes=num_classes, 
                        dropout_p=dropout_p)
# Print the modified model architecture
print(model)



criterion = nn.CrossEntropyLoss()


# Suppose X_train_scaled.shape = (19000, 600, 5)
# and y_train_encoded.shape = (19000,)
X_train_transposed = np.transpose(X_train_scaled, (0, 2, 1))  # (19000, 5, 600)

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        # Convert numpy arrays to PyTorch tensors
        self.X = torch.from_numpy(X).float()  # shape: (num_samples, channels, seq_len)
        self.y = torch.from_numpy(y).long()   # shape: (num_samples,)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # Return (features, label) for sample 'idx'
        return self.X[idx], self.y[idx]

train_dataset = TimeSeriesDataset(X_train_transposed, y_train_encoded)

batch_size = 512
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Suppose X_val_scaled.shape = (val_size, 600, 5)
# Suppose y_val_encoded.shape = (val_size,)

# 1) Transpose
X_val_transposed = np.transpose(np.concatenate((X_val_scaled, X_test_scaled), axis=0), (0, 2, 1))  # shape: (val_size, 5, 600)

# 2) Wrap in a Dataset
val_dataset = TimeSeriesDataset(X_val_transposed, np.concatenate((y_val_encoded, y_test_encoded), axis=0))

# 3) Create DataLoader (batch_size can match or differ from train)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


if __name__ == "__main__":

        # Assume train_loader and val_loader are predefined DataLoader instances
    # Also assume that the model is defined and instantiated as `model`

    # Initialize the device, model, criterion, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    l2_lambda = 1e-4  # You can adjust this value based on your needs

    # Initialize the optimizer with weight_decay for L2 regularization
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=l2_lambda)

    # Initialize the learning rate scheduler
    # Here, ReduceLROnPlateau reduces the LR by a factor of 0.1 if validation loss doesn't improve for 5 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=15, verbose=True)

    # Set the desired validation loss threshold
    validation_threshold = 0.25

    epoch_loss = 1
    epoch = 0

    while True:
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Training loop
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * X_batch.size(0)
            _, pred = torch.max(outputs, 1)
            correct += (pred == y_batch).sum().item()
            total += y_batch.size(0)
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        epoch += 1
        
        print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
        
        # Validation every 5 epochs
        if epoch % 5 == 0:
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            # To collect predictions and true labels for further metrics
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)

                    outputs = model(X_batch)              # shape: (batch_size, num_classes)
                    loss = criterion(outputs, y_batch)

                    val_loss += loss.item() * X_batch.size(0)

                    # Predictions
                    _, predicted = torch.max(outputs, 1)  # shape: (batch_size,)
                    
                    correct += (predicted == y_batch).sum().item()
                    total += y_batch.size(0)
                    
                    # Store predictions & labels for confusion matrix, etc.
                    all_preds.append(predicted.cpu().numpy())
                    all_labels.append(y_batch.cpu().numpy())

            # Convert lists of arrays into a single 1D array
            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)

            val_loss /= total
            val_acc = correct / total

            print(f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}")

            # --------------------------------------------------------------------------
            # Additional Metrics
            # --------------------------------------------------------------------------

            # 1) Confusion Matrix
            cm = confusion_matrix(all_labels, all_preds)
            print("Confusion Matrix:")
            print(cm)

            # 2) Classification Report
            #    If you have 3 classes: 0="short", 1="flat", 2="long" (example)
            target_names = ["short", "flat", "long"]  # adjust if needed
            report = classification_report(all_labels, all_preds, target_names=target_names)
            print("Classification Report:")
            print(report)
            
            # Step the scheduler with the validation loss
            scheduler.step(val_loss)
            print(f"Learning Rate after scheduler step: {optimizer.param_groups[0]['lr']:.6f}")
            
            # If the validation loss is acceptable, stop training
            if val_loss <= validation_threshold or optimizer.param_groups[0]['lr'] < 1e-8:
                print(f"Validation loss has reached the threshold of {validation_threshold:.4f}, stopping training.")
                break
