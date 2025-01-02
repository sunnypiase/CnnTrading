import torch
import torch.nn as nn

class HybridCNN(nn.Module):
    def __init__(self, num_features=5, seq_len=600, num_classes=3, dropout_p=0.5):
        super(HybridCNN, self).__init__()
        
        # Shared Convolutional Layer
        self.shared_conv1 = nn.Conv1d(
            in_channels=num_features,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.shared_relu = nn.ReLU()
        self.shared_dropout = nn.Dropout1d(p=0.4)  # Increased dropout
        
        # Feature-Specific Convolutional Layers
        self.feature_cnns = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Dropout1d(p=0.6),  # Increased dropout
                nn.Conv1d(128, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Dropout1d(p=0.4)   # Increased dropout
            )
            for _ in range(num_features)
        ])
        
        # Fully Connected Layers with Increased Dropout
        self.fc1 = nn.Linear(32 * seq_len + 32 * seq_len * num_features, 256)
        self.fc_dropout = nn.Dropout(p=0.8)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        """
        Input shape: (batch_size, num_features=5, seq_len=600)
        """
        # Shared Convolution
        shared_out = self.shared_conv1(x)  # Shape: (batch_size, 64, seq_len)
        shared_out = self.shared_relu(shared_out)
        shared_out = self.shared_dropout(shared_out)
        
        # Feature-Specific Convolutions
        feature_outputs = []
        for i, cnn in enumerate(self.feature_cnns):
            feature = x[:, i:i+1, :]  # Shape: (batch_size, 1, seq_len)
            feature_out = cnn(feature)  # Shape: (batch_size, 64, seq_len)
            feature_outputs.append(feature_out)
        
        # Concatenate All Outputs
        all_features = torch.cat([shared_out] + feature_outputs, dim=1)  # Shape: (batch_size, 64 + 64*num_features, seq_len)
        all_features = all_features.view(all_features.size(0), -1)       # Shape: (batch_size, (64 + 64*num_features)*seq_len)
        
        # Fully Connected Layers
        x = self.fc1(all_features)          # Shape: (batch_size, 256)
        x = self.fc_dropout(x)              # Dropout before activation
        x = nn.ReLU()(x)                    # Activation
        x = self.fc2(x)                     # Shape: (batch_size, num_classes)
        return x