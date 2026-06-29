import torch
import torch.nn as nn

class EdgeEEGNet(nn.Module):
    """Extreme-capacity student model (1,074 params, 0.004 MB) with high-res feature interception."""
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(EdgeEEGNet, self).__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv1d(16, 4, kernel_size=16, padding='same', bias=False),
            nn.BatchNorm1d(4)
        )
        
        # Split the original self.depthwise block BEFORE the AdaptiveAvgPool1d
        self.depthwise_features = nn.Sequential(
            nn.Conv1d(4, 8, kernel_size=1, groups=4, bias=False),
            nn.BatchNorm1d(8),
            nn.ELU()
        )
        
        # The pooling and dropout portion
        self.depthwise_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Dropout(dropout_rate)
        )
        
        self.classifier = nn.Linear(8, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_features=False):
        x = x[:, :16, :] 
        x1 = self.block1(x)
        
        # Intercept the features here: Shape is [Batch, 8, 1280]
        temporal_features = self.depthwise_features(x1)
        
        # Continue the graph normally: Shape becomes [Batch, 8, 1]
        pooled = self.depthwise_pool(temporal_features)
        
        x_flat = pooled.view(pooled.size(0), -1)
        logits = self.classifier(x_flat)
        
        if return_features:
            return logits, temporal_features
        return logits

class TeacherCGRU(nn.Module):
    """Upscaled High-Capacity Spatial-Temporal Teacher Model (~14.9M Params / 59 MB)."""
    def __init__(self, num_classes=2):
        super(TeacherCGRU, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(18, 64, kernel_size=32, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=16, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.gru = nn.GRU(
            input_size=128, 
            hidden_size=768, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True
        )
        self.fc = nn.Linear(1536, num_classes)

    def forward(self, x, return_features=False):
        cnn_out = self.cnn(x)
        gru_in = cnn_out.permute(0, 2, 1)
        gru_out, _ = self.gru(gru_in)
        
        features = gru_out[:, -1, :]
        logits = self.fc(features)
        
        if return_features:
            return logits, cnn_out # Shape: [Batch, 128, 320]
        return logits
