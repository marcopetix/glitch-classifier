import torch
import torch.nn as nn
import torch.nn.functional as F

class GravitySpyCNN(nn.Module):
    def __init__(self, input_shape=(1, 128, 128), num_classes=7, l2_reg=1e-4):
        super(GravitySpyCNN, self).__init__()

        self.input_shape = input_shape

        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=5)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)

        # Compute the flattened feature size after convolutions and pooling
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            x = self._forward_features(dummy_input)
            self.flat_dim = x.view(-1).shape[0]

        self.fc1 = nn.Linear(self.flat_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)
    
    def _forward_features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.dropout(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Example usage:
if __name__ == "__main__":
    model = GravitySpyCNN(input_shape=(1, 128, 128), num_classes=7)
    print(model)
    sample = torch.randn(4, 1, 128, 128)
    out = model(sample)
    print(out.shape)  # Should be [4, 7] for batch size 4 and 7 classes