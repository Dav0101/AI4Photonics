
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torcwa

class MSshaper(nn.Module):
    def __init__(self):
        super(MSshaper, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            stride=2,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            stride=2,
            padding=1
        )

        self.conv3 = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            stride=2,
            padding=1
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.sigmoid(self.conv3(x))
        
        x = self.fc2(x) # Niente softmax qui se usiamo CrossEntropyLoss (la include già)
        return x
    
model = MSshaper()
loss =