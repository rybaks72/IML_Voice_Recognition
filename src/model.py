import torch
import torch.nn as nn
import torch.nn.functional as F

# simple model copied from the pytorch tutorial adjusted to our needs
# for Milestone 1 we don't need anything fancy so this is the simplest way to
# get us to >= F1 requirement
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)   # input: 1 channel spectrogram (grayscale)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        #TODO calculate thia 29
        self.fc1 = nn.Linear(16 * 29 * 29, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)  # binary output

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #TODO x = x.view(x.size(0), -1)  # flatten all features
        x = x.view(-1, 16 * 29*29)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
