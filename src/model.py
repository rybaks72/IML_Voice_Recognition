import torch
import torch.nn as nn
import torch.nn.functional as F

# simple model copied from the pytorch tutorial adjusted to our needs
# for Milestone 1 we don't need anything fancy so this is the simplest way to
# get us to >= F1 requirement
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential( nn.Conv2d(1, 16, 5),
                                    #nn.BatchNorm2d(16),
                                    nn.ReLU())
                                    #nn.BatchNorm2d(16),
                                    #nn.MaxPool2d(2, 2) )

        self.conv2 = nn.Sequential( nn.Conv2d(16, 32, 5),
                                    #nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    #nn.BatchNorm2d(32),
                                    nn.MaxPool2d(2, 2) )

        self.conv3 = nn.Sequential( nn.Conv2d(32, 64, 5),
                                    #nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    #nn.BatchNorm2d(64),
                                    nn.MaxPool2d(2, 2) )
        self.conv4 = nn.Sequential( nn.Conv2d(64, 128, 5),
                                      # #nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                      # nn.BatchNorm2d(128),
                                    nn.MaxPool2d(2, 2) )
        # input: 1 channel spectrogram (grayscale)
        self.pool = nn.MaxPool2d(2, 2)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, 128, 130)
            out = self.conv1(dummy)
            out = self.conv2(out)
            out = self.conv3(out)
            out = self.conv4(out)
            flat_dim = out.numel()

        self.fc1 = nn.Linear(flat_dim, 120)

        self.dropout1 = nn.Dropout(p=0.2)

        self.fc2 = nn.Linear(120, 84)

        self.dropout2 = nn.Dropout(p=0.5)

        self.fc3 = nn.Linear(84, 2)# binary output



    def forward(self, x): #
        # x = self.pool(F.relu(self.conv1(x))) #
        # x = self.pool(F.relu(self.conv2(x)))
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.shape[0], -1) #a bit more robust
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
