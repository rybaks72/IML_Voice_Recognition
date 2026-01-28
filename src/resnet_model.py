import torch
import torch.nn as nn
import torch.nn.functional as F

# Definition of the Basic Residual Block
class BasicBlock(nn.Module):
    expansion = 1
 
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3):
        super(BasicBlock, self).__init__()
        

        # added padding calculation to safely chang ethe kernel size
        pad = (kernel_size - 1) // 2

        # First Conv layer + BN + ReLU
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Second Conv layer + BN (ReLU is applied after the shortcut addition)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=pad,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection (Identity mapping)
        self.shortcut = nn.Sequential()
        # If input and output dimensions (channels or spatial size) differ,
        # we need a 1x1 convolution to match the dimensions for addition.
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
           )


    def forward(self, x):
        # Store the original input for the shortcut connection
        identity = x

        # Pass through the two convolutional layers
        out = self.conv1(x)
        
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        # Add the shortcut (Residual connection: H(x) = F(x) + x)
        out += self.shortcut(identity)
        out = self.relu(out)
       # out = self.bn2(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(ResNet, self).__init__()
        self.in_channels = 32

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=3, bias=False),  # Input channel set to 1
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Residual Blocks Layers
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout = nn.Dropout(p=0.5)

        # Final Fully Connected Layer (Output classes set to 2)
        self.fc = nn.Linear(128 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        # Create a list of strides for the blocks in the layer (only the first block uses stride > 1)
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            # Create a block and update the in_channels for the next block
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        # Apply Global Average Pooling
        out = self.avgpool(out)

        # Flattening from (Batch, 512, 1, 1) to (Batch, 512)
        out = torch.flatten(out, 1)

        out = self.dropout(out)

        # Final Fully Connected Layer
        out = self.fc(out)
        return out

def ResNet18():
    '''
    This is a simplified ResNet-18 model obtained after the tests.
    2, 2, 1 corresponds to the number of blocks in each of the three layers.
    There is one dropout layer before the final fully connected layer.
    '''
    return ResNet(BasicBlock, [2, 2,1])