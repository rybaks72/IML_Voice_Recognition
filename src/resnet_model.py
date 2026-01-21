import torch
import torch.nn as nn
import torch.nn.functional as F

"""
ResNet (Residual Network) Architecture Rationale Summary:

## Core Mechanism: The Residual Block
* **The Problem (Vanishing Gradient):** In deep traditional CNNs, gradients vanish during backpropagation, causing earlier layers to stop learning effectively, which degrades performance.
* **The Solution (Skip Connection):** ResNet introduces a **Skip Connection** (Identity Mapping) that bypasses the main convolutional path F(x). 
* **The Math:** The block learns the residual F(x) such that **H(x) = F(x) + x**.  This ensures that gradients flow freely, allowing training of **very deep networks** (e.g., 18, 50 layers).

##  SimpleCNN vs. ResNet-18 Comparison

| Feature | SimpleCNN (Original Model) | ResNet-18 (Proposed Upgrade) |
| :--- | :--- | :--- |
| **Depth** | Shallow (4 Conv + 3 FC = 7 Layers) | **Deep (18 Layers)** |
| **Core Element** | Standard Convolutions | **Residual Blocks (Skip Connections)** |
| **Primary Advantage** | Simplicity, Fast setup | **Significantly higher accuracy potential** (by solving gradient problem) |
| **Spłaszczanie/Flattening** | Full Flattening (Manual size calculation needed) | **Global Average Pooling** (Robust to input size, fixed 512 features) |
| **Stability** | Lower (Batch Norm is commented out) | **High (uses Batch Normalization extensively)** |
"""

# #shortcut version 3

# class SELayer(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(SELayer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction, channel, bias=False),
#             # Sigmoid creates the "weight" (0 to 1) for each channel
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y.expand_as(x)



# 1. Definition of the Basic Residual Block
class BasicBlock(nn.Module):
    # For ResNet-18 and ResNet-34, the expansion factor is 1
    expansion = 1
 
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3):
        super(BasicBlock, self).__init__()
        # #shortcut version 4
        # self.stride = stride

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
        #self.bn1 = nn.BatchNorm2d(out_channels)

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

        #shortcut vrsion 4
        self.shortcut = nn.Sequential()

        # # shortcut version 3
        # self.se = SELayer(out_channels, reduction=16)

        #shortcut version1 and 3
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

        # # shortcut version 2
        # self.shortcut = nn.Sequential(
        #     nn.Conv2d(in_channels, self.expansion * out_channels, 
        #             kernel_size=1, stride=stride, bias=False),
        #     nn.BatchNorm2d(self.expansion * out_channels)
        # )

    def forward(self, x):
        # Store the original input for the shortcut connection
        identity = x

        # Pass through the two convolutional layers
        out = self.conv1(x)
        
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

       # out = self.se(out)

        # # shortcut version 4
        # # this replaces the 1x1 convolution shortcut
        # if out.shape != identity.shape:
        #     # average pooling to downsample spatial dimensions instead of conv layer (use stride to match spatial dimensions)
        #     if self.stride > 1:
        #         identity = F.avg_pool2d(identity, self.stride)
            
        #     #  check if spatial dimensions STILL don't match (runtime error before)
        #     if identity.shape[2:] != out.shape[2:]:
        #         identity = F.adaptive_avg_pool2d(identity, (out.shape[2], out.shape[3]))

        #     # match the channel dimensions (Padding with zeros)
        #     if out.shape[1] != identity.shape[1]:
        #         pad_channels = out.shape[1] - identity.shape[1]
        #         pad = torch.zeros(identity.shape[0], pad_channels, 
        #                           identity.shape[2], identity.shape[3]).to(x.device)
        #         identity = torch.cat([identity, pad], dim=1)
            
        # #  Add the identity to the output
        # out += identity

        # Add the shortcut (Residual connection: H(x) = F(x) + x)
        out += self.shortcut(identity)
        out = self.relu(out)
       # out = self.bn2(out)

        return out


# 2. Definition of the ResNet Architecture (ResNet-18 implementation)
class ResNet(nn.Module):
    # block: the building block (e.g., BasicBlock)
    # num_blocks: list defining the number of blocks in each of the four layers
    # num_classes: number of output classes (2 for Imposter/You)
    def __init__(self, block, num_blocks, num_classes=2):
        super(ResNet, self).__init__()
        self.in_channels = 32

        # Initial Convolutional Layer (handles 1-channel spectrogram input)
        # kernel_size=7, stride=2, MaxPool: typical for processing large images (spectrograms)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=3, bias=False),  # Input channel set to 1
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            #nn.BatchNorm2d(32),
            #nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Residual Blocks Layers
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        #self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Global Average Pooling (used instead of Flattening to normalize output size)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        #testing global max pooling
        #self.avgpool = nn.AdaptiveMaxPool2d((1, 1))

        #testing dropout layer
        #self.dropout = nn.Dropout(p=0.5)

        #testing multiple dropout layers
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.2)
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
       # out = self.dropout1(out)
        out = self.layer2(out)
        #out = self.dropout2(out)
        out = self.layer3(out)
       # out = self.dropout3(out)
       # out = self.layer4(out)

        # Apply Global Average Pooling
        out = self.avgpool(out)

        # Flattening from (Batch, 512, 1, 1) to (Batch, 512)
        out = torch.flatten(out, 1)

        out = self.dropout(out)

        # Final Fully Connected Layer
        out = self.fc(out)
        return out


# Chose ResNet18 b/c it's the fastest to train
def ResNet18():
    # Number of blocks in each layer for ResNet-18: [2, 2, 2, 2]
    # lets change to [1, 1, 1, 1]
    return ResNet(BasicBlock, [2, 2,1])