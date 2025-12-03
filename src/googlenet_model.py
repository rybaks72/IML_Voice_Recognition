import torch
import torch.nn as nn
from torchvision.models import googlenet, Googlenet_Weights


def GoogleNet(pretrained=False):
    """
    Creates a GoogleNet (Inception v1) model adapted for 1-channel spectrograms
    and binary classification (2 classes).

    Args:
        pretrained (bool): If True, loads ImageNet weights (advanced).
                           For now, we default to False (training from scratch).
    """

    # 1. Load the architecture from torchvision
    # aux_logits=False: GoogleNet normally has 3 outputs (2 auxiliary for training).
    # We disable them to keep your training loop simple (one loss, one output).
    if pretrained:
        weights = Googlenet_Weights.IMAGENET1K_V1
    else:
        weights = None

    model = googlenet(weights=weights, aux_logits=False, init_weights=True)

    # 2. Modify the First Layer (Input)
    # Original GoogleNet expects 3 channels (RGB). We have 1 (Grayscale Spectrogram).
    # We replace the first Convolutional layer.
    # Original: nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

    original_conv1 = model.conv1.conv

    model.conv1.conv = nn.Conv2d(
        in_channels=1,  # CHANGED from 3 to 1
        out_channels=original_conv1.out_channels,
        kernel_size=original_conv1.kernel_size,
        stride=original_conv1.stride,
        padding=original_conv1.padding,
        bias=original_conv1.bias
    )

    # 3. Modify the Last Layer (Output)
    # Original GoogleNet outputs 1000 classes (ImageNet). We need 2 (Imposter vs You).

    num_features = model.fc.in_features  # typically 1024 for GoogleNet
    model.fc = nn.Linear(num_features, 2)  # CHANGED to 2 classes

    return model
