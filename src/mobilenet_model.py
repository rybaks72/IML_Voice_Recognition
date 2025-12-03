import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


def MobileNetV2(pretrained=False):
    """
    Creates a MobileNetV2 model adapted for 1-channel spectrograms
    and binary classification (2 classes).

    Why MobileNet?
    It uses 'Depthwise Separable Convolutions' to be extremely lightweight and fast.
    Perfect for running voice verification on a phone or Raspberry Pi.
    """

    # 1. Load the architecture from torchvision
    if pretrained:
        weights = MobileNet_V2_Weights.IMAGENET1K_V1
    else:
        weights = None

    model = mobilenet_v2(weights=weights)

    # 2. Modify the First Layer (Input)
    # MobileNetV2 structure is: model.features[0][0] is the first Conv2d layer.
    # We need to change input channels from 3 (RGB) to 1 (Spectrogram).

    original_first_layer = model.features[0][0]

    model.features[0][0] = nn.Conv2d(
        in_channels=1,  # CHANGED from 3 to 1
        out_channels=original_first_layer.out_channels,
        kernel_size=original_first_layer.kernel_size,
        stride=original_first_layer.stride,
        padding=original_first_layer.padding,
        bias=False
    )

    # 3. Modify the Last Layer (Classifier)
    # The classifier in MobileNetV2 is a Sequential block:
    # (0): Dropout
    # (1): Linear (in_features=1280, out_features=1000)
    # We need to replace that Linear layer to output 2 classes.

    # model.last_channel is usually 1280 for MobileNetV2
    model.classifier[1] = nn.Linear(model.last_channel, 2)  # CHANGED to 2 classes

    return model

