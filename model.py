# model.py
# Shared DigitNet definition — must match the architecture used during training.
# Both the inference script and the visualizer import from here.

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)


class DigitNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
        )
        with torch.no_grad():
            _dummy   = torch.zeros(1, 1, 28, 28)
            flat_dim = self.features(_dummy).numel()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def load_model(checkpoint_path: str, device: torch.device) -> DigitNet:
    """
    Instantiate DigitNet and load saved weights.

    Parameters
    ----------
    checkpoint_path : path to the .pth state_dict file
    device          : torch.device to load the model onto

    Returns
    -------
    model in eval mode, ready for inference
    """
    model = DigitNet()
    # map_location ensures the checkpoint loads correctly even if it was
    # saved on GPU and you are now running on CPU (or vice-versa).
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()   # disables Dropout and switches BatchNorm to inference mode
    return model
