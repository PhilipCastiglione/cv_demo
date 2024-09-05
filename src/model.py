import os

import torch
from torch import nn

from src.config import CHECKPOINTS_DIR, DEVICE


class Model(nn.Module):
    """
    A simple CNN model for the FashionMNIST dataset.

    Uses 2 convolutional layers followed by max pooling and 2 fully connected layers.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 64 * 7 * 7)
        x = self.classifier(x)
        return x

    @classmethod
    def load_checkpoints(cls):
        model = cls().to(DEVICE)
        model.load_state_dict(
            torch.load(os.path.join(CHECKPOINTS_DIR, "model.pth"), weights_only=True)
        )
        return model

    def save_checkpoints(self):
        torch.save(self.state_dict(), os.path.join(CHECKPOINTS_DIR, "model.pth"))
