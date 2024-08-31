import os

import torch
from torch import nn

from config import CHECKPOINTS_DIR, DEVICE


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    @classmethod
    def load_checkpoints(cls):
        model = cls().to(DEVICE)
        model.load_state_dict(
            torch.load(os.path.join(CHECKPOINTS_DIR, "model.pth"), weights_only=True)
        )
        return model

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def save_checkpoints(self):
        torch.save(self.state_dict(), os.path.join(CHECKPOINTS_DIR, "model.pth"))
