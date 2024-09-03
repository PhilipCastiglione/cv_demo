import os

import torch

CHECKPOINTS_DIR = os.path.join(os.path.dirname(__file__), "../checkpoints")
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
LOG_DIR = os.path.join(os.path.dirname(__file__), "../runs")

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {DEVICE} device")
