from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from config import BATCH_SIZE, DATA_DIR


class Data:
    train: DataLoader
    test: DataLoader

    def __init__(self):
        self.path = DATA_DIR

    def get_data(self):
        training_data = datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor(),
        )

        test_data = datasets.FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor(),
        )

        self.train = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)

        self.test = DataLoader(test_data, batch_size=BATCH_SIZE)

    def inspect_data(self):
        for X, y in self.test:
            print(f"Shape of X [N, C, H, W]: {X.shape}")
            print(f"Shape of y: {y.shape} {y.dtype}")
            break

        for X, y in self.train:
            print(f"Shape of X [N, C, H, W]: {X.shape}")
            print(f"Shape of y: {y.shape} {y.dtype}")
            break
