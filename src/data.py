import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid

from config import BATCH_SIZE, DATA_DIR


class Data:
    train: DataLoader
    test: DataLoader

    def __init__(self):
        self.path = DATA_DIR
        self.classes = (
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle Boot",
        )

    def get_data(self):
        training_data = datasets.FashionMNIST(
            root=self.path,
            train=True,
            download=True,
            transform=ToTensor(),
        )

        test_data = datasets.FashionMNIST(
            root=self.path,
            train=False,
            download=True,
            transform=ToTensor(),
        )

        self.train = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)

        self.test = DataLoader(test_data, batch_size=BATCH_SIZE)

    def inspect_data(self):
        print(self.train.dataset, end="\n\n")
        print(self.test.dataset, end="\n\n")

        train_features, train_labels = next(iter(self.train))

        print(f"Feature batch shape: {train_features.size()}", end="\n\n")
        print(f"Labels batch shape: {train_labels.size()}", end="\n\n")

        print(f"Classes: {self.classes}", end="\n\n")

        print("First batch examples in image below.")

        img_grid = make_grid(train_features)
        img_grid = img_grid.mean(dim=0)

        npimg = img_grid.numpy()
        plt.imshow(npimg, cmap="Greys")
