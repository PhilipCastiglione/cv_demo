import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import writer
from torchvision import datasets
from torchvision.transforms import v2
from torchvision.utils import make_grid

from src.config import DATA_DIR


class Data:
    train: DataLoader
    test: DataLoader

    def __init__(self, batch_size: int, summary_writer: writer.SummaryWriter):
        self.path = DATA_DIR
        self.batch_size = batch_size
        self.summary_writer = summary_writer
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
            transform=v2.Compose(
                [
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.RandomHorizontalFlip(),
                    v2.RandomRotation((-5, 5)),
                ]
            ),
        )

        test_data = datasets.FashionMNIST(
            root=self.path,
            train=False,
            download=True,
            transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        )

        self.train = DataLoader(training_data, batch_size=self.batch_size, shuffle=True)

        self.test = DataLoader(test_data, batch_size=self.batch_size)

    def inspect_data(self):
        print(self.train.dataset, end="\n\n")
        print(self.test.dataset, end="\n\n")

        train_features, train_labels = next(iter(self.train))

        print(f"Feature batch shape: {train_features.size()}", end="\n\n")
        print(f"Labels batch shape: {train_labels.size()}", end="\n\n")

        print(f"Classes: {self.classes}", end="\n\n")

        print("First batch examples in image below.")

        features_image_grid = make_grid(train_features)
        plot_images = features_image_grid.mean(dim=0).numpy()

        plt.imshow(plot_images, cmap="Greys")

        self.summary_writer.add_image("First batch examples", features_image_grid)
        self.summary_writer.flush()
