from typing import Sized, cast

import torch
from torch import nn
from torch.optim.sgd import SGD

from config import DEVICE
from src.data import Data
from src.model import NeuralNetwork


def main():
    data = Data()
    data.get_data()
    data.inspect_data()

    model = NeuralNetwork().to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=1e-3)

    def train():
        size = len(cast(Sized, data.train.dataset))

        model.train()

        for batch, (X, y) in enumerate(data.train):
            X, y = X.to(DEVICE), y.to(DEVICE)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test():
        size = len(cast(Sized, data.test.dataset))
        num_batches = len(data.test)
        model.eval()
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in data.test:
                X, y = X.to(DEVICE), y.to(DEVICE)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size

        print(
            f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
        )

    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train()
        test()

    print("Done!")

    model.save_checkpoints()

    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    model.eval()
    x, y = data.test.dataset[0][0], data.test.dataset[0][1]
    with torch.no_grad():
        x = x.to(DEVICE)
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')


if __name__ == "__main__":
    main()
