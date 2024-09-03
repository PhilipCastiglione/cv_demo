from typing import Sized, cast

import torch
from torch.utils.tensorboard import writer

from src.config import DEVICE
from src.data import Data
from src.loss_function import LossFunction
from src.model import Model
from src.optimizer import Optimizer


class Pipeline:
    def __init__(
        self,
        data: Data,
        model: Model,
        loss_fn: LossFunction,
        optimizer: Optimizer,
        summary_writer: writer.SummaryWriter,
    ):
        self.data = data
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.summary_writer = summary_writer

    def process(self, epochs: int):
        for epoch in range(epochs):
            print(f"Epoch: {epoch + 1}\n")

            train_loss = self.__train(epoch)
            test_loss = self.__test()

            self.summary_writer.add_scalars(
                "Training Loss vs Test Loss",
                {"Training Loss": train_loss, "Test Loss": test_loss},
                epoch + 1,
            )
            self.summary_writer.add_scalar(
                "Loss Divergence %",
                (test_loss - train_loss) / train_loss * 100.0,
                epoch + 1,
            )

        self.summary_writer.flush()
        self.model.save_checkpoints()

    def __train(self, epoch: int) -> float:
        self.model.train()

        cumulative_loss = 0
        last_loss = 0

        for batch, (train_input, label) in enumerate(self.data.train):
            train_input, label = train_input.to(DEVICE), label.to(DEVICE)

            self.optimizer.zero_grad()

            prediction = self.model(train_input)

            loss = self.loss_fn(prediction, label)
            loss.backward()

            self.optimizer.step()

            cumulative_loss += loss.item()

            if (batch + 1) % 100 == 0:
                last_loss = cumulative_loss / 100
                total_batches = epoch * len(self.data.train) + batch + 1
                print(f"Training batch {total_batches} loss: {last_loss:.4f}")
                self.summary_writer.add_scalar(
                    "Training Loss", last_loss, total_batches
                )
                cumulative_loss = 0

        return last_loss

    def __test(self) -> float:
        self.model.eval()

        mean_loss = 0
        correct = 0

        with torch.no_grad():
            for test_input, label in self.data.test:
                test_input, label = test_input.to(DEVICE), label.to(DEVICE)
                prediction = self.model(test_input)
                mean_loss += self.loss_fn(prediction, label).item()
                correct += (
                    (prediction.argmax(1) == label).type(torch.float).sum().item()
                )

        mean_loss /= len(self.data.test)
        correct /= len(cast(Sized, self.data.test.dataset))

        print(
            f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {mean_loss:>8f} \n"
        )

        return mean_loss
