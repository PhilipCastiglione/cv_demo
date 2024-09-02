import torch

from config import DEVICE


class Pipeline:
    def __init__(self, data, model, loss_fn, optimizer):
        self.data = data
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def process(self, epochs):
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            self.__train()
            self.__test()

        self.model.save_checkpoints()

    def __train(self):
        size = len(self.data.train.dataset)

        self.model.train()

        for batch, (X, y) in enumerate(self.data.train):
            X, y = X.to(DEVICE), y.to(DEVICE)

            # Compute prediction error
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def __test(self):
        size = len(self.data.test.dataset)
        num_batches = len(self.data.test)

        self.model.eval()
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in self.data.test:
                X, y = X.to(DEVICE), y.to(DEVICE)
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size

        print(
            f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
        )
