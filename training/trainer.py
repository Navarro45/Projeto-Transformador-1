import torch
from tqdm import tqdm


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        validation_loader,
        device,
        epochs=5,
        learning_rate=0.001
    ):

        self.model = model

        self.train_loader = train_loader

        self.validation_loader = validation_loader

        self.device = device

        self.epochs = epochs

        self.criterion = torch.nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )

        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }

    def train(self):

        for epoch in range(self.epochs):

            self.model.train()

            running_loss = 0
            correct = 0
            total = 0

            progress_bar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch+1}/{self.epochs}"
            )

            for images, labels in progress_bar:

                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(images)

                loss = self.criterion(
                    outputs,
                    labels
                )

                loss.backward()

                self.optimizer.step()

                running_loss += loss.item()

                _, predicted = torch.max(
                    outputs,
                    1
                )

                total += labels.size(0)

                correct += (
                    predicted == labels
                ).sum().item()

            train_loss = (
                running_loss /
                len(self.train_loader)
            )

            train_acc = correct / total

            val_loss, val_acc = self.validate()

            self.history["train_loss"].append(
                train_loss
            )

            self.history["train_acc"].append(
                train_acc
            )

            self.history["val_loss"].append(
                val_loss
            )

            self.history["val_acc"].append(
                val_acc
            )

            print(f"Train Acc: {train_acc:.4f}")
            print(f"Val Acc: {val_acc:.4f}")

    def validate(self):

        self.model.eval()

        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in self.validation_loader:

                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)

                loss = self.criterion(
                    outputs,
                    labels
                )

                total_loss += loss.item()

                _, predicted = torch.max(
                    outputs,
                    1
                )

                total += labels.size(0)

                correct += (
                    predicted == labels
                ).sum().item()

        return (
            total_loss / len(self.validation_loader),
            correct / total
        )