import torch
import copy
from tqdm import tqdm


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        validation_loader,
        device,
        epochs=5,
        learning_rate=0.001,
        early_stopping_patience=5,
        early_stopping_min_delta=0.0,
        scheduler_patience=2,
        scheduler_factor=0.5,
        scheduler_min_lr=1e-6
    ):

        self.model = model

        self.train_loader = train_loader

        self.validation_loader = validation_loader

        self.device = device

        self.epochs = epochs

        if (
            early_stopping_patience is not None and
            early_stopping_patience < 0
        ):

            early_stopping_patience = None

        self.early_stopping_patience = early_stopping_patience

        self.early_stopping_min_delta = early_stopping_min_delta

        self.scheduler_patience = scheduler_patience

        if (
            self.scheduler_patience is not None and
            self.scheduler_patience < 0
        ):

            self.scheduler_patience = None

        self.scheduler_factor = scheduler_factor

        self.scheduler_min_lr = scheduler_min_lr

        self.criterion = torch.nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )

        self.scheduler = None

        if self.scheduler_patience is not None:

            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.scheduler_factor,
                patience=self.scheduler_patience,
                min_lr=self.scheduler_min_lr
            )

        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rate": []
        }

    def train(self):

        best_val_loss = float("inf")

        best_state = None

        epochs_without_improvement = 0

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

            current_lr = (
                self.optimizer
                .param_groups[0]["lr"]
            )

            self.history["learning_rate"].append(
                current_lr
            )

            print(f"Train Acc: {train_acc:.4f}")
            print(f"Val Acc: {val_acc:.4f}")
            print(f"Learning Rate: {current_lr:.8f}")

            if self.scheduler is not None:

                self.scheduler.step(
                    val_loss
                )

                next_lr = (
                    self.optimizer
                    .param_groups[0]["lr"]
                )

                if next_lr < current_lr:

                    print(
                        "Scheduler reduziu o learning rate "
                        f"para {next_lr:.8f}"
                    )

            improved = (
                val_loss <
                best_val_loss -
                self.early_stopping_min_delta
            )

            if improved:

                best_val_loss = val_loss

                best_state = copy.deepcopy(
                    self.model.state_dict()
                )

                epochs_without_improvement = 0

            else:

                epochs_without_improvement += 1

                if (
                    self.early_stopping_patience is not None and
                    epochs_without_improvement >= self.early_stopping_patience
                ):

                    print(
                        "Early stopping acionado "
                        f"na epoca {epoch + 1}. "
                        f"Melhor val_loss: {best_val_loss:.4f}"
                    )

                    break

        if best_state is not None:

            self.model.load_state_dict(
                best_state
            )

        return self.history

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
