from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm


@dataclass
class SpectralTrainingConfig:
    epochs: int = 100
    learning_rate: float = 1e-3
    batch_size: int = 64
    validation_split: float = 0.2
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-5
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6


class SpectralAutoencoderTrainer:
    def __init__(
        self,
        model,
        spectra: np.ndarray,
        device,
        config: SpectralTrainingConfig,
        seed: int = 42,
    ):
        self.model = model.to(device)
        self.spectra = spectra
        self.device = device
        self.config = config
        self.seed = seed
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            min_lr=config.scheduler_min_lr,
        )
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
        }

    def train(self):
        train_loader, validation_loader = self._loaders()

        best_val_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0

        for epoch in range(self.config.epochs):
            self.model.train()
            running_loss = 0.0

            progress = tqdm(
                train_loader,
                desc=f"Spectral AE {epoch + 1}/{self.config.epochs}",
            )

            for (batch,) in progress:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()

                reconstruction, _ = self.model(batch)
                loss = self.criterion(reconstruction, batch)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            train_loss = running_loss / max(1, len(train_loader))
            val_loss = self.validate(validation_loader)
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.history["train_loss"].append(float(train_loss))
            self.history["val_loss"].append(float(val_loss))
            self.history["learning_rate"].append(float(current_lr))

            self.scheduler.step(val_loss)

            improved = (
                val_loss <
                best_val_loss - self.config.early_stopping_min_delta
            )

            if improved:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.config.early_stopping_patience:
                    print(
                        "Early stopping espectral acionado "
                        f"na epoca {epoch + 1}. "
                        f"Melhor val_loss: {best_val_loss:.6f}"
                    )
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self.history

    def validate(self, loader) -> float:
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                reconstruction, _ = self.model(batch)
                loss = self.criterion(reconstruction, batch)
                running_loss += loss.item()

        return running_loss / max(1, len(loader))

    def embeddings(self, spectra: np.ndarray) -> np.ndarray:
        self.model.eval()
        data = torch.tensor(spectra, dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(data),
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        embeddings = []
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                embedding = self.model.encode(batch)
                embeddings.append(embedding.cpu().numpy())

        return np.concatenate(embeddings, axis=0)

    def _loaders(self) -> Tuple[DataLoader, DataLoader]:
        data = torch.tensor(self.spectra, dtype=torch.float32)
        dataset = TensorDataset(data)

        validation_size = max(
            1,
            int(len(dataset) * self.config.validation_split),
        )
        train_size = len(dataset) - validation_size

        if train_size < 1:
            raise ValueError("amostras insuficientes para treino e validacao")

        generator = torch.Generator()
        generator.manual_seed(self.seed)

        train_dataset, validation_dataset = random_split(
            dataset,
            [train_size, validation_size],
            generator=generator,
        )

        return (
            DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                generator=generator,
            ),
            DataLoader(
                validation_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
            ),
        )
