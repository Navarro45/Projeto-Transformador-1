from __future__ import annotations

import torch
import torch.nn as nn


class SpectralAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 256,
    ):
        super().__init__()

        mid_dim = max(latent_dim * 2, hidden_dim // 2)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, spectra: torch.Tensor) -> torch.Tensor:
        return self.encoder(spectra)

    def forward(self, spectra: torch.Tensor):
        embedding = self.encode(spectra)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding
