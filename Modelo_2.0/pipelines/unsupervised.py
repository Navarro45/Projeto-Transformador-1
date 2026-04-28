import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
import hdbscan

DATA_DIR = "data/train"


class Autoencoder(nn.Module):
    def __init__(self, input_dim=1024, latent_dim=32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def process_class(cls_name):
    print(f"\n🔬 Processando classe: {cls_name}")

    cls_path = os.path.join(DATA_DIR, cls_name)

    spectra = []
    paths = []

    for file in os.listdir(cls_path):
        if file.endswith(".npy"):
            path = os.path.join(cls_path, file)

            try:
                spec = np.load(path)
                spectra.append(spec)
                paths.append(path)
            except:
                continue

    if len(spectra) < 10:
        print("⚠️ Poucos dados, pulando...")
        return

    spectra = torch.tensor(np.array(spectra), dtype=torch.float32)

    # -------- AUTOENCODER --------
    model = Autoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(5):
        recon, z = model(spectra)
        loss = loss_fn(recon, spectra)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"{cls_name} AE Epoch {epoch}: {loss.item()}")

    embeddings = z.detach().numpy()

    # -------- PCA --------
    reduced = PCA(n_components=10).fit_transform(embeddings)

    # -------- CLUSTER --------
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
    labels = clusterer.fit_predict(reduced)

    # -------- SALVAR --------
    label_dict = {}

    for p, l in zip(paths, labels):
        if l != -1:
            label_dict[p] = int(l)

    save_path = os.path.join(cls_path, "pseudo_labels.npy")
    np.save(save_path, label_dict)

    print(f"✅ {cls_name}: {len(label_dict)} subclasses criadas")


def run():
    for cls in ["star", "galaxy", "quasar"]:
        process_class(cls)

    print("\n✅ Pipeline não supervisionado finalizado")