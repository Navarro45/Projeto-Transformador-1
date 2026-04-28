import torch
from src.spectral_model import SpectralCNN


def train_spectral_per_class(train_loader, config):
    models = {}

    for cls in [0, 1, 2]:
        print(f"\n🔬 Treinando modelo espectral para classe {cls}")

        spectra_list = []
        labels_list = []

        for _, spectra, labels, pseudo in train_loader:
            mask = (labels == cls) & (pseudo != -1)

            if mask.sum() == 0:
                continue

            spectra_list.append(spectra[mask])
            labels_list.append(pseudo[mask])

        if len(spectra_list) == 0:
            print(f"⚠️ Classe {cls} sem dados suficientes")
            continue

        X = torch.cat(spectra_list).to(config.DEVICE)
        y = torch.cat(labels_list).to(config.DEVICE)

        num_classes = len(torch.unique(y))

        model = SpectralCNN(config.SPECTRAL_LENGTH, num_classes).to(config.DEVICE)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.LR,
            weight_decay=1e-4
        )

        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(10):
            outputs = model(X)
            loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"[Classe {cls}] Epoch {epoch} Loss: {loss.item()}")

        models[cls] = model

    return models