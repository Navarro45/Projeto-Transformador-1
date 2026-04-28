import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class AstroDataset(Dataset):
    def __init__(self, root_dir, transform=None, spectral_length=1024):
        self.root_dir = root_dir
        self.transform = transform
        self.spectral_length = spectral_length

        self.samples = self._load_samples()
        self.pseudo_labels = self._load_pseudo_labels()

    def _load_samples(self):
        samples = []

        for label, cls in enumerate(["star", "galaxy", "quasar"]):
            cls_path = os.path.join(self.root_dir, cls)

            if not os.path.exists(cls_path):
                print(f"⚠️ Pasta não encontrada: {cls_path}")
                continue

            files = os.listdir(cls_path)
            print(f"🔎 {cls}: {len(files)} arquivos encontrados")

            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    base = os.path.splitext(file)[0]

                    img_path = os.path.join(cls_path, file)
                    spec_path = os.path.join(cls_path, base + ".npy")

                    samples.append((img_path, spec_path, label))

        print(f"\n📦 TOTAL FINAL DE SAMPLES: {len(samples)}\n")

        return samples

    def _load_pseudo_labels(self):
        pseudo = {}

        for cls in ["star", "galaxy", "quasar"]:
            path = os.path.join(self.root_dir, cls, "pseudo_labels.npy")

            if os.path.exists(path):
                data = np.load(path, allow_pickle=True).item()
                pseudo.update(data)

        print(f"📊 Pseudo-labels carregados: {len(pseudo)}")

        return pseudo

    def _process_spectrum(self, spec_path):
        if not os.path.exists(spec_path):
            return None

        spec = np.load(spec_path)

        spec = (spec - np.mean(spec)) / (np.std(spec) + 1e-8)

        if len(spec) < self.spectral_length:
            pad = self.spectral_length - len(spec)
            spec = np.pad(spec, (0, pad))
        else:
            spec = spec[:self.spectral_length]

        return torch.tensor(spec, dtype=torch.float32)

    def __len__(self):
        size = len(self.samples)

        if size == 0:
            print(f"⚠️ Dataset vazio em: {self.root_dir}")

        return size

    def __getitem__(self, idx):
        img_path, spec_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        spectrum = self._process_spectrum(spec_path)

        if spectrum is None:
            spectrum = torch.zeros(self.spectral_length, dtype=torch.float32)

        # 🔥 pega pseudo-label
        pseudo_label = self.pseudo_labels.get(spec_path, -1)

        return image, spectrum, label, pseudo_label