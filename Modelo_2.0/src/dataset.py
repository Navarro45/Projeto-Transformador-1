import os
import csv
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from src.spectrum_generator import SpectrumGenerator


class AstroDataset(Dataset):
    def __init__(self, root_dir, transform=None, spectral_length=1024, config=None):
        self.root_dir = root_dir
        self.transform = transform
        self.spectral_length = spectral_length
        self.config = config
        self.spectrum_generator = None
        self.generate_if_missing = False

        if config is not None:
            self.generate_if_missing = getattr(config, "GENERATE_SPECTRUM_IF_MISSING", False)
            if self.generate_if_missing:
                self.spectrum_generator = SpectrumGenerator(config)
        self.metadata_by_key = self._load_metadata()

        self.samples = self._load_samples()
        self.pseudo_labels = self._load_pseudo_labels()

    def _load_metadata(self):
        by_key = {}
        if self.config is None:
            return by_key
        metadata_path = self.config.PATHS.get("metadata", "")
        if not metadata_path or not os.path.exists(metadata_path):
            return by_key

        with open(metadata_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                class_name = row.get("class_name", "").strip().lower()
                base_name = row.get("base_name", "").strip()
                if class_name and base_name:
                    by_key[f"{class_name}/{base_name}"] = row
        return by_key

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

                    meta_key = f"{cls}/{base}"
                    metadata = self.metadata_by_key.get(meta_key, {})
                    samples.append((img_path, spec_path, label, metadata))

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

    def _generate_spectrum_from_image(self, image_path, metadata=None):
        if self.spectrum_generator is None:
            return None
        spec = self.spectrum_generator.generate(image_path, metadata=metadata)
        return torch.tensor(spec, dtype=torch.float32)

    def __len__(self):
        size = len(self.samples)

        if size == 0:
            print(f"⚠️ Dataset vazio em: {self.root_dir}")

        return size

    def __getitem__(self, idx):
        img_path, spec_path, label, metadata = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        spectrum = self._process_spectrum(spec_path)

        if spectrum is None and self.generate_if_missing:
            spectrum = self._generate_spectrum_from_image(img_path, metadata)

        if spectrum is None:
            spectrum = torch.zeros(self.spectral_length, dtype=torch.float32)

        # 🔥 pega pseudo-label
        pseudo_label = self.pseudo_labels.get(spec_path, -1)

        return image, spectrum, label, pseudo_label