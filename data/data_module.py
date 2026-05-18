from torchvision import datasets, transforms
from torch.utils.data import (DataLoader, random_split,Subset)

import random


class DataModule:

    def __init__(
        self,
        train_dir,
        test_dir,
        img_size=224,
        batch_size=32,
        validation_split=0.2,

        max_train_images=None,
        max_validation_images=None,
        max_test_images=None,

        seed=42
    ):

        random.seed(seed)

        self.transform = transforms.Compose([

            transforms.Resize(
                (img_size, img_size)
            ),

            transforms.ToTensor()
        ])

        # =========================
        # DATASETS
        # =========================

        full_train_dataset = datasets.ImageFolder(
            train_dir,
            transform=self.transform
        )

        full_test_dataset = datasets.ImageFolder(
            test_dir,
            transform=self.transform
        )

        # =========================
        # SPLIT
        # =========================

        train_size = int(
            (1 - validation_split)
            * len(full_train_dataset)
        )

        val_size = (
            len(full_train_dataset)
            - train_size
        )

        train_dataset, validation_dataset = random_split(
            full_train_dataset,
            [train_size, val_size]
        )

        # =========================
        # LIMIT TRAIN
        # =========================

        if max_train_images is not None:

            indices = random.sample(
                range(len(train_dataset)),
                min(
                    max_train_images,
                    len(train_dataset)
                )
            )

            train_dataset = Subset(
                train_dataset,
                indices
            )

        # =========================
        # LIMIT VALIDATION
        # =========================

        if max_validation_images is not None:

            indices = random.sample(
                range(len(validation_dataset)),
                min(
                    max_validation_images,
                    len(validation_dataset)
                )
            )

            validation_dataset = Subset(
                validation_dataset,
                indices
            )

        # =========================
        # LIMIT TEST
        # =========================

        if max_test_images is not None:

            indices = random.sample(
                range(len(full_test_dataset)),
                min(
                    max_test_images,
                    len(full_test_dataset)
                )
            )

            full_test_dataset = Subset(
                full_test_dataset,
                indices
            )

        # =========================
        # LOADERS
        # =========================

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        self.validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        self.test_loader = DataLoader(
            full_test_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        self.class_names = (
            full_train_dataset.classes
        )

        print(
            f"Train images: {len(train_dataset)}"
        )

        print(
            f"Validation images: "
            f"{len(validation_dataset)}"
        )

        print(
            f"Test images: "
            f"{len(full_test_dataset)}"
        )