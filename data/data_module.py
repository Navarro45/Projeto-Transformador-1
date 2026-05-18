from torchvision import datasets, transforms

from torch.utils.data import (DataLoader,random_split)


class DataModule:

    def __init__(
        self,
        train_dir,
        test_dir,
        img_size=224,
        batch_size=32,
        validation_split=0.2
    ):

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

        full_train_dataset = datasets.ImageFolder(
            train_dir,
            transform=self.transform
        )

        self.test_dataset = datasets.ImageFolder(
            test_dir,
            transform=self.transform
        )

        train_size = int(
            (1 - validation_split)
            * len(full_train_dataset)
        )

        val_size = len(full_train_dataset) - train_size

        self.train_dataset, self.validation_dataset = random_split(
            full_train_dataset,
            [train_size, val_size]
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        self.validation_loader = DataLoader(
            self.validation_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        self.class_names = (
            full_train_dataset.classes
        )