import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

def get_dataloaders(data_dir, batch_size):
    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=get_transforms(True))
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=get_transforms(False))
    test_ds  = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=get_transforms(False))

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size),
        DataLoader(test_ds, batch_size=batch_size)
    )