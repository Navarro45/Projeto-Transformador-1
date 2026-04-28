import torch.nn as nn
import torchvision.models as models

class ImageModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.model = models.resnet18(weights="IMAGENET1K_V1")

        # congelar tudo
        for param in self.model.parameters():
            param.requires_grad = False

        # liberar apenas última camada convolucional
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        in_features = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)