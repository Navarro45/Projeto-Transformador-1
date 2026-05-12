import torch.nn as nn
import torchvision.models as models
import torch

class ImageModel(nn.Module):
    def __init__(self, num_classes=3, center_focus_sigma=0.35, center_focus_strength=0.55):
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
        self.center_focus_sigma = center_focus_sigma
        self.center_focus_strength = center_focus_strength
        self.register_buffer("center_focus_mask", torch.empty(0))

    def _get_center_focus_mask(self, x):
        _, _, h, w = x.shape

        if (
            self.center_focus_mask.numel() > 0
            and self.center_focus_mask.shape[-2:] == (h, w)
            and self.center_focus_mask.device == x.device
        ):
            return self.center_focus_mask

        yy = torch.linspace(-1.0, 1.0, h, device=x.device).view(h, 1)
        xx = torch.linspace(-1.0, 1.0, w, device=x.device).view(1, w)
        distance2 = xx.pow(2) + yy.pow(2)

        gaussian = torch.exp(-distance2 / (2 * (self.center_focus_sigma ** 2)))
        gaussian = gaussian / gaussian.max()

        mask = (1.0 - self.center_focus_strength) + (self.center_focus_strength * gaussian)
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

        self.center_focus_mask = mask
        return mask

    def forward(self, x):
        # Prioriza pixels centrais sem zerar completamente as bordas.
        x = x * self._get_center_focus_mask(x)
        return self.model(x)