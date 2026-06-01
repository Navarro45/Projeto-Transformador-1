import torch.nn as nn
from torchvision import models
from .modelo_base import Modelo_Base


class ConvNeXtModel(Modelo_Base):

    def __init__(
        self,
        num_classes,
        device
    ):

        super().__init__(
            num_classes,
            device,
            model_name="convnext_tiny"
        )

        self.build_model()

    def build_model(self):

        self.model = models.convnext_tiny(
            weights="DEFAULT"
        )

        in_features = (
            self.model.classifier[2].in_features
        )

        self.classifier = nn.Linear(
            in_features,
            self.num_classes
        )

        self.model.classifier = nn.Identity()

        self.model = self.model.to(
            self.device
        )

        self.classifier = self.classifier.to(
            self.device
        )

    def forward_features(self, x):

        return self.model(x)

    def get_target_layer(self):

        return self.model.features[-1]

    def supports_gradcam(self):

        return True
